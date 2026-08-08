"""Chunked GLB conversion: global remesh + per-chunk bake + multi-primitive GLB."""

from __future__ import annotations

import gc
import math
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import cumesh
import numpy as np
import o_voxel.postprocess as op
import torch
import trimesh

from genrecon.vendor.cumesh_remeshing import remesh_narrow_band_dc


def _release_cuda_memory() -> None:
    """Release dead Python/CUDA allocations at an explicit pipeline boundary.

    CuMesh and cuBVH allocate outside PyTorch's caching allocator, while O-Voxel
    also creates large temporary PyTorch tensors.  Without an explicit collection
    point, freed PyTorch blocks can stay reserved and starve a later CuMesh
    ``cudaMalloc`` even though no live tensor needs those blocks anymore.
    """

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _export_scene_atomic(scene: trimesh.Scene, destination: Path) -> None:
    """Export a GLB cache entry atomically so interrupted writes are never resumed."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.stem}.",
        suffix=destination.suffix,
        dir=destination.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        scene.export(str(temporary))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _feature_locked_mask(vertices: torch.Tensor, faces: torch.Tensor, feature_angle_deg: float) -> torch.Tensor:
    """Return a (V,) bool mask of vertices that should be held fixed during smoothing.

    A vertex is locked if it touches a "feature" edge: a boundary edge, a non-manifold
    edge, or an edge whose dihedral angle exceeds ``feature_angle_deg``. Smoothing the
    rest leaves crease/sharp geometry intact while flat regions get denoised.
    """
    V = vertices.shape[0]
    F_count = faces.shape[0]
    f_long = faces.long()

    v0 = vertices[f_long[:, 0]]
    v1 = vertices[f_long[:, 1]]
    v2 = vertices[f_long[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)
    fn = fn / (fn.norm(dim=-1, keepdim=True) + 1e-12)

    # Canonical (sorted-pair) edge keys, three per face.
    e_per_face = torch.cat([f_long[:, [0, 1]], f_long[:, [1, 2]], f_long[:, [2, 0]]], dim=0)
    e_min = e_per_face.min(dim=1).values
    e_max = e_per_face.max(dim=1).values
    edge_keys = e_min.long() * V + e_max.long()
    face_per_edge = torch.arange(F_count, device=faces.device).repeat(3)

    unique_keys, inverse, counts = torch.unique(edge_keys, return_inverse=True, return_counts=True)
    n_edges = unique_keys.shape[0]

    # Pair faces sharing each edge: sort by inverse, take positions 0/1 within each group.
    sort_idx = torch.argsort(inverse)
    sorted_inv = inverse[sort_idx]
    sorted_face = face_per_edge[sort_idx]
    cum_starts = torch.cumsum(counts, dim=0) - counts
    pos_in_group = torch.arange(sorted_face.shape[0], device=faces.device) - cum_starts[sorted_inv]

    face_pair = torch.full((n_edges, 2), -1, dtype=torch.long, device=faces.device)
    m0 = pos_in_group == 0
    m1 = pos_in_group == 1
    face_pair[sorted_inv[m0], 0] = sorted_face[m0]
    face_pair[sorted_inv[m1], 1] = sorted_face[m1]

    boundary_edge = counts == 1
    nonmanifold_edge = counts > 2
    interior_edge = counts == 2

    cos_thresh = math.cos(math.radians(feature_angle_deg))
    n0 = fn[face_pair[:, 0].clamp(min=0)]
    n1 = fn[face_pair[:, 1].clamp(min=0)]
    cos_dihedral = (n0 * n1).sum(dim=-1).clamp(-1.0, 1.0)

    feature_edge = boundary_edge | nonmanifold_edge
    feature_edge[interior_edge] |= cos_dihedral[interior_edge] < cos_thresh

    feat_keys = unique_keys[feature_edge]
    feat_a = (feat_keys // V).long()
    feat_b = (feat_keys % V).long()
    locked = torch.zeros(V, dtype=torch.bool, device=vertices.device)
    locked[feat_a] = True
    locked[feat_b] = True
    return locked


def _vertex_neighbors(faces: torch.Tensor, V: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a directed neighbor list for the uniform graph Laplacian.

    Returns ``(src, dst, counts)``: edges go ``src → dst`` (one per ordered pair, deduped),
    and ``counts[v]`` is the number of distinct neighbors of vertex ``v``.
    """
    f_long = faces.long()
    pairs = torch.cat(
        [
            f_long[:, [0, 1]],
            f_long[:, [1, 0]],
            f_long[:, [1, 2]],
            f_long[:, [2, 1]],
            f_long[:, [2, 0]],
            f_long[:, [0, 2]],
        ],
        dim=0,
    )
    keys = pairs[:, 0] * V + pairs[:, 1]
    keys = torch.unique(keys)
    src = (keys // V).long()
    dst = (keys % V).long()
    counts = torch.zeros(V, device=faces.device, dtype=torch.long)
    counts.scatter_add_(0, src, torch.ones_like(src))
    return src, dst, counts


def _smoothing_step(
    vertices: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    counts: torch.Tensor,
    factor: float,
    locked: torch.Tensor,
) -> torch.Tensor:
    """Single uniform-Laplacian step: ``v += factor * (mean(neighbors) - v)`` for unlocked vertices."""
    nb_sum = torch.zeros_like(vertices)
    idx3 = src.unsqueeze(-1).expand(-1, 3)
    nb_sum.scatter_add_(0, idx3, vertices[dst])
    mean_nb = nb_sum / counts.clamp(min=1).unsqueeze(-1)
    delta = factor * (mean_nb - vertices)
    delta = torch.where(locked.unsqueeze(-1), torch.zeros_like(delta), delta)
    return vertices + delta


def feature_preserving_taubin(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    iterations: int,
    lam: float,
    mu: float,
    feature_angle_deg: float,
    verbose: bool = False,
) -> torch.Tensor:
    """Taubin λ-μ smoothing that locks vertices on sharp / boundary edges.

    Smooths flat regions (denoising the SLat-decoder per-voxel quilt) while preserving
    creases, corners, and mesh boundaries. Topology unchanged; only positions move.
    """
    V = vertices.shape[0]
    locked = _feature_locked_mask(vertices, faces, feature_angle_deg)
    src, dst, counts = _vertex_neighbors(faces, V)
    if verbose:
        print(
            f"[smooth] {locked.sum().item():,}/{V:,} vertices locked "
            f"(feature_angle={feature_angle_deg}°), "
            f"iters={iterations}, lam={lam}, mu={mu}"
        )
    for _ in range(iterations):
        vertices = _smoothing_step(vertices, src, dst, counts, lam, locked)
        vertices = _smoothing_step(vertices, src, dst, counts, mu, locked)
    return vertices


def _to_tensor(x: Any, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    if isinstance(x, np.ndarray):
        return torch.tensor(x, device=device, dtype=dtype)
    return torch.tensor(np.asarray(x), device=device, dtype=dtype)


def _voxel_size_scalar(voxel_size: Any) -> float:
    if isinstance(voxel_size, (int, float)):
        return float(voxel_size)
    if isinstance(voxel_size, torch.Tensor):
        return float(voxel_size.max().item()) if voxel_size.dim() else float(voxel_size.item())
    arr = np.asarray(voxel_size)
    return float(arr.max())


def _project_back_in_batches(
    vertices: torch.Tensor,
    source_vertices: torch.Tensor,
    source_faces: torch.Tensor,
    bvh: Any,
    amount: float,
    *,
    batch_size: int = 250_000,
) -> torch.Tensor:
    """Apply CuMesh's barycentric project-back without a scene-sized gather.

    ``remesh_narrow_band_dc`` normally performs this operation before returning,
    while all of its sparse-grid and topology temporaries are still alive.  At
    scene scale, ``source_vertices[source_faces[face_id]]`` alone can require
    hundreds of MiB.  Running the identical per-vertex expression after remesh
    returns retires those temporaries, and batching bounds the gather size.
    """

    if amount <= 0 or vertices.shape[0] == 0:
        return vertices
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    with torch.no_grad():
        for start in range(0, vertices.shape[0], batch_size):
            stop = min(start + batch_size, vertices.shape[0])
            vertex_batch = vertices[start:stop]
            distances, face_id, uvw = bvh.unsigned_distance(vertex_batch, return_uvw=True)
            source_triangles = source_vertices[source_faces[face_id.long()]]
            projected = (source_triangles * uvw.unsqueeze(-1)).sum(dim=1)
            vertex_batch -= amount * (vertex_batch - projected)
            del distances, face_id, uvw, source_triangles, projected, vertex_batch

    return vertices


def _assign_faces_to_nearest_chunk(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    chunk_centers: torch.Tensor,
    *,
    batch_size: int = 250_000,
    verbose: bool = False,
) -> torch.Tensor:
    """Assign faces by centroid without materializing a scene-sized distance matrix."""

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if chunk_centers.ndim != 2 or chunk_centers.shape[0] == 0 or chunk_centers.shape[1] != 3:
        raise ValueError("chunk_centers must have shape (N, 3) with N > 0")

    num_faces = faces.shape[0]
    assignment = torch.empty(num_faces, dtype=torch.long, device=vertices.device)
    num_batches = math.ceil(num_faces / batch_size) if num_faces else 0
    progress_interval = max(1, num_batches // 10) if num_batches else 1

    with torch.no_grad():
        for batch_index, start in enumerate(range(0, num_faces, batch_size), start=1):
            stop = min(start + batch_size, num_faces)
            face_batch = faces[start:stop].long()
            centroids = vertices[face_batch].mean(dim=1)
            distances = torch.cdist(centroids, chunk_centers)
            assignment[start:stop] = distances.argmin(dim=1)
            del face_batch, centroids, distances

            if verbose and (batch_index % progress_interval == 0 or batch_index == num_batches):
                print(
                    "[chunked_to_glb] face assignment batches "
                    f"{batch_index}/{num_batches}"
                )

    return assignment


def chunked_to_glb(
    vertices_world: torch.Tensor,
    faces: torch.Tensor,
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: dict,
    aabb_world: Any,
    voxel_size_world: Any,
    chunk_centers_world: torch.Tensor,
    chunk_size_world: float,
    chunk_indices: Sequence[int],
    *,
    do_fill_holes: bool = True,
    hole_perim: float = 1.0,
    do_remesh: bool = True,
    remesh_res: int | None = None,
    remesh_band: float = 1.0,
    remesh_project: float = 0.9,
    smooth_iters: int = 0,
    smooth_lambda: float = 0.5,
    smooth_mu: float = -0.53,
    smooth_feature_angle: float = 25.0,
    simplify_threshold: int = 5_000_000,
    texture_size: int = 4096,
    dump_geometry_dir: str | Path | None = None,
    chunks_save_dir: str | Path | None = None,
    verbose: bool = True,
) -> trimesh.Scene:
    """Global remesh, partition by face centroid, per-chunk bake, combine into one GLB.

    All inputs in world frame. attr_volume + coords + aabb_world + voxel_size_world define
    the global sparse attribute grid; passed through unchanged to per-chunk to_glb calls.

    If chunks_save_dir is given, every successfully-baked chunk is written there as
    chunk_NNN.glb immediately after bake. On a subsequent run, any chunk whose GLB
    already exists at that path is loaded from disk instead of re-baked, making the
    per-scene pipeline resumable across crashes (e.g. VRAM OOM on a single chunk).
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vertices_world = vertices_world.to(device, dtype=torch.float32)
    faces = faces.to(device)
    face_dtype = faces.dtype
    # Attribute tensors are unused during the memory-heavy global remesh.  Keep
    # them on their input device (CPU for the resumable CLI) until chunk baking.
    chunk_centers_world = chunk_centers_world.to(device, dtype=torch.float32)

    aabb_t = _to_tensor(aabb_world, device)
    assert aabb_t.shape == (2, 3), f"aabb_world must be (2,3), got {aabb_t.shape}"

    # ── Step 1: Global cleanup + remesh ─────────────────────────────────
    if verbose:
        print(f"[chunked_to_glb] input: {vertices_world.shape[0]:,} verts, {faces.shape[0]:,} faces")

    t0 = time.perf_counter()

    if do_fill_holes:
        pre_mesh = cumesh.CuMesh()
        pre_mesh.init(vertices_world, faces)
        pre_mesh.fill_holes(max_hole_perimeter=hole_perim)
        if verbose:
            print(
                f"[chunked_to_glb] after global fill_holes: "
                f"{pre_mesh.num_vertices:,} verts, {pre_mesh.num_faces:,} faces"
            )
        pre_v, pre_f = pre_mesh.read()
        # CuMesh now owns the filled generation; the original CUDA inputs are
        # no longer needed during remeshing or projection.
        del vertices_world, faces
        _release_cuda_memory()
    else:
        if verbose:
            print("[chunked_to_glb] skipping fill_holes")
        pre_v, pre_f = vertices_world, faces

    if do_remesh:
        bvh = cumesh.cuBVH(pre_v, pre_f)

        if remesh_res is None:
            extent = float((aabb_t[1] - aabb_t[0]).max().item())
            vs_scalar = _voxel_size_scalar(voxel_size_world)
            remesh_res = int(round(extent / vs_scalar))
        if verbose:
            print(
                f"[chunked_to_glb] remeshing at resolution={remesh_res}, band={remesh_band}, project={remesh_project}"
            )

        center = aabb_t.mean(dim=0)
        scale = float((aabb_t[1] - aabb_t[0]).max().item())

        # Use the repository-owned, memory-bounded derivative of the pinned
        # CuMesh implementation.  This must not depend on a hand-edited copy in
        # site-packages; fresh environments need the same scene-scale behavior.
        new_v, new_f = remesh_narrow_band_dc(
            pre_v,
            pre_f,
            center=center,
            scale=(remesh_res + 3 * remesh_band) / remesh_res * scale,
            resolution=remesh_res,
            band=remesh_band,
            # Project after this function returns so its sparse-grid/topology
            # temporaries are gone, then use bounded barycentric gathers.
            project_back=0.0,
            verbose=verbose,
            bvh=bvh,
        )

        _release_cuda_memory()
        if remesh_project > 0:
            if verbose:
                print(
                    "[chunked_to_glb] projecting remesh back to source "
                    "in 250,000-vertex batches"
                )
            new_v = _project_back_in_batches(
                new_v,
                pre_v,
                pre_f,
                bvh,
                remesh_project,
            )

        # ``new_v``/``new_f`` own the remesh result.  Retire every input-side
        # allocation before constructing the output CuMesh: at scene scale the
        # old fill-hole mesh + BVH + source tensors and the new mesh cannot all
        # coexist within 16 GiB, even though either generation fits by itself.
        del bvh, center
        if do_fill_holes:
            pre_mesh.clear_cache()
            del pre_mesh
        del pre_v, pre_f
        if not do_fill_holes:
            del vertices_world, faces
        _release_cuda_memory()

        mesh = cumesh.CuMesh()
        mesh.init(new_v, new_f)
        if verbose:
            print(f"[chunked_to_glb] after remesh: {mesh.num_vertices:,} verts, {mesh.num_faces:,} faces")
        # DC remesh produces a clean manifold; o_voxel.to_glb's remesh branch also skips
        # post-remesh cleanup. Per-chunk to_glb(remesh=False) will run its own local cleanup.
        gv, gf = mesh.read()
        mesh.clear_cache()
        del mesh, new_v, new_f
        _release_cuda_memory()
    else:
        if verbose:
            print("[chunked_to_glb] skipping remesh")
        gv, gf = pre_v, pre_f
        # With remeshing disabled, ``gv``/``gf`` keep the selected generation
        # alive while the redundant owners and aliases can be discarded.
        if do_fill_holes:
            pre_mesh.clear_cache()
            del pre_mesh
            del pre_v, pre_f
        else:
            del pre_v, pre_f, vertices_world, faces
        _release_cuda_memory()

    elapsed_remesh = time.perf_counter() - t0
    if verbose:
        print(f"[chunked_to_glb] global stage: {elapsed_remesh:.1f}s")

    # ── Step 1.5: Optional feature-preserving Taubin smoothing ──────────
    if smooth_iters > 0:
        t_smooth = time.perf_counter()
        gv = feature_preserving_taubin(
            gv.contiguous(),
            gf.contiguous(),
            iterations=smooth_iters,
            lam=smooth_lambda,
            mu=smooth_mu,
            feature_angle_deg=smooth_feature_angle,
            verbose=verbose,
        )
        if verbose:
            print(f"[chunked_to_glb] smoothing: {time.perf_counter() - t_smooth:.1f}s")

    # ── Optional debug dump: write the global pre-bake mesh as a flat PLY ──
    if dump_geometry_dir is not None:
        dump_dir = Path(dump_geometry_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        global_path = dump_dir / "global_mesh.ply"
        trimesh.Trimesh(
            vertices=gv.detach().cpu().numpy(),
            faces=gf.detach().cpu().numpy(),
            process=False,
        ).export(str(global_path))
        if verbose:
            print(f"[chunked_to_glb] dumped {global_path} " f"({gv.shape[0]:,} verts, {gf.shape[0]:,} faces)")

    # ── Step 2: Spatial partition by face centroid (nearest chunk center) ─
    assignment = _assign_faces_to_nearest_chunk(
        gv,
        gf,
        chunk_centers_world,
        verbose=verbose,
    )  # (F,) → index into chunk_indices
    if verbose:
        for k, idx in enumerate(chunk_indices):
            print(f"[chunked_to_glb] chunk {idx:03d}: {(assignment == k).sum().item():,} faces")
    # Retire temporary allocator blocks from the bounded assignment batches
    # before moving the attribute volumes onto the GPU.
    _release_cuda_memory()

    # These tensors are first consumed by O-Voxel's per-chunk texture bake.
    attr_volume = attr_volume.to(device)
    coords = coords.to(device)

    # ── Step 3 + 4: Per-chunk bake + combine into trimesh.Scene ─────────
    scene = trimesh.Scene()
    n_verts = gv.shape[0]

    if chunks_save_dir is not None:
        chunks_save_dir = Path(chunks_save_dir)
        chunks_save_dir.mkdir(parents=True, exist_ok=True)

    def _add_loaded_to_scene(loaded: Any, chunk_idx: int) -> None:
        """trimesh.load returns a Scene with one or more geometries; preserve them."""
        geoms = list(loaded.geometry.items()) if hasattr(loaded, "geometry") else [(None, loaded)]
        for geom_name, geom in geoms:
            new_name = f"chunk_{chunk_idx:03d}" if len(geoms) == 1 else f"chunk_{chunk_idx:03d}_{geom_name}"
            scene.add_geometry(geom, geom_name=new_name)

    for k, chunk_idx in enumerate(chunk_indices):
        face_mask = assignment == k
        n_chunk_faces = int(face_mask.sum().item())
        if n_chunk_faces == 0:
            if verbose:
                print(f"[chunked_to_glb] chunk {chunk_idx:03d}: empty, skipping")
            del face_mask
            _release_cuda_memory()
            continue

        # Resume: if a per-chunk GLB exists from a prior run, load it and skip bake.
        chunk_path = chunks_save_dir / f"chunk_{chunk_idx:03d}.glb" if chunks_save_dir is not None else None
        if chunk_path is not None and chunk_path.exists() and dump_geometry_dir is None:
            if verbose:
                print(f"[chunked_to_glb] chunk {chunk_idx:03d}: resume from {chunk_path}")
            try:
                _add_loaded_to_scene(trimesh.load(str(chunk_path), force="scene"), chunk_idx)
                del face_mask
                _release_cuda_memory()
                continue
            except Exception as e:
                print(f"[chunked_to_glb] chunk {chunk_idx:03d}: WARN failed to load cached GLB ({e}); re-baking")

        chunk_faces_global = gf[face_mask].long()
        used_v_ids = torch.unique(chunk_faces_global.flatten())
        remap = torch.full((n_verts,), -1, dtype=torch.long, device=gv.device)
        remap[used_v_ids] = torch.arange(used_v_ids.shape[0], device=gv.device)
        chunk_v = gv[used_v_ids].contiguous()
        chunk_f = remap[chunk_faces_global].to(face_dtype).contiguous()

        # Only compact chunk geometry is required by O-Voxel.  Drop the global
        # remap and selection temporaries before its CuMesh/BVH allocations.
        del face_mask, chunk_faces_global, used_v_ids, remap
        _release_cuda_memory()

        # Dump-only path: write the per-chunk pre-bake mesh and skip op.to_glb.
        if dump_geometry_dir is not None:
            chunk_path = Path(dump_geometry_dir) / f"chunk_{chunk_idx:03d}_geom.ply"
            trimesh.Trimesh(
                vertices=chunk_v.detach().cpu().numpy(),
                faces=chunk_f.detach().cpu().numpy(),
                process=False,
            ).export(str(chunk_path))
            if verbose:
                print(
                    f"[chunked_to_glb] dumped {chunk_path} " f"({chunk_v.shape[0]:,} verts, {chunk_f.shape[0]:,} faces)"
                )
            del chunk_v, chunk_f
            _release_cuda_memory()
            continue

        if n_chunk_faces > simplify_threshold:
            decimation_target = simplify_threshold
        else:
            decimation_target = n_chunk_faces

        if verbose:
            print(
                f"\n[chunked_to_glb] baking chunk {chunk_idx:03d}: "
                f"{chunk_v.shape[0]:,} verts, {chunk_f.shape[0]:,} faces, "
                f"decimation_target={decimation_target:,}"
            )

        t1 = time.perf_counter()
        chunk_glb = op.to_glb(
            vertices=chunk_v,
            faces=chunk_f,
            attr_volume=attr_volume,
            coords=coords,
            attr_layout=attr_layout,
            aabb=aabb_t,
            voxel_size=voxel_size_world,
            decimation_target=decimation_target,
            texture_size=texture_size,
            remesh=False,
            verbose=verbose,
        )
        elapsed_chunk = time.perf_counter() - t1
        if verbose:
            print(f"[chunked_to_glb] chunk {chunk_idx:03d}: baked in {elapsed_chunk:.1f}s")

        # Persist each chunk to disk before adding to the combined scene, so a crash
        # on a later chunk doesn't lose this one.
        if chunk_path is not None:
            try:
                export_obj = chunk_glb if isinstance(chunk_glb, trimesh.Scene) else trimesh.Scene([chunk_glb])
                _export_scene_atomic(export_obj, chunk_path)
                if verbose:
                    print(f"[chunked_to_glb] chunk {chunk_idx:03d}: saved {chunk_path}")
            except Exception as e:
                print(f"[chunked_to_glb] chunk {chunk_idx:03d}: WARN failed to save {chunk_path}: {e}")

        scene.add_geometry(chunk_glb, geom_name=f"chunk_{chunk_idx:03d}")
        del chunk_glb, chunk_v, chunk_f
        _release_cuda_memory()

    return scene
