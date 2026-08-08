"""Memory-bounded CuMesh narrow-band dual-contouring remesh.

This module is a repository-owned derivative of ``cumesh/remeshing.py`` from
CuMesh commit ``12289e1062f0603f2f0d0771b02e1395d247f26f``.  CuMesh is MIT
licensed; its notice is retained in ``genrecon/vendor/cumesh-LICENSE``.

The geometry equations and split decisions match the pinned upstream
implementation.  The differences are allocation lifetime only:

* UDF queries and hash insertions use bounded batches;
* the voxel hashmap is released during vertex dual contouring and rebuilt for
  topology instead of overlapping both scene-sized hashmaps;
* topology generation, vertex-index compaction, and quad split selection use
  bounded accelerator batches while full integer topology stays on CPU; and
* the final triangle tensor is transferred to the accelerator only once.

Keeping this implementation in the repository is intentional.  A hand-edited
``site-packages/cumesh/remeshing.py`` is not captured by ``pip freeze`` and
therefore cannot reproduce the 16-GiB reconstruction path on a fresh machine.
"""

from __future__ import annotations

from typing import Any

import torch
from cumesh import _C
from cumesh.bvh import cuBVH
from tqdm import tqdm


CUMESH_COMMIT = "12289e1062f0603f2f0d0771b02e1395d247f26f"
TOPOLOGY_BATCH_SIZE = 250_000
SPARSE_BATCH_SIZE = 1_000_000


def _init_hashmap(
    resolution: int,
    capacity: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    volume = resolution * resolution * resolution

    if volume < 2**32:
        key_dtype = torch.uint32
    elif volume < 2**64:
        key_dtype = torch.uint64
    else:
        raise ValueError(f"The spatial size is too large to fit in a hashmap: {volume} > 2^64")

    hashmap_keys = torch.full(
        (capacity,),
        torch.iinfo(key_dtype).max,
        dtype=key_dtype,
        device=device,
    )
    hashmap_values = torch.empty((capacity,), dtype=torch.uint32, device=device)
    return hashmap_keys, hashmap_values


def _release_device_memory(device: torch.device) -> None:
    """Finish queued work before releasing cached CUDA allocations."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def _insert_indexed_coords_batched(
    hashmap: tuple[torch.Tensor, torch.Tensor],
    coords: torch.Tensor,
    resolution: int,
    *,
    batch_size: int = SPARSE_BATCH_SIZE,
) -> None:
    """Insert ``coords -> global row index`` without a scene-sized padded copy.

    CuMesh's convenience entry point derives values from the launch-local row
    index.  Calling it on slices would therefore store incorrect indices.  Use
    the explicit-value entry point so every batch stores the same global value
    as the original single launch.
    """

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.dtype != torch.int32:
        raise ValueError(f"coords must be int32 with shape (N, 3), got {coords.dtype} {tuple(coords.shape)}")
    if coords.shape[0] >= 2**32:
        raise ValueError("CuMesh's uint32 hashmap values support fewer than 2^32 coordinates")

    for start in range(0, coords.shape[0], batch_size):
        stop = min(start + batch_size, coords.shape[0])
        padded_coords = torch.empty(
            (stop - start, 4),
            dtype=torch.int32,
            device=coords.device,
        )
        padded_coords[:, 0] = 0
        padded_coords[:, 1:] = coords[start:stop]
        global_indices = torch.arange(
            start,
            stop,
            dtype=torch.int64,
            device=coords.device,
        ).to(torch.uint32)
        _C.hashmap_insert_3d_cuda(
            *hashmap,
            padded_coords,
            global_indices,
            resolution,
            resolution,
            resolution,
        )
        del padded_coords, global_indices


def _grid_distances_batched(
    grid_vertices: torch.Tensor,
    bvh: Any,
    center: torch.Tensor,
    scale: float,
    resolution: int,
    epsilon: float,
    *,
    batch_size: int = SPARSE_BATCH_SIZE,
) -> torch.Tensor:
    """Evaluate the vertex UDF without materializing all query points."""

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    distances = torch.empty(
        (grid_vertices.shape[0],),
        dtype=torch.float32,
        device=grid_vertices.device,
    )
    for start in range(0, grid_vertices.shape[0], batch_size):
        stop = min(start + batch_size, grid_vertices.shape[0])
        points = (grid_vertices[start:stop].float() / resolution - 0.5) * scale + center
        batch_distances = bvh.unsigned_distance(points)[0]
        distances[start:stop] = batch_distances
        del points, batch_distances
    distances.sub_(epsilon)
    return distances


def _build_topology_batched(
    coords: torch.Tensor,
    intersected_cpu: torch.Tensor,
    hashmap_voxels: tuple[torch.Tensor, torch.Tensor],
    edge_neighbor_voxel_offset: torch.Tensor,
    resolution: int,
    *,
    batch_size: int = TOPOLOGY_BATCH_SIZE,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate valid quads in bounded accelerator batches.

    Output order is the original row-major voxel/direction order.  Integer
    results are copied to CPU immediately, preventing the upstream
    ``(N, 3, 4, 3)`` neighbor tensor from becoming scene-sized on the GPU.
    """

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if intersected_cpu.device.type != "cpu":
        raise ValueError("intersected_cpu must reside on CPU")
    if intersected_cpu.shape != (coords.shape[0], 3):
        raise ValueError(
            f"intersected_cpu must have shape {(coords.shape[0], 3)}, "
            f"got {tuple(intersected_cpu.shape)}"
        )

    device = coords.device
    quad_batches: list[torch.Tensor] = []
    direction_batches: list[torch.Tensor] = []
    total_batches = (coords.shape[0] + batch_size - 1) // batch_size
    report_every = max(1, total_batches // 10)
    valid_quad_count = 0
    for batch_index, start in enumerate(range(0, coords.shape[0], batch_size), start=1):
        stop = min(start + batch_size, coords.shape[0])
        coords_batch = coords[start:stop]
        intersected_batch = intersected_cpu[start:stop].to(device)
        edge_neighbor_voxels = coords_batch.reshape(-1, 1, 1, 3) + edge_neighbor_voxel_offset
        intersection_mask = intersected_batch != 0
        connected_voxels = edge_neighbor_voxels[intersection_mask]
        connected_directions = intersected_batch[intersection_mask]
        connected_count = connected_voxels.shape[0]

        if connected_count:
            connected_keys = torch.empty(
                (connected_count * 4, 4),
                dtype=torch.int32,
                device=device,
            )
            connected_keys[:, 0] = 0
            connected_keys[:, 1:] = connected_voxels.reshape(-1, 3)
            connected_indices = _C.hashmap_lookup_3d_cuda(
                *hashmap_voxels,
                connected_keys,
                resolution,
                resolution,
                resolution,
            ).reshape(connected_count, 4).int()
            connected_valid = (connected_indices != 0xFFFFFFFF).all(dim=1)
            quad_batch = connected_indices[connected_valid].int().cpu()
            direction_batch = connected_directions[connected_valid].int().cpu()
            if quad_batch.shape[0]:
                quad_batches.append(quad_batch)
                direction_batches.append(direction_batch)
                valid_quad_count += quad_batch.shape[0]
            del connected_keys, connected_indices, connected_valid

        del coords_batch, intersected_batch, edge_neighbor_voxels
        del intersection_mask, connected_voxels, connected_directions
        if verbose and (batch_index % report_every == 0 or batch_index == total_batches):
            print(
                f"[cumesh-low-memory] topology batches {batch_index}/{total_batches}, "
                f"valid quads={valid_quad_count:,}",
                flush=True,
            )

    if not quad_batches:
        return (
            torch.empty((0, 4), dtype=torch.int32),
            torch.empty((0,), dtype=torch.int32),
        )
    return torch.cat(quad_batches, dim=0), torch.cat(direction_batches, dim=0)


def _remap_quads_batched(
    quad_indices: torch.Tensor,
    unique_vertices: torch.Tensor,
    voxel_count: int,
    *,
    batch_size: int = TOPOLOGY_BATCH_SIZE,
) -> torch.Tensor:
    """Remap quads in place without allocating a second full index tensor."""

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    vertex_map = torch.zeros(
        (voxel_count,),
        dtype=torch.int32,
        device=quad_indices.device,
    )
    vertex_map[unique_vertices] = torch.arange(
        unique_vertices.shape[0],
        dtype=torch.int32,
        device=quad_indices.device,
    )
    for start in range(0, quad_indices.shape[0], batch_size):
        stop = min(start + batch_size, quad_indices.shape[0])
        quad_indices[start:stop] = vertex_map[quad_indices[start:stop]]
    return quad_indices


def _unique_referenced_vertices(
    quad_indices_cpu: torch.Tensor,
    voxel_count: int,
    *,
    batch_size: int = TOPOLOGY_BATCH_SIZE,
) -> torch.Tensor:
    """Return sorted referenced indices without sorting all flattened quads."""

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if quad_indices_cpu.device.type != "cpu":
        raise ValueError("quad_indices_cpu must reside on CPU")
    referenced = torch.zeros((voxel_count,), dtype=torch.bool, device="cpu")
    for start in range(0, quad_indices_cpu.shape[0], batch_size):
        stop = min(start + batch_size, quad_indices_cpu.shape[0])
        referenced[quad_indices_cpu[start:stop].long()] = True
    return torch.nonzero(referenced, as_tuple=False).flatten().to(torch.int32)


def _quad_split_tables(device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        torch.tensor([0, 1, 2, 0, 2, 3], dtype=torch.long, device=device),
        torch.tensor([0, 2, 1, 0, 3, 2], dtype=torch.long, device=device),
        torch.tensor([0, 1, 3, 3, 1, 2], dtype=torch.long, device=device),
        torch.tensor([0, 3, 1, 3, 2, 1], dtype=torch.long, device=device),
    )


def _triangulate_quads_batched(
    mesh_vertices: torch.Tensor,
    quad_indices_cpu: torch.Tensor,
    intersected_directions_cpu: torch.Tensor,
    *,
    batch_size: int = TOPOLOGY_BATCH_SIZE,
) -> torch.Tensor:
    """Choose the same diagonal as CuMesh while bounding accelerator memory.

    ``quad_indices_cpu`` and ``intersected_directions_cpu`` deliberately remain
    on CPU.  Each batch is moved to the mesh device, evaluated with the pinned
    upstream split formula, and copied back into a preallocated CPU tensor.
    """

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if quad_indices_cpu.device.type != "cpu" or intersected_directions_cpu.device.type != "cpu":
        raise ValueError("quad topology inputs must remain on CPU")
    if quad_indices_cpu.ndim != 2 or quad_indices_cpu.shape[1] != 4:
        raise ValueError(f"quad_indices_cpu must have shape (N, 4), got {tuple(quad_indices_cpu.shape)}")
    if intersected_directions_cpu.shape != (quad_indices_cpu.shape[0],):
        raise ValueError(
            "intersected_directions_cpu must have one entry per quad, got "
            f"{tuple(intersected_directions_cpu.shape)}"
        )

    device = mesh_vertices.device
    split_1_n, split_1_p, split_2_n, split_2_p = _quad_split_tables(device)
    quad_count = quad_indices_cpu.shape[0]
    triangles_cpu = torch.empty((quad_count * 2, 3), dtype=torch.int32, device="cpu")

    for start in range(0, quad_count, batch_size):
        stop = min(start + batch_size, quad_count)
        quad_batch = quad_indices_cpu[start:stop].to(device)
        direction_batch = intersected_directions_cpu[start:stop].to(device)

        attempt_0 = torch.where(
            (direction_batch == 1).unsqueeze(1),
            quad_batch[:, split_1_p],
            quad_batch[:, split_1_n],
        )
        normals_0a = torch.cross(
            mesh_vertices[attempt_0[:, 1]] - mesh_vertices[attempt_0[:, 0]],
            mesh_vertices[attempt_0[:, 2]] - mesh_vertices[attempt_0[:, 0]],
            dim=-1,
        )
        normals_0b = torch.cross(
            mesh_vertices[attempt_0[:, 2]] - mesh_vertices[attempt_0[:, 1]],
            mesh_vertices[attempt_0[:, 3]] - mesh_vertices[attempt_0[:, 1]],
            dim=-1,
        )
        alignment_0 = (normals_0a * normals_0b).sum(dim=1).abs()

        attempt_1 = torch.where(
            (direction_batch == 1).unsqueeze(1),
            quad_batch[:, split_2_p],
            quad_batch[:, split_2_n],
        )
        normals_1a = torch.cross(
            mesh_vertices[attempt_1[:, 1]] - mesh_vertices[attempt_1[:, 0]],
            mesh_vertices[attempt_1[:, 2]] - mesh_vertices[attempt_1[:, 0]],
            dim=-1,
        )
        normals_1b = torch.cross(
            mesh_vertices[attempt_1[:, 2]] - mesh_vertices[attempt_1[:, 1]],
            mesh_vertices[attempt_1[:, 3]] - mesh_vertices[attempt_1[:, 1]],
            dim=-1,
        )
        alignment_1 = (normals_1a * normals_1b).sum(dim=1).abs()

        selected = torch.where(
            (alignment_0 > alignment_1).unsqueeze(1),
            attempt_0,
            attempt_1,
        )
        triangles_cpu[2 * start : 2 * stop] = selected.reshape(-1, 3).cpu()
        del quad_batch, direction_batch, attempt_0, attempt_1
        del normals_0a, normals_0b, normals_1a, normals_1b
        del alignment_0, alignment_1, selected

    return triangles_cpu.to(device)


def remesh_narrow_band_dc(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    center: torch.Tensor,
    scale: float,
    resolution: int,
    band: float = 1,
    project_back: float = 0,
    verbose: bool = False,
    bvh: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remesh with CuMesh narrow-band UDF/DC using bounded topology buffers."""

    assert vertices.ndim == 2 and vertices.shape[1] == 3 and vertices.dtype == torch.float32
    assert faces.ndim == 2 and faces.shape[1] == 3 and faces.dtype == torch.int32
    assert center.ndim == 1 and center.shape[0] == 3

    device = vertices.device
    edge_neighbor_voxel_offset = torch.tensor(
        [
            [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
            [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
        ],
        dtype=torch.int32,
        device=device,
    ).unsqueeze(0)
    offsets = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=torch.int32,
        device=device,
    )

    if bvh is None:
        if verbose:
            print("Building BVH...")
        bvh = cuBVH(vertices, faces)

    epsilon = band * scale / resolution
    base_resolution = resolution
    while base_resolution > 32:
        assert base_resolution % 2 == 0, "Failed to find a base resolution that is a multiple of 2"
        base_resolution //= 2

    coords = torch.stack(
        torch.meshgrid(
            torch.arange(base_resolution, device=device),
            torch.arange(base_resolution, device=device),
            torch.arange(base_resolution, device=device),
            indexing="ij",
        ),
        dim=-1,
    ).int().reshape(-1, 3)

    progress = tqdm(
        total=int(torch.log2(torch.tensor(resolution // base_resolution)).item()) + 1,
        desc="Building Sparse Grid",
        disable=not verbose,
    )

    while True:
        cell_size = scale / base_resolution
        points = ((coords.float() + 0.5) / base_resolution - 0.5) * scale + center
        distances = bvh.unsigned_distance(points)[0]
        distances -= epsilon
        distances = torch.abs(distances)
        subdivision_mask = distances < 0.87 * cell_size
        coords = coords[subdivision_mask]

        if base_resolution >= resolution:
            break

        base_resolution *= 2
        coords *= 2
        coords = (coords.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1, 3)
        progress.update(1)

    del points, distances, subdivision_mask, offsets
    _release_device_memory(device)

    voxel_count = coords.shape[0]
    hashmap_voxels = _init_hashmap(resolution, 2 * voxel_count, device)
    _insert_indexed_coords_batched(
        hashmap_voxels,
        coords,
        resolution,
    )

    coords = coords.contiguous()
    grid_vertices = _C.get_sparse_voxel_grid_active_vertices(
        *hashmap_voxels,
        coords,
        resolution,
        resolution,
        resolution,
    )
    grid_vertex_count = grid_vertices.shape[0]
    if verbose:
        print(
            f"[cumesh-low-memory] active voxels={voxel_count:,}, "
            f"grid vertices={grid_vertex_count:,}",
            flush=True,
        )
    # Active-vertex enumeration is the voxel hashmap's last consumer until
    # topology.  Release it before the UDF and vertex-hash phases, then rebuild
    # the identical mapping from ``coords`` later.
    del hashmap_voxels
    _release_device_memory(device)
    vertex_distances = _grid_distances_batched(
        grid_vertices,
        bvh,
        center,
        scale,
        resolution,
        epsilon,
    )
    progress.update(1)
    progress.close()

    if verbose:
        print("Running Dual Contouring...")

    hashmap_vertices = _init_hashmap(resolution + 1, 2 * grid_vertex_count, device)
    _insert_indexed_coords_batched(
        hashmap_vertices,
        grid_vertices,
        resolution + 1,
    )
    # The hashmap now owns the coordinate -> distance-row mapping.  DC reads
    # only that mapping and ``vertex_distances``; the 1.8-GiB grid coordinate
    # tensor in the 1920-resolution scene can be retired before output buffers
    # are allocated.
    del grid_vertices
    _release_device_memory(device)
    dual_vertices, intersected = _C.simple_dual_contour(
        *hashmap_vertices,
        coords,
        vertex_distances,
        resolution + 1,
        resolution + 1,
        resolution + 1,
    )
    del hashmap_vertices, vertex_distances
    _release_device_memory(device)

    # Topology is integer-only.  Keep the DC edge flags and generated quads on
    # CPU, retaining only the compact batch under construction on the GPU.
    intersected_cpu = intersected.cpu()
    del intersected
    _release_device_memory(device)

    hashmap_voxels = _init_hashmap(resolution, 2 * voxel_count, device)
    _insert_indexed_coords_batched(
        hashmap_voxels,
        coords,
        resolution,
    )
    quad_indices_cpu, intersected_directions_cpu = _build_topology_batched(
        coords,
        intersected_cpu,
        hashmap_voxels,
        edge_neighbor_voxel_offset,
        resolution,
        verbose=verbose,
    )
    del intersected_cpu, hashmap_voxels, coords, edge_neighbor_voxel_offset
    _release_device_memory(device)

    unique_vertices_cpu = _unique_referenced_vertices(
        quad_indices_cpu,
        voxel_count,
    )
    unique_vertices_device = unique_vertices_cpu.to(device)
    compact_vertices = dual_vertices[unique_vertices_device]
    del dual_vertices
    quad_indices_cpu = _remap_quads_batched(
        quad_indices_cpu,
        unique_vertices_cpu,
        voxel_count,
    )
    del unique_vertices_cpu, unique_vertices_device
    _release_device_memory(device)

    mesh_vertices = (compact_vertices / resolution - 0.5) * scale + center
    del compact_vertices
    mesh_triangles = _triangulate_quads_batched(
        mesh_vertices,
        quad_indices_cpu,
        intersected_directions_cpu,
    )
    del quad_indices_cpu, intersected_directions_cpu
    _release_device_memory(device)

    if project_back > 0:
        if verbose:
            print("Projecting back to original mesh...")
        _, face_ids, barycentrics = bvh.unsigned_distance(mesh_vertices, return_uvw=True)
        original_triangles = vertices[faces[face_ids.long()]]
        projected_vertices = (original_triangles * barycentrics.unsqueeze(-1)).sum(dim=1)
        mesh_vertices -= project_back * (mesh_vertices - projected_vertices)

    return mesh_vertices, mesh_triangles.int()
