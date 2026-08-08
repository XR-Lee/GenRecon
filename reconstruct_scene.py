"""
input:
    - sparse structure gen checkpoint
    - shape slat and texture slat checkpoint (either both at resolution 512 or both at resolution 1024)
    - mode: Sage_gt or Scannet_gt
    - path: e.g. <PATH_TO_SAGE_TEST_SET>/renders_room/0a1be5f3 for Sage_gt and <PATH_TO_SCANNETPP_V2_OFFICIAL>/data/2a1b555966 for Scannet_gt
    - output_path
does:
    - gets chunks: inference/get_chunks.py
    - gets images for chunks: inference/get_images.py
    - sets up pipeline (either 512 or 1024 depending on input checkpoints):  genrecon/pipelines/full_scene_images_to_3d.py
    - runs pipeline
    - saves output mesh + coords as .ply
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from genrecon.pipelines.full_scene_images_to_3d import FullSceneImagesTo3DPipeline
from inference.get_chunks import (
    BaseChunker,
    IphoneChunker,
    SageGtChunker,
    ScannetChunker,
    ScannetGtChunker,
    ScannetIphoneChunker,
)
from inference.get_images import (
    IphoneImageSelecter,
    SageImageSelecter,
    ScannetImageSelecter,
    ScannetIphoneImageSelecter,
)
from inference.projection_ownership import build_projection_ownership
from inference.transform_to_original import (
    save_coords_to_original,
    save_mesh_to_original,
)


def _next_halving_friendly(n: int) -> int:
    """Smallest m >= n such that halving m down to <= 32 never lands on an odd > 32.

    Required by ``cumesh.remeshing.remesh_narrow_band_dc``, which asserts
    ``base_resolution % 2 == 0`` while the resolution is > 32.
    """

    def ok(x: int) -> bool:
        while x > 32:
            if x % 2 != 0:
                return False
            x //= 2
        return True

    while not ok(n):
        n += 1
    return n


def _seed_everything(seed: int) -> None:
    """Seed all global RNGs so the run is reproducible from --seed alone."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _set_cond2d_scene_views(sel, view_by_chunk: dict[int, int], cameras_json: Path) -> None:
    """Replace selected per-chunk cond2D inputs with deterministic scene crops."""
    positions = {chunk_id: index for index, chunk_id in enumerate(sel.chunk_indices)}
    unknown = sorted(set(view_by_chunk) - positions.keys())
    if unknown:
        raise ValueError(f"fixed cond2D chunks are not present in the selected scene: {unknown}")
    invalid = sorted(
        (chunk_id, view_index)
        for chunk_id, view_index in view_by_chunk.items()
        if not 0 <= view_index < len(sel.scene_images_512)
    )
    if invalid:
        raise ValueError(
            f"fixed cond2D scene views are outside [0, {len(sel.scene_images_512)}): {invalid}"
        )

    document = json.loads(cameras_json.read_text())
    chunks_json = {int(chunk["chunk_index"]): chunk for chunk in document["chunks"]}
    for chunk_id, scene_view_index in view_by_chunk.items():
        position = positions[chunk_id]
        sel.cond2d_images_512[position] = sel.scene_images_512[scene_view_index]
        sel.cond2d_images_1024[position] = sel.scene_images_1024[scene_view_index]
        sel.cond2d_intrinsics[position] = sel.scene_intrinsics[scene_view_index]
        sel.cond2d_extrinsics_c0[position] = sel.scene_extrinsics_c0[scene_view_index]
        source = document["scene"][scene_view_index]
        chunks_json[chunk_id]["cond2d_view"] = {
            "img_path": source["img_path"],
            "scene_view_index": scene_view_index,
            "extrinsics_c0": source["extrinsics_c0"],
            "intrinsics": source["intrinsics"],
        }
    cameras_json.write_text(json.dumps(document, indent=2) + "\n")


def _share_cond2d_scene_view(sel, chunk_ids: list[int], scene_view_index: int, cameras_json: Path) -> None:
    """Use one saved scene crop as the cond2D context for selected chunks."""
    _set_cond2d_scene_views(sel, {chunk_id: scene_view_index for chunk_id in chunk_ids}, cameras_json)


def _camera_mask_key(view: dict) -> tuple:
    intrinsics = np.asarray(view["intrinsics"], dtype=np.float64)
    return (Path(view["img_path"]).name, *np.round(intrinsics.reshape(-1), 6).tolist())


def _match_mask_views(cameras_json: Path, mask_cameras_json: Path) -> list[int | None]:
    current = json.loads(cameras_json.read_text())["scene"]
    canonical = json.loads(mask_cameras_json.read_text())["scene"]
    canonical_by_key: dict[tuple, int] = {}
    for index, view in enumerate(canonical):
        key = _camera_mask_key(view)
        if key in canonical_by_key:
            raise ValueError(f"mask camera key is not unique: {key}")
        canonical_by_key[key] = index
    return [canonical_by_key.get(_camera_mask_key(view)) for view in current]


def _mask_cond2d_scene_views(
    sel,
    chunk_ids: list[int],
    mask_dir: Path,
    cameras_json: Path,
    background_keep: float,
    mask_view_indices: list[int | None] | None = None,
) -> None:
    """Apply crop-aligned instance masks to selected cond2D images only."""
    positions = {chunk_id: index for index, chunk_id in enumerate(sel.chunk_indices)}
    unknown = sorted(set(chunk_ids) - positions.keys())
    if unknown:
        raise ValueError(f"masked cond2D chunks are not present in the selected scene: {unknown}")
    document = json.loads(cameras_json.read_text())
    chunks_json = {int(chunk["chunk_index"]): chunk for chunk in document["chunks"]}
    for chunk_id in chunk_ids:
        view_index = chunks_json[chunk_id]["cond2d_view"].get("scene_view_index")
        if view_index is None:
            raise ValueError("cond2D masking requires a fixed/shared scene_view_index")
        mask_view_index = (
            mask_view_indices[int(view_index)] if mask_view_indices is not None else int(view_index)
        )
        if mask_view_index is None:
            raise ValueError(
                f"cond2D scene view {view_index} has no matching canonical instance mask"
            )
        mask_path = mask_dir / f"view_{mask_view_index:03d}.png"
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        mask_np = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
        mask = torch.from_numpy(mask_np)[None, None]
        position = positions[chunk_id]
        for images in (sel.cond2d_images_512, sel.cond2d_images_1024):
            image = images[position]
            resized = torch.nn.functional.interpolate(
                mask,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            weight = background_keep + (1.0 - background_keep) * resized
            images[position] = image * weight
        chunks_json[chunk_id]["cond2d_view"]["instance_mask"] = str(mask_path)
        chunks_json[chunk_id]["cond2d_view"]["instance_mask_view_index"] = mask_view_index
        chunks_json[chunk_id]["cond2d_view"]["mask_background_keep"] = background_keep
    cameras_json.write_text(json.dumps(document, indent=2) + "\n")


def _mask_scene_object_features(
    sel,
    mask_dir: Path,
    summary_path: Path,
    cameras_json: Path,
    min_training_points: int,
    mask_view_indices: list[int | None] | None = None,
    *,
    mask_rgb_before_encoding: bool = True,
) -> set[int]:
    """Select object views and optionally remove non-instance RGB before DINO."""
    summary = json.loads(summary_path.read_text())
    records = {int(view["view_index"]): view for view in summary["views"]}
    if mask_view_indices is None:
        mask_view_indices = list(range(len(sel.scene_images_512)))
    if len(mask_view_indices) != len(sel.scene_images_512):
        raise ValueError("mask view mapping must match the selected scene-view count")
    enabled = {
        view_index
        for view_index, mask_view_index in enumerate(mask_view_indices)
        if mask_view_index is not None
        and mask_view_index in records
        and records[mask_view_index].get("active_parts")
        and int(records[mask_view_index].get("training_points", 0)) >= min_training_points
    }
    if mask_rgb_before_encoding:
        for view_index, mask_view_index in enumerate(mask_view_indices):
            if mask_view_index is None:
                mask_np = np.zeros((1024, 1024), dtype=np.float32)
            else:
                mask_path = mask_dir / f"view_{mask_view_index:03d}.png"
                if not mask_path.is_file():
                    raise FileNotFoundError(mask_path)
                mask_np = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
                if view_index not in enabled:
                    mask_np.fill(0.0)
            mask = torch.from_numpy(mask_np)[None, None]
            for images in (sel.scene_images_512, sel.scene_images_1024):
                image = images[view_index]
                resized = torch.nn.functional.interpolate(
                    mask,
                    size=image.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                images[view_index] = image * resized

    document = json.loads(cameras_json.read_text())
    for view_index, view in enumerate(document["scene"]):
        mask_view_index = mask_view_indices[view_index]
        view["object_feature_mask_view_index"] = mask_view_index
        view["object_feature_mask"] = (
            str(mask_dir / f"view_{mask_view_index:03d}.png")
            if mask_view_index is not None
            else None
        )
        view["object_feature_enabled"] = view_index in enabled
        view["object_feature_rgb_mode"] = (
            "masked_before_dino" if mask_rgb_before_encoding else "context_encoded_before_token_gate"
        )
        view["object_feature_training_points"] = (
            int(records[mask_view_index].get("training_points", 0))
            if mask_view_index is not None and mask_view_index in records
            else 0
        )
    cameras_json.write_text(json.dumps(document, indent=2) + "\n")
    return enabled


def _cond2d_scene_view_indices(sel, cameras_json: Path) -> list[int]:
    document = json.loads(cameras_json.read_text())
    chunks = {int(chunk["chunk_index"]): chunk for chunk in document["chunks"]}
    result = []
    for chunk_id in sel.chunk_indices:
        view_index = chunks[chunk_id]["cond2d_view"].get("scene_view_index")
        if view_index is None:
            raise ValueError("object feature-only cond2D requires fixed scene-view indices")
        result.append(int(view_index))
    return result


def _load_custom_chunk_layout(path: Path, output: Path):
    """Load explicit world-space centers while preserving the chunker's transform format."""
    document = json.loads(path.read_text())
    chunk_size = float(document["chunk_size_m"])
    centers = [[float(value) for value in center] for center in document["centers"]]
    if chunk_size <= 0 or not centers or any(len(center) != 3 for center in centers):
        raise ValueError("custom chunk layout requires positive chunk_size_m and non-empty [x,y,z] centers")
    relative_lattice = (np.asarray(centers) - np.asarray(centers[0])) / chunk_size * 16.0
    if not np.allclose(relative_lattice, np.round(relative_lattice), atol=1e-5):
        raise ValueError("custom chunk centers must align to the 1/16-chunk model lattice")
    _, m_o2c, m_c2o, rel_t = (
        centers,
        *BaseChunker()._get_transforms(centers, chunk_size, output),
    )
    source = {"source": str(path), "chunk_size_m": chunk_size, "centers": centers}
    (output / "custom_chunk_layout.json").write_text(json.dumps(source, indent=2) + "\n")
    return centers, m_o2c, m_c2o, rel_t


def _farthest_point_indices(points: np.ndarray, count: int) -> np.ndarray:
    """Deterministically spread a small anchor bank over an instance point cluster."""
    if count <= 0 or len(points) == 0:
        return np.empty(0, dtype=np.int64)
    count = min(count, len(points))
    first = int(np.argmax(np.linalg.norm(points - points.mean(axis=0), axis=1)))
    selected = [first]
    min_distance_sq = np.sum((points - points[first]) ** 2, axis=1)
    while len(selected) < count:
        index = int(np.argmax(min_distance_sq))
        selected.append(index)
        min_distance_sq = np.minimum(min_distance_sq, np.sum((points - points[index]) ** 2, axis=1))
    return np.asarray(selected, dtype=np.int64)


def _prepare_instance_anchors(
    points_path: Path,
    roi_boxes: list[list[float]],
    *,
    max_reprojection_error: float,
    min_track_length: int,
    token_count: int,
    holdout_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, dict]:
    rows = []
    for line in points_path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        error = float(fields[7])
        track_length = (len(fields) - 8) // 2
        if error > max_reprojection_error or track_length < min_track_length:
            continue
        xyz = np.asarray([float(fields[1]), float(fields[2]), float(fields[3])], dtype=np.float64)
        in_roi = any(
            x0 <= xyz[0] <= x1 and y0 <= xyz[1] <= y1 and z0 <= xyz[2] <= z1
            for x0, x1, y0, y1, z0, z1 in roi_boxes
        )
        if in_roi:
            observed_image_ids = [int(fields[index]) for index in range(8, len(fields), 2)]
            rows.append((int(fields[0]), xyz, error, track_length, observed_image_ids))
    if len(rows) < token_count:
        raise ValueError(f"instance ROI has {len(rows)} quality points, fewer than {token_count} tokens")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    holdout_count = int(round(len(rows) * holdout_fraction))
    holdout_rows = [rows[index] for index in order[:holdout_count]]
    training_rows = [rows[index] for index in order[holdout_count:]]
    training_xyz = np.asarray([row[1] for row in training_rows])
    selected_local = _farthest_point_indices(training_xyz, token_count)
    token_rows = [training_rows[index] for index in selected_local]
    token_xyz = torch.tensor(np.asarray([row[1] for row in token_rows]), dtype=torch.float32)
    diagnostics = {
        "schema": "genrecon.instance-anchor-bank",
        "points_path": str(points_path),
        "roi_boxes_world": roi_boxes,
        "quality_roi_points": len(rows),
        "training_points": len(training_rows),
        "holdout_points": len(holdout_rows),
        "holdout_fraction": holdout_fraction,
        "token_count": len(token_rows),
        "seed": seed,
        "token_point_ids": [row[0] for row in token_rows],
        "holdout_point_ids": [row[0] for row in holdout_rows],
        "training_point_ids": [row[0] for row in training_rows],
        "token_xyz_world": [row[1].tolist() for row in token_rows],
        "token_observation_image_ids": [row[4] for row in token_rows],
    }
    return token_xyz, diagnostics


def _colmap_image_ids_by_name(images_path: Path) -> dict[str, int]:
    lines = [
        line.strip()
        for line in images_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if len(lines) % 2:
        raise ValueError(f"COLMAP images file must contain two lines per image: {images_path}")
    result = {}
    for line in lines[::2]:
        fields = line.split()
        result[Path(fields[9]).name] = int(fields[0])
    return result


def _world_boxes_to_chunk0(
    boxes: list[list[float]] | None,
    world_to_chunk0: torch.Tensor,
) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]] | None:
    if not boxes:
        return None
    transformed = []
    matrix = world_to_chunk0.to(dtype=torch.float64)
    for x0, x1, y0, y1, z0, z1 in boxes:
        corners = torch.tensor(
            [[x, y, z, 1.0] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)],
            dtype=torch.float64,
        )
        points = (corners @ matrix.T)[:, :3]
        transformed.append(
            (
                tuple(float(value) for value in points.min(dim=0).values),
                tuple(float(value) for value in points.max(dim=0).values),
            )
        )
    return transformed


MODES = {
    "Sage_gt": (SageGtChunker, SageImageSelecter, lambda p: p / "transforms.json"),
    "Scannet_gt": (
        ScannetGtChunker,
        ScannetImageSelecter,
        lambda p: p / "dslr" / "nerfstudio" / "transforms_undistorted.json",
    ),
    "Scannet_colmap": (
        ScannetChunker,
        ScannetImageSelecter,
        lambda p: p / "dslr" / "nerfstudio" / "transforms_undistorted.json",
    ),
    "Scannet_iphone": (
        ScannetIphoneChunker,
        ScannetIphoneImageSelecter,
        lambda p: p / "iphone" / "colmap" / "cameras.txt",
    ),
    "Iphone": (
        IphoneChunker,
        IphoneImageSelecter,
        lambda p: p / "colmap" / "cameras.txt",
    ),
}


def _save_plys(
    out_path: Path,
    scene_mesh,
    coords_list,
    coords_resolution: int,
    chunk_indices,
    *,
    mesh_transform: torch.Tensor,
    coords_transform,
    label: str,
) -> None:
    """Save the merged mesh + per-chunk coords as .ply under ``out_path``.

    ``mesh_transform`` lifts the mesh; ``coords_transform(chunk_idx)`` returns
    the per-chunk transform for its coords (identity for the chunk-local save,
    ``m_c2o[chunk_idx]`` for the world save).
    """
    save_mesh_to_original(out_path / "mesh.ply", scene_mesh, mesh_transform)
    print(label)
    for coords, chunk_idx in zip(coords_list, chunk_indices):
        save_coords_to_original(
            out_path / f"coords_{chunk_idx:03d}.ply", coords, coords_resolution, coords_transform(chunk_idx)
        )


def _save_to_glb_inputs(
    out_path: Path,
    scene_mesh,
    *,
    vertices: torch.Tensor,
    voxel_size: float,
    origin,
    label: str,
) -> None:
    """Build the chunked-GLB conversion inputs and save them to ``to_glb_inputs.pt``.

    The grid is padded up to satisfy the remesh halving constraint (see
    ``_next_halving_friendly``). ``vertices``/``voxel_size``/``origin`` must
    already be in the target frame (chunk-local or world); ``label`` is a prefix
    for the log line (e.g. ``"chunk-local "`` or ``""``).
    """
    gx, gy, gz = (int(v) for v in scene_mesh.voxel_shape[2:])
    gx_p, gy_p, gz_p = (_next_halving_friendly(v) for v in (gx, gy, gz))
    if (gx_p, gy_p, gz_p) != (gx, gy, gz):
        print(f"[chunked] padding grid ({gx}, {gy}, {gz}) → ({gx_p}, {gy_p}, {gz_p}) for remesh halving constraint")
    aabb = [
        list(origin),
        [origin[i] + (gx_p, gy_p, gz_p)[i] * voxel_size for i in range(3)],
    ]
    to_glb_inputs = {
        "vertices": vertices.detach().cpu(),
        "faces": scene_mesh.faces.detach().cpu(),
        "attr_volume": scene_mesh.attrs.detach().cpu(),
        "coords": scene_mesh.coords.detach().cpu(),
        "attr_layout": scene_mesh.layout,
        "aabb": aabb,
        "voxel_size": voxel_size,
    }
    torch.save(to_glb_inputs, out_path / "to_glb_inputs.pt")
    print(f"[chunked] saved {label}to_glb_inputs to {out_path / 'to_glb_inputs.pt'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=list(MODES))
    parser.add_argument("--path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--ss_ckpt", default=None)
    parser.add_argument("--shape_ckpt", default=None)
    parser.add_argument("--tex_ckpt", default=None)
    parser.add_argument(
        "--ss_config",
        default="configs/gen/ss_flow_img/genrecon.json",
        help="Training config matching --ss_ckpt.",
    )
    parser.add_argument(
        "--shape_config",
        default="configs/gen/slat_flow_img2shape/genrecon_512.json",
        help="Training config matching --shape_ckpt.",
    )
    parser.add_argument(
        "--tex_config",
        default="configs/gen/slat_flow_imgshape2tex/genrecon_512.json",
        help="Training config matching --tex_ckpt.",
    )
    parser.add_argument("--pipeline", choices=["512"], default="512")
    parser.add_argument(
        "--pipeline_config",
        default=None,
        help="Path to a pipeline config .json (sampler configs etc). "
        "Defaults to ImagesTo3DPipeline.DEFAULT_PIPELINE_CONFIG_FILE "
        "(configs/pipelines/original.json).",
    )
    parser.add_argument("--num_imgs_per_scene", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save_imgs",
        action="store_true",
        help="Save the selected scene images to <output_path>/scene/ and the "
        "per-chunk 2D cond views to <output_path>/chunk_<idx>/cond2d.png.",
    )
    parser.add_argument(
        "--boundary_sensitive_slat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable boundary-sensitive overlap aggregation for shape + tex SLat "
        "(on by default; disable with --no-boundary_sensitive_slat).",
    )
    parser.add_argument(
        "--boundary_width_slat",
        type=int,
        default=1,
        help="Boundary width (in latent voxels) for SLat aggregation.",
    )
    parser.add_argument(
        "--min_overlap_factor",
        type=int,
        default=4,
        help="Sets BaseChunker.min_overlap = min_overlap_factor * atomic_distance "
        "(atomic_distance = 1/16). Integer; default 4 -> 1/4 chunk overlap.",
    )
    parser.add_argument(
        "--occ_threshold",
        type=float,
        default=-1.0,
        help="Threshold applied to SS occupancy logits when extracting coords "
        "(default -1.0, more permissive than sigmoid > 0.5). Higher = more "
        "confident / sparser; lower = more permissive / denser.",
    )
    parser.add_argument(
        "--chunk_size_factor",
        type=float,
        default=1.11,
        help="Override chunk_size = delta_z * chunk_size_factor (default 1.11). "
        "Applied to all multiplicative-chunk modes. Not supported for Sage_gt "
        "(additive chunk_size); passing it explicitly there is an error.",
    )
    parser.add_argument(
        "--colmap_subdir",
        default="colmap",
        help="Subdirectory under --path holding cameras.txt/images.txt/points3D.txt "
        "for mode=Iphone (e.g. colmap_mast3r, colmap_hloc). Ignored for other modes.",
    )
    parser.add_argument(
        "--min_points_per_chunk",
        type=int,
        default=None,
        help="Override the chunker's min-points-per-chunk threshold. ScannetMixin "
        "defaults to 500, which is too aggressive for sparse-point datasets like "
        "vfront_gtpoints_10k (only ~10k pts/scene). Lower to e.g. 30 for those.",
    )
    parser.add_argument(
        "--manual_z_bounds",
        type=float,
        nargs=2,
        metavar=("FLOOR", "CEILING"),
        default=None,
        help="Iphone mode only. Override the vertical room bounds in COLMAP world "
        "coordinates while retaining point-derived XY bounds.",
    )
    parser.add_argument(
        "--skip_point_cleaning",
        action="store_true",
        help="Skip the statistical+radius outlier filters in _get_clean_points. "
        "Those filters are tuned for dense COLMAP clouds (millions of pts) and "
        "reject ~all points on sparse GT clouds (e.g. vfront_gtpoints_10k).",
    )
    parser.add_argument(
        "--max_reproj_error",
        type=float,
        default=None,
        help="Iphone mode only. Reject COLMAP points above this reprojection "
        "error in pixels. Omitted keeps the input-adapter default.",
    )
    parser.add_argument(
        "--min_track_len",
        type=int,
        default=None,
        help="Iphone mode only. Reject COLMAP points observed by fewer images. "
        "Omitted keeps the input-adapter default.",
    )
    parser.add_argument(
        "--stat_std_ratio",
        type=float,
        default=None,
        help="Iphone mode only. Override std_ratio for open3d statistical outlier "
        "removal in _get_clean_points (default 2.0). Larger = less aggressive.",
    )
    parser.add_argument(
        "--radius_nb_points",
        type=int,
        default=None,
        help="Iphone mode only. Override nb_points for open3d radius outlier "
        "removal in _get_clean_points (default 10). Smaller = less aggressive.",
    )
    parser.add_argument(
        "--radius_m",
        type=float,
        default=None,
        help="Iphone mode only. Override radius (meters) for open3d radius outlier "
        "removal in _get_clean_points (default 0.1). Larger = less aggressive.",
    )
    parser.add_argument(
        "--validation_crop_idx",
        type=int,
        default=None,
        help="If set, bypass the chunker and read the crop transform from "
        "<path>/../../crops/<scene_id>.json[chunks][idx]. Mesh + to_glb_inputs are "
        "saved in chunk-local coords (matching the validation GT GLB frame). "
        "Sage_gt mode only.",
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="iPhone modes only: revert to the legacy single-square crop around "
        "the principal point. Default (flag omitted) crops non-square frames "
        "into two square views along the longer axis, doubling the scene-wide "
        "input pool. Ignored for non-iPhone modes.",
    )
    parser.add_argument(
        "--proj_batch_voxels",
        type=int,
        default=None,
        help="Override FullSceneImagesTo3DPipeline.proj_batch_voxels (default "
        "8192). Streams the voxel dim through the global projection+aggregator "
        "in chunks of this size; smaller = lower peak VRAM, more iterations. "
        "Pure performance knob, no numerical effect. Drop to 2048/1024 when "
        "running with very many views (e.g. all-frames iPhone captures).",
    )
    parser.add_argument(
        "--joint_decode_max_chunks_per_group",
        type=int,
        default=None,
        help="Override the joint decoder's per-group chunk cap. Intended as an "
        "explicit low-VRAM retry control; omitted keeps the pipeline default.",
    )
    parser.add_argument(
        "--joint_decode_max_inflated_voxels",
        type=int,
        default=None,
        help="Override the joint decoder's per-group inflated-voxel cap. Intended "
        "as an explicit low-VRAM retry control; omitted keeps the pipeline default.",
    )
    parser.add_argument(
        "--overlap_diagnostics",
        default=None,
        help="Write per-step, pre-aggregation overlap disagreement metrics to this JSON path. "
        "Diagnostics are read-only and do not change MultiDiffusion aggregation.",
    )
    parser.add_argument(
        "--overlap_diagnostics_roi_box",
        type=float,
        nargs=6,
        action="append",
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="Optional repeatable world-space box. Diagnostics add roi_* metrics over the union "
        "of overlap voxels inside these boxes.",
    )
    parser.add_argument(
        "--chunk_layout_json",
        type=Path,
        help="Bypass automatic chunk placement and load {chunk_size_m, centers} in world coordinates.",
    )
    parser.add_argument(
        "--fixed_cond2d_scene_views",
        type=Path,
        help="JSON object mapping chunk ids to zero-based scene crop indices. Used to hold cond2D "
        "fixed across chunk-coordinate ablations.",
    )
    parser.add_argument(
        "--instance_anchor_roi_box",
        type=float,
        nargs=6,
        action="append",
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="Repeatable world-space instance box used to select high-confidence COLMAP anchor tracks.",
    )
    parser.add_argument(
        "--instance_anchor_chunks",
        type=int,
        nargs="+",
        help="Chunk ids that receive the same track-level DINO anchor token bank.",
    )
    parser.add_argument("--instance_anchor_token_count", type=int, default=64)
    parser.add_argument("--instance_anchor_holdout_fraction", type=float, default=0.3)
    parser.add_argument(
        "--cond2d_mask_dir",
        type=Path,
        help="Directory containing crop-aligned view_NNN.png instance masks.",
    )
    parser.add_argument(
        "--instance_mask_cameras_json",
        type=Path,
        help="Canonical cameras.json for the mask files; masks are matched by image basename and crop intrinsics.",
    )
    parser.add_argument(
        "--cond2d_mask_chunks",
        type=int,
        nargs="+",
        help="Chunk ids whose fixed cond2D scene crops are multiplied by instance masks.",
    )
    parser.add_argument("--cond2d_mask_background_keep", type=float, default=0.0)
    parser.add_argument(
        "--projection_ownership_mode",
        choices=["mask", "depth"],
        help="Apply SAM patch ownership to scene-wide cond3D projection. Depth mode also uses "
        "COLMAP track depth and hard free-space occupancy suppression.",
    )
    parser.add_argument(
        "--projection_ownership_mask_dir",
        type=Path,
        help="Directory containing union view_NNN.png and optional view_NNN_part_P.png masks.",
    )
    parser.add_argument(
        "--projection_ownership_anchors_json",
        type=Path,
        help="Fixed point-ID split JSON; only training_point_ids provide projection depth.",
    )
    parser.add_argument(
        "--projection_ownership_roi_box",
        type=float,
        nargs=6,
        action="append",
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="Repeatable world-space object/part box where projection ownership is active.",
    )
    parser.add_argument("--projection_ownership_surface_band_m", type=float, default=0.08)
    parser.add_argument("--projection_ownership_depth_fill_radius", type=float, default=2.0)
    parser.add_argument("--projection_ownership_mask_patch_threshold", type=float, default=0.25)
    parser.add_argument("--projection_ownership_min_free_views", type=int, default=2)
    parser.add_argument(
        "--projection_ownership_global_filter",
        action="store_true",
        help="Apply SAM/depth feature filtering throughout each chunk, not only inside the diagnostic ROI.",
    )
    parser.add_argument(
        "--projection_ownership_support_box",
        type=float,
        nargs=6,
        action="append",
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="Repeatable world-space object support envelope used by strict object-only generation.",
    )
    parser.add_argument(
        "--projection_ownership_hard_support",
        action="store_true",
        help="Delete sparse occupancy outside --projection_ownership_support_box and crop decoded faces.",
    )
    parser.add_argument(
        "--projection_ownership_seed_surface",
        action="store_true",
        help="Retain sparse cells with COLMAP track-depth surface votes as positive object evidence.",
    )
    parser.add_argument(
        "--projection_ownership_allow_unknown",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In depth mode, retain object-mask patches without nearby track depth.",
    )
    parser.add_argument(
        "--object_feature_only",
        action="store_true",
        help="Select object views and zero non-object DINO patch tokens. By default RGB is also "
        "masked before DINO; --object_feature_encode_context keeps full RGB during encoding.",
    )
    parser.add_argument(
        "--object_feature_encode_context",
        action="store_true",
        help="Encode full scene/cond2D RGB with DINO before zeroing non-object patch tokens. "
        "This preserves context inside retained object/global tokens while keeping the same token gate.",
    )
    parser.add_argument(
        "--object_feature_drop_global_tokens",
        action="store_true",
        help="With object-feature conditioning, also zero DINO global/register tokens so only owned patch tokens remain.",
    )
    parser.add_argument(
        "--object_feature_mask_summary",
        type=Path,
        help="SAM summary.json used to reject scene crops with too few registered object prompts.",
    )
    parser.add_argument("--object_feature_min_training_points", type=int, default=3)
    parser.add_argument(
        "--shared_cond2d_chunks",
        type=int,
        nargs="+",
        default=None,
        help="Diagnostic proxy for shared instance context: replace cond2D for these chunk ids "
        "with --shared_cond2d_scene_view while leaving scene-wide cond3D unchanged.",
    )
    parser.add_argument(
        "--shared_cond2d_scene_view",
        type=int,
        default=None,
        help="Zero-based scene crop index used by --shared_cond2d_chunks.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    missing = [
        n
        for n, v in [("--ss_ckpt", args.ss_ckpt), ("--shape_ckpt", args.shape_ckpt), ("--tex_ckpt", args.tex_ckpt)]
        if v is None
    ]
    if missing:
        parser.error(f"{', '.join(missing)} required.")

    if args.validation_crop_idx is not None and args.mode != "Sage_gt":
        parser.error("--validation_crop_idx is only supported for --mode Sage_gt.")
    if args.validation_crop_idx is not None and args.chunk_layout_json is not None:
        parser.error("--validation_crop_idx and --chunk_layout_json are mutually exclusive.")
    if args.overlap_diagnostics_roi_box and args.overlap_diagnostics is None:
        parser.error("--overlap_diagnostics_roi_box requires --overlap_diagnostics.")
    for box in (
        (args.overlap_diagnostics_roi_box or [])
        + (args.instance_anchor_roi_box or [])
        + (args.projection_ownership_roi_box or [])
        + (args.projection_ownership_support_box or [])
    ):
        if box[0] > box[1] or box[2] > box[3] or box[4] > box[5]:
            parser.error("each ROI box requires MIN <= MAX on every axis.")
    if (args.instance_anchor_roi_box is None) != (args.instance_anchor_chunks is None):
        parser.error("--instance_anchor_roi_box and --instance_anchor_chunks must be used together.")
    if args.instance_anchor_roi_box is not None and args.mode != "Iphone":
        parser.error("instance anchor experiments currently require --mode Iphone.")
    if args.instance_anchor_token_count <= 0:
        parser.error("--instance_anchor_token_count must be positive.")
    if not 0 <= args.instance_anchor_holdout_fraction < 1:
        parser.error("--instance_anchor_holdout_fraction must be in [0, 1).")
    if (args.cond2d_mask_dir is None) != (args.cond2d_mask_chunks is None):
        parser.error("--cond2d_mask_dir and --cond2d_mask_chunks must be used together.")
    if args.cond2d_mask_dir is not None and args.fixed_cond2d_scene_views is None and args.shared_cond2d_chunks is None:
        parser.error("cond2D masking requires fixed or shared scene views.")
    if not 0 <= args.cond2d_mask_background_keep <= 1:
        parser.error("--cond2d_mask_background_keep must be in [0, 1].")
    ownership_values = (
        args.projection_ownership_mask_dir,
        args.projection_ownership_anchors_json,
        args.projection_ownership_roi_box,
    )
    if args.projection_ownership_mode is None and any(value is not None for value in ownership_values):
        parser.error("projection ownership paths/ROI require --projection_ownership_mode.")
    if args.projection_ownership_mode is not None and any(value is None for value in ownership_values):
        parser.error(
            "--projection_ownership_mode requires mask dir, anchors JSON, and at least one ROI box."
        )
    if args.projection_ownership_mode is not None and args.pipeline != "512":
        parser.error("projection ownership is currently calibrated for the 512 pipeline only.")
    if args.projection_ownership_surface_band_m <= 0:
        parser.error("--projection_ownership_surface_band_m must be positive.")
    if args.projection_ownership_depth_fill_radius < 0:
        parser.error("--projection_ownership_depth_fill_radius must be non-negative.")
    if not 0 <= args.projection_ownership_mask_patch_threshold <= 1:
        parser.error("--projection_ownership_mask_patch_threshold must be in [0, 1].")
    if args.projection_ownership_min_free_views <= 0:
        parser.error("--projection_ownership_min_free_views must be positive.")
    if args.projection_ownership_hard_support and not args.projection_ownership_support_box:
        parser.error("--projection_ownership_hard_support requires at least one support box.")
    if (
        args.projection_ownership_global_filter
        or args.projection_ownership_hard_support
        or args.projection_ownership_seed_surface
    ) and args.projection_ownership_mode is None:
        parser.error("global/hard/seed projection ownership requires --projection_ownership_mode.")
    if args.projection_ownership_seed_surface and args.projection_ownership_mode != "depth":
        parser.error("--projection_ownership_seed_surface requires depth ownership.")
    if args.object_feature_min_training_points <= 0:
        parser.error("--object_feature_min_training_points must be positive.")
    if args.object_feature_only:
        if args.object_feature_mask_summary is None:
            parser.error("--object_feature_only requires --object_feature_mask_summary.")
        if args.instance_mask_cameras_json is None:
            parser.error("--object_feature_only requires --instance_mask_cameras_json.")
        if args.cond2d_mask_dir is None or args.projection_ownership_mode is None:
            parser.error("--object_feature_only requires cond2D masks and projection ownership.")
        required_background_keep = 1.0 if args.object_feature_encode_context else 0.0
        if args.cond2d_mask_background_keep != required_background_keep:
            parser.error(
                "--object_feature_only requires --cond2d_mask_background_keep "
                f"{required_background_keep:g} for the selected RGB encoding mode."
            )
        if not args.projection_ownership_global_filter or not args.projection_ownership_hard_support:
            parser.error("--object_feature_only requires global filtering and hard object support.")
        if args.pipeline != "512":
            parser.error("--object_feature_only is currently supported by the 512 pipeline only.")
    elif args.object_feature_encode_context:
        parser.error("--object_feature_encode_context requires --object_feature_only.")
    if args.object_feature_drop_global_tokens and not args.object_feature_only:
        parser.error("--object_feature_drop_global_tokens requires --object_feature_only.")
    if args.joint_decode_max_chunks_per_group is not None and args.joint_decode_max_chunks_per_group <= 0:
        parser.error("--joint_decode_max_chunks_per_group must be positive.")
    if args.joint_decode_max_inflated_voxels is not None and args.joint_decode_max_inflated_voxels <= 0:
        parser.error("--joint_decode_max_inflated_voxels must be positive.")
    if (args.shared_cond2d_chunks is None) != (args.shared_cond2d_scene_view is None):
        parser.error("--shared_cond2d_chunks and --shared_cond2d_scene_view must be used together.")
    if args.shared_cond2d_chunks is not None and len(set(args.shared_cond2d_chunks)) != len(args.shared_cond2d_chunks):
        parser.error("--shared_cond2d_chunks must not contain duplicate ids.")
    if args.max_reproj_error is not None and args.max_reproj_error < 0:
        parser.error("--max_reproj_error must be non-negative.")
    if args.min_track_len is not None and args.min_track_len < 0:
        parser.error("--min_track_len must be non-negative.")
    if args.manual_z_bounds is not None and args.manual_z_bounds[0] >= args.manual_z_bounds[1]:
        parser.error("--manual_z_bounds requires FLOOR < CEILING.")

    _seed_everything(args.seed)

    scene_path = Path(args.path)
    out_path = Path(args.output_path)
    out_path.mkdir(parents=True, exist_ok=True)

    with (out_path / "args.json").open("w", encoding="utf-8") as f:
        json.dump(
            {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            f,
            indent=2,
            sort_keys=True,
        )

    pipeline = FullSceneImagesTo3DPipeline.from_finetuned(
        stage_models={
            "sparse_structure_flow_model": args.ss_ckpt,
            f"shape_slat_flow_model_{args.pipeline}": args.shape_ckpt,
            f"tex_slat_flow_model_{args.pipeline}": args.tex_ckpt,
        },
        pipeline_config_file=args.pipeline_config,
        stage_config_files={
            "sparse_structure_flow_model": args.ss_config,
            f"shape_slat_flow_model_{args.pipeline}": args.shape_config,
            f"tex_slat_flow_model_{args.pipeline}": args.tex_config,
        },
    )
    if args.proj_batch_voxels is not None:
        pipeline.proj_batch_voxels = args.proj_batch_voxels
    pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    chunker_cls, selecter_cls, transforms_json = MODES[args.mode]
    if args.mode == "Iphone":
        transforms_json = lambda p, sub=args.colmap_subdir: p / sub / "cameras.txt"
    chunker_kwargs: dict = {"min_overlap_factor": args.min_overlap_factor}
    if args.mode == "Sage_gt":
        # Sage_gt uses additive chunk_size; the multiplicative factor doesn't apply.
        # Silently skip the default, but still error if the user passed it explicitly.
        if any(a == "--chunk_size_factor" or a.startswith("--chunk_size_factor=") for a in sys.argv[1:]):
            parser.error("--chunk_size_factor is not supported for mode=Sage_gt (additive chunk_size).")
    elif args.chunk_size_factor is not None:
        chunker_kwargs["chunk_size_factor"] = args.chunk_size_factor
    if args.mode == "Iphone":
        chunker_kwargs["colmap_subdir"] = args.colmap_subdir
        if args.max_reproj_error is not None:
            chunker_kwargs["max_reproj_error"] = args.max_reproj_error
        if args.min_track_len is not None:
            chunker_kwargs["min_track_len"] = args.min_track_len
        if args.manual_z_bounds is not None:
            chunker_kwargs["manual_z_bounds"] = tuple(args.manual_z_bounds)
        if args.stat_std_ratio is not None:
            chunker_kwargs["stat_std_ratio"] = args.stat_std_ratio
        if args.radius_nb_points is not None:
            chunker_kwargs["radius_nb_points"] = args.radius_nb_points
        if args.radius_m is not None:
            chunker_kwargs["radius_m"] = args.radius_m
    elif any(
        v is not None
        for v in (
            args.max_reproj_error,
            args.min_track_len,
            args.manual_z_bounds,
            args.stat_std_ratio,
            args.radius_nb_points,
            args.radius_m,
        )
    ):
        parser.error(
            "--max_reproj_error/--min_track_len/--manual_z_bounds/"
            "--stat_std_ratio/--radius_nb_points/--radius_m are Iphone-mode only."
        )
    if args.min_points_per_chunk is not None:
        chunker_kwargs["min_points_per_chunk"] = args.min_points_per_chunk
    if args.skip_point_cleaning:
        chunker_kwargs["skip_point_cleaning"] = True

    if args.validation_crop_idx is not None:
        scene_id = scene_path.name
        crops_json = scene_path.parent.parent / "crops" / f"{scene_id}.json"
        with crops_json.open("r", encoding="utf-8") as f:
            crops_data = json.load(f)
        crop = crops_data["chunks"][args.validation_crop_idx]
        m_o2c = [torch.tensor(crop["M_original_to_chunk"], dtype=torch.float32)]
        m_c2o = [torch.tensor(crop["M_chunk_to_original"], dtype=torch.float32)]
        rel_t = [torch.zeros(3, dtype=torch.float32)]
        with (out_path / "crop_transform.json").open("w", encoding="utf-8") as f:
            json.dump(crop, f, indent=2)
        print(f"[validation_crop] using crops/{scene_id}.json[chunks][{args.validation_crop_idx}]")
    elif args.chunk_layout_json is not None:
        _, m_o2c, m_c2o, rel_t = _load_custom_chunk_layout(args.chunk_layout_json, out_path)
        print(f"[chunk_layout] loaded {len(m_o2c)} centers from {args.chunk_layout_json}")
    else:
        _, m_o2c, m_c2o, rel_t = chunker_cls(**chunker_kwargs).get_chunks(scene_path, out_path)
    selecter_kwargs: dict = {}
    if args.mode in ("Scannet_iphone", "Iphone"):
        selecter_kwargs["center_crop"] = args.center_crop
    sel = selecter_cls(**selecter_kwargs).get_images(
        m_o2c,
        transforms_json(scene_path),
        args.num_imgs_per_scene,
        out_path,
        seed=args.seed,
    )
    rel_t_kept = [rel_t[i] for i in sel.chunk_indices]
    mask_view_indices = None
    if args.instance_mask_cameras_json is not None:
        mask_view_indices = _match_mask_views(
            out_path / "cameras.json",
            args.instance_mask_cameras_json,
        )
        print(
            f"[instance_masks] matched {sum(index is not None for index in mask_view_indices)}/"
            f"{len(mask_view_indices)} scene crops to canonical masks"
        )
    if args.fixed_cond2d_scene_views is not None:
        raw_mapping = json.loads(args.fixed_cond2d_scene_views.read_text())
        if not isinstance(raw_mapping, dict):
            raise ValueError("--fixed_cond2d_scene_views must contain a JSON object")
        fixed_mapping = {int(chunk_id): int(view_index) for chunk_id, view_index in raw_mapping.items()}
        _set_cond2d_scene_views(sel, fixed_mapping, out_path / "cameras.json")
        print(f"[fixed_cond2d] {fixed_mapping}")
    if args.shared_cond2d_chunks is not None:
        _share_cond2d_scene_view(
            sel,
            args.shared_cond2d_chunks,
            args.shared_cond2d_scene_view,
            out_path / "cameras.json",
        )
        print(
            f"[shared_cond2d] chunks={args.shared_cond2d_chunks} use scene view "
            f"{args.shared_cond2d_scene_view}"
        )
    if args.cond2d_mask_dir is not None:
        _mask_cond2d_scene_views(
            sel,
            args.cond2d_mask_chunks,
            args.cond2d_mask_dir,
            out_path / "cameras.json",
            args.cond2d_mask_background_keep,
            mask_view_indices,
        )
        print(
            f"[cond2d_masks] chunks={args.cond2d_mask_chunks} dir={args.cond2d_mask_dir} "
            f"background_keep={args.cond2d_mask_background_keep}"
        )

    object_enabled_views: set[int] | None = None
    object_cond2d_view_indices: list[int] | None = None
    if args.object_feature_only:
        object_enabled_views = _mask_scene_object_features(
            sel,
            args.projection_ownership_mask_dir,
            args.object_feature_mask_summary,
            out_path / "cameras.json",
            args.object_feature_min_training_points,
            mask_view_indices,
            mask_rgb_before_encoding=not args.object_feature_encode_context,
        )
        object_cond2d_view_indices = _cond2d_scene_view_indices(sel, out_path / "cameras.json")
        rejected = sorted(set(range(len(sel.scene_images_512))) - object_enabled_views)
        print(
            f"[object_features] enabled scene views={sorted(object_enabled_views)} "
            f"rejected={rejected} cond2d={object_cond2d_view_indices} "
            f"rgb_mode={'context_encoded' if args.object_feature_encode_context else 'pre_masked'}"
        )

    overlap_roi_chunk0 = _world_boxes_to_chunk0(args.overlap_diagnostics_roi_box, m_o2c[0])
    projection_ownership = None
    projection_ownership_diagnostics: dict = {}
    if args.projection_ownership_mode is not None:
        ownership_roi_chunk0 = _world_boxes_to_chunk0(args.projection_ownership_roi_box, m_o2c[0])
        support_bounds_chunk0 = _world_boxes_to_chunk0(
            args.projection_ownership_support_box,
            m_o2c[0],
        )
        projection_ownership, projection_ownership_diagnostics = build_projection_ownership(
            camera_document=json.loads((out_path / "cameras.json").read_text()),
            mask_dir=args.projection_ownership_mask_dir,
            anchors_document=json.loads(args.projection_ownership_anchors_json.read_text()),
            colmap_images_path=scene_path / args.colmap_subdir / "images.txt",
            colmap_points_path=scene_path / args.colmap_subdir / "points3D.txt",
            roi_boxes_world=args.projection_ownership_roi_box,
            roi_bounds_chunk0=ownership_roi_chunk0,
            world_to_chunk0=m_o2c[0],
            chunk_size_m=float(m_c2o[0][0, 0].item()),
            mode=args.projection_ownership_mode,
            support_bounds_chunk0=support_bounds_chunk0,
            enabled_view_indices=object_enabled_views,
            mask_view_indices=mask_view_indices,
            global_feature_filter=args.projection_ownership_global_filter,
            hard_support=args.projection_ownership_hard_support,
            seed_surface=args.projection_ownership_seed_surface,
            patch_resolution=32,
            mask_patch_threshold=args.projection_ownership_mask_patch_threshold,
            depth_fill_radius_patches=args.projection_ownership_depth_fill_radius,
            surface_band_m=args.projection_ownership_surface_band_m,
            min_free_views=args.projection_ownership_min_free_views,
            allow_unknown=args.projection_ownership_allow_unknown,
            part_count=len(args.projection_ownership_roi_box),
        )
        (out_path / "projection_ownership.json").write_text(
            json.dumps(projection_ownership_diagnostics, indent=2) + "\n"
        )
        print(
            f"[projection_ownership] mode={args.projection_ownership_mode} "
            f"depth_known={projection_ownership_diagnostics['depth_known_fraction_of_mask']:.1%} "
            f"training_tracks={projection_ownership_diagnostics['training_tracks_in_roi']}"
        )
    instance_anchor_points_chunk0 = None
    instance_anchor_view_mask = None
    instance_anchor_diagnostics: dict = {}
    if args.instance_anchor_roi_box is not None:
        anchor_world, instance_anchor_diagnostics = _prepare_instance_anchors(
            scene_path / args.colmap_subdir / "points3D.txt",
            args.instance_anchor_roi_box,
            max_reprojection_error=args.max_reproj_error if args.max_reproj_error is not None else 2.0,
            min_track_length=args.min_track_len if args.min_track_len is not None else 3,
            token_count=args.instance_anchor_token_count,
            holdout_fraction=args.instance_anchor_holdout_fraction,
            seed=args.seed,
        )
        ones = torch.ones(len(anchor_world), 1, dtype=anchor_world.dtype)
        anchor_h = torch.cat([anchor_world, ones], dim=1)
        instance_anchor_points_chunk0 = (anchor_h @ m_o2c[0].T)[:, :3]
        instance_anchor_diagnostics["target_chunk_ids"] = args.instance_anchor_chunks
        instance_anchor_diagnostics["token_xyz_chunk0"] = instance_anchor_points_chunk0.tolist()
        image_ids = _colmap_image_ids_by_name(scene_path / args.colmap_subdir / "images.txt")
        camera_document = json.loads((out_path / "cameras.json").read_text())
        scene_image_ids = [image_ids[Path(view["img_path"]).name] for view in camera_document["scene"]]
        instance_anchor_view_mask = torch.tensor(
            [
                [image_id in observed for image_id in scene_image_ids]
                for observed in instance_anchor_diagnostics["token_observation_image_ids"]
            ],
            dtype=torch.bool,
        )
        instance_anchor_diagnostics["scene_image_ids"] = scene_image_ids
        instance_anchor_diagnostics["track_scene_view_count"] = instance_anchor_view_mask.sum(dim=1).tolist()
        (out_path / "instance_anchors.json").write_text(
            json.dumps(instance_anchor_diagnostics, indent=2) + "\n"
        )
        print(
            f"[instance_anchors] selected {len(anchor_world)} tokens from "
            f"{instance_anchor_diagnostics['training_points']} training points; "
            f"held out {instance_anchor_diagnostics['holdout_points']}"
        )

    if args.save_imgs:
        scene_dir = out_path / "scene"
        scene_dir.mkdir(parents=True, exist_ok=True)
        for view_idx, img in enumerate(sel.scene_images_1024):
            arr = (img.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).save(scene_dir / f"view_{view_idx:03d}.png")
        for chunk_idx, img in zip(sel.chunk_indices, sel.cond2d_images_1024):
            chunk_dir = out_path / f"chunk_{chunk_idx:03d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            arr = (img.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).save(chunk_dir / "cond2d.png")

    ss_sampler_params: dict = {}

    slat_sampler_params: dict = {}
    if args.boundary_sensitive_slat:
        slat_sampler_params = {"boundary_sensitive": True, "boundary_width": args.boundary_width_slat}
    overlap_diagnostic_records: list[dict] | None = [] if args.overlap_diagnostics else None
    scene_mesh, coords_list = pipeline.run(
        sel,
        rel_t_kept,
        seed=args.seed,
        pipeline_type=args.pipeline,
        sparse_structure_sampler_params=ss_sampler_params,
        shape_slat_sampler_params=slat_sampler_params,
        tex_slat_sampler_params=slat_sampler_params,
        occ_threshold=args.occ_threshold,
        joint_decode_max_chunks_per_group=args.joint_decode_max_chunks_per_group,
        joint_decode_max_inflated_voxels=args.joint_decode_max_inflated_voxels,
        overlap_diagnostics=overlap_diagnostic_records,
        overlap_diagnostics_roi_bounds=overlap_roi_chunk0,
        instance_anchor_points_chunk0=instance_anchor_points_chunk0,
        instance_anchor_view_mask=instance_anchor_view_mask,
        instance_anchor_chunk_ids=args.instance_anchor_chunks,
        instance_anchor_diagnostics=instance_anchor_diagnostics,
        projection_ownership=projection_ownership,
        projection_ownership_diagnostics=projection_ownership_diagnostics,
        object_feature_only=args.object_feature_only,
        object_feature_keep_global_tokens=not args.object_feature_drop_global_tokens,
        cond2d_scene_view_indices=object_cond2d_view_indices,
    )
    if instance_anchor_points_chunk0 is not None:
        (out_path / "instance_anchors.json").write_text(
            json.dumps(instance_anchor_diagnostics, indent=2) + "\n"
        )
    if projection_ownership is not None:
        (out_path / "projection_ownership.json").write_text(
            json.dumps(projection_ownership_diagnostics, indent=2) + "\n"
        )

    if args.overlap_diagnostics:
        diagnostics_path = Path(args.overlap_diagnostics)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        with diagnostics_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "schema": "genrecon.overlap-diagnostics",
                    "schema_version": 1,
                    "seed": args.seed,
                    "chunk_indices": list(sel.chunk_indices),
                    "roi_boxes_world": args.overlap_diagnostics_roi_box or [],
                    "roi_boxes_chunk0": overlap_roi_chunk0 or [],
                    "records": overlap_diagnostic_records,
                },
                f,
                indent=2,
            )
        print(f"[overlap_diagnostics] wrote {len(overlap_diagnostic_records):,} records to {diagnostics_path}")

    coords_resolution = pipeline.models[f"shape_slat_flow_model_{args.pipeline}"].resolution

    if args.validation_crop_idx is not None:
        # Chunk-local save: the GT GLB at crops/<id>_<idx>.glb lives in this exact frame,
        # and m_c2o[0] from the validation JSON is generally rotated, so the axis-aligned
        # world-lift shortcut below would be wrong. Lift later via crop_transform.json.
        identity = torch.eye(4, dtype=torch.float32)
        _save_plys(
            out_path,
            scene_mesh,
            coords_list,
            coords_resolution,
            sel.chunk_indices,
            mesh_transform=identity,
            coords_transform=lambda i: identity,
            label="\n PLY saved (chunk-local)!",
        )
        _save_to_glb_inputs(
            out_path,
            scene_mesh,
            vertices=scene_mesh.vertices,
            voxel_size=scene_mesh.voxel_size,
            origin=[scene_mesh.origin[i].item() for i in range(3)],
            label="chunk-local ",
        )

        # Chunk-local "world": chunk 0 sits at the origin and the cube has unit
        # side length, so chunked_to_glb.py can consume this directly. The
        # generated scene.glb will be in chunk-local coords (matching gt.glb).
        # Lift to true world later via M_chunk_to_original if needed.
        chunk_inputs = {
            "chunk_centers_world": torch.zeros(len(sel.chunk_indices), 3, dtype=torch.float32),
            "chunk_size_world": 1.0,
            "chunk_indices": list(sel.chunk_indices),
            "M_chunk_to_original": m_c2o[0].detach().cpu(),
        }
        torch.save(chunk_inputs, out_path / "chunk_inputs.pt")
        print(f"[chunked] saved chunk metadata to {out_path / 'chunk_inputs.pt'}")
    else:
        # Joint frame = chunker's chunk-0 local frame, so m_c2o[0] lifts it to world.
        chunk_size = m_c2o[0][0, 0].item()
        chunk_center0 = m_c2o[0][:3, 3].to(scene_mesh.vertices.device, dtype=scene_mesh.vertices.dtype)
        _save_plys(
            out_path,
            scene_mesh,
            coords_list,
            coords_resolution,
            sel.chunk_indices,
            mesh_transform=m_c2o[0],
            coords_transform=lambda i: m_c2o[i],
            label="\n PLY saved!",
        )

        # ── Save inputs for chunked GLB conversion (testing harness) ──
        _save_to_glb_inputs(
            out_path,
            scene_mesh,
            vertices=scene_mesh.vertices * chunk_size + chunk_center0,
            voxel_size=scene_mesh.voxel_size * chunk_size,
            origin=[scene_mesh.origin[i].item() * chunk_size + chunk_center0[i].item() for i in range(3)],
            label="",
        )

        chunk_inputs = {
            "chunk_centers_world": torch.stack([m_c2o[i][:3, 3] for i in sel.chunk_indices]).detach().cpu(),
            "chunk_size_world": chunk_size,
            "chunk_indices": list(sel.chunk_indices),
        }
        torch.save(chunk_inputs, out_path / "chunk_inputs.pt")
        print(f"[chunked] saved chunk metadata to {out_path / 'chunk_inputs.pt'}")


if __name__ == "__main__":
    main()
