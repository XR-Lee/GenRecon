from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class ColmapTrackPoint:
    point_id: int
    xyz: np.ndarray
    image_ids: frozenset[int]


def read_colmap_image_ids(path: Path) -> dict[str, int]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    if len(lines) % 2:
        raise ValueError(f"COLMAP images file must contain two lines per image: {path}")
    return {Path(line.split()[9]).name: int(line.split()[0]) for line in lines[::2]}


def read_colmap_track_points(path: Path) -> dict[int, ColmapTrackPoint]:
    result: dict[int, ColmapTrackPoint] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 8 or (len(fields) - 8) % 2:
            raise ValueError(f"Malformed COLMAP point row: {line[:160]}")
        point_id = int(fields[0])
        result[point_id] = ColmapTrackPoint(
            point_id=point_id,
            xyz=np.asarray([float(fields[1]), float(fields[2]), float(fields[3])], dtype=np.float64),
            image_ids=frozenset(int(fields[index]) for index in range(8, len(fields), 2)),
        )
    return result


def point_in_boxes(point: np.ndarray, boxes: list[list[float]]) -> bool:
    return any(
        x0 <= point[0] <= x1 and y0 <= point[1] <= y1 and z0 <= point[2] <= z1
        for x0, x1, y0, y1, z0, z1 in boxes
    )


def downsample_mask(path: Path, patch_resolution: int, threshold: float) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    image = Image.open(path).convert("L")
    reduced = np.asarray(
        image.resize((patch_resolution, patch_resolution), Image.Resampling.BOX),
        dtype=np.float32,
    ) / 255.0
    return reduced >= threshold


def fill_sparse_patch_depth(
    sparse_depth: np.ndarray,
    observed: np.ndarray,
    allowed: np.ndarray,
    max_patch_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-fill sparse patch depths within a bounded image-space radius."""
    if sparse_depth.shape != observed.shape or observed.shape != allowed.shape:
        raise ValueError("depth, observed, and allowed patch grids must have the same shape")
    if not observed.any():
        return np.zeros_like(sparse_depth), np.zeros_like(observed), np.full_like(sparse_depth, np.inf)
    distance, nearest = distance_transform_edt(~observed, return_indices=True)
    filled = sparse_depth[tuple(nearest)]
    valid = allowed & (distance <= max_patch_distance)
    return filled.astype(np.float32), valid, distance.astype(np.float32)


def build_projection_ownership(
    *,
    camera_document: dict[str, Any],
    mask_dir: Path,
    anchors_document: dict[str, Any],
    colmap_images_path: Path,
    colmap_points_path: Path,
    roi_boxes_world: list[list[float]],
    roi_bounds_chunk0: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
    world_to_chunk0: torch.Tensor,
    chunk_size_m: float,
    mode: str,
    support_bounds_chunk0: list[tuple[tuple[float, float, float], tuple[float, float, float]]] | None = None,
    enabled_view_indices: set[int] | None = None,
    mask_view_indices: list[int | None] | None = None,
    global_feature_filter: bool = False,
    hard_support: bool = False,
    seed_surface: bool = False,
    patch_resolution: int = 32,
    mask_patch_threshold: float = 0.25,
    depth_fill_radius_patches: float = 2.0,
    surface_band_m: float = 0.08,
    min_free_views: int = 2,
    allow_unknown: bool = True,
    part_count: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if mode not in {"mask", "depth"}:
        raise ValueError(f"projection ownership mode must be mask or depth, got {mode!r}")
    if chunk_size_m <= 0 or surface_band_m <= 0:
        raise ValueError("chunk_size_m and surface_band_m must be positive")
    if min_free_views <= 0:
        raise ValueError("min_free_views must be positive")
    if hard_support and not support_bounds_chunk0:
        raise ValueError("hard_support requires at least one support bound")

    image_ids = read_colmap_image_ids(colmap_images_path)
    tracks = read_colmap_track_points(colmap_points_path)
    training_ids = set(int(value) for value in anchors_document["training_point_ids"])
    training_tracks = [
        tracks[point_id]
        for point_id in sorted(training_ids & tracks.keys())
        if point_in_boxes(tracks[point_id].xyz, roi_boxes_world)
    ]
    transform = np.asarray(world_to_chunk0, dtype=np.float64)

    if mask_view_indices is not None and len(mask_view_indices) != len(camera_document["scene"]):
        raise ValueError("mask_view_indices must contain one source index per scene view")

    masks: list[np.ndarray] = []
    part_masks: list[np.ndarray] = []
    depth_maps: list[np.ndarray] = []
    depth_valid: list[np.ndarray] = []
    view_records: list[dict[str, Any]] = []

    for view_index, camera in enumerate(camera_document["scene"]):
        mask_view_index = mask_view_indices[view_index] if mask_view_indices is not None else view_index
        view_enabled = (
            mask_view_index is not None
            and (enabled_view_indices is None or view_index in enabled_view_indices)
        )
        if mask_view_index is None:
            union_patch = np.zeros((patch_resolution, patch_resolution), dtype=bool)
        else:
            union_patch = downsample_mask(
                mask_dir / f"view_{mask_view_index:03d}.png",
                patch_resolution,
                mask_patch_threshold,
            )
            if not view_enabled:
                union_patch.fill(False)
        current_parts = []
        for part_index in range(part_count):
            part_path = (
                mask_dir / f"view_{mask_view_index:03d}_part_{part_index}.png"
                if mask_view_index is not None
                else None
            )
            if part_path is not None and part_path.is_file():
                part_mask = downsample_mask(part_path, patch_resolution, mask_patch_threshold)
                if not view_enabled:
                    part_mask.fill(False)
                current_parts.append(part_mask)
            else:
                current_parts.append(np.zeros_like(union_patch))
        if current_parts:
            part_stack = np.stack(current_parts, axis=0)
            # The union file remains authoritative for compatibility with the
            # deployed E run; part masks preserve attribution for diagnostics.
        else:
            part_stack = np.empty((0, patch_resolution, patch_resolution), dtype=bool)

        image_name = Path(camera["img_path"]).name
        if image_name not in image_ids:
            raise KeyError(f"scene view {view_index} image is absent from COLMAP: {image_name}")
        image_id = image_ids[image_name]
        extrinsics = np.asarray(camera["extrinsics_c0"], dtype=np.float64)
        intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
        patch_depths: list[list[list[float]]] = [
            [[] for _ in range(patch_resolution)] for _ in range(patch_resolution)
        ]
        projected_tracks = 0
        for track in training_tracks:
            if image_id not in track.image_ids:
                continue
            point_h = np.append(track.xyz, 1.0)
            point_chunk0 = transform @ point_h
            point_camera = extrinsics @ point_chunk0
            depth = float(point_camera[2])
            if depth <= 1e-8:
                continue
            uv1 = intrinsics @ (point_camera[:3] / depth)
            u, v = float(uv1[0]), float(uv1[1])
            if not (0.0 <= u < 1.0 and 0.0 <= v < 1.0):
                continue
            patch_x = min(patch_resolution - 1, int(np.floor(u * patch_resolution)))
            patch_y = min(patch_resolution - 1, int(np.floor(v * patch_resolution)))
            patch_depths[patch_y][patch_x].append(depth)
            projected_tracks += 1

        sparse_depth = np.zeros((patch_resolution, patch_resolution), dtype=np.float32)
        observed = np.zeros((patch_resolution, patch_resolution), dtype=bool)
        for patch_y in range(patch_resolution):
            for patch_x in range(patch_resolution):
                values = patch_depths[patch_y][patch_x]
                if values:
                    sparse_depth[patch_y, patch_x] = float(np.median(values))
                    observed[patch_y, patch_x] = True
        filled, known, distance = fill_sparse_patch_depth(
            sparse_depth,
            observed,
            union_patch,
            depth_fill_radius_patches,
        )
        masks.append(union_patch)
        part_masks.append(part_stack)
        depth_maps.append(filled)
        depth_valid.append(known)
        view_records.append(
            {
                "view_index": view_index,
                "image": image_name,
                "image_id": image_id,
                "mask_view_index": mask_view_index,
                "feature_enabled": view_enabled,
                "mask_patches": int(union_patch.sum()),
                "observed_depth_patches": int(observed.sum()),
                "filled_depth_patches": int(known.sum()),
                "projected_training_track_observations": projected_tracks,
                "median_fill_distance_patches": float(np.median(distance[known])) if known.any() else None,
                "part_mask_patches": [int(mask.sum()) for mask in part_stack],
            }
        )

    mask_tensor = torch.from_numpy(np.stack(masks)).reshape(len(masks), -1).bool()
    depth_tensor = torch.from_numpy(np.stack(depth_maps)).reshape(len(depth_maps), -1).float()
    valid_tensor = torch.from_numpy(np.stack(depth_valid)).reshape(len(depth_valid), -1).bool()
    if part_count:
        part_tensor = torch.from_numpy(np.stack(part_masks)).bool()
    else:
        part_tensor = torch.empty(len(masks), 0, patch_resolution, patch_resolution, dtype=torch.bool)
    roi_tensor = torch.tensor(roi_bounds_chunk0, dtype=torch.float32)
    ownership: dict[str, Any] = {
        "mode": mode,
        "mask": mask_tensor,
        "part_mask": part_tensor,
        "surface_depth": depth_tensor,
        "depth_valid": valid_tensor,
        "roi_bounds": roi_tensor,
        "support_bounds": torch.tensor(support_bounds_chunk0 or [], dtype=torch.float32),
        "global_feature_filter": bool(global_feature_filter),
        "hard_support": bool(hard_support),
        "seed_surface": bool(seed_surface),
        "surface_band": float(surface_band_m / chunk_size_m),
        "min_free_views": int(min_free_views),
        "allow_unknown": bool(allow_unknown),
        "patch_resolution": int(patch_resolution),
    }
    total_mask = int(mask_tensor.sum().item())
    total_depth = int(valid_tensor.sum().item())
    summary = {
        "schema": "genrecon.projection-ownership",
        "mode": mode,
        "mask_dir": str(mask_dir),
        "patch_resolution": patch_resolution,
        "mask_patch_threshold": mask_patch_threshold,
        "depth_fill_radius_patches": depth_fill_radius_patches,
        "surface_band_m": surface_band_m,
        "surface_band_chunk0": surface_band_m / chunk_size_m,
        "min_free_views": min_free_views,
        "allow_unknown": allow_unknown,
        "training_point_ids": len(training_ids),
        "training_tracks_in_roi": len(training_tracks),
        "roi_boxes_world": roi_boxes_world,
        "roi_bounds_chunk0": [
            {"lower": list(lower), "upper": list(upper)} for lower, upper in roi_bounds_chunk0
        ],
        "support_bounds_chunk0": [
            {"lower": list(lower), "upper": list(upper)}
            for lower, upper in (support_bounds_chunk0 or [])
        ],
        "global_feature_filter": bool(global_feature_filter),
        "hard_support": bool(hard_support),
        "seed_surface": bool(seed_surface),
        "enabled_view_indices": sorted(enabled_view_indices) if enabled_view_indices is not None else None,
        "mask_view_indices": mask_view_indices,
        "scene_views": len(masks),
        "mask_patches": total_mask,
        "depth_known_patches": total_depth,
        "depth_known_fraction_of_mask": total_depth / total_mask if total_mask else 0.0,
        "part_count": part_count,
        "views": view_records,
    }
    return ownership, summary
