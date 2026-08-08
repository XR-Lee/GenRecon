#!/usr/bin/env python3
"""Render COLMAP camera views and compare source images with PLY/GLB output.

Large meshes can be rendered in independent processes with ``--stage ply`` and
``--stage glb``. ``--stage analyze`` then produces comparisons and metrics.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
from dataclasses import dataclass
from html import escape as html_escape
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    intrinsic: np.ndarray


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    camera_id: int
    name: str
    world_to_camera: np.ndarray
    observations: np.ndarray


@dataclass(frozen=True)
class ColmapPoint:
    xyz: np.ndarray
    reprojection_error: float
    track_length: int


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n")


def quaternion_to_rotation(qvec: Iterable[float]) -> np.ndarray:
    q = np.asarray(tuple(qvec), dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"Expected a four-component quaternion, got {q.shape}")
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("COLMAP quaternion must be finite and non-zero")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def read_colmap_cameras(path: str | Path) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        values = line.split()
        if len(values) < 5:
            raise ValueError(f"Malformed COLMAP camera line: {line}")
        camera_id = int(values[0])
        model = values[1]
        width, height = int(values[2]), int(values[3])
        params = [float(value) for value in values[4:]]
        if model == "PINHOLE" and len(params) == 4:
            fx, fy, cx, cy = params
        elif model == "SIMPLE_PINHOLE" and len(params) == 3:
            fx, cx, cy = params
            fy = fx
        else:
            raise ValueError(
                f"Unsupported camera model {model!r}; use undistorted PINHOLE/SIMPLE_PINHOLE inputs"
            )
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        cameras[camera_id] = ColmapCamera(camera_id, model, width, height, intrinsic)
    if not cameras:
        raise ValueError(f"No COLMAP cameras found in {path}")
    return cameras


def read_colmap_images(path: str | Path) -> dict[str, ColmapImage]:
    lines = Path(path).read_text().splitlines()
    images: dict[str, ColmapImage] = {}
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#"):
            continue
        values = line.split()
        if len(values) != 10:
            raise ValueError(f"Malformed COLMAP image header: {line[:160]}")
        image_id = int(values[0])
        rotation = quaternion_to_rotation(float(value) for value in values[1:5])
        translation = np.asarray([float(value) for value in values[5:8]], dtype=np.float64)
        camera_id = int(values[8])
        name = values[9]
        world_to_camera = np.eye(4, dtype=np.float64)
        world_to_camera[:3, :3] = rotation
        world_to_camera[:3, 3] = translation

        observations = np.empty((0, 3), dtype=np.float64)
        if index < len(lines):
            point_values = lines[index].strip().split()
            index += 1
            if point_values:
                if len(point_values) % 3:
                    raise ValueError(f"Malformed POINTS2D line following {name}")
                observations = np.asarray(point_values, dtype=np.float64).reshape(-1, 3)
        images[Path(name).name] = ColmapImage(
            image_id=image_id,
            camera_id=camera_id,
            name=name,
            world_to_camera=world_to_camera,
            observations=observations,
        )
    if not images:
        raise ValueError(f"No COLMAP images found in {path}")
    return images


def read_colmap_points(path: str | Path) -> dict[int, ColmapPoint]:
    points: dict[int, ColmapPoint] = {}
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        values = line.split()
        if len(values) < 8 or (len(values) - 8) % 2:
            raise ValueError(f"Malformed COLMAP point line: {line[:160]}")
        point_id = int(values[0])
        points[point_id] = ColmapPoint(
            xyz=np.asarray([float(value) for value in values[1:4]], dtype=np.float64),
            reprojection_error=float(values[7]),
            track_length=(len(values) - 8) // 2,
        )
    return points


def genrecon_glb_to_world_matrix() -> np.ndarray:
    """Map exported glTF ``(x, z, -y)`` coordinates back to COLMAP world."""

    return np.asarray(
        [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )


def _input_view_names(cameras_json: Path | None) -> set[str]:
    if cameras_json is None:
        return set()
    document = json.loads(cameras_json.read_text())
    return {Path(entry["img_path"]).name for entry in document["scene"]}


def _view_dir(output: Path, group: str, image_name: str) -> Path:
    return output / "views" / group / Path(image_name).stem


def _resolve_image(images_root: Path, record: ColmapImage) -> Path:
    candidates = [images_root / record.name, images_root / Path(record.name).name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not resolve source image {record.name!r} under {images_root}")


def _scaled_camera(camera: ColmapCamera, output_width: int) -> tuple[int, int, np.ndarray]:
    output_height = max(1, int(round(camera.height * output_width / camera.width)))
    sx, sy = output_width / camera.width, output_height / camera.height
    intrinsic = camera.intrinsic.copy()
    intrinsic[0, :] *= sx
    intrinsic[1, :] *= sy
    return output_width, output_height, intrinsic


def _save_source_and_camera(
    view_path: Path,
    source_path: Path,
    record: ColmapImage,
    camera: ColmapCamera,
    group: str,
    width: int,
    height: int,
    intrinsic: np.ndarray,
) -> None:
    view_path.mkdir(parents=True, exist_ok=True)
    original_path = view_path / "original.jpg"
    if not original_path.is_file():
        with Image.open(source_path) as image:
            image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS).save(
                original_path, quality=95, subsampling=0
            )
    _write_json(
        view_path / "camera.json",
        {
            "camera_id": camera.camera_id,
            "camera_model": camera.model,
            "group": group,
            "image_id": record.image_id,
            "image_name": Path(record.name).name,
            "intrinsic_render": intrinsic,
            "render_height": height,
            "render_width": width,
            "source_height": camera.height,
            "source_width": camera.width,
            "world_to_camera": record.world_to_camera,
        },
    )


def render_asset(
    *,
    asset_kind: str,
    asset_path: Path,
    cameras: dict[int, ColmapCamera],
    images: dict[str, ColmapImage],
    images_root: Path,
    input_names: set[str],
    output: Path,
    width: int,
) -> None:
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")
    import open3d as o3d

    first_camera = cameras[next(iter(cameras))]
    render_width, render_height, _ = _scaled_camera(first_camera, width)
    renderer = o3d.visualization.rendering.OffscreenRenderer(render_width, render_height)
    renderer.scene.set_background(np.asarray([0, 0, 0, 0], dtype=np.float32))

    if asset_kind == "ply":
        print(f"[fidelity] loading PLY: {asset_path}")
        mesh = o3d.io.read_triangle_mesh(str(asset_path), enable_post_processing=False, print_progress=True)
        if not mesh.has_triangles():
            raise ValueError(f"PLY has no triangles: {asset_path}")
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.base_color = np.asarray([1, 1, 1, 1], dtype=np.float32)
        renderer.scene.add_geometry("ply", mesh, material)
        coordinate_map = np.eye(4, dtype=np.float64)
        geometry_summary = {
            "triangles": len(mesh.triangles),
            "vertices": len(mesh.vertices),
            "vertex_colors": mesh.has_vertex_colors(),
        }
    elif asset_kind == "glb":
        print(f"[fidelity] loading GLB: {asset_path}")
        model = o3d.io.read_triangle_model(str(asset_path), print_progress=True)
        if not model.meshes:
            raise ValueError(f"GLB has no meshes: {asset_path}")
        for material in model.materials:
            # Isolate baked albedo from viewer-specific IBL and tone mapping.
            material.shader = "defaultUnlit"
        renderer.scene.add_model("glb", model)
        coordinate_map = genrecon_glb_to_world_matrix()
        geometry_summary = {
            "materials": len(model.materials),
            "mesh_primitives": len(model.meshes),
            "triangles": sum(len(item.mesh.triangles) for item in model.meshes),
            "vertices": sum(len(item.mesh.vertices) for item in model.meshes),
        }
    else:
        raise ValueError(f"Unknown asset kind: {asset_kind}")

    rendered: list[dict[str, Any]] = []
    ordered_images = sorted(images.values(), key=lambda item: item.name)
    for view_index, record in enumerate(ordered_images, start=1):
        camera = cameras[record.camera_id]
        view_width, view_height, intrinsic = _scaled_camera(camera, width)
        if (view_width, view_height) != (render_width, render_height):
            raise ValueError("All cameras must have the same dimensions for one render pass")
        image_name = Path(record.name).name
        group = "input" if image_name in input_names else "heldout"
        view_path = _view_dir(output, group, image_name)
        source_path = _resolve_image(images_root, record)
        _save_source_and_camera(
            view_path, source_path, record, camera, group, view_width, view_height, intrinsic
        )

        # Open3D expects world-to-camera. The right-side map converts GLB to world.
        extrinsic = record.world_to_camera @ coordinate_map
        renderer.setup_camera(intrinsic, extrinsic, view_width, view_height)
        color = np.asarray(renderer.render_to_image())[:, :, :3]
        depth = np.asarray(renderer.render_to_depth_image(z_in_view_space=True))
        mask = np.isfinite(depth) & (depth > 0)

        rgba = np.concatenate([color, (mask.astype(np.uint8) * 255)[..., None]], axis=2)
        Image.fromarray(rgba).save(view_path / f"{asset_kind}_render.png")
        Image.fromarray(mask.astype(np.uint8) * 255).save(view_path / f"{asset_kind}_mask.png")
        depth_mm = np.zeros(depth.shape, dtype=np.uint16)
        depth_mm[mask] = np.clip(np.rint(depth[mask] * 1000), 1, np.iinfo(np.uint16).max).astype(
            np.uint16
        )
        Image.fromarray(depth_mm).save(view_path / f"{asset_kind}_depth_mm.png")
        coverage = float(mask.mean())
        rendered.append({"group": group, "image": image_name, "mask_coverage": coverage})
        print(
            f"[fidelity] {asset_kind} {view_index:02d}/{len(ordered_images)} "
            f"{image_name}: coverage={coverage:.3f}"
        )

    _write_json(
        output / f"render_{asset_kind}.json",
        {
            "asset": str(asset_path),
            "asset_kind": asset_kind,
            "coordinate_map_asset_to_world": coordinate_map,
            "geometry": geometry_summary,
            "render_height": render_height,
            "render_width": render_width,
            "shader": "defaultUnlit",
            "views": rendered,
        },
    )
    del renderer
    gc.collect()


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _load_render(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    return rgba[:, :, :3], rgba[:, :, 3] > 127


def _load_depth(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        depth_mm = np.asarray(image, dtype=np.uint16)
    return depth_mm.astype(np.float32) / 1000.0


def masked_psnr(reference: np.ndarray, estimate: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    difference = reference.astype(np.float64) / 255.0 - estimate.astype(np.float64) / 255.0
    mse = float(np.mean(np.square(difference[mask])))
    return float("inf") if mse == 0 else float(-10 * math.log10(mse))


def masked_ssim(reference: np.ndarray, estimate: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    x = reference.astype(np.float32) / 255.0
    y = estimate.astype(np.float32) / 255.0
    scores: list[np.ndarray] = []
    c1, c2 = 0.01**2, 0.03**2
    for channel in range(3):
        xc, yc = x[:, :, channel], y[:, :, channel]
        mu_x = cv2.GaussianBlur(xc, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(yc, (11, 11), 1.5)
        sigma_x = cv2.GaussianBlur(xc * xc, (11, 11), 1.5) - mu_x * mu_x
        sigma_y = cv2.GaussianBlur(yc * yc, (11, 11), 1.5) - mu_y * mu_y
        sigma_xy = cv2.GaussianBlur(xc * yc, (11, 11), 1.5) - mu_x * mu_y
        numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        scores.append(numerator / np.maximum(denominator, 1e-12))
    core = cv2.erode(mask.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    valid = core if np.any(core) else mask
    return float(np.mean(np.stack(scores, axis=-1)[valid]))


def _affine_color_match(
    reference: np.ndarray, estimate: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, list[dict[str, float]]]:
    reference_float = reference.astype(np.float64) / 255.0
    estimate_float = estimate.astype(np.float64) / 255.0
    samples = np.flatnonzero(mask.reshape(-1))
    if samples.size > 100_000:
        samples = samples[np.linspace(0, samples.size - 1, 100_000, dtype=np.int64)]
    corrected = estimate_float.copy()
    parameters: list[dict[str, float]] = []
    for channel in range(3):
        source = estimate_float[:, :, channel].reshape(-1)[samples]
        target = reference_float[:, :, channel].reshape(-1)[samples]
        design = np.column_stack([source, np.ones_like(source)])
        slope, offset = np.linalg.lstsq(design, target, rcond=None)[0]
        slope = float(np.clip(slope, 0.25, 4.0))
        offset = float(np.clip(offset, -0.5, 0.5))
        corrected[:, :, channel] = np.clip(estimate_float[:, :, channel] * slope + offset, 0, 1)
        parameters.append({"offset": offset, "slope": slope})
    return np.rint(corrected * 255).astype(np.uint8), parameters


def _edge_f1(reference: np.ndarray, estimate: np.ndarray, mask: np.ndarray, tolerance: int = 3) -> float:
    core = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    if not np.any(core):
        return float("nan")
    ref_edge = (cv2.Canny(cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY), 80, 160) > 0) & core
    est_edge = (cv2.Canny(cv2.cvtColor(estimate, cv2.COLOR_RGB2GRAY), 80, 160) > 0) & core
    if not np.any(ref_edge) or not np.any(est_edge):
        return float("nan")
    distance_to_ref = cv2.distanceTransform((~ref_edge).astype(np.uint8), cv2.DIST_L2, 3)
    distance_to_est = cv2.distanceTransform((~est_edge).astype(np.uint8), cv2.DIST_L2, 3)
    precision = float(np.mean(distance_to_ref[est_edge] <= tolerance))
    recall = float(np.mean(distance_to_est[ref_edge] <= tolerance))
    return 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))


class MaskedLpips:
    def __init__(self, enabled: bool) -> None:
        self.model = None
        self.device = "cpu"
        if not enabled:
            return
        import lpips
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = lpips.LPIPS(net="alex", verbose=False).eval().to(self.device)

    def __call__(self, reference: np.ndarray, estimate: np.ndarray, mask: np.ndarray) -> float:
        if self.model is None:
            return float("nan")
        import torch

        patch_size, stride = 128, 64
        height, width = mask.shape
        candidates: list[tuple[int, int]] = []
        for top in range(0, max(1, height - patch_size + 1), stride):
            for left in range(0, max(1, width - patch_size + 1), stride):
                patch_mask = mask[top : top + patch_size, left : left + patch_size]
                if patch_mask.shape == (patch_size, patch_size) and patch_mask.mean() >= 0.8:
                    candidates.append((top, left))
        if not candidates:
            return float("nan")
        if len(candidates) > 12:
            indices = np.linspace(0, len(candidates) - 1, 12, dtype=np.int64)
            candidates = [candidates[index] for index in indices]

        reference_tensors = []
        estimate_tensors = []
        for top, left in candidates:
            ref = reference[top : top + patch_size, left : left + patch_size].copy()
            est = estimate[top : top + patch_size, left : left + patch_size].copy()
            valid = mask[top : top + patch_size, left : left + patch_size]
            est[~valid] = ref[~valid]
            reference_tensors.append(torch.from_numpy(ref).permute(2, 0, 1).float() / 127.5 - 1)
            estimate_tensors.append(torch.from_numpy(est).permute(2, 0, 1).float() / 127.5 - 1)
        values = []
        with torch.no_grad():
            for start in range(0, len(reference_tensors), 6):
                ref_batch = torch.stack(reference_tensors[start : start + 6]).to(self.device)
                est_batch = torch.stack(estimate_tensors[start : start + 6]).to(self.device)
                values.extend(self.model(ref_batch, est_batch).flatten().cpu().tolist())
        return float(np.mean(values))


def image_metrics(
    reference: np.ndarray,
    estimate: np.ndarray,
    mask: np.ndarray,
    lpips_metric: MaskedLpips,
) -> tuple[dict[str, Any], np.ndarray]:
    if not np.any(mask):
        empty = float("nan")
        return {
            "color_affine": [],
            "coverage": 0.0,
            "edge_f1_3px": empty,
            "lpips_masked_patches": empty,
            "mae": empty,
            "psnr_db": empty,
            "psnr_exposure_compensated_db": empty,
            "ssim": empty,
            "ssim_exposure_compensated": empty,
        }, estimate
    corrected, affine = _affine_color_match(reference, estimate, mask)
    absolute = np.abs(reference.astype(np.float32) - estimate.astype(np.float32)) / 255.0
    return {
        "color_affine": affine,
        "coverage": float(mask.mean()),
        "edge_f1_3px": _edge_f1(reference, estimate, mask),
        "lpips_masked_patches": lpips_metric(reference, estimate, mask),
        "mae": float(np.mean(absolute[mask])),
        "psnr_db": masked_psnr(reference, estimate, mask),
        "psnr_exposure_compensated_db": masked_psnr(reference, corrected, mask),
        "ssim": masked_ssim(reference, estimate, mask),
        "ssim_exposure_compensated": masked_ssim(reference, corrected, mask),
    }, corrected


def _distance_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        empty = float("nan")
        return {
            "count": 0,
            "mean_m": empty,
            "median_m": empty,
            "p90_m": empty,
            "within_0.02m": empty,
            "within_0.05m": empty,
            "within_0.10m": empty,
        }
    return {
        "count": int(values.size),
        "mean_m": float(np.mean(values)),
        "median_m": float(np.median(values)),
        "p90_m": float(np.percentile(values, 90)),
        "within_0.02m": float(np.mean(values <= 0.02)),
        "within_0.05m": float(np.mean(values <= 0.05)),
        "within_0.10m": float(np.mean(values <= 0.10)),
    }


def rendered_geometry_metrics(ply_depth: np.ndarray, glb_depth: np.ndarray) -> dict[str, Any]:
    ply_mask, glb_mask = ply_depth > 0, glb_depth > 0
    intersection = ply_mask & glb_mask
    union = ply_mask | glb_mask
    differences = np.abs(ply_depth[intersection] - glb_depth[intersection])
    return {
        "depth_absolute_difference": _distance_summary(differences),
        "intersection_coverage": float(intersection.mean()),
        "mask_iou": float(intersection.sum() / union.sum()) if np.any(union) else float("nan"),
        "ply_coverage": float(ply_mask.mean()),
        "glb_coverage": float(glb_mask.mean()),
    }


def sparse_depth_metrics(
    record: ColmapImage,
    camera: ColmapCamera,
    points: dict[int, ColmapPoint],
    depth: np.ndarray,
    *,
    max_reprojection_error: float = 2.0,
    min_track_length: int = 3,
) -> dict[str, Any]:
    height, width = depth.shape
    sx, sy = width / camera.width, height / camera.height
    eligible = 0
    signed_differences: list[float] = []
    for x, y, point_id_float in record.observations:
        point_id = int(point_id_float)
        point = points.get(point_id)
        if (
            point is None
            or point.reprojection_error > max_reprojection_error
            or point.track_length < min_track_length
        ):
            continue
        camera_point = record.world_to_camera[:3, :3] @ point.xyz + record.world_to_camera[:3, 3]
        if camera_point[2] <= 0:
            continue
        column = int(np.clip(round(x * sx), 0, width - 1))
        row = int(np.clip(round(y * sy), 0, height - 1))
        eligible += 1
        rendered_depth = float(depth[row, column])
        if rendered_depth > 0:
            signed_differences.append(rendered_depth - float(camera_point[2]))
    signed = np.asarray(signed_differences, dtype=np.float64)
    summary = _distance_summary(np.abs(signed))
    summary.update(
        {
            "eligible_points": eligible,
            "mesh_depth_coverage": float(signed.size / eligible) if eligible else float("nan"),
            "signed_mean_m": float(np.mean(signed)) if signed.size else float("nan"),
        }
    )
    return summary


def _overlay(reference: np.ndarray, estimate: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = reference.astype(np.float32).copy()
    result[mask] = 0.5 * reference[mask] + 0.5 * estimate[mask]
    return np.rint(result).astype(np.uint8)


def _error_heatmap(reference: np.ndarray, estimate: np.ndarray, mask: np.ndarray) -> np.ndarray:
    error = np.mean(np.abs(reference.astype(np.float32) - estimate.astype(np.float32)), axis=2)
    scaled = np.clip(error / 128.0 * 255.0, 0, 255).astype(np.uint8)
    heatmap = cv2.cvtColor(cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    heatmap[~mask] = 0
    return heatmap


def _depth_heatmap(ply_depth: np.ndarray, glb_depth: np.ndarray) -> np.ndarray:
    valid = (ply_depth > 0) & (glb_depth > 0)
    difference_cm = np.abs(ply_depth - glb_depth) * 100
    scaled = np.clip(difference_cm / 10.0 * 255.0, 0, 255).astype(np.uint8)
    heatmap = cv2.cvtColor(cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    heatmap[~valid] = 0
    return heatmap


def _labeled_panel(image: np.ndarray, label: str, width: int = 480) -> np.ndarray:
    height = int(round(image.shape[0] * width / image.shape[1]))
    resized = np.asarray(Image.fromarray(image).resize((width, height), Image.Resampling.LANCZOS))
    canvas = Image.new("RGB", (width, height + 34), (24, 24, 24))
    canvas.paste(Image.fromarray(resized), (0, 34))
    ImageDraw.Draw(canvas).text((12, 9), label, fill=(245, 245, 245))
    return np.asarray(canvas)


def _write_comparison(
    view_path: Path,
    original: np.ndarray,
    ply: np.ndarray,
    ply_mask: np.ndarray,
    glb: np.ndarray,
    glb_mask: np.ndarray,
    ply_depth: np.ndarray,
    glb_depth: np.ndarray,
) -> None:
    ply_overlay = _overlay(original, ply, ply_mask)
    glb_overlay = _overlay(original, glb, glb_mask)
    glb_error = _error_heatmap(original, glb, glb_mask)
    geometry_error = _depth_heatmap(ply_depth, glb_depth)
    Image.fromarray(ply_overlay).save(view_path / "ply_overlay.jpg", quality=94)
    Image.fromarray(glb_overlay).save(view_path / "glb_overlay.jpg", quality=94)
    Image.fromarray(glb_error).save(view_path / "glb_rgb_error.png")
    Image.fromarray(geometry_error).save(view_path / "ply_glb_depth_error.png")
    panels = [
        _labeled_panel(original, "Original COLMAP view"),
        _labeled_panel(ply, "PLY vertex color (transparent = missing)"),
        _labeled_panel(glb, "GLB baked albedo (transparent = missing)"),
        _labeled_panel(ply_overlay, "Original / PLY 50-50 overlay"),
        _labeled_panel(glb_overlay, "Original / GLB 50-50 overlay"),
        _labeled_panel(glb_error, "GLB RGB error (0..128 intensity)"),
    ]
    top = np.concatenate(panels[:3], axis=1)
    bottom = np.concatenate(panels[3:], axis=1)
    Image.fromarray(np.concatenate([top, bottom])).save(
        view_path / "comparison.jpg", quality=92, subsampling=0
    )


def _flatten_metrics(prefix: str, value: dict[str, Any], output: dict[str, Any]) -> None:
    for key, item in value.items():
        name = f"{prefix}_{key}" if prefix else key
        if isinstance(item, dict):
            _flatten_metrics(name, item, output)
        elif isinstance(item, (int, float, np.integer, np.floating)):
            output[name] = float(item)


def _aggregate(rows: list[dict[str, Any]], group: str) -> dict[str, Any]:
    selected = rows if group == "all" else [row for row in rows if row["group"] == group]
    numeric_keys = sorted(
        {
            key
            for row in selected
            for key, value in row.items()
            if key not in {"group", "image"} and isinstance(value, (int, float))
        }
    )
    metrics: dict[str, Any] = {}
    for key in numeric_keys:
        values = np.asarray([row.get(key, float("nan")) for row in selected], dtype=np.float64)
        finite = values[np.isfinite(values)]
        metrics[key] = {
            "count": int(finite.size),
            "mean": float(np.mean(finite)) if finite.size else float("nan"),
            "median": float(np.median(finite)) if finite.size else float("nan"),
        }
    return {"metrics": metrics, "views": len(selected)}


def _metric_stat(summary: dict[str, Any], group: str, key: str, statistic: str = "mean") -> float:
    return float(summary["subsets"][group]["metrics"].get(key, {}).get(statistic, float("nan")))


def _metric_mean(summary: dict[str, Any], group: str, key: str) -> float:
    return _metric_stat(summary, group, key, "mean")


def _metric_median(summary: dict[str, Any], group: str, key: str) -> float:
    return _metric_stat(summary, group, key, "median")


def _format_metric(value: float, digits: int = 3) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"


def _write_contact_sheets(output: Path) -> None:
    contact_root = output / "contact_sheets"
    contact_root.mkdir(parents=True, exist_ok=True)
    all_comparisons: dict[str, list[Path]] = {}
    for group in ("input", "heldout"):
        comparisons = sorted((output / "views" / group).glob("*/comparison.jpg"))
        all_comparisons[group] = comparisons
        for page_index, start in enumerate(range(0, len(comparisons), 8), start=1):
            images = []
            for path in comparisons[start : start + 8]:
                image = Image.open(path).convert("RGB")
                image.thumbnail((1200, 540), Image.Resampling.LANCZOS)
                images.append(image.copy())
            if not images:
                continue
            canvas = Image.new(
                "RGB", (max(image.width for image in images), sum(image.height for image in images))
            )
            top = 0
            for image in images:
                canvas.paste(image, (0, top))
                top += image.height
            canvas.save(contact_root / f"{group}_{page_index:02d}.jpg", quality=90)

    representatives: list[Path] = []
    for group in ("input", "heldout"):
        candidates = all_comparisons[group]
        if candidates:
            representatives.extend(
                candidates[index]
                for index in np.linspace(0, len(candidates) - 1, min(4, len(candidates)), dtype=int)
            )
    overview_images = []
    for path in representatives:
        image = Image.open(path).convert("RGB")
        image.thumbnail((1100, 500), Image.Resampling.LANCZOS)
        overview_images.append(image.copy())
    if overview_images:
        canvas = Image.new(
            "RGB", (max(image.width for image in overview_images), sum(image.height for image in overview_images))
        )
        top = 0
        for image in overview_images:
            canvas.paste(image, (0, top))
            top += image.height
        canvas.save(output / "overview.jpg", quality=92)


def _write_report(output: Path, summary: dict[str, Any], scene_label: str) -> None:
    input_views = int(summary["subsets"]["input"]["views"])
    heldout_views = int(summary["subsets"]["heldout"]["views"])
    total_views = int(summary["subsets"]["all"]["views"])
    overview_views = min(4, input_views) + min(4, heldout_views)
    quick_links = [
        f"- `overview.jpg`：{overview_views} 个代表视角。",
        f"- `contact_sheets/`：全部 {total_views} 个视角的分页对照。",
        "- `views/{input,heldout}/<image>/comparison.jpg`：逐视角六联图。",
        "- `metrics.csv`、`summary.json`：逐视角和聚合数值。",
    ]
    if (output / "sfm_glb_geometry" / "README.md").is_file():
        quick_links.append(
            "- `sfm_glb_geometry/README.md`：COLMAP/SfM 点到 GLB 三角面的精确 3D 一致性。"
        )
    if (output / "sfm_glb_geometry" / "rgb_overlays" / "overview.jpg").is_file():
        quick_links.append(
            "- `sfm_glb_geometry/rgb_overlays/overview.jpg`：将几何误差叠加回原始 RGB 视角。"
        )

    lines = [
        f"# {scene_label} 相机同视角几何与图像 Fidelity",
        "",
        "所有渲染均使用原始 COLMAP 内外参。"
        f"`input` 是 GenRecon 实际使用的 {input_views} 张物理相机，"
        f"`heldout` 是未输入模型的 {heldout_views} 张注册相机，后者更适合判断跨视角保真度。",
        "",
        "## 快速入口",
        "",
        *quick_links,
        "",
        "## 聚合结果（除注明外为逐视角均值）",
        "",
        "| 子集 | 资产 | 覆盖率 | PSNR | 曝光补偿 PSNR | SSIM | masked LPIPS | Edge F1@3px | SfM 深度覆盖 | SfM 深度中位误差（跨视角中位） | SfM @10cm |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("input", "heldout", "all"):
        for asset in ("ply", "glb"):
            prefix = f"{asset}_image"
            sparse_prefix = f"{asset}_sfm"
            values = [
                f"{_metric_mean(summary, group, prefix + '_coverage'):.1%}",
                _format_metric(_metric_mean(summary, group, prefix + "_psnr_db"), 2),
                _format_metric(
                    _metric_mean(summary, group, prefix + "_psnr_exposure_compensated_db"), 2
                ),
                _format_metric(_metric_mean(summary, group, prefix + "_ssim"), 3),
                _format_metric(_metric_mean(summary, group, prefix + "_lpips_masked_patches"), 3),
                _format_metric(_metric_mean(summary, group, prefix + "_edge_f1_3px"), 3),
                f"{_metric_mean(summary, group, sparse_prefix + '_mesh_depth_coverage'):.1%}",
                _format_metric(_metric_median(summary, group, sparse_prefix + "_median_m") * 100, 1)
                + "cm",
                f"{_metric_mean(summary, group, sparse_prefix + '_within_0.10m'):.1%}",
            ]
            lines.append(f"| {group} | {asset.upper()} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## PLY 与 GLB 几何一致性",
            "",
            "| 子集 | mask IoU | 共同覆盖率 | 深度中位差 | 深度 P90 差 | 深度 @2cm | 深度 @5cm |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group in ("input", "heldout", "all"):
        lines.append(
            "| "
            + group
            + " | "
            + " | ".join(
                [
                    _format_metric(_metric_mean(summary, group, "geometry_mask_iou")),
                    f"{_metric_mean(summary, group, 'geometry_intersection_coverage'):.1%}",
                    _format_metric(
                        _metric_mean(summary, group, "geometry_depth_absolute_difference_median_m") * 100,
                        2,
                    )
                    + "cm",
                    _format_metric(
                        _metric_mean(summary, group, "geometry_depth_absolute_difference_p90_m") * 100,
                        2,
                    )
                    + "cm",
                    f"{_metric_mean(summary, group, 'geometry_depth_absolute_difference_within_0.02m'):.1%}",
                    f"{_metric_mean(summary, group, 'geometry_depth_absolute_difference_within_0.05m'):.1%}",
                ]
            )
            + " |"
        )

    worst_views = sorted(
        (
            row
            for row in summary["views"]
            if np.isfinite(float(row.get("ply_sfm_median_m", float("nan"))))
        ),
        key=lambda row: float(row["ply_sfm_median_m"]),
        reverse=True,
    )[:3]
    lines.extend(
        [
            "",
            "## SfM 深度异常视角",
            "",
            "主表使用跨视角中位数描述典型误差；完整的逐视角均值、中位数和样本数仍保留在 `summary.json`。",
            "",
            "| 视角 | 子集 | mesh 深度覆盖 | 深度中位误差 | 深度 P90 | 有效 SfM 点 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in worst_views:
        lines.append(
            f"| {row['image']} | {row['group']} | {float(row['ply_sfm_mesh_depth_coverage']):.1%} | "
            f"{float(row['ply_sfm_median_m']):.2f}m | {float(row['ply_sfm_p90_m']):.2f}m | "
            f"{int(float(row['ply_sfm_count']))} |"
        )
    lines.extend(
        [
            "",
            "上表保留当前场景误差最大的三个注册视角，不把它们作为统计噪声删除；主表使用跨视角中位数同时表达典型表现。",
            "",
            "## 指标解释",
            "",
            "- 覆盖率只统计 mesh 在该相机中的有效深度像素；透明区域是未生成几何，不参与 RGB 指标。",
            "- GLB 使用无环境光的 `defaultUnlit` albedo 渲染，隔离烘焙纹理；PBR 查看器中的亮度会随 IBL 改变。",
            "- PSNR/SSIM/LPIPS 只衡量覆盖区域。曝光补偿 PSNR 对每通道拟合一个全局仿射变换。",
            "- masked LPIPS 使用覆盖率至少 80% 的 128x128 patch；数值越低越好。",
            "- SfM 深度误差将重投影误差不超过 2px、轨迹长度至少 3 的 COLMAP 点与渲染 z-buffer 比较。",
            "- PLY↔GLB 深度差衡量 GLB 分块、清理和简化造成的几何变化。",
            "",
            "这些不是数据集官方图像或深度评测分数。原图没有逐像素真值深度，且模型输入可能经过方形裁剪；",
            "因此本目录同时提供覆盖率、SfM 稀疏几何检查和完整可视化，不能只看单个 PSNR 数字。",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines))


def _write_html(output: Path, rows: list[dict[str, Any]], scene_label: str) -> None:
    cards = []
    for row in rows:
        stem = Path(row["image"]).stem
        relative = f"views/{row['group']}/{stem}"
        cards.append(
            f'<article><a href="{relative}/comparison.jpg"><img loading="lazy" '
            f'src="{relative}/comparison.jpg" alt="{stem} comparison"></a>'
            f'<h2>{stem} <span>{row["group"]}</span></h2></article>'
        )
    navigation = [
        '<a href="README.md">方法与聚合指标</a>',
        '<a href="overview.jpg">代表视角</a>',
        '<a href="metrics.csv">CSV</a>',
    ]
    if (output / "sfm_glb_geometry" / "README.md").is_file():
        navigation.append('<a href="sfm_glb_geometry/README.md">SfM↔GLB 几何</a>')
    if (output / "sfm_glb_geometry" / "rgb_overlays" / "overview.jpg").is_file():
        navigation.append(
            '<a href="sfm_glb_geometry/rgb_overlays/overview.jpg">几何 Error Overlay</a>'
        )
    label = html_escape(scene_label)
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{label} fidelity</title><style>
body{{margin:0;background:#111;color:#eee;font:14px system-ui,sans-serif}}header{{padding:20px 3vw;border-bottom:1px solid #444}}
main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:12px;padding:12px}}
article{{background:#1d1d1d;border:1px solid #3a3a3a;border-radius:4px;overflow:hidden}}img{{display:block;width:100%;height:auto}}
h1{{font-size:22px;margin:0 0 6px}}h2{{font-size:14px;margin:8px 12px 10px}}span{{color:#aaa;font-weight:400}}
a{{color:#9dc7ff}}p{{margin:4px 0}}</style></head><body><header><h1>{label} 同相机视角 Fidelity</h1>
<p>{' · '.join(navigation)}</p>
</header><main>{''.join(cards)}</main></body></html>"""
    (output / "index.html").write_text(html)


def analyze(
    *,
    cameras: dict[int, ColmapCamera],
    images: dict[str, ColmapImage],
    points: dict[int, ColmapPoint],
    input_names: set[str],
    output: Path,
    scene_label: str,
    use_lpips: bool,
) -> None:
    lpips_metric = MaskedLpips(use_lpips)
    rows: list[dict[str, Any]] = []
    for view_index, record in enumerate(sorted(images.values(), key=lambda item: item.name), start=1):
        image_name = Path(record.name).name
        group = "input" if image_name in input_names else "heldout"
        view_path = _view_dir(output, group, image_name)
        original = _load_rgb(view_path / "original.jpg")
        ply, ply_mask = _load_render(view_path / "ply_render.png")
        glb, glb_mask = _load_render(view_path / "glb_render.png")
        ply_depth = _load_depth(view_path / "ply_depth_mm.png")
        glb_depth = _load_depth(view_path / "glb_depth_mm.png")

        ply_image_metrics, _ = image_metrics(original, ply, ply_mask, lpips_metric)
        glb_image_metrics, _ = image_metrics(original, glb, glb_mask, lpips_metric)
        geometry = rendered_geometry_metrics(ply_depth, glb_depth)
        camera = cameras[record.camera_id]
        ply_sfm = sparse_depth_metrics(record, camera, points, ply_depth)
        glb_sfm = sparse_depth_metrics(record, camera, points, glb_depth)
        metrics = {
            "geometry": geometry,
            "glb": {"image": glb_image_metrics, "sfm_depth": glb_sfm},
            "group": group,
            "image": image_name,
            "ply": {"image": ply_image_metrics, "sfm_depth": ply_sfm},
        }
        _write_json(view_path / "metrics.json", metrics)
        _write_comparison(
            view_path, original, ply, ply_mask, glb, glb_mask, ply_depth, glb_depth
        )

        row: dict[str, Any] = {"group": group, "image": image_name}
        _flatten_metrics("ply_image", ply_image_metrics, row)
        _flatten_metrics("glb_image", glb_image_metrics, row)
        _flatten_metrics("ply_sfm", ply_sfm, row)
        _flatten_metrics("glb_sfm", glb_sfm, row)
        _flatten_metrics("geometry", geometry, row)
        rows.append(row)
        print(f"[fidelity] analyze {view_index:02d}/{len(images)} {image_name}")

    fieldnames = ["image", "group"] + sorted(
        {key for row in rows for key in row if key not in {"image", "group"}}
    )
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "metric_scope": {
            "image_metrics": "rendered mesh mask only",
            "lpips": "128x128 patches with at least 80% mesh coverage",
            "sfm_filter": {"max_reprojection_error_px": 2.0, "min_track_length": 3},
        },
        "subsets": {group: _aggregate(rows, group) for group in ("all", "input", "heldout")},
        "views": rows,
    }
    _write_json(output / "summary.json", summary)
    _write_contact_sheets(output)
    _write_report(output, summary, scene_label)
    _write_html(output, rows, scene_label)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("ply", "glb", "analyze", "all"), default="all")
    parser.add_argument("--ply", type=Path, required=True, help="World-coordinate vertex-color PLY")
    parser.add_argument("--glb", type=Path, required=True, help="GenRecon PBR GLB")
    parser.add_argument("--cameras", type=Path, required=True, help="COLMAP cameras.txt")
    parser.add_argument("--images", type=Path, required=True, help="COLMAP images.txt")
    parser.add_argument("--points", type=Path, required=True, help="COLMAP points3D.txt")
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--input-cameras-json", type=Path, help="GenRecon cameras.json for input/heldout split")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scene-label",
        help="Display label for reports; defaults to the output directory's parent name",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--no-lpips", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.width < 128:
        raise SystemExit("--width must be at least 128 pixels")
    for path in (args.ply, args.glb, args.cameras, args.images, args.points, args.images_root):
        if not path.exists():
            raise SystemExit(f"Required input does not exist: {path}")
    args.output.mkdir(parents=True, exist_ok=True)
    scene_label = args.scene_label or args.output.parent.name
    cameras = read_colmap_cameras(args.cameras)
    images = read_colmap_images(args.images)
    unknown_cameras = sorted({record.camera_id for record in images.values()} - set(cameras))
    if unknown_cameras:
        raise SystemExit(f"images.txt references unknown cameras: {unknown_cameras}")
    input_names = _input_view_names(args.input_cameras_json)
    unknown_inputs = sorted(input_names - set(images))
    if unknown_inputs:
        raise SystemExit(f"Input cameras are absent from images.txt: {unknown_inputs}")

    _write_json(
        args.output / "config.json",
        {
            "cameras": args.cameras,
            "glb": args.glb,
            "images": args.images,
            "images_root": args.images_root,
            "input_cameras_json": args.input_cameras_json,
            "input_views": sorted(input_names),
            "output": args.output,
            "ply": args.ply,
            "registered_views": len(images),
            "render_width": args.width,
            "scene_label": scene_label,
        },
    )

    common = dict(
        cameras=cameras,
        images=images,
        images_root=args.images_root,
        input_names=input_names,
        output=args.output,
        width=args.width,
    )
    if args.stage in {"ply", "all"}:
        render_asset(asset_kind="ply", asset_path=args.ply, **common)
    if args.stage in {"glb", "all"}:
        render_asset(asset_kind="glb", asset_path=args.glb, **common)
    if args.stage in {"analyze", "all"}:
        points = read_colmap_points(args.points)
        analyze(
            cameras=cameras,
            images=images,
            points=points,
            input_names=input_names,
            output=args.output,
            scene_label=scene_label,
            use_lpips=not args.no_lpips,
        )


if __name__ == "__main__":
    main()
