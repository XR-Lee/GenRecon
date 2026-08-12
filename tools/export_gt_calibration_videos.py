#!/usr/bin/env python3
"""Export audited RGB/reference/GenRecon videos for representative GT units."""

from __future__ import annotations

import argparse
import gc
import hashlib
import html
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from tools.evaluate_mesh_pointcloud import load_meshlab_transforms
    from tools.evaluate_view_fidelity import (
        quaternion_to_rotation,
        read_colmap_cameras,
        read_colmap_images,
    )
    from tools.export_foundation_genrecon_videos import (
        encode_video,
        ffmpeg_executable,
        probe_video,
    )
except ModuleNotFoundError:
    from evaluate_mesh_pointcloud import load_meshlab_transforms
    from evaluate_view_fidelity import (
        quaternion_to_rotation,
        read_colmap_cameras,
        read_colmap_images,
    )
    from export_foundation_genrecon_videos import (
        encode_video,
        ffmpeg_executable,
        probe_video,
    )

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "configs" / "eval" / "gt_calibration_visualization_v1.json"
DEFAULT_REGISTRY = ROOT / "data" / "gt-calibration-v1" / "registry.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "gt-calibration-v1" / "visualizations-v1"
SCHEMA_VERSION = 1
AVAILABLE_STATUS = "available"
BLOCKED_STATUS = "blocked-auth"
MISSING_PREDICTION_STATUS = "missing-prediction"
PANEL_SIZE = (640, 480)
COMPARISON_SIZE = (1920, 720)
FONT_REGULAR = Path("/usr/share/fonts/truetype/lato/Lato-Medium.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/lato/Lato-Semibold.ttf")


class VisualizationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ViewCamera:
    index: int
    role: str
    order: int
    source_label: str
    rgb_path: Path
    source_width: int
    source_height: int
    intrinsic: np.ndarray
    distortion: np.ndarray | None
    world_to_camera: np.ndarray
    panel_intrinsic: np.ndarray
    panel_content_box: tuple[int, int, int, int]


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            VisualizationError(f"Non-finite JSON constant {value} in {path}")
        ),
    )


def write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def root_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _fit_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    start_size: int,
    *,
    minimum: int = 12,
    bold: bool = False,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for size in range(start_size, minimum - 1, -1):
        font = _font(size, bold=bold)
        if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
            return font
    return _font(minimum, bold=bold)


def _validate_matrix(matrix: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise VisualizationError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise VisualizationError(f"{label} has an invalid homogeneous row")
    return value


def panel_camera(
    intrinsic: np.ndarray,
    source_width: int,
    source_height: int,
    panel_width: int = PANEL_SIZE[0],
    panel_height: int = PANEL_SIZE[1],
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Source dimensions must be positive")
    content_scale = min(panel_width / source_width, panel_height / source_height)
    content_width = max(1, int(round(source_width * content_scale)))
    content_height = max(1, int(round(source_height * content_scale)))
    offset_x = (panel_width - content_width) // 2
    offset_y = (panel_height - content_height) // 2
    sx = content_width / source_width
    sy = content_height / source_height
    output = np.asarray(intrinsic, dtype=np.float64).copy()
    if output.shape != (3, 3) or not np.isfinite(output).all():
        raise VisualizationError("Camera intrinsic must be a finite 3x3 matrix")
    output[0, 0] *= sx
    output[0, 1] *= sx
    output[0, 2] = output[0, 2] * sx + offset_x
    output[1, 0] *= sy
    output[1, 1] *= sy
    output[1, 2] = output[1, 2] * sy + offset_y
    return output, (offset_x, offset_y, content_width, content_height)


def _intrinsics_from_record(
    record: dict[str, Any], image_size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray | None, int, int]:
    value = record.get("intrinsics")
    distortion: np.ndarray | None = None
    if isinstance(value, list):
        intrinsic = np.asarray(value, dtype=np.float64)
        width, height = record.get("image_size", image_size)
    elif isinstance(value, dict) and "params" in value:
        model = value.get("model")
        params = np.asarray(value["params"], dtype=np.float64)
        width, height = int(value["width"]), int(value["height"])
        if model == "OPENCV" and params.shape == (8,):
            fx, fy, cx, cy, k1, k2, p1, p2 = params
            distortion = np.asarray([k1, k2, p1, p2], dtype=np.float64)
        elif model == "PINHOLE" and params.shape == (4,):
            fx, fy, cx, cy = params
        elif model == "SIMPLE_PINHOLE" and params.shape == (3,):
            fx, cx, cy = params
            fy = fx
        else:
            raise VisualizationError(f"Unsupported camera model in GT split: {model}")
        intrinsic = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    elif isinstance(value, dict):
        required = {"fx", "fy", "cx", "cy"}
        if not required <= set(value):
            raise VisualizationError(f"Camera intrinsics are missing {sorted(required - set(value))}")
        width = int(value.get("width", image_size[0]))
        height = int(value.get("height", image_size[1]))
        intrinsic = np.asarray(
            [[value["fx"], 0, value["cx"]], [0, value["fy"], value["cy"]], [0, 0, 1]],
            dtype=np.float64,
        )
    else:
        raise VisualizationError("Camera split record has no supported intrinsics")
    width, height = int(width), int(height)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise VisualizationError("Camera intrinsic must be finite and 3x3")
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise VisualizationError("Camera focal lengths must be positive")
    return intrinsic, distortion, width, height


def _world_to_camera_from_record(record: dict[str, Any]) -> np.ndarray:
    if "world_to_camera" in record:
        return _validate_matrix(np.asarray(record["world_to_camera"]), "world_to_camera")
    if "camera_to_world" in record:
        camera_to_world = _validate_matrix(
            np.asarray(record["camera_to_world"]), "camera_to_world"
        )
        return np.linalg.inv(camera_to_world)
    if "qvec_wxyz" in record and "tvec_world_to_camera" in record:
        output = np.eye(4, dtype=np.float64)
        output[:3, :3] = quaternion_to_rotation(record["qvec_wxyz"])
        output[:3, 3] = np.asarray(record["tvec_world_to_camera"], dtype=np.float64)
        return output
    raise VisualizationError("Camera split record has no supported pose")


def _build_generic_views(unit_dir: Path, camera_path: Path) -> list[ViewCamera]:
    camera_document = load_json(camera_path)
    views: list[ViewCamera] = []
    for role in ("conditioning", "heldout"):
        records = camera_document.get(role)
        if not isinstance(records, list) or len(records) != 8:
            raise VisualizationError(f"{camera_path} must contain exactly 8 {role} records")
        for order, record in enumerate(records):
            rgb_path = resolve_path(record["rgb"], unit_dir)
            if not rgb_path.is_file():
                raise VisualizationError(f"Missing frozen RGB view: {rgb_path}")
            with Image.open(rgb_path) as opened:
                image_size = opened.size
            intrinsic, distortion, width, height = _intrinsics_from_record(record, image_size)
            if image_size != (width, height):
                raise VisualizationError(
                    f"RGB/camera size mismatch for {rgb_path}: {image_size} != {(width, height)}"
                )
            panel_intrinsic, content_box = panel_camera(intrinsic, width, height)
            views.append(
                ViewCamera(
                    index=len(views),
                    role=role,
                    order=order,
                    source_label=str(
                        record.get(
                            "source_image",
                            record.get("source_view", record.get("source_frame", rgb_path.name)),
                        )
                    ),
                    rgb_path=rgb_path,
                    source_width=width,
                    source_height=height,
                    intrinsic=intrinsic,
                    distortion=distortion,
                    world_to_camera=_world_to_camera_from_record(record),
                    panel_intrinsic=panel_intrinsic,
                    panel_content_box=content_box,
                )
            )
    return views


def _build_eth3d_views(manifest: dict[str, Any], unit_dir: Path) -> list[ViewCamera]:
    input_document = manifest["input"]
    cameras = read_colmap_cameras(resolve_path(input_document["camera_model"], unit_dir))
    images = read_colmap_images(resolve_path(input_document["images_model"], unit_dir))
    images_root = resolve_path(input_document["images_root"], unit_dir)
    views: list[ViewCamera] = []
    for role, key in (("conditioning", "conditioning_views"), ("heldout", "heldout_views")):
        records = input_document[key]
        if len(records) != 8:
            raise VisualizationError(f"ETH3D representative must contain 8 {role} views")
        for order, name in enumerate(records):
            image_record = images.get(Path(name).name)
            if image_record is None:
                raise VisualizationError(f"ETH3D image is absent from COLMAP model: {name}")
            camera = cameras[image_record.camera_id]
            rgb_path = (images_root / name).resolve()
            if not rgb_path.is_file():
                raise VisualizationError(f"Missing ETH3D source image: {rgb_path}")
            with Image.open(rgb_path) as opened:
                image_size = opened.size
            if image_size != (camera.width, camera.height):
                raise VisualizationError(f"ETH3D RGB/camera size mismatch for {rgb_path}")
            panel_intrinsic, content_box = panel_camera(
                camera.intrinsic, camera.width, camera.height
            )
            views.append(
                ViewCamera(
                    index=len(views),
                    role=role,
                    order=order,
                    source_label=Path(name).name,
                    rgb_path=rgb_path,
                    source_width=camera.width,
                    source_height=camera.height,
                    intrinsic=camera.intrinsic,
                    distortion=None,
                    world_to_camera=image_record.world_to_camera,
                    panel_intrinsic=panel_intrinsic,
                    panel_content_box=content_box,
                )
            )
    return views


def build_views(manifest: dict[str, Any], unit_dir: Path) -> list[ViewCamera]:
    if manifest["dataset"] == "eth3d-indoor-training":
        views = _build_eth3d_views(manifest, unit_dir)
    else:
        camera_value = manifest.get("input", {}).get("cameras")
        if not camera_value:
            raise VisualizationError(f"Unit {manifest['unit_id']} has no frozen camera split")
        views = _build_generic_views(unit_dir, resolve_path(camera_value, unit_dir))
    if len(views) != 16:
        raise VisualizationError(f"Expected 16 frozen views, got {len(views)}")
    return views


def _fit_image_to_panel(image: Image.Image, *, resample: Image.Resampling) -> Image.Image:
    source = image.convert("RGB")
    scale = min(PANEL_SIZE[0] / source.width, PANEL_SIZE[1] / source.height)
    width = max(1, int(round(source.width * scale)))
    height = max(1, int(round(source.height * scale)))
    resized = source.resize((width, height), resample)
    panel = Image.new("RGB", PANEL_SIZE, (8, 10, 12))
    panel.paste(resized, ((PANEL_SIZE[0] - width) // 2, (PANEL_SIZE[1] - height) // 2))
    return panel


def _prepare_source_frame(view: ViewCamera) -> Image.Image:
    with Image.open(view.rgb_path) as opened:
        rgb = np.asarray(opened.convert("RGB"))
    if view.distortion is not None:
        rgb = cv2.undistort(rgb, view.intrinsic, view.distortion, None, view.intrinsic)
    return _fit_image_to_panel(Image.fromarray(rgb), resample=Image.Resampling.LANCZOS)


def point_allocations(counts: Sequence[int], maximum: int) -> list[int]:
    if maximum <= 0 or any(count < 0 for count in counts):
        raise ValueError("Point counts and maximum must be non-negative, with positive maximum")
    total = sum(counts)
    if total <= maximum:
        return list(counts)
    raw = np.asarray(counts, dtype=np.float64) * (maximum / total)
    allocation = np.floor(raw).astype(np.int64)
    remainder = maximum - int(allocation.sum())
    order = np.argsort(-(raw - allocation), kind="stable")
    for index in order[:remainder]:
        allocation[index] += 1
    return allocation.astype(int).tolist()


def _sample_indices(count: int, sample_count: int, seed: int) -> np.ndarray:
    if sample_count >= count:
        return np.arange(count, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(count, sample_count, replace=False))


def _load_reference_point_cloud(
    paths: list[Path],
    transforms: dict[str, np.ndarray],
    maximum: int,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    import open3d as o3d

    candidates: list[tuple[np.ndarray, np.ndarray, int, str]] = []
    counts: list[int] = []
    for source_index, path in enumerate(paths):
        cloud = o3d.io.read_point_cloud(str(path), print_progress=True)
        points = np.asarray(cloud.points)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            raise VisualizationError(f"Reference point cloud is empty or invalid: {path}")
        if not np.isfinite(points).all():
            raise VisualizationError(f"Reference point cloud has non-finite points: {path}")
        colors = np.asarray(cloud.colors)
        if colors.shape != points.shape:
            colors = np.tile(np.asarray([[0.66, 0.76, 0.79]]), (len(points), 1))
        count = len(points)
        candidate_count = min(count, maximum)
        indices = _sample_indices(count, candidate_count, seed + source_index * 1009)
        sampled_points = points[indices].astype(np.float64, copy=True)
        sampled_colors = np.clip(colors[indices], 0.0, 1.0).astype(np.float64, copy=True)
        transform = transforms.get(path.name, np.eye(4, dtype=np.float64))
        sampled_points = sampled_points @ transform[:3, :3].T + transform[:3, 3]
        candidates.append((sampled_points, sampled_colors, count, path.name))
        counts.append(count)
        del cloud, points, colors
        gc.collect()

    allocations = point_allocations(counts, maximum)
    point_batches: list[np.ndarray] = []
    color_batches: list[np.ndarray] = []
    sources: list[dict[str, Any]] = []
    for source_index, ((points, colors, count, name), allocation) in enumerate(
        zip(candidates, allocations)
    ):
        if len(points) > allocation:
            indices = _sample_indices(len(points), allocation, seed + 50000 + source_index * 1009)
            points = points[indices]
            colors = colors[indices]
        point_batches.append(points)
        color_batches.append(colors)
        sources.append({"name": name, "source_points": count, "render_points": len(points)})
    output_points = np.concatenate(point_batches, axis=0)
    output_colors = np.concatenate(color_batches, axis=0)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(output_points)
    cloud.colors = o3d.utility.Vector3dVector(output_colors)
    return cloud, {
        "kind": "pointcloud",
        "source_points": int(sum(counts)),
        "render_points": int(len(output_points)),
        "sources": sources,
        "bounds_m": np.stack((output_points.min(axis=0), output_points.max(axis=0))).tolist(),
    }


def _render_geometry(
    *,
    kind: str,
    manifest: dict[str, Any],
    unit_dir: Path,
    views: list[ViewCamera],
    output_dir: Path,
    maximum_points: int,
    point_size: float,
    seed: int,
    registry_root: Path,
) -> dict[str, Any]:
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")
    import open3d as o3d

    output_dir.mkdir(parents=True, exist_ok=True)
    renderer = o3d.visualization.rendering.OffscreenRenderer(*PANEL_SIZE)
    renderer.scene.set_background(np.asarray([0.03, 0.035, 0.04, 1.0], dtype=np.float32))
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.base_color = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    source_paths: list[Path]
    source_hashes: list[str]
    if kind == "reference":
        reference = manifest["reference"]
        source_paths = [resolve_path(value, unit_dir) for value in reference["paths"]]
        source_hashes = [sha256_file(path) for path in source_paths]
        if reference["kind"] == "mesh":
            mesh = o3d.io.read_triangle_mesh(
                str(source_paths[0]), enable_post_processing=False, print_progress=True
            )
            if not mesh.has_triangles():
                raise VisualizationError(f"Reference mesh has no triangles: {source_paths[0]}")
            renderer.scene.add_geometry("geometry", mesh, material)
            geometry = {
                "kind": "mesh",
                "vertices": len(mesh.vertices),
                "triangles": len(mesh.triangles),
                "vertex_colors": mesh.has_vertex_colors(),
            }
        elif reference["kind"] == "pointcloud":
            transforms = (
                load_meshlab_transforms(resolve_path(reference["alignment_mlp"], unit_dir))
                if reference.get("alignment_mlp")
                else {}
            )
            cloud, geometry = _load_reference_point_cloud(
                source_paths, transforms, maximum_points, seed
            )
            material.point_size = float(point_size)
            renderer.scene.add_geometry("geometry", cloud, material)
        else:
            raise VisualizationError(f"Unsupported reference kind: {reference['kind']}")
        provenance = {
            "role": "evaluation-reference",
            "reference_kind": reference["kind"],
            "reference_scope": reference.get("scope"),
            "reference_roi": reference.get("roi"),
            "reference_source_type": reference.get("source_type"),
        }
    elif kind == "prediction":
        prediction_value = manifest.get("prediction_mesh")
        if not prediction_value:
            raise VisualizationError(f"Unit {manifest['unit_id']} has no prediction mesh")
        prediction_path = resolve_path(prediction_value, registry_root)
        source_paths = [prediction_path]
        source_hashes = [sha256_file(prediction_path)]
        mesh = o3d.io.read_triangle_mesh(
            str(prediction_path), enable_post_processing=False, print_progress=True
        )
        if not mesh.has_triangles():
            raise VisualizationError(f"Prediction mesh has no triangles: {prediction_path}")
        renderer.scene.add_geometry("geometry", mesh, material)
        geometry = {
            "kind": "mesh",
            "vertices": len(mesh.vertices),
            "triangles": len(mesh.triangles),
            "vertex_colors": mesh.has_vertex_colors(),
        }
        provenance = {"role": "genrecon-prediction", "reference_geometry_used": False}
    else:
        raise VisualizationError(f"Unsupported render kind: {kind}")

    frame_records: list[dict[str, Any]] = []
    for view in views:
        renderer.setup_camera(
            view.panel_intrinsic,
            view.world_to_camera,
            PANEL_SIZE[0],
            PANEL_SIZE[1],
        )
        color = np.asarray(renderer.render_to_image())[:, :, :3]
        depth = np.asarray(renderer.render_to_depth_image(z_in_view_space=True))
        mask = np.isfinite(depth) & (depth > 0)
        frame_path = output_dir / f"{view.index:06d}.png"
        mask_path = output_dir.parent.parent / "masks" / kind / f"{view.index:06d}.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(color.astype(np.uint8)).save(frame_path, compress_level=6)
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path, compress_level=6)
        coverage = float(mask.mean())
        masked_std = float(color[mask].std()) if np.any(mask) else 0.0
        frame_records.append(
            {
                "index": view.index,
                "role": view.role,
                "order": view.order,
                "source_label": view.source_label,
                "frame": root_relative(frame_path),
                "mask": root_relative(mask_path),
                "coverage": coverage,
                "masked_rgb_std": masked_std,
            }
        )
        print(
            f"[gt-visualization-render] {manifest['unit_id']} {kind} "
            f"{view.index + 1:02d}/16 coverage={coverage:.4f}"
        )
    del renderer
    gc.collect()
    summary = {
        "schema": "genrecon.gt-calibration-geometry-render",
        "schema_version": SCHEMA_VERSION,
        "unit_id": manifest["unit_id"],
        "kind": kind,
        "renderer": "Open3D OffscreenRenderer, defaultUnlit",
        "panel_size": list(PANEL_SIZE),
        "point_size_px": point_size if geometry["kind"] == "pointcloud" else None,
        "geometry": geometry,
        "sources": [
            {
                "path": root_relative(path),
                "size_bytes": path.stat().st_size,
                "sha256": digest,
            }
            for path, digest in zip(source_paths, source_hashes)
        ],
        "provenance": provenance,
        "frames": frame_records,
    }
    write_json(output_dir.parent.parent / f"render_{kind}.json", summary)
    return summary


def _render_subprocess(
    *,
    kind: str,
    entry: dict[str, Any],
    args: argparse.Namespace,
    candidate_dir: Path,
) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_render",
        "--kind",
        kind,
        "--unit-id",
        entry["unit_id"],
        "--plan",
        str(args.plan),
        "--registry",
        str(args.registry),
        "--candidate-dir",
        str(candidate_dir),
    ]
    subprocess.run(command, check=True, cwd=ROOT)


def _copy_cached_prediction(
    *,
    entry: dict[str, Any],
    manifest: dict[str, Any],
    views: list[ViewCamera],
    candidate_dir: Path,
    registry_root: Path,
) -> dict[str, Any]:
    cache = entry["prediction_render_cache"]
    expected_sha256 = cache["expected_prediction_sha256"]
    prediction_path = resolve_path(manifest["prediction_mesh"], registry_root)
    render_manifest_path = resolve_path(cache["render_manifest"], ROOT)
    render_manifest = load_json(render_manifest_path)
    cached_asset = resolve_path(render_manifest["asset"], ROOT)
    actual_hashes = {
        "registry_prediction": sha256_file(prediction_path),
        "cached_render_asset": sha256_file(cached_asset),
    }
    if any(value != expected_sha256 for value in actual_hashes.values()):
        raise VisualizationError(
            f"Prediction render cache SHA mismatch for {manifest['unit_id']}: {actual_hashes}"
        )
    if render_manifest.get("asset_kind") != "ply" or render_manifest.get("shader") != "defaultUnlit":
        raise VisualizationError("Cached prediction render has incompatible protocol")

    views_root = resolve_path(cache["views_root"], ROOT)
    output_dir = candidate_dir / "frames" / "prediction"
    mask_dir = candidate_dir / "masks" / "prediction"
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    frame_records = []
    for view in views:
        stem = Path(view.source_label).stem
        matches = list(views_root.glob(f"*/*{stem}*/ply_render.png"))
        if len(matches) != 1:
            raise VisualizationError(f"Expected one cached prediction render for {stem}, got {matches}")
        camera_path = matches[0].with_name("camera.json")
        camera = load_json(camera_path)
        cached_pose = np.asarray(camera["world_to_camera"], dtype=np.float64)
        if not np.allclose(cached_pose, view.world_to_camera, atol=1e-9):
            raise VisualizationError(f"Cached prediction camera mismatch for {stem}")
        with Image.open(matches[0]) as opened:
            rgba = opened.convert("RGBA")
            rgb_panel = _fit_image_to_panel(rgba.convert("RGB"), resample=Image.Resampling.LANCZOS)
            if "A" in opened.getbands():
                alpha = opened.getchannel("A")
            else:
                array = np.asarray(opened.convert("RGB"))
                alpha = Image.fromarray((np.any(array > 3, axis=2).astype(np.uint8) * 255))
            mask_panel = _fit_image_to_panel(
                Image.merge("RGB", (alpha, alpha, alpha)), resample=Image.Resampling.NEAREST
            ).convert("L")
        frame_path = output_dir / f"{view.index:06d}.png"
        mask_path = mask_dir / f"{view.index:06d}.png"
        rgb_panel.save(frame_path, compress_level=6)
        mask_array = np.asarray(mask_panel) > 127
        Image.fromarray(mask_array.astype(np.uint8) * 255).save(mask_path, compress_level=6)
        rgb_array = np.asarray(rgb_panel)
        frame_records.append(
            {
                "index": view.index,
                "role": view.role,
                "order": view.order,
                "source_label": view.source_label,
                "frame": root_relative(frame_path),
                "mask": root_relative(mask_path),
                "coverage": float(mask_array.mean()),
                "masked_rgb_std": float(rgb_array[mask_array].std()) if np.any(mask_array) else 0.0,
                "cached_render": root_relative(matches[0]),
                "cached_camera": root_relative(camera_path),
            }
        )
    summary = {
        "schema": "genrecon.gt-calibration-geometry-render",
        "schema_version": SCHEMA_VERSION,
        "unit_id": manifest["unit_id"],
        "kind": "prediction",
        "renderer": render_manifest["shader"],
        "cache_reuse": {
            "render_manifest": root_relative(render_manifest_path),
            "render_manifest_sha256": sha256_file(render_manifest_path),
            "expected_prediction_sha256": expected_sha256,
            "actual_hashes": actual_hashes,
            "camera_matrices_revalidated": True,
        },
        "panel_size": list(PANEL_SIZE),
        "geometry": render_manifest["geometry"],
        "sources": [
            {
                "path": root_relative(prediction_path),
                "size_bytes": prediction_path.stat().st_size,
                "sha256": expected_sha256,
            }
        ],
        "provenance": {"role": "genrecon-prediction", "reference_geometry_used": False},
        "frames": frame_records,
    }
    write_json(candidate_dir / "render_prediction.json", summary)
    return summary


def _status_panel(title: str, body: Sequence[str], *, size: tuple[int, int] = PANEL_SIZE) -> Image.Image:
    canvas = Image.new("RGB", size, (22, 26, 29))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 12, size[1]), fill=(206, 151, 55))
    draw.text((42, 50), title, font=_fit_font(draw, title, size[0] - 84, 28, bold=True), fill=(247, 248, 248))
    y = 118
    for paragraph in body:
        lines = textwrap.wrap(paragraph, width=48) or [""]
        for line in lines:
            draw.text((42, y), line, font=_font(18), fill=(187, 195, 199))
            y += 27
        y += 12
    return canvas


def _source_frames(views: list[ViewCamera], candidate_dir: Path) -> list[dict[str, Any]]:
    output_dir = candidate_dir / "frames" / "source"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for view in views:
        frame = _prepare_source_frame(view)
        frame_path = output_dir / f"{view.index:06d}.png"
        frame.save(frame_path, compress_level=6)
        array = np.asarray(frame)
        records.append(
            {
                "index": view.index,
                "role": view.role,
                "order": view.order,
                "source_label": view.source_label,
                "source_rgb": root_relative(view.rgb_path),
                "source_rgb_sha256": sha256_file(view.rgb_path),
                "frame": root_relative(frame_path),
                "frame_sha256": sha256_file(frame_path),
                "rgb_std": float(array.std()),
                "world_to_camera": view.world_to_camera.tolist(),
                "intrinsic_source": view.intrinsic.tolist(),
                "intrinsic_panel": view.panel_intrinsic.tolist(),
                "source_size": [view.source_width, view.source_height],
                "panel_content_box": list(view.panel_content_box),
                "source_undistorted_for_visualization": view.distortion is not None,
            }
        )
    return records


def _missing_prediction_frames(
    views: list[ViewCamera], candidate_dir: Path
) -> dict[str, Any]:
    output_dir = candidate_dir / "frames" / "prediction"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for view in views:
        frame = _status_panel(
            "NO GENRECON PREDICTION",
            [
                "The calibration package is prepared, but GenRecon has not been run for this unit.",
                "This panel is a status card, not a geometry render.",
            ],
        )
        path = output_dir / f"{view.index:06d}.png"
        frame.save(path, compress_level=6)
        records.append(
            {
                "index": view.index,
                "role": view.role,
                "order": view.order,
                "source_label": view.source_label,
                "frame": root_relative(path),
                "coverage": None,
                "masked_rgb_std": None,
                "status": MISSING_PREDICTION_STATUS,
            }
        )
    summary = {
        "schema": "genrecon.gt-calibration-geometry-render",
        "schema_version": SCHEMA_VERSION,
        "kind": "prediction-status",
        "availability": MISSING_PREDICTION_STATUS,
        "reference_geometry_used": False,
        "frames": records,
    }
    write_json(candidate_dir / "render_prediction.json", summary)
    return summary


def _compose_comparison(
    *,
    entry: dict[str, Any],
    manifest: dict[str, Any],
    views: list[ViewCamera],
    candidate_dir: Path,
    prediction_available: bool,
) -> list[dict[str, Any]]:
    output_dir = candidate_dir / "frames" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    prediction_subtitle = (
        "GenRecon prediction mesh | same frozen camera"
        if prediction_available
        else "Unavailable | status card, not a render"
    )
    panel_titles = (
        ("REFERENCE RGB", "Frozen calibration input/heldout image"),
        (
            "GT / REFERENCE GEOMETRY",
            f"{manifest['gt_tier']} | evaluation-only geometry",
        ),
        ("GENRECON PREDICTION", prediction_subtitle),
    )
    for view in views:
        panels = []
        for kind in ("source", "reference", "prediction"):
            with Image.open(candidate_dir / "frames" / kind / f"{view.index:06d}.png") as opened:
                panels.append(opened.convert("RGB"))
        canvas = Image.new("RGB", COMPARISON_SIZE, (15, 18, 20))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, COMPARISON_SIZE[0], 8), fill=(26, 139, 149))
        heading = f"{entry['dataset_label']}  /  {entry['scene_label']}"
        draw.text((24, 20), heading, font=_fit_font(draw, heading, 1030, 26, bold=True), fill=(245, 247, 247))
        role = view.role.upper()
        role_color = (64, 178, 134) if view.role == "conditioning" else (224, 170, 70)
        role_width = draw.textbbox((0, 0), role, font=_font(16, bold=True))[2] + 24
        draw.rounded_rectangle(
            (COMPARISON_SIZE[0] - role_width - 24, 19, COMPARISON_SIZE[0] - 24, 49),
            radius=4,
            fill=role_color,
        )
        draw.text(
            (COMPARISON_SIZE[0] - role_width - 12, 25),
            role,
            font=_font(16, bold=True),
            fill=(10, 20, 19),
        )
        for column, (panel, (title, subtitle)) in enumerate(zip(panels, panel_titles)):
            x = column * PANEL_SIZE[0]
            draw.rectangle((x, 68, x + PANEL_SIZE[0], 119), fill=(31, 36, 39))
            draw.text((x + 18, 77), title, font=_font(18, bold=True), fill=(241, 243, 243))
            draw.text((x + 18, 99), subtitle, font=_fit_font(draw, subtitle, 600, 13), fill=(157, 168, 173))
            canvas.paste(panel, (x, 120))
            if column:
                draw.line((x, 68, x, 600), fill=(80, 89, 93), width=2)
        footer = (
            f"View {view.index + 1:02d}/16  |  {view.source_label}  |  "
            f"track={manifest['track']}  |  GT is never used as GenRecon conditioning"
        )
        draw.text((24, 625), footer, font=_fit_font(draw, footer, 1870, 18), fill=(213, 218, 220))
        scope = manifest["reference"].get("scope", manifest["reference"].get("roi", "declared reference scope"))
        note = f"Reference scope: {scope}. Black background denotes no rendered geometry. Visual review is not a metric score."
        draw.text((24, 663), note, font=_fit_font(draw, note, 1870, 15), fill=(139, 151, 157))
        output_path = output_dir / f"{view.index:06d}.png"
        canvas.save(output_path, compress_level=6)
        records.append(
            {
                "index": view.index,
                "role": view.role,
                "order": view.order,
                "source_label": view.source_label,
                "frame": root_relative(output_path),
                "sha256": sha256_file(output_path),
            }
        )
    return records


def _encode_candidate_videos(
    candidate_dir: Path, fps: float, ffmpeg: str
) -> dict[str, Any]:
    videos = {}
    for kind in ("source", "reference", "prediction", "comparison"):
        destination = candidate_dir / f"{kind}.mp4"
        encode_video(
            ffmpeg=ffmpeg,
            source_pattern=candidate_dir / "frames" / kind / "%06d.png",
            destination=destination,
            fps=fps,
            report_log=candidate_dir / f"ffmpeg_{kind}.log",
        )
        videos[kind] = probe_video(destination)
    return videos


def _make_overview(candidate_dir: Path, frame_count: int) -> tuple[Path, Path]:
    selected = sorted(
        set(
            [
                0,
                max(0, frame_count // 3),
                max(0, 2 * frame_count // 3),
                frame_count - 1,
            ]
        )
    )
    contact = Image.new("RGB", (1920, 720), (20, 24, 27))
    for index, frame_index in enumerate(selected[:4]):
        with Image.open(
            candidate_dir / "frames" / "comparison" / f"{frame_index:06d}.png"
        ) as opened:
            image = opened.convert("RGB").resize(
                (960, 360), Image.Resampling.LANCZOS
            )
        x = (index % 2) * 960
        y = (index // 2) * 360
        contact.paste(image, (x, y))
    overview = candidate_dir / "overview.jpg"
    contact.save(overview, quality=92, subsampling=0)
    with Image.open(candidate_dir / "frames" / "comparison" / "000000.png") as first:
        poster_image = first.convert("RGB").resize((960, 360), Image.Resampling.LANCZOS)
    poster = candidate_dir / "poster.jpg"
    poster_image.save(poster, quality=92, subsampling=0)
    return overview, poster


def _unit_context(
    unit_id: str, plan: dict[str, Any], registry_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, Path]:
    entries = {entry["unit_id"]: entry for entry in plan["datasets"]}
    if unit_id not in entries:
        raise VisualizationError(f"Unit is not in visualization plan: {unit_id}")
    registry = load_json(registry_path)
    units = {unit["unit_id"]: unit for unit in registry["units"]}
    if unit_id not in units:
        raise VisualizationError(f"Unit is not in GT registry: {unit_id}")
    entry = entries[unit_id]
    registry_row = units[unit_id]
    manifest_path = resolve_path(registry_row["manifest"], registry_path.parent)
    manifest = load_json(manifest_path)
    if manifest["dataset"] != entry["dataset"] or registry_row["dataset"] != entry["dataset"]:
        raise VisualizationError(f"Dataset mismatch for representative {unit_id}")
    return entry, registry_row, manifest, manifest_path, manifest_path.parent


def _build_available_candidate(
    entry: dict[str, Any],
    registry_row: dict[str, Any],
    manifest: dict[str, Any],
    manifest_path: Path,
    unit_dir: Path,
    candidate_dir: Path,
    args: argparse.Namespace,
    plan: dict[str, Any],
) -> dict[str, Any]:
    if manifest["status"] != "prepared" or registry_row["status"] != "prepared":
        raise VisualizationError(f"Available representative is not prepared: {entry['unit_id']}")
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    candidate_dir.mkdir(parents=True)
    views = build_views(manifest, unit_dir)
    source_records = _source_frames(views, candidate_dir)
    _render_subprocess(
        kind="reference", entry=entry, args=args, candidate_dir=candidate_dir
    )
    reference_render = load_json(candidate_dir / "render_reference.json")
    prediction_available = bool(manifest.get("prediction_mesh"))
    if prediction_available:
        if entry.get("prediction_render_cache"):
            prediction_render = _copy_cached_prediction(
                entry=entry,
                manifest=manifest,
                views=views,
                candidate_dir=candidate_dir,
                registry_root=args.registry.parent,
            )
        else:
            _render_subprocess(
                kind="prediction", entry=entry, args=args, candidate_dir=candidate_dir
            )
            prediction_render = load_json(candidate_dir / "render_prediction.json")
        prediction_status = AVAILABLE_STATUS
    else:
        prediction_render = _missing_prediction_frames(views, candidate_dir)
        prediction_status = MISSING_PREDICTION_STATUS
    comparison_records = _compose_comparison(
        entry=entry,
        manifest=manifest,
        views=views,
        candidate_dir=candidate_dir,
        prediction_available=prediction_available,
    )
    ffmpeg = ffmpeg_executable(args.ffmpeg)
    videos = _encode_candidate_videos(candidate_dir, plan["playback_fps"], ffmpeg)
    overview, poster = _make_overview(candidate_dir, len(views))
    reference_by_index = {item["index"]: item for item in reference_render["frames"]}
    prediction_by_index = {item["index"]: item for item in prediction_render["frames"]}
    comparison_by_index = {item["index"]: item for item in comparison_records}
    frame_records = []
    for source in source_records:
        index = source["index"]
        reference = reference_by_index[index]
        prediction = prediction_by_index[index]
        frame_records.append(
            {
                **source,
                "reference_frame": reference["frame"],
                "reference_frame_sha256": sha256_file(ROOT / reference["frame"]),
                "reference_coverage": reference["coverage"],
                "reference_masked_rgb_std": reference["masked_rgb_std"],
                "prediction_frame": prediction["frame"],
                "prediction_frame_sha256": sha256_file(ROOT / prediction["frame"]),
                "prediction_coverage": prediction.get("coverage"),
                "prediction_masked_rgb_std": prediction.get("masked_rgb_std"),
                "comparison_frame": comparison_by_index[index]["frame"],
                "comparison_frame_sha256": comparison_by_index[index]["sha256"],
            }
        )
    document = {
        "schema": "genrecon.gt-calibration-visualization-candidate",
        "schema_version": SCHEMA_VERSION,
        "dataset": entry["dataset"],
        "dataset_label": entry["dataset_label"],
        "unit_id": entry["unit_id"],
        "scene_label": entry["scene_label"],
        "selection_reason": entry["selection_reason"],
        "build_contract": {
            "plan_sha256": sha256_file(args.plan),
            "registry_sha256": sha256_file(args.registry),
            "tool_sha256": sha256_file(Path(__file__)),
        },
        "status": AVAILABLE_STATUS,
        "prediction_status": prediction_status,
        "protocol": {
            "camera_policy": "exact frozen 8 conditioning + 8 heldout cameras; no interpolation",
            "frame_order": "conditioning 0-7 followed by heldout 0-7",
            "playback_fps": plan["playback_fps"],
            "panel_size": list(PANEL_SIZE),
            "comparison_size": list(COMPARISON_SIZE),
            "reference_geometry_policy": "evaluation-only; never GenRecon conditioning",
            "missing_prediction_policy": "explicit status cards; never substitute GT geometry",
        },
        "inputs": {
            "manifest": root_relative(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "gt_tier": manifest["gt_tier"],
            "track": manifest["track"],
            "capture_kind": manifest.get("capture_kind"),
            "license_status": manifest.get("license_status"),
        },
        "reference_render": reference_render,
        "prediction_render": prediction_render,
        "frames": frame_records,
        "videos": videos,
        "overview": root_relative(overview),
        "overview_sha256": sha256_file(overview),
        "poster": root_relative(poster),
        "poster_sha256": sha256_file(poster),
        "limitations": manifest.get("evaluation", {}).get("limitations", [])
        + [
            "The GT/reference render is a visualization of evaluation geometry, not a model output.",
            "Video playback uses frozen sparse views and is not a continuous source-time trajectory.",
            "A nonblank render does not establish geometric or photometric accuracy.",
        ],
    }
    write_json(candidate_dir / "manifest.json", document)
    return document


def _build_blocked_candidate(
    entry: dict[str, Any], registry_row: dict[str, Any], candidate_dir: Path, args: argparse.Namespace
) -> dict[str, Any]:
    expected = entry.get("expected_status")
    if registry_row["status"] != expected or expected != BLOCKED_STATUS:
        raise VisualizationError(f"Unexpected blocker status for {entry['unit_id']}")
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    frame_dir = candidate_dir / "frames" / "status"
    frame_dir.mkdir(parents=True)
    card = _status_panel(
        "OMNIOBJECT3D: BLOCKED-AUTH",
        [
            "No authorized official RGB, camera, or dense scan package is available locally.",
            "The configured semantic request slot is not claimed as an official object ID.",
            "No reference image or geometry render has been fabricated.",
        ],
        size=COMPARISON_SIZE,
    )
    for index in range(8):
        card.save(frame_dir / f"{index:06d}.png", compress_level=6)
    ffmpeg = ffmpeg_executable(args.ffmpeg)
    video_path = candidate_dir / "status.mp4"
    encode_video(
        ffmpeg=ffmpeg,
        source_pattern=frame_dir / "%06d.png",
        destination=video_path,
        fps=2.0,
        report_log=candidate_dir / "ffmpeg_status.log",
    )
    poster = candidate_dir / "poster.jpg"
    card.resize((960, 360), Image.Resampling.LANCZOS).save(
        poster, quality=92, subsampling=0
    )
    document = {
        "schema": "genrecon.gt-calibration-visualization-candidate",
        "schema_version": SCHEMA_VERSION,
        "dataset": entry["dataset"],
        "dataset_label": entry["dataset_label"],
        "unit_id": entry["unit_id"],
        "scene_label": entry["scene_label"],
        "selection_reason": entry["selection_reason"],
        "build_contract": {
            "plan_sha256": sha256_file(args.plan),
            "registry_sha256": sha256_file(args.registry),
            "tool_sha256": sha256_file(Path(__file__)),
        },
        "status": BLOCKED_STATUS,
        "prediction_status": BLOCKED_STATUS,
        "blocker": registry_row.get("blocker"),
        "videos": {"status": probe_video(video_path)},
        "poster": root_relative(poster),
        "poster_sha256": sha256_file(poster),
        "limitations": [
            "No authorized OmniObject3D source package is present.",
            "No representative source image, reference geometry, prediction, or metric is shown.",
        ],
    }
    write_json(candidate_dir / "manifest.json", document)
    return document


def _concat_overview(videos: list[Path], output: Path, ffmpeg: str) -> dict[str, Any]:
    list_path = output.parent / "overview_concat.txt"
    list_path.write_text(
        "".join(f"file '{path.resolve()}'\n" for path in videos), encoding="utf-8"
    )
    temporary = output.with_name(f".{output.stem}.tmp.mp4")
    temporary.unlink(missing_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-an",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(temporary),
        ],
        check=True,
        cwd=ROOT,
    )
    temporary.replace(output)
    return probe_video(output)


def _write_index_html(output: Path, rows: list[dict[str, Any]], overview: Path) -> None:
    cards = []
    for row in rows:
        manifest_path = output / "candidates" / row["unit_id"] / "manifest.json"
        poster = Path(row["poster"]).resolve().relative_to(output.resolve())
        if row["status"] == AVAILABLE_STATUS:
            video = Path(row["videos"]["comparison"]["path"]).resolve().relative_to(output.resolve())
            links = "".join(
                f'<a href="{html.escape(str(Path(item["path"]).resolve().relative_to(output.resolve())))}">{kind}</a>'
                for kind, item in row["videos"].items()
            )
            detail = f"{row['inputs']['gt_tier']} | prediction: {row['prediction_status']}"
        else:
            video = Path(row["videos"]["status"]["path"]).resolve().relative_to(output.resolve())
            links = '<span class="muted">No authorized source/reference/prediction assets</span>'
            detail = "blocked-auth | no fabricated representative"
        cards.append(
            f"""
<article class="dataset">
  <header><div><h2>{html.escape(row['dataset_label'])}</h2><p>{html.escape(row['scene_label'])}</p></div><span class="status {html.escape(row['status'])}">{html.escape(row['status'])}</span></header>
  <video controls preload="metadata" poster="{html.escape(str(poster))}" src="{html.escape(str(video))}"></video>
  <div class="meta">{html.escape(detail)}</div>
  <nav>{links}<a href="{html.escape(str(manifest_path.relative_to(output)))}">manifest</a></nav>
</article>"""
        )
    overview_relative = overview.resolve().relative_to(output.resolve())
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GT calibration representative videos</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;background:#eef1f2;color:#202629;font-family:Inter,system-ui,sans-serif;letter-spacing:0}} .top{{background:#fff;border-bottom:1px solid #cbd2d5;padding:20px 24px}} .top div{{max-width:1500px;margin:auto}} h1{{font-size:26px;margin:0 0 6px}} .top p{{margin:0;color:#5c686e;font-size:14px}} main{{max-width:1500px;margin:auto;padding:22px 24px 40px}} .reel{{background:#15191b;border:1px solid #394247;border-radius:6px;overflow:hidden;margin-bottom:20px}} .reel header{{padding:12px 15px;color:#fff;font-weight:700}} video{{display:block;width:100%;background:#080a0b}} .grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}} .dataset{{background:#fff;border:1px solid #cbd2d5;border-radius:6px;overflow:hidden}} .dataset header{{display:flex;justify-content:space-between;align-items:start;padding:13px 15px}} h2{{font-size:17px;margin:0}} .dataset header p{{font-size:13px;color:#647076;margin:4px 0 0}} .status{{font:700 12px ui-monospace,monospace;padding:5px 7px;border-radius:3px;background:#2b7b65;color:#fff}} .status.blocked-auth{{background:#a46b1e}} .meta{{padding:10px 15px;border-top:1px solid #e1e5e7;color:#556168;font-size:13px}} nav{{display:flex;gap:14px;flex-wrap:wrap;padding:11px 15px}} a{{color:#096d78;text-decoration:none;font-size:13px;font-weight:700}} a:hover{{text-decoration:underline}} .muted{{font-size:13px;color:#727e84}} @media(max-width:900px){{.grid{{grid-template-columns:1fr}}main{{padding:14px}}}}
</style></head><body><section class="top"><div><h1>GT calibration representative videos</h1><p>Frozen RGB | evaluation reference geometry | GenRecon prediction. Missing assets remain explicit.</p></div></section><main><section class="reel"><header>All datasets overview</header><video controls preload="metadata" src="{html.escape(str(overview_relative))}"></video></section><section class="grid">{''.join(cards)}</section></main></body></html>"""
    (output / "index.html").write_text(document, encoding="utf-8")


def _candidate_reusable(
    manifest_path: Path, *, plan_path: Path, registry_path: Path
) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        document = load_json(manifest_path)
    except (OSError, VisualizationError, json.JSONDecodeError):
        return False
    expected = {
        "plan_sha256": sha256_file(plan_path),
        "registry_sha256": sha256_file(registry_path),
        "tool_sha256": sha256_file(Path(__file__)),
    }
    if document.get("build_contract") != expected:
        return False
    for video in document.get("videos", {}).values():
        path = Path(video.get("path", ""))
        if (
            not path.is_file()
            or path.stat().st_size != video.get("size_bytes")
            or sha256_file(path) != video.get("sha256")
        ):
            return False
    return True


def build_all(args: argparse.Namespace) -> dict[str, Any]:
    args.plan = args.plan.resolve()
    args.registry = args.registry.resolve()
    args.output = args.output.resolve()
    plan = load_json(args.plan)
    if plan.get("schema") != "genrecon.gt-calibration-visualization-plan":
        raise VisualizationError(f"Unexpected visualization plan: {args.plan}")
    registry = load_json(args.registry)
    registry_units = {item["unit_id"]: item for item in registry["units"]}
    args.output.mkdir(parents=True, exist_ok=True)
    selected = set(args.unit)
    unknown = selected - {entry["unit_id"] for entry in plan["datasets"]}
    if unknown:
        raise VisualizationError(f"Unknown requested representative units: {sorted(unknown)}")
    rows = []
    for entry in plan["datasets"]:
        if selected and entry["unit_id"] not in selected:
            continue
        registry_row = registry_units[entry["unit_id"]]
        candidate_dir = args.output / "candidates" / entry["unit_id"]
        manifest_path = candidate_dir / "manifest.json"
        if not args.force and _candidate_reusable(
            manifest_path, plan_path=args.plan, registry_path=args.registry
        ):
            row = load_json(manifest_path)
            print(f"[gt-visualization] reuse {entry['unit_id']}")
        elif registry_row["status"] == BLOCKED_STATUS:
            row = _build_blocked_candidate(entry, registry_row, candidate_dir, args)
        else:
            context = _unit_context(entry["unit_id"], plan, args.registry)
            _, registry_row, manifest, source_manifest_path, unit_dir = context
            row = _build_available_candidate(
                entry,
                registry_row,
                manifest,
                source_manifest_path,
                unit_dir,
                candidate_dir,
                args,
                plan,
            )
        rows.append(row)
    if selected:
        return {"rows": rows}
    ffmpeg = ffmpeg_executable(args.ffmpeg)
    clip_paths = []
    for row in rows:
        video = row["videos"]["comparison" if row["status"] == AVAILABLE_STATUS else "status"]
        clip_paths.append(Path(video["path"]))
    overview_path = args.output / "all_datasets_overview.mp4"
    overview = _concat_overview(clip_paths, overview_path, ffmpeg)
    index = {
        "schema": "genrecon.gt-calibration-visualization-index",
        "schema_version": SCHEMA_VERSION,
        "plan": root_relative(args.plan),
        "plan_sha256": sha256_file(args.plan),
        "registry": root_relative(args.registry),
        "registry_sha256": sha256_file(args.registry),
        "tool": root_relative(Path(__file__)),
        "tool_sha256": sha256_file(Path(__file__)),
        "representatives": [
            {
                "dataset": row["dataset"],
                "dataset_label": row["dataset_label"],
                "unit_id": row["unit_id"],
                "scene_label": row["scene_label"],
                "status": row["status"],
                "prediction_status": row["prediction_status"],
                "manifest": root_relative(args.output / "candidates" / row["unit_id"] / "manifest.json"),
            }
            for row in rows
        ],
        "overview_video": overview,
        "counts": {
            "datasets": len(rows),
            "available_datasets": sum(row["status"] == AVAILABLE_STATUS for row in rows),
            "blocked_datasets": sum(row["status"] == BLOCKED_STATUS for row in rows),
            "datasets_with_prediction": sum(
                row["prediction_status"] == AVAILABLE_STATUS for row in rows
            ),
            "datasets_missing_prediction": sum(
                row["prediction_status"] == MISSING_PREDICTION_STATUS for row in rows
            ),
            "candidate_videos": sum(len(row["videos"]) for row in rows),
            "total_videos": 1 + sum(len(row["videos"]) for row in rows),
        },
    }
    write_json(args.output / "index.json", index)
    _write_index_html(args.output, rows, overview_path)
    (args.output / "README.md").write_text(
        "# GT calibration representative videos\n\n"
        "Open `index.html` for the review grid. Each prepared representative uses "
        "the frozen 8 conditioning + 8 heldout cameras. Columns are reference RGB, "
        "evaluation-only GT/reference geometry, and GenRecon prediction. Missing "
        "predictions and blocked source packages are never replaced by GT geometry.\n",
        encoding="utf-8",
    )
    return index


def validate_output(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    index_path = output / "index.json"
    index = load_json(index_path)
    errors: list[str] = []
    representatives = index.get("representatives", [])
    if len(representatives) != 7:
        errors.append(f"expected 7 dataset representatives, found {len(representatives)}")
    candidate_video_count = 0
    frame_count = 0
    reference_coverages: list[float] = []
    prediction_coverages: list[float] = []
    for item in representatives:
        manifest_path = ROOT / item["manifest"]
        if not manifest_path.is_file():
            errors.append(f"missing candidate manifest: {manifest_path}")
            continue
        document = load_json(manifest_path)
        if document["status"] == AVAILABLE_STATUS:
            frames = document.get("frames", [])
            if len(frames) != 16:
                errors.append(f"{item['unit_id']}: expected 16 frames, found {len(frames)}")
            for frame in frames:
                frame_count += 1
                for path_key, hash_key in (
                    ("frame", "frame_sha256"),
                    ("reference_frame", "reference_frame_sha256"),
                    ("prediction_frame", "prediction_frame_sha256"),
                    ("comparison_frame", "comparison_frame_sha256"),
                ):
                    path = ROOT / frame[path_key]
                    if not path.is_file() or sha256_file(path) != frame[hash_key]:
                        errors.append(f"{item['unit_id']}: invalid {path_key} for frame {frame['index']}")
                if frame["rgb_std"] <= 5.0:
                    errors.append(f"{item['unit_id']}: blank source frame {frame['index']}")
                reference_coverage = frame["reference_coverage"]
                reference_coverages.append(reference_coverage)
                if reference_coverage <= 0.0001 or frame["reference_masked_rgb_std"] <= 0.5:
                    errors.append(f"{item['unit_id']}: blank reference render {frame['index']}")
                if document["prediction_status"] == AVAILABLE_STATUS:
                    coverage = frame["prediction_coverage"]
                    prediction_coverages.append(coverage)
                    if coverage <= 0.0001 or frame["prediction_masked_rgb_std"] <= 0.5:
                        errors.append(f"{item['unit_id']}: blank prediction render {frame['index']}")
            expected_video_keys = {"source", "reference", "prediction", "comparison"}
        else:
            expected_video_keys = {"status"}
        if set(document["videos"]) != expected_video_keys:
            errors.append(f"{item['unit_id']}: unexpected video set {sorted(document['videos'])}")
        for kind, expected in document["videos"].items():
            candidate_video_count += 1
            path = Path(expected["path"])
            if not path.is_file() or sha256_file(path) != expected["sha256"]:
                errors.append(f"{item['unit_id']}: invalid {kind} video hash")
                continue
            actual = probe_video(path)
            for key in ("decoded_frame_count", "reported_frame_count", "width", "height"):
                if actual[key] != expected[key]:
                    errors.append(f"{item['unit_id']}: {kind} video {key} mismatch")
            if actual["decoded_frame_count"] <= 0 or actual["minimum_decoded_rgb_std"] <= 0.5:
                errors.append(f"{item['unit_id']}: {kind} video is blank")
    overview_path = Path(index["overview_video"]["path"])
    if not overview_path.is_file() or sha256_file(overview_path) != index["overview_video"]["sha256"]:
        errors.append("overview video hash mismatch")
    else:
        actual_overview = probe_video(overview_path)
        if actual_overview["decoded_frame_count"] != index["overview_video"]["decoded_frame_count"]:
            errors.append("overview video frame count mismatch")
    if candidate_video_count != index["counts"]["candidate_videos"]:
        errors.append("candidate video count does not match index")
    result = {
        "schema": "genrecon.gt-calibration-visualization-validation",
        "schema_version": SCHEMA_VERSION,
        "result": "pass" if not errors else "fail",
        "counts": {
            "datasets": len(representatives),
            "available_datasets": sum(item["status"] == AVAILABLE_STATUS for item in representatives),
            "blocked_datasets": sum(item["status"] == BLOCKED_STATUS for item in representatives),
            "frozen_camera_frames": frame_count,
            "candidate_videos": candidate_video_count,
            "total_videos": candidate_video_count + (1 if overview_path.is_file() else 0),
        },
        "coverage": {
            "reference_minimum": min(reference_coverages) if reference_coverages else None,
            "reference_median": float(np.median(reference_coverages)) if reference_coverages else None,
            "prediction_minimum": min(prediction_coverages) if prediction_coverages else None,
            "prediction_median": float(np.median(prediction_coverages)) if prediction_coverages else None,
        },
        "errors": errors,
    }
    write_json(output / "validation.json", result)
    print(f"[gt-visualization-validation] {result['result']} {result['counts']}")
    if errors:
        raise VisualizationError("Visualization validation failed: " + "; ".join(errors[:8]))
    return result


def render_internal(args: argparse.Namespace) -> None:
    plan = load_json(args.plan.resolve())
    entry, _, manifest, _, unit_dir = _unit_context(
        args.unit_id, plan, args.registry.resolve()
    )
    views = build_views(manifest, unit_dir)
    _render_geometry(
        kind=args.kind,
        manifest=manifest,
        unit_dir=unit_dir,
        views=views,
        output_dir=args.candidate_dir.resolve() / "frames" / args.kind,
        maximum_points=int(plan["reference_max_points"]),
        point_size=float(plan["point_size_px"]),
        seed=int(plan["seed"]),
        registry_root=args.registry.resolve().parent,
    )
    print(f"[gt-visualization-render] complete {entry['unit_id']} {args.kind}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    all_parser = subparsers.add_parser("all", help="Build all representative videos and validate")
    all_parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    all_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    all_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    all_parser.add_argument("--ffmpeg", default=None)
    all_parser.add_argument("--unit", action="append", default=[])
    all_parser.add_argument("--force", action="store_true")
    validate_parser = subparsers.add_parser("validate", help="Validate existing videos")
    validate_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    render_parser = subparsers.add_parser("_render")
    render_parser.add_argument("--kind", choices=("reference", "prediction"), required=True)
    render_parser.add_argument("--unit-id", required=True)
    render_parser.add_argument("--plan", type=Path, required=True)
    render_parser.add_argument("--registry", type=Path, required=True)
    render_parser.add_argument("--candidate-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "_render":
        render_internal(args)
        return 0
    if args.command == "validate":
        validate_output(args)
        return 0
    index = build_all(args)
    if not args.unit:
        validate_output(args)
        print(f"[gt-visualization] wrote {args.output / 'index.html'}")
    else:
        print(f"[gt-visualization] built {len(index['rows'])} selected unit(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
