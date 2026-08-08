#!/usr/bin/env python3
"""Prepare native COLMAP-A scenes for GenRecon without foundation geometry.

The adapter undistorts registered RGB/masks/observations to PINHOLE cameras,
keeps measured COLMAP tracks and errors, and applies one documented world
similarity for z-up plus proxy camera-height scale.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from tools.prepare_foundation_sfm import (
        align_z_up_and_proxy_scale,
        rotation_to_qvec,
        sha256_file,
        utc_now,
        write_json,
    )
except ModuleNotFoundError:
    from prepare_foundation_sfm import (
        align_z_up_and_proxy_scale,
        rotation_to_qvec,
        sha256_file,
        utc_now,
        write_json,
    )

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPRODUCTS_ROOT = ROOT / "data" / "internet-zero-shot" / "sfm-preproducts-v1"
DEFAULT_RAW_ROOT = ROOT / "data" / "internet-zero-shot" / "raw-candidates-v1"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "internet-zero-shot" / "native-sfm-v1"
DEFAULT_CANDIDATES = (
    "raw-001-copped-hall",
    "raw-010-waco-fire-station",
    "raw-016-harnden-tavern",
)
SCHEMA_VERSION = 1
VISUAL_DISPOSITIONS = {
    "raw-001-copped-hall": "proceed_engineering_pilot",
    "raw-010-waco-fire-station": "reject_current_shot_person_dominated",
    "raw-016-harnden-tavern": "reject_current_shot_person_dominated",
}


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant {value} in {path}")
        ),
    )


def quaternion_to_rotation(qvec: list[float]) -> np.ndarray:
    quaternion = np.asarray(qvec, dtype=np.float64)
    quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
    w, x, y, z = quaternion
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def read_cameras(path: Path) -> dict[int, dict[str, Any]]:
    cameras: dict[int, dict[str, Any]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        values = line.split()
        camera_id = int(values[0])
        cameras[camera_id] = {
            "camera_id": camera_id,
            "model": values[1],
            "width": int(values[2]),
            "height": int(values[3]),
            "params": [float(value) for value in values[4:]],
        }
    if not cameras:
        raise ValueError(f"No COLMAP cameras in {path}")
    return cameras


def read_images(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#"):
            continue
        values = line.split(maxsplit=9)
        if len(values) != 10:
            raise ValueError(f"Malformed COLMAP image header: {line[:160]}")
        points_line = lines[index].strip() if index < len(lines) else ""
        index += 1
        point_values = points_line.split()
        if point_values and len(point_values) % 3:
            raise ValueError(f"Malformed POINTS2D for {values[9]}")
        points2d = []
        for offset in range(0, len(point_values), 3):
            points2d.append(
                [
                    float(point_values[offset]),
                    float(point_values[offset + 1]),
                    int(point_values[offset + 2]),
                ]
            )
        rotation = quaternion_to_rotation([float(value) for value in values[1:5]])
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = rotation
        extrinsic[:3, 3] = [float(value) for value in values[5:8]]
        records.append(
            {
                "image_id": int(values[0]),
                "camera_id": int(values[8]),
                "name": Path(values[9]).name,
                "extrinsic": extrinsic,
                "points2d": points2d,
            }
        )
    if not records:
        raise ValueError(f"No COLMAP images in {path}")
    return records


def read_points(path: Path) -> list[dict[str, Any]]:
    points = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        values = line.split()
        if len(values) < 8 or (len(values) - 8) % 2:
            raise ValueError(f"Malformed COLMAP point: {line[:160]}")
        points.append(
            {
                "point_id": int(values[0]),
                "xyz": np.asarray([float(value) for value in values[1:4]], dtype=np.float64),
                "rgb": np.asarray([int(value) for value in values[4:7]], dtype=np.uint8),
                "error": float(values[7]),
                "track": [int(value) for value in values[8:]],
            }
        )
    if not points:
        raise ValueError(f"No COLMAP points in {path}")
    return points


def camera_calibration(camera: dict[str, Any]) -> dict[str, Any]:
    model = camera["model"]
    params = camera["params"]
    width, height = camera["width"], camera["height"]
    if model == "SIMPLE_RADIAL":
        focal, cx, cy, k1 = params
        fx = fy = focal
        distortion = np.asarray([k1, 0.0, 0.0, 0.0], dtype=np.float64)
    elif model == "RADIAL":
        focal, cx, cy, k1, k2 = params
        fx = fy = focal
        distortion = np.asarray([k1, k2, 0.0, 0.0], dtype=np.float64)
    elif model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params
        distortion = np.asarray([k1, k2, p1, p2], dtype=np.float64)
    elif model == "PINHOLE":
        fx, fy, cx, cy = params
        distortion = np.zeros(4, dtype=np.float64)
    elif model == "SIMPLE_PINHOLE":
        focal, cx, cy = params
        fx = fy = focal
        distortion = np.zeros(4, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported native COLMAP camera model: {model}")
    intrinsic = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    if np.any(distortion):
        new_intrinsic, roi = cv2.getOptimalNewCameraMatrix(
            intrinsic, distortion, (width, height), alpha=0, newImgSize=(width, height)
        )
        x, y, output_width, output_height = (int(value) for value in roi)
        if output_width <= 0 or output_height <= 0:
            raise ValueError(f"Empty undistortion ROI for camera {camera['camera_id']}")
        map_x, map_y = cv2.initUndistortRectifyMap(
            intrinsic,
            distortion,
            None,
            new_intrinsic,
            (width, height),
            cv2.CV_32FC1,
        )
        cropped_intrinsic = new_intrinsic.copy()
        cropped_intrinsic[0, 2] -= x
        cropped_intrinsic[1, 2] -= y
    else:
        x, y, output_width, output_height = 0, 0, width, height
        map_x = map_y = None
        cropped_intrinsic = intrinsic.copy()
    return {
        "source_model": model,
        "source_intrinsic": intrinsic,
        "distortion": distortion,
        "new_intrinsic_uncropped": (
            new_intrinsic if np.any(distortion) else intrinsic.copy()
        ),
        "intrinsic": cropped_intrinsic,
        "roi": (x, y, output_width, output_height),
        "map_x": map_x,
        "map_y": map_y,
        "width": output_width,
        "height": output_height,
    }


def transform_observations(
    points2d: list[list[float]], calibration: dict[str, Any]
) -> list[list[float]]:
    if not points2d:
        return []
    xy = np.asarray([[item[0], item[1]] for item in points2d], dtype=np.float64)
    if np.any(calibration["distortion"]):
        xy = cv2.undistortPoints(
            xy.reshape(-1, 1, 2),
            calibration["source_intrinsic"],
            calibration["distortion"],
            P=calibration["new_intrinsic_uncropped"],
        ).reshape(-1, 2)
        xy[:, 0] -= calibration["roi"][0]
        xy[:, 1] -= calibration["roi"][1]
    return [
        [float(position[0]), float(position[1]), int(source[2])]
        for position, source in zip(xy, points2d)
    ]


def dynamic_fraction_lookup(mask_summary: dict[str, Any]) -> dict[str, float | None]:
    return {
        Path(item.get("image", item.get("name", ""))).stem: item.get("dynamic_fraction")
        for item in mask_summary.get("frames", [])
        if item.get("image") or item.get("name")
    }


def dynamic_mask_path(candidate_dir: Path, image_name: str) -> Path:
    choices = [
        candidate_dir / "masks_dynamic" / f"{Path(image_name).name}.png",
        candidate_dir / "masks_dynamic" / f"{Path(image_name).stem}.png",
    ]
    for path in choices:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Dynamic mask missing for {image_name}")


def undistort_rgba(
    source_path: Path,
    dynamic_path: Path,
    calibration: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(source_path) as opened:
        rgb = np.asarray(opened.convert("RGB"))
    with Image.open(dynamic_path) as opened:
        dynamic = np.asarray(opened.convert("L"))
    if dynamic.shape != rgb.shape[:2]:
        raise ValueError(f"Mask/RGB dimensions differ: {dynamic_path}")
    valid = np.full(dynamic.shape, 255, dtype=np.uint8)
    if calibration["map_x"] is not None:
        rgb = cv2.remap(
            rgb,
            calibration["map_x"],
            calibration["map_y"],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        dynamic = cv2.remap(
            dynamic,
            calibration["map_x"],
            calibration["map_y"],
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
        valid = cv2.remap(
            valid,
            calibration["map_x"],
            calibration["map_y"],
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    x, y, width, height = calibration["roi"]
    rgb = rgb[y : y + height, x : x + width]
    dynamic = dynamic[y : y + height, x : x + width]
    valid = valid[y : y + height, x : x + width]
    dynamic_output = np.where((dynamic > 127) | (valid <= 127), 255, 0).astype(np.uint8)
    alpha = (255 - dynamic_output).astype(np.uint8)
    return np.concatenate([rgb, alpha[..., None]], axis=2), dynamic_output


def write_native_ply(path: Path, points: list[dict[str, Any]], aligned_xyz: np.ndarray) -> None:
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("error", "<f4"),
            ("track_length", "<u2"),
        ]
    )
    vertices = np.empty(len(points), dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = aligned_xyz.T.astype(np.float32)
    colors = np.stack([item["rgb"] for item in points])
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    vertices["error"] = [item["error"] for item in points]
    vertices["track_length"] = [len(item["track"]) // 2 for item in points]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property float error\nproperty ushort track_length\nend_header\n"
    ).encode("ascii")
    path.write_bytes(header + vertices.tobytes())


def camera_up_diagnostic(extrinsics: np.ndarray) -> dict[str, Any]:
    camera_to_world = np.transpose(extrinsics[:, :3, :3], (0, 2, 1))
    up_vectors = -camera_to_world[:, :, 1]
    mean_up = np.mean(up_vectors, axis=0)
    mean_up /= max(float(np.linalg.norm(mean_up)), 1e-12)
    angles = np.degrees(
        np.arccos(np.clip(np.abs(up_vectors @ mean_up), -1.0, 1.0))
    )
    return {
        "mean_up_old_world": mean_up,
        "angle_deg": {
            "median": float(np.median(angles)),
            "p90": float(np.percentile(angles, 90)),
            "max": float(np.max(angles)),
        },
    }


def reprojection_diagnostic(
    images: list[dict[str, Any]],
    points: list[dict[str, Any]],
    calibrations: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    point_lookup = {item["point_id"]: item["xyz_aligned"] for item in points}
    residuals = []
    for image in images:
        intrinsic = calibrations[image["camera_id"]]["intrinsic"]
        rotation = image["extrinsic_aligned"][:3, :3]
        translation = image["extrinsic_aligned"][:3, 3]
        for x, y, point_id in image["points2d_aligned"]:
            if point_id < 0 or point_id not in point_lookup:
                continue
            camera_xyz = rotation @ point_lookup[point_id] + translation
            if camera_xyz[2] <= 1e-9:
                continue
            projected = intrinsic @ camera_xyz
            projected = projected[:2] / projected[2]
            residuals.append(float(np.linalg.norm(projected - [x, y])))
    values = np.asarray(residuals, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean_px": float(np.mean(values)),
        "median_px": float(np.median(values)),
        "p90_px": float(np.percentile(values, 90)),
        "max_px": float(np.max(values)),
    }


def candidate_inputs(preproducts_root: Path, candidate_id: str) -> dict[str, Path]:
    source = preproducts_root / "candidates" / candidate_id
    return {
        "source": source,
        "rgb": source / "rgb",
        "masks": source / "masks_dynamic",
        "colmap": source / "colmap",
        "frames": source / "frames.json",
        "quality": source / "quality.json",
    }


def prepare_candidate(
    candidate: dict[str, Any],
    *,
    preproducts_root: Path,
    raw_root: Path,
    output_root: Path,
    proxy_camera_height: float,
    force: bool,
) -> dict[str, Any]:
    candidate_id = candidate["candidate_id"]
    inputs = candidate_inputs(preproducts_root, candidate_id)
    output = output_root / "candidates" / candidate_id
    manifest_path = output / "manifest.json"
    if manifest_path.is_file() and not force:
        previous = load_json(manifest_path)
        if previous.get("status") == "completed":
            print(f"[native] {candidate_id}: already prepared")
            return previous
    if force and output.exists():
        shutil.rmtree(output)
    (output / "rgb").mkdir(parents=True, exist_ok=True)
    (output / "masks_dynamic").mkdir(parents=True, exist_ok=True)
    (output / "colmap_sfm").mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    quality = load_json(inputs["quality"])
    frames_source = load_json(inputs["frames"])
    cameras = read_cameras(inputs["colmap"] / "cameras.txt")
    images = read_images(inputs["colmap"] / "images.txt")
    points = read_points(inputs["colmap"] / "points3D.txt")
    calibrations = {camera_id: camera_calibration(camera) for camera_id, camera in cameras.items()}
    extrinsics = np.stack([item["extrinsic"] for item in images])
    xyz = np.stack([item["xyz"] for item in points])
    quality_indices = np.asarray(
        [
            index
            for index, point in enumerate(points)
            if point["error"] <= 2.0 and len(point["track"]) // 2 >= 4
        ],
        dtype=np.int64,
    )
    if quality_indices.size < 500:
        raise ValueError(f"Too few quality COLMAP points for alignment: {quality_indices.size}")
    _, aligned_extrinsics, alignment = align_z_up_and_proxy_scale(
        xyz[quality_indices], extrinsics, proxy_camera_height
    )
    transform = np.asarray(alignment["old_world_to_proxy_metric_z_up"], dtype=np.float64)
    aligned_xyz = (xyz @ transform[:3, :3].T) + transform[:3, 3]
    for point, position in zip(points, aligned_xyz):
        point["xyz_aligned"] = position
    frames_lookup = {Path(item["name"]).stem: item for item in frames_source["frames"]}
    dynamic_summary = load_json(inputs["source"] / "mask_summary.json")
    dynamic_by_name = dynamic_fraction_lookup(dynamic_summary)
    output_frames = []
    for image, aligned_extrinsic in zip(images, aligned_extrinsics):
        image["extrinsic_aligned"] = aligned_extrinsic
        image["points2d_aligned"] = transform_observations(
            image["points2d"], calibrations[image["camera_id"]]
        )
        source_path = inputs["rgb"] / image["name"]
        mask_path = dynamic_mask_path(inputs["source"], image["name"])
        rgba, dynamic = undistort_rgba(
            source_path, mask_path, calibrations[image["camera_id"]]
        )
        output_name = f"{Path(image['name']).stem}.png"
        Image.fromarray(rgba, mode="RGBA").save(output / "rgb" / output_name)
        Image.fromarray(dynamic, mode="L").save(output / "masks_dynamic" / output_name)
        source_frame = frames_lookup[Path(image["name"]).stem]
        output_frames.append(
            {
                "index": len(output_frames),
                "name": output_name,
                "source_name": image["name"],
                "source_index": source_frame["index"],
                "timestamp_s": source_frame["timestamp_s"],
                "width": rgba.shape[1],
                "height": rgba.shape[0],
                "dynamic_fraction": float(np.mean(dynamic > 127)),
                "source_dynamic_fraction": dynamic_by_name.get(Path(image["name"]).stem),
            }
        )
    camera_lines = [
        "# Native COLMAP cameras undistorted to PINHOLE",
        "# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]",
    ]
    for camera_id in sorted(cameras):
        calibration = calibrations[camera_id]
        intrinsic = calibration["intrinsic"]
        camera_lines.append(
            f"{camera_id} PINHOLE {calibration['width']} {calibration['height']} "
            f"{intrinsic[0,0]:.17g} {intrinsic[1,1]:.17g} "
            f"{intrinsic[0,2]:.17g} {intrinsic[1,2]:.17g}"
        )
    image_lines = [
        "# Native COLMAP images transformed by one world Sim(3)",
        "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME",
    ]
    for image in images:
        qvec = rotation_to_qvec(image["extrinsic_aligned"][:3, :3])
        translation = image["extrinsic_aligned"][:3, 3]
        output_name = f"{Path(image['name']).stem}.png"
        image_lines.append(
            f"{image['image_id']} "
            + " ".join(f"{value:.17g}" for value in qvec)
            + " "
            + " ".join(f"{value:.17g}" for value in translation)
            + f" {image['camera_id']} {output_name}"
        )
        image_lines.append(
            " ".join(
                f"{x:.8f} {y:.8f} {point_id}"
                for x, y, point_id in image["points2d_aligned"]
            )
        )
    point_lines = [
        "# Native observed COLMAP points; XYZ transformed by one world Sim(3)",
        "# POINT3D_ID X Y Z R G B ERROR TRACK[]",
    ]
    for point in points:
        point_lines.append(
            f"{point['point_id']} "
            + " ".join(f"{value:.17g}" for value in point["xyz_aligned"])
            + " "
            + " ".join(str(int(value)) for value in point["rgb"])
            + f" {point['error']:.17g} "
            + " ".join(str(value) for value in point["track"])
        )
    colmap_output = output / "colmap_sfm"
    (colmap_output / "cameras.txt").write_text("\n".join(camera_lines) + "\n")
    (colmap_output / "images.txt").write_text("\n".join(image_lines) + "\n")
    (colmap_output / "points3D.txt").write_text("\n".join(point_lines) + "\n")
    write_native_ply(output / "sfm_points.ply", points, aligned_xyz)
    reprojection = reprojection_diagnostic(images, points, calibrations)
    raw_index = load_json(raw_root / "index.json")
    raw_candidate = next(
        item for item in raw_index["candidates"] if item["candidate_id"] == candidate_id
    )
    alignment["camera_up_diagnostic"] = camera_up_diagnostic(extrinsics)
    alignment["status"] = "proxy-not-ground-truth-metric"
    alignment["source"] = "native_masked_colmap"
    source_video = Path(frames_source["source_video"])
    manifest = {
        "schema": "genrecon.native-sfm-scene",
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "status": "completed",
        "track": "native-sfm",
        "candidate_id": candidate_id,
        "title": candidate["title"],
        "source_grade": quality["quality_gate"]["grade"],
        "visual_disposition": VISUAL_DISPOSITIONS[candidate_id],
        "rights": {
            "license": raw_candidate.get("license"),
            "license_status": raw_candidate.get("license_status"),
            "source_page": raw_candidate.get("source_page"),
            "capture_date": raw_candidate.get("capture_date"),
            "catalog_date": raw_candidate.get("catalog_date"),
        },
        "source": {
            "preproducts_directory": str(inputs["source"].resolve()),
            "colmap_directory": str(inputs["colmap"].resolve()),
            "source_video": str(source_video.resolve()),
            "source_video_sha256": frames_source["source_sha256"],
            "quality_json_sha256": sha256_file(inputs["quality"]),
        },
        "selection": {
            "policy": "all_registered_images_from_best_native_colmap_model",
            "selected": [item["name"] for item in output_frames],
            "selected_dynamic_fraction": [item["dynamic_fraction"] for item in output_frames],
        },
        "counts": {
            "registered_images": len(images),
            "points": len(points),
            "quality_points_error_le_2_track_ge_4": int(quality_indices.size),
            "observations": sum(len(item["points2d"]) for item in images),
        },
        "camera_conversion": {
            str(camera_id): {
                "source_model": cameras[camera_id]["model"],
                "source_width": cameras[camera_id]["width"],
                "source_height": cameras[camera_id]["height"],
                "source_params": cameras[camera_id]["params"],
                "output_model": "PINHOLE",
                "output_width": calibrations[camera_id]["width"],
                "output_height": calibrations[camera_id]["height"],
                "output_intrinsic": calibrations[camera_id]["intrinsic"],
                "roi": calibrations[camera_id]["roi"],
            }
            for camera_id in cameras
        },
        "alignment": alignment,
        "reprojection_after_adapter": reprojection,
        "frames": output_frames,
        "limitations": [
            "Geometry, camera poses, tracks, and point errors are native masked-COLMAP observations.",
            "Metric scale is a 1.6 m handheld-camera-height proxy, not a measured scene scale.",
            "Gravity uses mean camera up and is not an IMU measurement.",
            "RGBA alpha suppresses dynamic-mask pixels for GenRecon while preserving RGB values.",
            "Technical SfM-A does not override visual, privacy, rights, or zero-shot gates.",
        ],
        "elapsed_s": time.monotonic() - start,
    }
    write_json(manifest_path, manifest)
    write_json(
        output / "frames.json",
        {
            "schema": "genrecon.native-sfm-registered-frames",
            "schema_version": SCHEMA_VERSION,
            "candidate_id": candidate_id,
            "source_video": str(source_video.resolve()),
            "source_sha256": frames_source["source_sha256"],
            "selected_start_s": frames_source["selected_start_s"],
            "selected_end_s": frames_source["selected_end_s"],
            "fps": frames_source["fps"],
            "frame_count": len(output_frames),
            "frames": output_frames,
        },
    )
    print(
        f"[native] {candidate_id}: images={len(images)} points={len(points)} "
        f"scale={alignment['scale_factor']:.5f} reproj_p90={reprojection['p90_px']:.3f}px"
    )
    return manifest


def run_preflight(
    candidate_id: str,
    output_root: Path,
    *,
    views: int,
    seed: int,
    force: bool,
) -> dict[str, Any]:
    scene = output_root / "candidates" / candidate_id
    destination = scene / "genrecon_preflight"
    status_path = destination / "preflight.json"
    if status_path.is_file() and not force:
        previous = load_json(status_path)
        if previous.get("status") == "passed":
            print(f"[native] {candidate_id}: preflight already passed")
            return previous
    if force and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    captured = io.StringIO()
    try:
        from inference.get_chunks import IphoneChunker
        from inference.get_images import IphoneImageSelecter

        manifest = load_json(scene / "manifest.json")
        with contextlib.redirect_stdout(captured):
            centers, original_to_chunks, _, _ = IphoneChunker(
                colmap_subdir="colmap_sfm",
                max_reproj_error=2.0,
                min_track_len=4,
            ).get_chunks(scene, destination)
            selected = IphoneImageSelecter(center_crop=False).get_images(
                original_to_chunks,
                scene / "colmap_sfm" / "cameras.txt",
                min(views, manifest["counts"]["registered_images"]),
                destination,
                seed=seed,
            )
        log = captured.getvalue()
        (destination / "preflight.log").write_text(log)
        fallback_count = log.count("falling back to closest camera")
        import trimesh

        cloud = trimesh.load(destination / "clean_points.ply", process=False)
        clean_points = int(len(cloud.vertices))
        status = "passed" if centers and clean_points >= 500 and fallback_count == 0 else "failed"
        result = {
            "schema": "genrecon.native-sfm-preflight",
            "schema_version": SCHEMA_VERSION,
            "created_utc": utc_now(),
            "candidate_id": candidate_id,
            "status": status,
            "colmap_subdir": "colmap_sfm",
            "max_reproj_error_px": 2.0,
            "min_track_length": 4,
            "clean_point_count": clean_points,
            "chunk_count": len(centers),
            "scene_image_crop_count": int(selected.scene_images_512.shape[0]),
            "condition_view_count": len(selected.cond2d_images_512),
            "condition_chunk_indices": selected.chunk_indices,
            "closest_camera_fallback_count": fallback_count,
            "elapsed_s": time.monotonic() - start,
        }
        if status != "passed":
            result["error"] = "Native-SfM preflight requires usable chunks, >=500 points, and zero fallback"
        write_json(status_path, result)
        print(
            f"[native] {candidate_id}: preflight={status} clean={clean_points} "
            f"chunks={len(centers)} fallback={fallback_count}"
        )
        return result
    except Exception as error:
        (destination / "preflight.log").write_text(captured.getvalue())
        result = {
            "schema": "genrecon.native-sfm-preflight",
            "schema_version": SCHEMA_VERSION,
            "created_utc": utc_now(),
            "candidate_id": candidate_id,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_s": time.monotonic() - start,
        }
        write_json(status_path, result)
        print(f"[native] {candidate_id}: preflight failed: {error}")
        return result


def selected_candidates(preproducts_root: Path, requested: list[str]) -> list[dict[str, Any]]:
    index = load_json(preproducts_root / "index.json")
    identifiers = list(dict.fromkeys(requested or DEFAULT_CANDIDATES))
    by_id = {item["candidate_id"]: item for item in index["candidates"]}
    missing = sorted(set(identifiers) - set(by_id))
    if missing:
        raise ValueError(f"Unknown preproduct candidates: {missing}")
    candidates = [by_id[identifier] for identifier in identifiers]
    non_a = [
        item["candidate_id"]
        for item in candidates
        if item["quality"]["quality_gate"]["grade"] != "A"
    ]
    if non_a:
        raise ValueError(f"Native track accepts only SfM-A candidates: {non_a}")
    return candidates


def summary_rows(candidates: list[dict[str, Any]], output_root: Path) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        directory = output_root / "candidates" / candidate_id
        manifest = load_json(directory / "manifest.json") if (directory / "manifest.json").is_file() else {}
        preflight = (
            load_json(directory / "genrecon_preflight" / "preflight.json")
            if (directory / "genrecon_preflight" / "preflight.json").is_file()
            else {}
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                "title": candidate["title"],
                "status": manifest.get("status", "missing"),
                "grade": f"SfM-{manifest.get('source_grade', '-')}",
                "preflight": preflight.get("status", "missing"),
                "registered_images": manifest.get("counts", {}).get("registered_images"),
                "points": manifest.get("counts", {}).get("points"),
                "quality_points": manifest.get("counts", {}).get(
                    "quality_points_error_le_2_track_ge_4"
                ),
                "proxy_scale": manifest.get("alignment", {}).get("scale_factor"),
                "reprojection_p90_px": manifest.get("reprojection_after_adapter", {}).get("p90_px"),
                "clean_points": preflight.get("clean_point_count"),
                "chunks": preflight.get("chunk_count"),
                "fallback": preflight.get("closest_camera_fallback_count"),
                "visual_disposition": manifest.get("visual_disposition"),
                "license": manifest.get("rights", {}).get("license"),
                "license_status": manifest.get("rights", {}).get("license_status"),
            }
        )
    return rows


def make_overview(rows: list[dict[str, Any]], output_root: Path) -> None:
    canvas = Image.new("RGB", (1500, 930), (238, 241, 242))
    draw = ImageDraw.Draw(canvas)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    text_font = ImageFont.truetype(str(font_path), 15) if font_path.is_file() else ImageFont.load_default()
    for index, row in enumerate(rows):
        left = index * 500
        draw.rectangle((left + 8, 8, left + 492, 920), fill=(255, 255, 255), outline=(190, 198, 203))
        draw.text((left + 20, 20), f"{row['candidate_id']}  {row['grade']}", fill=(25, 30, 34), font=text_font)
        draw.text(
            (left + 20, 44),
            f"preflight={row['preflight']} chunks={row['chunks']} fallback={row['fallback']}",
            fill=(70, 78, 84),
            font=text_font,
        )
        image_path = output_root.parent / "sfm-preproducts-v1" / "candidates" / row["candidate_id"] / "qc_contact.jpg"
        if not image_path.is_file():
            image_path = DEFAULT_PREPRODUCTS_ROOT / "candidates" / row["candidate_id"] / "qc_contact.jpg"
        with Image.open(image_path) as opened:
            preview = opened.convert("RGB")
            preview.thumbnail((460, 820), Image.Resampling.LANCZOS)
        canvas.paste(preview, (left + 20 + (460 - preview.width) // 2, 82))
    canvas.save(output_root / "overview.jpg", quality=90, subsampling=0)


def summarize(candidates: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    rows = summary_rows(candidates, output_root)
    summary = {
        "schema": "genrecon.native-sfm-index",
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "track": "native-sfm",
        "summary": {
            "candidate_count": len(rows),
            "completed": sum(item["status"] == "completed" for item in rows),
            "preflight": dict(sorted(Counter(item["preflight"] for item in rows).items())),
            "registered_images": sum(item["registered_images"] or 0 for item in rows),
            "points": sum(item["points"] or 0 for item in rows),
            "clean_points": sum(item["clean_points"] or 0 for item in rows),
            "chunks": sum(item["chunks"] or 0 for item in rows),
            "fallback": sum(item["fallback"] or 0 for item in rows),
        },
        "candidates": rows,
    }
    write_json(output_root / "index.json", summary)
    fields = list(rows[0]) if rows else []
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    make_overview(rows, output_root)
    (output_root / "README.md").write_text(
        """# Native SfM GenRecon Inputs v1

This track contains native masked-COLMAP observations, not foundation pseudo
geometry. RGB is undistorted RGBA, `colmap_sfm/` is PINHOLE, and one documented
world Sim(3) provides z-up plus proxy camera-height scale.

```text
candidates/<candidate_id>/
  rgb/
  masks_dynamic/
  colmap_sfm/
  sfm_points.ply
  frames.json
  manifest.json
  genrecon_preflight/
```

`index.json`, `summary.csv`, and `validation.json` are the machine-readable
entry points. Metric scale remains a proxy and visual/privacy/rights gates are
independent of SfM-A.
""",
        encoding="utf-8",
    )
    return summary


def validate(candidates: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    summary = summarize(candidates, output_root)
    errors = []
    rgba_count = 0
    alpha_pixels = 0
    point_count = 0
    for row in summary["candidates"]:
        candidate_id = row["candidate_id"]
        directory = output_root / "candidates" / candidate_id
        manifest = load_json(directory / "manifest.json")
        if row["preflight"] != "passed" or row["fallback"] != 0:
            errors.append(f"{candidate_id}: native preflight must pass with zero fallback")
        images = read_images(directory / "colmap_sfm" / "images.txt")
        points = read_points(directory / "colmap_sfm" / "points3D.txt")
        point_count += len(points)
        if len(images) != manifest["counts"]["registered_images"]:
            errors.append(f"{candidate_id}: registered image count mismatch")
        if len(points) != manifest["counts"]["points"]:
            errors.append(f"{candidate_id}: point count mismatch")
        for frame in manifest["frames"]:
            rgba_path = directory / "rgb" / frame["name"]
            mask_path = directory / "masks_dynamic" / frame["name"]
            with Image.open(rgba_path) as opened:
                rgba = np.asarray(opened.convert("RGBA"))
            with Image.open(mask_path) as opened:
                dynamic = np.asarray(opened.convert("L"))
            if rgba.shape[:2] != dynamic.shape or not np.array_equal(rgba[..., 3], 255 - dynamic):
                errors.append(f"{candidate_id}/{frame['name']}: alpha/mask mismatch")
            if set(np.unique(dynamic)) - {0, 255}:
                errors.append(f"{candidate_id}/{frame['name']}: mask is not binary")
            rgba_count += 1
            alpha_pixels += dynamic.size
        if manifest["reprojection_after_adapter"]["p90_px"] > 2.5:
            errors.append(f"{candidate_id}: adapted reprojection p90 exceeds 2.5 px")
    strict_json = 0
    for path in output_root.rglob("*.json"):
        try:
            load_json(path)
            strict_json += 1
        except Exception as error:
            errors.append(f"strict JSON {path}: {error}")
    result = {
        "schema": "genrecon.native-sfm-validation",
        "schema_version": SCHEMA_VERSION,
        "validated_utc": utc_now(),
        "result": "pass" if not errors else "fail",
        "counts": {
            **summary["summary"],
            "rgba_images": rgba_count,
            "alpha_pixels": alpha_pixels,
            "colmap_points": point_count,
            "strict_json_files": strict_json,
        },
        "checks": {
            "native_colmap_preserved": not any("count mismatch" in error for error in errors),
            "rgba_alpha_matches_dynamic_mask": not any("alpha/mask" in error for error in errors),
            "binary_dynamic_masks": not any("not binary" in error for error in errors),
            "adapted_reprojection_p90_le_2_5px": not any(
                "reprojection" in error for error in errors
            ),
            "zero_fallback_preflight": not any("preflight" in error for error in errors),
            "strict_json": not any(error.startswith("strict JSON") for error in errors),
        },
        "errors": errors,
    }
    write_json(output_root / "validation.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "preflight", "summarize", "validate", "all"))
    parser.add_argument("--preproducts-root", type=Path, default=DEFAULT_PREPRODUCTS_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--proxy-camera-height", type=float, default=1.6)
    parser.add_argument("--genrecon-views", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-preflight", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.proxy_camera_height <= 0 or args.genrecon_views <= 0:
        raise SystemExit("Proxy camera height and GenRecon views must be positive")
    candidates = selected_candidates(args.preproducts_root, args.candidate)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.stage in {"prepare", "all"}:
        write_json(
            args.output_root / "config.json",
            {
                "schema": "genrecon.native-sfm-config",
                "schema_version": SCHEMA_VERSION,
                "created_utc": utc_now(),
                "candidate_ids": [item["candidate_id"] for item in candidates],
                "preproducts_root": str(args.preproducts_root.resolve()),
                "raw_root": str(args.raw_root.resolve()),
                "output_root": str(args.output_root.resolve()),
                "protocol": {
                    "source_gate": "SfM-A only",
                    "output_camera_model": "PINHOLE",
                    "dynamic_alpha": "alpha = 255 - undistorted dynamic mask",
                    "gravity": "negative mean native COLMAP camera y axis",
                    "proxy_camera_height_m": args.proxy_camera_height,
                    "point_filter_max_reprojection_error_px": 2.0,
                    "point_filter_min_track_length": 4,
                    "genrecon_views": args.genrecon_views,
                    "seed": args.seed,
                },
                "source_index_sha256": sha256_file(args.preproducts_root / "index.json"),
                "tool": str(Path(__file__).resolve()),
                "tool_sha256": sha256_file(Path(__file__).resolve()),
            },
        )
        for candidate in candidates:
            prepare_candidate(
                candidate,
                preproducts_root=args.preproducts_root,
                raw_root=args.raw_root,
                output_root=args.output_root,
                proxy_camera_height=args.proxy_camera_height,
                force=args.force,
            )
    if args.stage in {"preflight", "all"}:
        for candidate in candidates:
            run_preflight(
                candidate["candidate_id"],
                args.output_root,
                views=args.genrecon_views,
                seed=args.seed,
                force=args.force_preflight,
            )
    if args.stage in {"summarize", "all"}:
        summarize(candidates, args.output_root)
    if args.stage in {"validate", "all"}:
        result = validate(candidates, args.output_root)
        print(f"[native-validation] {result['result']} {result['counts']}")
        if result["result"] != "pass":
            for error in result["errors"]:
                print(f"[native-validation] ERROR: {error}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
