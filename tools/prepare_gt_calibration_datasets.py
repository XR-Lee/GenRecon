#!/usr/bin/env python3
"""Build the frozen mixed-GT calibration registry and lightweight input packages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile

import numpy as np
import py7zr
import trimesh
from PIL import Image

try:
    from tools.prepare_da3_scannetpp import (
        _camera_center,
        read_cameras_binary,
        read_image_headers_binary,
    )
except ModuleNotFoundError:
    from prepare_da3_scannetpp import (
        _camera_center,
        read_cameras_binary,
        read_image_headers_binary,
    )

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "configs" / "eval" / "gt_calibration_v1.json"
DEFAULT_OUTPUT = ROOT / "data" / "gt-calibration-v1"
SCHEMA = "genrecon.gt-calibration-unit"
REGISTRY_SCHEMA = "genrecon.gt-calibration-registry"
PLAN_SCHEMA = "genrecon.gt-calibration-plan"
_HASH_CACHE: dict[tuple[str, int, int], str] = {}
SEVEN_INTRINSICS = {"fx": 585.0, "fy": 585.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480}
REDWOOD_INTRINSICS = {"fx": 525.0, "fy": 525.0, "cx": 319.5, "cy": 239.5, "width": 640, "height": 480}


class CalibrationBuildError(RuntimeError):
    pass


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            CalibrationBuildError(f"Non-finite JSON constant {value} in {path}")
        ),
    )


def write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key in _HASH_CACHE:
        return _HASH_CACHE[key]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    value = digest.hexdigest()
    _HASH_CACHE[key] = value
    return value


def file_prefix(path: Path, size: int) -> bytes:
    with path.open("rb") as handle:
        return handle.read(size)


def relpath(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), base.resolve())


def uniform_indices(count: int, requested: int, *, excluded: set[int] | None = None) -> list[int]:
    if count <= 0 or requested <= 0:
        return []
    excluded = excluded or set()
    eligible = [index for index in range(count) if index not in excluded]
    if len(eligible) <= requested:
        return eligible
    positions = np.linspace(0, len(eligible) - 1, requested)
    selected = [eligible[int(round(value))] for value in positions]
    return list(dict.fromkeys(selected))


def safe_zip_names(archive: ZipFile) -> list[str]:
    names = archive.namelist()
    for name in names:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise CalibrationBuildError(f"Unsafe ZIP path {name!r}")
    return names


def safe_7z_extract(path: Path, output: Path) -> None:
    with py7zr.SevenZipFile(path) as archive:
        names = archive.getnames()
        for name in names:
            member = Path(name)
            if member.is_absolute() or ".." in member.parts:
                raise CalibrationBuildError(f"Unsafe 7z path {name!r} in {path}")
        archive.extractall(path=output)


def _write_source_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [
        {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in paths
    ]


def _manifest_path(output: Path, unit_id: str) -> Path:
    return output / "units" / unit_id / "manifest.json"


def _write_manifest(output: Path, document: dict[str, Any]) -> dict[str, Any]:
    path = _manifest_path(output, document["unit_id"])
    write_json(path, document)
    return {
        "unit_id": document["unit_id"],
        "physical_scene_group": document["physical_scene_group"],
        "dataset": document["dataset"],
        "track": document["track"],
        "gt_tier": document["gt_tier"],
        "status": document["status"],
        "manifest": relpath(path, output),
        "prediction_mesh": document.get("prediction_mesh"),
        "blocker": document.get("blocker"),
    }


def _base_manifest(
    *,
    unit_id: str,
    dataset: str,
    track: str,
    gt_tier: str,
    capture_kind: str,
    license_status: str,
    status: str = "prepared",
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "unit_id": unit_id,
        "physical_scene_group": f"{dataset}:{unit_id}",
        "dataset": dataset,
        "track": track,
        "gt_tier": gt_tier,
        "capture_kind": capture_kind,
        "license_status": license_status,
        "status": status,
    }


def _parse_colmap_image_names(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    if len(lines) % 2:
        raise CalibrationBuildError(f"COLMAP images file has an odd number of records: {path}")
    return [line.split()[9] for line in lines[::2]]


def prepare_scannetpp(plan: dict[str, Any], output: Path) -> list[dict[str, Any]]:
    dataset = plan["datasets"]["da3-scannetpp"]
    rows = []
    for scene_id in dataset["unit_ids"]:
        adapted = ROOT / "data" / "da3-adapted" / scene_id
        selection_path = adapted / "selection.json"
        source_scene = ROOT / "data" / "da3-bench" / "extracted" / "scannetpp" / scene_id
        merge_root = source_scene / "merge_dslr_iphone"
        source_model = merge_root / "colmap" / "sparse_render_rgb"
        reference = source_scene / "scans" / "mesh_aligned_0.05.ply"
        prediction = ROOT / "outputs" / scene_id / "reconstruction" / "mesh.ply"
        required = [
            selection_path,
            reference,
            adapted / "iphone" / "colmap" / "cameras.txt",
            adapted / "iphone" / "colmap" / "images.txt",
            source_model / "cameras.bin",
            source_model / "images.bin",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        status = "prepared" if not missing else "missing-local-source"
        unit_id = f"scannetpp-{scene_id}"
        unit_dir = output / "units" / unit_id
        manifest_path = _manifest_path(output, unit_id)
        selection = load_json(selection_path) if selection_path.is_file() else {}
        camera_records: list[dict[str, Any]] = []
        conditioning_paths: list[str] = []
        heldout_paths: list[str] = []
        conditioning_depths: list[str] = []
        heldout_depths: list[str] = []
        source_view_files: list[Path] = []
        camera_path = unit_dir / "cameras.json"
        if status == "prepared":
            cameras = read_cameras_binary(source_model / "cameras.bin")
            headers = sorted(
                (
                    item
                    for item in read_image_headers_binary(source_model / "images.bin")
                    if item.name.startswith("iphone/")
                ),
                key=lambda item: item.name,
            )
            by_name = {item.name: item for item in headers}
            conditioning_names = [item["original_name"] for item in selection["views"]]
            missing_names = sorted(set(conditioning_names).difference(by_name))
            if missing_names:
                raise CalibrationBuildError(
                    f"ScanNet++ selected views missing from source model {scene_id}: {missing_names}"
                )
            remaining = [item for item in headers if item.name not in set(conditioning_names)]
            heldout_headers = [remaining[index] for index in uniform_indices(len(remaining), 8)]
            role_headers = {
                "conditioning": [by_name[name] for name in conditioning_names],
                "heldout": heldout_headers,
            }
            for role, selected_headers in role_headers.items():
                for order, header in enumerate(selected_headers):
                    source_rgb = merge_root / "images" / header.name
                    source_depth = merge_root / "render_depth" / f"{Path(header.name).stem}.png"
                    if not source_rgb.is_file() or not source_depth.is_file():
                        raise CalibrationBuildError(
                            f"Missing ScanNet++ RGB/depth pair: {source_rgb}, {source_depth}"
                        )
                    rgb = unit_dir / "rgb" / role / f"{order:03d}{source_rgb.suffix.lower()}"
                    depth = unit_dir / "depth" / role / f"{order:03d}.png"
                    rgb.parent.mkdir(parents=True, exist_ok=True)
                    depth.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_rgb, rgb)
                    shutil.copy2(source_depth, depth)
                    camera = cameras[header.camera_id]
                    record = {
                        "role": role,
                        "order": order,
                        "source_image": header.name,
                        "rgb": relpath(rgb, unit_dir),
                        "depth": relpath(depth, unit_dir),
                        "qvec_wxyz": list(header.qvec),
                        "tvec_world_to_camera": list(header.tvec),
                        "camera_center_world_m": _camera_center(header.qvec, header.tvec),
                        "camera_id": header.camera_id,
                        "intrinsics": {
                            "model": camera.model_name,
                            "width": camera.width,
                            "height": camera.height,
                            "params": list(camera.params),
                        },
                    }
                    camera_records.append(record)
                    target_views = conditioning_paths if role == "conditioning" else heldout_paths
                    target_depths = conditioning_depths if role == "conditioning" else heldout_depths
                    target_views.append(relpath(rgb, manifest_path.parent))
                    target_depths.append(relpath(depth, manifest_path.parent))
                    source_view_files.extend([source_rgb, source_depth])
            write_json(
                camera_path,
                {
                    "schema": "genrecon.gt-camera-split",
                    "schema_version": 1,
                    "pose_convention": "world-to-camera-colmap-qvec-tvec",
                    "conditioning": [item for item in camera_records if item["role"] == "conditioning"],
                    "heldout": [item for item in camera_records if item["role"] == "heldout"],
                },
            )
        manifest = _base_manifest(
            unit_id=unit_id,
            dataset="da3-scannetpp",
            track=dataset["track"],
            gt_tier=dataset["gt_tier"],
            capture_kind=dataset["capture_kind"],
            license_status=dataset["license_status"],
            status=status,
        )
        manifest.update(
            {
                "source": {
                    "scene_id": scene_id,
                    "selection": str(selection_path.resolve()),
                    "records": _write_source_records([*required, *source_view_files]) if status == "prepared" else [],
                },
                "input": {
                    "conditioning_views": conditioning_paths,
                    "heldout_views": heldout_paths,
                    "conditioning_depths": conditioning_depths,
                    "heldout_depths": heldout_depths,
                    "cameras": relpath(camera_path, manifest_path.parent) if camera_records else None,
                    "camera_model": relpath(adapted / "iphone" / "colmap" / "cameras.txt", manifest_path.parent),
                    "images_model": relpath(adapted / "iphone" / "colmap" / "images.txt", manifest_path.parent),
                    "pose_source": "provided-registered-iphone-colmap",
                    "split_policy": "freeze-existing-8-conditioning; uniformly-select-8-disjoint-registered-iphone-heldout",
                },
                "reference": {
                    "kind": "mesh",
                    "paths": [relpath(reference, manifest_path.parent)],
                    "roi": "paper-like-unclipped",
                    "scope": "raw-global",
                    "source_type": "submillimeter-laser-scan-mesh",
                },
                "evaluation": {
                    "alignment": "provided-metric-world-frame",
                    "limitations": [
                        "GenRecon scanner observation envelope is unavailable; raw and optional GT-AABB protocols remain separate.",
                        "Heldout images are disjoint from conditioning views but participate in the provided global COLMAP model; they are not geometry-independent.",
                    ],
                },
                "prediction_mesh": relpath(prediction, output) if prediction.is_file() else None,
                "blocker": missing or None,
            }
        )
        rows.append(_write_manifest(output, manifest))
    return rows


def _eth3d_source_scene(scene: str, source_root: Path) -> Path:
    existing = ROOT / "data" / "eth3d" / "raw" / scene
    if existing.is_dir():
        return existing
    return source_root / "eth3d" / "extracted" / scene


def prepare_eth3d(plan: dict[str, Any], output: Path, source_root: Path) -> list[dict[str, Any]]:
    dataset = plan["datasets"]["eth3d-indoor-training"]
    extracted_root = source_root / "eth3d" / "extracted"
    extracted_root.mkdir(parents=True, exist_ok=True)
    for scene in dataset["unit_ids"]:
        source_scene = _eth3d_source_scene(scene, source_root)
        if source_scene.is_dir():
            continue
        for suffix in ("dslr_undistorted", "dslr_scan_eval"):
            archive = source_root / "eth3d" / f"{scene}_{suffix}.7z"
            if not archive.is_file():
                raise CalibrationBuildError(f"Missing ETH3D archive {archive}")
            safe_7z_extract(archive, extracted_root)

    prediction_map = {
        "delivery_area": ROOT / "outputs" / "eth3d" / "delivery_area" / "model_scale" / "mesh.ply",
        "pipes": ROOT / "outputs" / "eth3d" / "pipes" / "high_precision" / "mesh.ply",
    }
    rows = []
    for scene in dataset["unit_ids"]:
        source_scene = _eth3d_source_scene(scene, source_root)
        images_root = source_scene / "images"
        colmap = source_scene / "dslr_calibration_undistorted"
        scan_dir = source_scene / "dslr_scan_eval"
        images_txt = colmap / "images.txt"
        names = _parse_colmap_image_names(images_txt)
        selected = uniform_indices(len(names), min(16, len(names)))
        input_positions = selected[::2][:8]
        heldout_positions = [value for value in selected if value not in input_positions][:8]
        scans = sorted(scan_dir.glob("scan*.ply"))
        alignment = scan_dir / "scan_alignment.mlp"
        manifest = _base_manifest(
            unit_id=f"eth3d-{scene}",
            dataset="eth3d-indoor-training",
            track=dataset["track"],
            gt_tier=dataset["gt_tier"],
            capture_kind=dataset["capture_kind"],
            license_status=dataset["license_status"],
        )
        manifest_path = _manifest_path(output, manifest["unit_id"])
        archives = [
            source_root / "eth3d" / f"{scene}_{suffix}.7z"
            for suffix in ("dslr_undistorted", "dslr_scan_eval")
            if (source_root / "eth3d" / f"{scene}_{suffix}.7z").is_file()
        ]
        local_source_records = [
            colmap / "cameras.txt",
            images_txt,
            colmap / "points3D.txt",
            alignment,
            *scans,
        ]
        manifest.update(
            {
                "source": {
                    "scene_id": scene,
                    "records": _write_source_records(archives or local_source_records),
                },
                "input": {
                    "conditioning_views": [names[index] for index in input_positions],
                    "heldout_views": [names[index] for index in heldout_positions],
                    "images_root": relpath(images_root, manifest_path.parent),
                    "camera_model": relpath(colmap / "cameras.txt", manifest_path.parent),
                    "images_model": relpath(images_txt, manifest_path.parent),
                    "points_model": relpath(colmap / "points3D.txt", manifest_path.parent),
                    "pose_source": "provided-eth3d-dslr-calibration",
                },
                "reference": {
                    "kind": "pointcloud",
                    "paths": [relpath(path, manifest_path.parent) for path in scans],
                    "alignment_mlp": relpath(alignment, manifest_path.parent),
                    "roi": "full-public-scan-eval-cloud",
                    "source_type": "independent-laser-scan-evaluation-cloud",
                },
                "evaluation": {
                    "alignment": "official-meshlab-scan-alignment",
                    "limitations": ["Common suite point-cloud scores are not the official ETH3D depth-map evaluator."],
                },
                "prediction_mesh": relpath(prediction_map[scene], output) if scene in prediction_map and prediction_map[scene].is_file() else None,
            }
        )
        rows.append(_write_manifest(output, manifest))
    return rows


def _sequence_number(value: str) -> int:
    match = re.search(r"\d+", value)
    if not match:
        raise CalibrationBuildError(f"Invalid 7-Scenes sequence identifier {value!r}")
    return int(match.group())


def _nested_zip(outer: ZipFile, scene: str, sequence: int) -> ZipFile:
    name = f"{scene}/seq-{sequence:02d}.zip"
    try:
        payload = outer.read(name)
    except KeyError as exc:
        raise CalibrationBuildError(f"Missing nested sequence {name}") from exc
    nested = ZipFile(io.BytesIO(payload))
    safe_zip_names(nested)
    return nested


def _sequence_frame_count(nested: ZipFile) -> int:
    indices = {
        int(match.group(1))
        for name in nested.namelist()
        if (match := re.search(r"frame-(\d+)\.color\.png$", name))
    }
    if not indices:
        raise CalibrationBuildError("7-Scenes nested sequence contains no RGB frames")
    return max(indices) + 1


def _seven_frame_bytes(nested: ZipFile, sequence: int, frame: int, kind: str) -> bytes:
    suffix = {"rgb": "color.png", "depth": "depth.png", "pose": "pose.txt"}[kind]
    return nested.read(f"seq-{sequence:02d}/frame-{frame:06d}.{suffix}")


def _parse_pose(payload: bytes) -> np.ndarray:
    values = np.fromstring(payload.decode("ascii"), sep=" ", dtype=np.float64)
    if values.size != 16:
        raise CalibrationBuildError(f"Expected a 4x4 pose, found {values.size} values")
    matrix = values.reshape(4, 4)
    if not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5):
        raise CalibrationBuildError("Invalid camera-to-world pose")
    return matrix


def _backproject_depth(depth: np.ndarray, pose_c2w: np.ndarray, intrinsics: dict[str, float], pixel_stride: int) -> np.ndarray:
    rows = np.arange(0, depth.shape[0], pixel_stride)
    cols = np.arange(0, depth.shape[1], pixel_stride)
    uu, vv = np.meshgrid(cols, rows)
    z = depth[vv, uu].astype(np.float64) / 1000.0
    valid = (z > 0.05) & (z < 20.0) & (depth[vv, uu] != 65535)
    z = z[valid]
    x = (uu[valid] - intrinsics["cx"]) * z / intrinsics["fx"]
    y = (vv[valid] - intrinsics["cy"]) * z / intrinsics["fy"]
    camera = np.column_stack((x, y, z))
    return (camera @ pose_c2w[:3, :3].T + pose_c2w[:3, 3]).astype(np.float32)


def _voxel_downsample(points: np.ndarray, voxel_m: float, max_points: int, seed: int) -> np.ndarray:
    finite = points[np.isfinite(points).all(axis=1)]
    keys = np.floor(finite / voxel_m).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    result = finite[np.sort(indices)]
    if len(result) > max_points:
        rng = np.random.default_rng(seed)
        result = result[np.sort(rng.choice(len(result), max_points, replace=False))]
    return result


def _extract_seven_views(
    nested: ZipFile,
    sequence: int,
    indices: list[int],
    destination: Path,
    role: str,
) -> list[dict[str, Any]]:
    records = []
    for order, frame in enumerate(indices):
        stem = f"{role}_{order:03d}"
        rgb = destination / "rgb" / f"{stem}.png"
        depth = destination / "depth" / f"{stem}.png"
        rgb.parent.mkdir(parents=True, exist_ok=True)
        depth.parent.mkdir(parents=True, exist_ok=True)
        rgb.write_bytes(_seven_frame_bytes(nested, sequence, frame, "rgb"))
        depth.write_bytes(_seven_frame_bytes(nested, sequence, frame, "depth"))
        pose = _parse_pose(_seven_frame_bytes(nested, sequence, frame, "pose"))
        records.append(
            {
                "role": role,
                "order": order,
                "sequence": sequence,
                "frame": frame,
                "rgb": relpath(rgb, destination),
                "depth": relpath(depth, destination),
                "camera_to_world": pose.tolist(),
                "intrinsics": SEVEN_INTRINSICS,
            }
        )
    return records


def prepare_seven_scenes(plan: dict[str, Any], output: Path, source_root: Path, *, force: bool) -> list[dict[str, Any]]:
    dataset = plan["datasets"]["seven-scenes"]
    rows = []
    tsdf_archive = source_root / "seven-scenes" / "tsdf.zip"
    for scene in dataset["unit_ids"]:
        archive_path = source_root / "seven-scenes" / f"{scene}.zip"
        unit_id = f"seven-scenes-{scene}"
        unit_dir = output / "units" / unit_id
        reference_path = unit_dir / "reference" / "fusion_reference.ply"
        camera_path = unit_dir / "cameras.json"
        if force and unit_dir.exists():
            shutil.rmtree(unit_dir)
        with ZipFile(archive_path) as outer:
            safe_zip_names(outer)
            train = outer.read(f"{scene}/TrainSplit.txt").decode("ascii").splitlines()
            test = outer.read(f"{scene}/TestSplit.txt").decode("ascii").splitlines()
            train_sequences = [_sequence_number(item) for item in train if item.strip()]
            test_sequences = [_sequence_number(item) for item in test if item.strip()]
            with _nested_zip(outer, scene, train_sequences[0]) as nested:
                count = _sequence_frame_count(nested)
                input_indices = uniform_indices(count, 8)
                input_records = _extract_seven_views(nested, train_sequences[0], input_indices, unit_dir, "conditioning")
            with _nested_zip(outer, scene, test_sequences[0]) as nested:
                count = _sequence_frame_count(nested)
                heldout_indices = uniform_indices(count, 8)
                heldout_records = _extract_seven_views(nested, test_sequences[0], heldout_indices, unit_dir, "heldout")

            sampled_frames = 0
            if not reference_path.is_file():
                batches = []
                for sequence in sorted(set(train_sequences + test_sequences)):
                    with _nested_zip(outer, scene, sequence) as nested:
                        count = _sequence_frame_count(nested)
                        for frame in range(0, count, 50):
                            depth = np.asarray(Image.open(io.BytesIO(_seven_frame_bytes(nested, sequence, frame, "depth"))))
                            pose = _parse_pose(_seven_frame_bytes(nested, sequence, frame, "pose"))
                            batches.append(_backproject_depth(depth, pose, SEVEN_INTRINSICS, pixel_stride=8))
                            sampled_frames += 1
                points = _voxel_downsample(np.concatenate(batches, axis=0), 0.01, 2_000_000, 42)
                reference_path.parent.mkdir(parents=True, exist_ok=True)
                trimesh.PointCloud(points).export(reference_path)
            else:
                for sequence in sorted(set(train_sequences + test_sequences)):
                    with _nested_zip(outer, scene, sequence) as nested:
                        count = _sequence_frame_count(nested)
                        sampled_frames += len(range(0, count, 50))
        write_json(
            camera_path,
            {
                "schema": "genrecon.gt-camera-split",
                "schema_version": 1,
                "pose_convention": "camera-to-world",
                "conditioning": input_records,
                "heldout": heldout_records,
            },
        )
        manifest = _base_manifest(
            unit_id=unit_id,
            dataset="seven-scenes",
            track=dataset["track"],
            gt_tier=dataset["gt_tier"],
            capture_kind=dataset["capture_kind"],
            license_status=dataset["license_status"],
        )
        manifest_path = _manifest_path(output, unit_id)
        manifest.update(
            {
                "source": {"scene_id": scene, "records": _write_source_records([archive_path, tsdf_archive])},
                "input": {
                    "conditioning_views": [record["rgb"] for record in input_records],
                    "heldout_views": [record["rgb"] for record in heldout_records],
                    "conditioning_depths": [record["depth"] for record in input_records],
                    "heldout_depths": [record["depth"] for record in heldout_records],
                    "cameras": relpath(camera_path, manifest_path.parent),
                    "pose_source": "official-kinectfusion-camera-to-world",
                    "intrinsics_status": "official-default-depth-intrinsics-rgb-uncalibrated",
                },
                "reference": {
                    "kind": "pointcloud",
                    "paths": [relpath(reference_path, manifest_path.parent)],
                    "roi": "all-official-train-and-test-sequences-sampled",
                    "source_type": "clean-depth-fusion-reference",
                    "generation": {"frame_stride": 50, "pixel_stride": 8, "voxel_m": 0.01, "max_points": 2_000_000, "sampled_frames": sampled_frames},
                },
                "evaluation": {
                    "alignment": "official-kinectfusion-world-frame",
                    "limitations": ["Reference depth and poses are KinectFusion-derived rather than an independent laser scan.", "Raw RGB and depth cameras are not calibrated; official default depth intrinsics are used."],
                },
                "prediction_mesh": None,
            }
        )
        rows.append(_write_manifest(output, manifest))
    return rows


def _read_redwood_poses(path: Path) -> list[np.ndarray]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) % 5:
        raise CalibrationBuildError(f"Redwood trajectory must contain 5 lines per frame: {path}")
    poses = []
    for start in range(0, len(lines), 5):
        header = [int(value) for value in lines[start].split()]
        matrix = np.asarray([[float(value) for value in row.split()] for row in lines[start + 1 : start + 5]])
        if len(header) != 3 or matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise CalibrationBuildError(f"Invalid Redwood pose record at line {start + 1}")
        poses.append(matrix)
    return poses


def _extract_flat_zip_frames(archive_path: Path, indices: list[int], destination: Path, suffix: str) -> list[Path]:
    paths = []
    with ZipFile(archive_path) as archive:
        names = safe_zip_names(archive)
        by_index = {int(Path(name).stem): name for name in names if Path(name).suffix.lower() == suffix}
        for order, index in enumerate(indices):
            if index not in by_index:
                raise CalibrationBuildError(f"Frame {index} is missing from {archive_path}")
            output = destination / f"{order:03d}{suffix}"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(archive.read(by_index[index]))
            paths.append(output)
    return paths


def prepare_redwood(plan: dict[str, Any], output: Path, source_root: Path, *, force: bool) -> list[dict[str, Any]]:
    dataset = plan["datasets"]["redwood-reconstruction"]
    rows = []
    for scene in dataset["unit_ids"]:
        unit_id = f"redwood-{scene}"
        unit_dir = output / "units" / unit_id
        if force and unit_dir.exists():
            shutil.rmtree(unit_dir)
        primary = int(dataset["primary_trajectory"])
        secondary = int(dataset["secondary_trajectory"])
        primary_pose_path = source_root / "redwood" / f"{scene}{primary}-traj.txt"
        secondary_pose_path = source_root / "redwood" / f"{scene}{secondary}-traj.txt"
        primary_poses = _read_redwood_poses(primary_pose_path)
        secondary_poses = _read_redwood_poses(secondary_pose_path)
        input_indices = uniform_indices(len(primary_poses), 8)
        heldout_indices = uniform_indices(len(secondary_poses), 8)
        input_rgb = _extract_flat_zip_frames(source_root / "redwood" / f"{scene}{primary}-color.zip", input_indices, unit_dir / "rgb" / "conditioning", ".jpg")
        input_depth = _extract_flat_zip_frames(source_root / "redwood" / f"{scene}{primary}-depth-clean.zip", input_indices, unit_dir / "depth" / "conditioning", ".png")
        heldout_rgb = _extract_flat_zip_frames(source_root / "redwood" / f"{scene}{secondary}-color.zip", heldout_indices, unit_dir / "rgb" / "heldout", ".jpg")
        heldout_depth = _extract_flat_zip_frames(source_root / "redwood" / f"{scene}{secondary}-depth-clean.zip", heldout_indices, unit_dir / "depth" / "heldout", ".png")
        reference_path = unit_dir / "reference" / f"{scene}.ply"
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(source_root / "redwood" / f"{scene}.ply.zip") as archive:
            safe_zip_names(archive)
            reference_path.write_bytes(archive.read(f"{scene}.ply"))
        camera_path = unit_dir / "cameras.json"
        conditioning = [
            {"order": order, "source_frame": index, "rgb": relpath(input_rgb[order], unit_dir), "depth": relpath(input_depth[order], unit_dir), "camera_to_world": primary_poses[index].tolist(), "intrinsics": REDWOOD_INTRINSICS}
            for order, index in enumerate(input_indices)
        ]
        heldout = [
            {"order": order, "source_frame": index, "rgb": relpath(heldout_rgb[order], unit_dir), "depth": relpath(heldout_depth[order], unit_dir), "camera_to_world": secondary_poses[index].tolist(), "intrinsics": REDWOOD_INTRINSICS}
            for order, index in enumerate(heldout_indices)
        ]
        write_json(camera_path, {"schema": "genrecon.gt-camera-split", "schema_version": 1, "pose_convention": "camera-to-world", "conditioning": conditioning, "heldout": heldout})
        archives = [source_root / "redwood" / f"{scene}.ply.zip"] + [source_root / "redwood" / f"{scene}{trajectory}-{kind}" for trajectory in (primary, secondary) for kind in ("color.zip", "depth-clean.zip", "traj.txt")]
        manifest = _base_manifest(unit_id=unit_id, dataset="redwood-reconstruction", track=dataset["track"], gt_tier=dataset["gt_tier"], capture_kind=dataset["capture_kind"], license_status=dataset["license_status"])
        manifest_path = _manifest_path(output, unit_id)
        manifest.update(
            {
                "source": {"scene_id": scene, "records": _write_source_records(archives)},
                "input": {
                    "conditioning_views": [item["rgb"] for item in conditioning],
                    "heldout_views": [item["rgb"] for item in heldout],
                    "conditioning_depths": [item["depth"] for item in conditioning],
                    "heldout_depths": [item["depth"] for item in heldout],
                    "cameras": relpath(camera_path, manifest_path.parent),
                    "pose_source": "official-synthetic-trajectory",
                },
                "reference": {"kind": "pointcloud", "paths": [relpath(reference_path, manifest_path.parent)], "roi": "full-exact-synthetic-surface", "source_type": "synthetic-exact-pointcloud"},
                "evaluation": {"alignment": "provided-synthetic-world-frame", "limitations": ["Synthetic imagery and exact geometry do not measure real-sensor domain robustness."]},
                "prediction_mesh": None,
            }
        )
        rows.append(_write_manifest(output, manifest))
    return rows


def _parse_dtu_camera(payload: bytes) -> dict[str, Any]:
    lines = [line.strip() for line in payload.decode("ascii").splitlines() if line.strip()]
    try:
        extrinsic_index = lines.index("extrinsic")
        intrinsic_index = lines.index("intrinsic")
    except ValueError as exc:
        raise CalibrationBuildError("Invalid DTU camera file") from exc
    w2c = np.asarray([[float(value) for value in row.split()] for row in lines[extrinsic_index + 1 : extrinsic_index + 5]], dtype=np.float64)
    intrinsics = np.asarray([[float(value) for value in row.split()] for row in lines[intrinsic_index + 1 : intrinsic_index + 4]], dtype=np.float64)
    if w2c.shape != (4, 4) or intrinsics.shape != (3, 3):
        raise CalibrationBuildError("Invalid DTU camera matrix shape")
    if not np.isfinite(w2c).all() or not np.isfinite(intrinsics).all():
        raise CalibrationBuildError("DTU camera contains non-finite values")
    if not np.allclose(w2c[3], [0, 0, 0, 1], atol=1e-8):
        raise CalibrationBuildError("Invalid DTU homogeneous extrinsic row")
    if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
        raise CalibrationBuildError("DTU camera has a non-positive focal length")
    w2c[:3, 3] *= 0.001
    return {"world_to_camera": w2c.tolist(), "intrinsics": intrinsics.tolist(), "source_units": "millimeters", "output_units": "meters"}


def _scaled_pointcloud_from_zip(archive: ZipFile, name: str, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(name) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)
    loaded = trimesh.load(str(destination), process=False)
    if not isinstance(loaded, (trimesh.PointCloud, trimesh.Trimesh)):
        raise CalibrationBuildError(f"DTU reference is not point-like: {name}")
    points = np.asarray(loaded.vertices, dtype=np.float64) * 0.001
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise CalibrationBuildError(f"DTU reference has invalid vertices: {name}")
    if not np.isfinite(points).all():
        raise CalibrationBuildError(f"DTU reference has non-finite vertices: {name}")
    colors = getattr(loaded.visual, "vertex_colors", None) if isinstance(loaded, trimesh.Trimesh) else getattr(loaded, "colors", None)
    cloud = trimesh.PointCloud(points, colors=colors if colors is not None and len(colors) == len(points) else None)
    cloud.export(destination)
    return {"points": len(points), "bounds_m": np.stack((points.min(axis=0), points.max(axis=0))).tolist()}


def prepare_dtu(plan: dict[str, Any], output: Path, source_root: Path, *, force: bool) -> list[dict[str, Any]]:
    dataset = plan["datasets"]["dtu-mvs"]
    input_archive_path = source_root / "dtu" / "dtu-mvsnet-test.zip"
    reference_archive_path = source_root / "dtu" / "Points.zip"
    rows = []
    with ZipFile(input_archive_path) as inputs, ZipFile(reference_archive_path) as references:
        safe_zip_names(inputs)
        safe_zip_names(references)
        for scan in dataset["unit_ids"]:
            unit_id = f"dtu-{scan}"
            unit_dir = output / "units" / unit_id
            if force and unit_dir.exists():
                shutil.rmtree(unit_dir)
            all_indices = list(range(49))
            selected = uniform_indices(len(all_indices), 16)
            input_indices = selected[::2][:8]
            heldout_indices = [index for index in selected if index not in input_indices][:8]
            camera_records = {"conditioning": [], "heldout": []}
            for role, indices in (("conditioning", input_indices), ("heldout", heldout_indices)):
                for order, index in enumerate(indices):
                    image_name = f"dtu/{scan}/images/{index:08d}.jpg"
                    camera_name = f"dtu/{scan}/cams/{index:08d}_cam.txt"
                    image_path = unit_dir / "rgb" / role / f"{order:03d}.jpg"
                    camera_file = unit_dir / "camera_files" / role / f"{order:03d}.txt"
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    camera_file.parent.mkdir(parents=True, exist_ok=True)
                    image_path.write_bytes(inputs.read(image_name))
                    payload = inputs.read(camera_name)
                    camera_file.write_bytes(payload)
                    camera_records[role].append({"order": order, "source_view": index, "rgb": relpath(image_path, unit_dir), "camera_file": relpath(camera_file, unit_dir), **_parse_dtu_camera(payload)})
            reference_path = unit_dir / "reference" / f"{scan}.ply"
            number = int(scan.removeprefix("scan"))
            reference_stats = _scaled_pointcloud_from_zip(references, f"Points/stl/stl{number:03d}_total.ply", reference_path)
            camera_path = unit_dir / "cameras.json"
            write_json(camera_path, {"schema": "genrecon.gt-camera-split", "schema_version": 1, "pose_convention": "world-to-camera", **camera_records})
            manifest = _base_manifest(unit_id=unit_id, dataset="dtu-mvs", track=dataset["track"], gt_tier=dataset["gt_tier"], capture_kind=dataset["capture_kind"], license_status=dataset["license_status"])
            manifest_path = _manifest_path(output, unit_id)
            manifest.update(
                {
                    "source": {"scan_id": scan, "input_mirror": dataset["input_mirror"], "records": _write_source_records([input_archive_path, reference_archive_path])},
                    "input": {"conditioning_views": [item["rgb"] for item in camera_records["conditioning"]], "heldout_views": [item["rgb"] for item in camera_records["heldout"]], "cameras": relpath(camera_path, manifest_path.parent), "pose_source": "preprocessed-official-DTU-camera-matrices", "scale_to_meters": 0.001},
                    "reference": {"kind": "pointcloud", "paths": [relpath(reference_path, manifest_path.parent)], "roi": "full-structured-light-reference", "source_type": "structured-light-pointcloud", "stats": reference_stats},
                    "evaluation": {"alignment": "provided-DTU-calibrated-frame-scaled-to-meters", "limitations": ["Input images are the standard MVSNet-preprocessed DTU testing archive; GT points are from the official DTU Points.zip.", "Common suite metrics do not replace the official DTU observability-mask evaluator."]},
                    "prediction_mesh": None,
                }
            )
            rows.append(_write_manifest(output, manifest))
    return rows


def prepare_tanks_and_temples(plan: dict[str, Any], output: Path, source_root: Path) -> list[dict[str, Any]]:
    dataset = plan["datasets"]["tanks-and-temples-training"]
    unit_id = "tanks-and-temples-meetingroom"
    source = source_root / "tanks-and-temples"
    unit_dir = output / "units" / unit_id
    video = source / "Meetingroom.mp4"
    images = source / "Meetingroom.zip"
    camera_log = source / "Meetingroom_COLMAP.log"
    crop_json = source / "Meetingroom_crop.json"
    scans_archive = source / "Meetingroom_individual_scans.zip"
    alignment = source / "Meetingroom_alignment.txt"
    input_required = [video, images, camera_log, crop_json]
    input_valid = (
        all(path.is_file() for path in input_required)
        and file_prefix(video, 8)[4:8] == b"ftyp"
        and file_prefix(images, 4) == b"PK\x03\x04"
    )
    scans_valid = scans_archive.is_file() and file_prefix(scans_archive, 4) == b"PK\x03\x04"
    alignment_valid = alignment.is_file() and alignment.stat().st_size > 0
    camera_records: list[dict[str, Any]] = []
    reference_paths: list[Path] = []
    if input_valid:
        poses = _read_redwood_poses(camera_log)
        with ZipFile(images) as archive:
            names = safe_zip_names(archive)
            image_names = sorted(name for name in names if Path(name).suffix.lower() in {".jpg", ".jpeg", ".png"})
            if len(image_names) != len(poses):
                raise CalibrationBuildError(
                    f"T&T image/pose count mismatch: {len(image_names)} != {len(poses)}"
                )
            selected = uniform_indices(len(image_names), 16)
            input_indices = selected[::2][:8]
            heldout_indices = [index for index in selected if index not in input_indices][:8]
            for role, indices in (("conditioning", input_indices), ("heldout", heldout_indices)):
                for order, index in enumerate(indices):
                    path = unit_dir / "rgb" / role / f"{order:03d}.jpg"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    payload = archive.read(image_names[index])
                    path.write_bytes(payload)
                    with Image.open(io.BytesIO(payload)) as image:
                        width, height = image.size
                    camera_records.append(
                        {
                            "role": role,
                            "order": order,
                            "source_image": image_names[index],
                            "rgb": relpath(path, unit_dir),
                            "camera_to_world": poses[index].tolist(),
                            "image_size": [width, height],
                            "intrinsics": None,
                        }
                    )
        write_json(
            unit_dir / "cameras.json",
            {
                "schema": "genrecon.gt-camera-split",
                "schema_version": 1,
                "pose_convention": "camera-to-world",
                "intrinsics_status": "not-published-in-log-requires-colmap-recovery",
                "conditioning": [item for item in camera_records if item["role"] == "conditioning"],
                "heldout": [item for item in camera_records if item["role"] == "heldout"],
            },
        )
    if scans_valid:
        reference_dir = unit_dir / "reference" / "individual_scans"
        with ZipFile(scans_archive) as archive:
            names = safe_zip_names(archive)
            for name in names:
                if Path(name).suffix.lower() != ".ply":
                    continue
                path = reference_dir / Path(name).name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.read(name))
                reference_paths.append(path)
    if input_valid and scans_valid and alignment_valid:
        status = "blocked-adapter"
        blocker = (
            "Official sources are complete, but camera intrinsics recovery and the "
            "COLMAP-to-GT alignment adapter must run before evaluation."
        )
    elif input_valid:
        status = "blocked-reference-download"
        blocker = "Official Meetingroom laser scans/alignment are quota-blocked on Google Drive."
    else:
        status = "blocked-download"
        blocker = "Official source video/image/camera assets are incomplete."
    manifest = _base_manifest(unit_id=unit_id, dataset="tanks-and-temples-training", track=dataset["track"], gt_tier=dataset["gt_tier"], capture_kind=dataset["capture_kind"], license_status=dataset["license_status"], status=status)
    manifest_path = _manifest_path(output, unit_id)
    manifest.update(
        {
            "source": {"scene_id": "Meetingroom", "records": _write_source_records([*input_required, *([scans_archive] if scans_valid else []), *([alignment] if alignment_valid else [])]) if input_valid else []},
            "input": {
                "conditioning_views": [item["rgb"] for item in camera_records if item["role"] == "conditioning"],
                "heldout_views": [item["rgb"] for item in camera_records if item["role"] == "heldout"],
                "source_video": str(video.resolve()),
                "image_archive": str(images.resolve()),
                "cameras": relpath(unit_dir / "cameras.json", manifest_path.parent) if camera_records else None,
                "pose_source": "official-training-COLMAP-camera-log" if camera_records else "blocked",
            },
            "reference": {
                "kind": "pointcloud",
                "paths": [relpath(path, manifest_path.parent) for path in reference_paths],
                "alignment_file": relpath(alignment, manifest_path.parent) if alignment_valid else None,
                "official_crop_json": relpath(crop_json, manifest_path.parent) if crop_json.is_file() else None,
                "roi": "official-training-crop",
                "source_type": "prealigned-individual-laser-scans",
            },
            "evaluation": {
                "alignment": "official-training-alignment" if alignment_valid else "blocked",
                "limitations": [
                    "Official camera log omits intrinsics; they must be recovered from COLMAP before GenRecon input adaptation.",
                    "The source remains outside the common metric table until official GT and alignment downloads validate.",
                ],
            },
            "prediction_mesh": None,
            "blocker": blocker,
        }
    )
    return [_write_manifest(output, manifest)]


def prepare_omni_blockers(plan: dict[str, Any], output: Path) -> list[dict[str, Any]]:
    dataset = plan["datasets"]["omniobject3d"]
    rows = []
    for object_id in dataset["unit_ids"]:
        manifest = _base_manifest(unit_id=f"omniobject3d-{object_id}", dataset="omniobject3d", track=dataset["track"], gt_tier=dataset["gt_tier"], capture_kind=dataset["capture_kind"], license_status=dataset["license_status"], status="blocked-auth")
        manifest.update(
            {
                "source": {"object_id": object_id, "url": dataset["source_url"], "records": []},
                "input": {"conditioning_views": [], "heldout_views": [], "pose_source": "blocked"},
                "reference": {"kind": "mesh", "paths": [], "roi": "full-object", "source_type": "professional-real-object-scan"},
                "evaluation": {"alignment": "canonical-object-frame-pending", "limitations": ["Official source requires an approved OpenXLab account and AK/SK credentials."]},
                "prediction_mesh": None,
                "blocker": "OpenXLab login and AK/SK are unavailable; access was not bypassed.",
            }
        )
        rows.append(_write_manifest(output, manifest))
    return rows


def build_registry(plan_path: Path, output: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    document = {
        "schema": REGISTRY_SCHEMA,
        "schema_version": 1,
        "plan": relpath(plan_path, output),
        "plan_sha256": sha256_file(plan_path),
        "summary": {
            "unit_count": len(rows),
            "statuses": dict(sorted(Counter(row["status"] for row in rows).items())),
            "datasets": dict(sorted(Counter(row["dataset"] for row in rows).items())),
            "tracks": dict(sorted(Counter(row["track"] for row in rows).items())),
            "gt_tiers": dict(sorted(Counter(row["gt_tier"] for row in rows).items())),
            "predictions_available": sum(bool(row.get("prediction_mesh")) for row in rows),
        },
        "units": sorted(rows, key=lambda row: row["unit_id"]),
    }
    write_json(output / "registry.json", document)
    return document


def write_summary(output: Path, registry: dict[str, Any]) -> None:
    fields = ["unit_id", "dataset", "track", "gt_tier", "status", "prediction_mesh", "blocker", "manifest"]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in registry["units"]:
            writer.writerow({field: row.get(field) for field in fields})


def _strict_json_files(output: Path) -> tuple[int, list[str]]:
    count = 0
    errors = []
    for path in sorted(output.rglob("*.json")):
        if path.name in {"validation.json", "source_validation.json"}:
            continue
        try:
            load_json(path)
            count += 1
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    return count, errors


def validate(output: Path, plan: dict[str, Any]) -> dict[str, Any]:
    registry = load_json(output / "registry.json")
    expected = sum(len(dataset["unit_ids"]) for dataset in plan["datasets"].values())
    errors = []
    if plan.get("schema") != PLAN_SCHEMA:
        errors.append("plan schema mismatch")
    if plan.get("expected_unit_count", expected) != expected:
        errors.append(
            f"plan expected_unit_count {plan.get('expected_unit_count')} != dataset total {expected}"
        )
    if registry.get("schema") != REGISTRY_SCHEMA:
        errors.append("registry schema mismatch")
    registry_plan = (output / registry.get("plan", "")).resolve()
    if not registry_plan.is_file() or sha256_file(registry_plan) != registry.get("plan_sha256"):
        errors.append("registry plan path/hash mismatch")
    if len(registry.get("units", [])) != expected:
        errors.append(f"registry unit count {len(registry.get('units', []))} != {expected}")
    units = registry.get("units", [])
    expected_summary = {
        "unit_count": len(units),
        "statuses": dict(sorted(Counter(row["status"] for row in units).items())),
        "datasets": dict(sorted(Counter(row["dataset"] for row in units).items())),
        "tracks": dict(sorted(Counter(row["track"] for row in units).items())),
        "gt_tiers": dict(sorted(Counter(row["gt_tier"] for row in units).items())),
        "predictions_available": sum(bool(row.get("prediction_mesh")) for row in units),
    }
    if registry.get("summary") != expected_summary:
        errors.append("registry summary does not match unit rows")
    expected_manifests = {row["manifest"] for row in units}
    actual_manifests = {
        relpath(path, output) for path in (output / "units").glob("*/manifest.json")
    }
    if actual_manifests != expected_manifests:
        errors.append("unit manifest set does not match registry rows")
    seen = set()
    reference_files = 0
    reference_vertices = 0
    reference_faces = 0
    reference_bytes = 0
    source_records = 0
    source_hash_bytes = 0
    conditioning_views = 0
    heldout_views = 0
    decoded_images = 0
    decoded_depths = 0
    prediction_files = 0
    prediction_bytes = 0
    for row in registry.get("units", []):
        if row["unit_id"] in seen:
            errors.append(f"duplicate unit {row['unit_id']}")
        seen.add(row["unit_id"])
        manifest_path = output / row["manifest"]
        if not manifest_path.is_file():
            errors.append(f"missing manifest {manifest_path}")
            continue
        manifest = load_json(manifest_path)
        if manifest.get("schema") != SCHEMA:
            errors.append(f"manifest schema mismatch for {row['unit_id']}")
        for key in ("unit_id", "dataset", "track", "gt_tier", "physical_scene_group"):
            if manifest.get(key) != row.get(key):
                errors.append(f"manifest/registry {key} mismatch for {row['unit_id']}")
        if manifest.get("status") != row["status"]:
            errors.append(f"status mismatch for {row['unit_id']}")
        if row.get("prediction_mesh"):
            prediction = (output / row["prediction_mesh"]).resolve()
            if not prediction.is_file() or prediction.stat().st_size == 0:
                errors.append(f"missing prediction mesh for {row['unit_id']}: {prediction}")
            else:
                prediction_files += 1
                prediction_bytes += prediction.stat().st_size
        if manifest["status"] == "prepared":
            conditioning_views += len(manifest["input"]["conditioning_views"])
            heldout_views += len(manifest["input"]["heldout_views"])
            if not manifest["input"]["conditioning_views"]:
                errors.append(f"prepared unit has no conditioning views: {row['unit_id']}")
            if set(manifest["input"]["conditioning_views"]) & set(
                manifest["input"]["heldout_views"]
            ):
                errors.append(f"conditioning/heldout overlap for {row['unit_id']}")
            input_root_value = manifest["input"].get("root") or manifest["input"].get("images_root")
            input_root = (
                (manifest_path.parent / input_root_value).resolve()
                if input_root_value
                else manifest_path.parent
            )
            for role in ("conditioning_views", "heldout_views"):
                for value in manifest["input"][role]:
                    path = Path(value)
                    if not path.is_absolute():
                        path = (input_root / path).resolve()
                    if not path.is_file() or path.stat().st_size == 0:
                        errors.append(f"missing {role} image for {row['unit_id']}: {path}")
                        continue
                    try:
                        with Image.open(path) as image:
                            width, height = image.size
                            image.verify()
                        if width <= 0 or height <= 0:
                            raise CalibrationBuildError("non-positive image dimensions")
                        decoded_images += 1
                    except Exception as exc:
                        errors.append(f"invalid {role} image for {row['unit_id']}: {path}: {exc}")
            for role in ("conditioning_depths", "heldout_depths"):
                for value in manifest["input"].get(role, []):
                    path = Path(value)
                    if not path.is_absolute():
                        path = (input_root / path).resolve()
                    try:
                        with Image.open(path) as image:
                            width, height = image.size
                            image.verify()
                        if width <= 0 or height <= 0:
                            raise CalibrationBuildError("non-positive depth dimensions")
                        decoded_depths += 1
                    except Exception as exc:
                        errors.append(f"invalid {role} image for {row['unit_id']}: {path}: {exc}")
            camera_value = manifest["input"].get("cameras")
            if camera_value:
                camera_path = (manifest_path.parent / camera_value).resolve()
                try:
                    camera_split = load_json(camera_path)
                    if len(camera_split.get("conditioning", [])) != len(
                        manifest["input"]["conditioning_views"]
                    ) or len(camera_split.get("heldout", [])) != len(
                        manifest["input"]["heldout_views"]
                    ):
                        raise CalibrationBuildError("camera split count does not match input views")
                except Exception as exc:
                    errors.append(f"invalid camera split for {row['unit_id']}: {camera_path}: {exc}")
            for value in manifest["reference"]["paths"]:
                path = (manifest_path.parent / value).resolve()
                if not path.is_file() or path.stat().st_size == 0:
                    errors.append(f"missing reference for {row['unit_id']}: {path}")
                    continue
                reference_files += 1
                reference_bytes += path.stat().st_size
                try:
                    loaded = trimesh.load(str(path), process=False)
                    if not isinstance(loaded, (trimesh.Trimesh, trimesh.PointCloud)):
                        raise CalibrationBuildError(
                            f"unsupported geometry payload {type(loaded).__name__}"
                        )
                    vertices = np.asarray(loaded.vertices)
                    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
                        raise CalibrationBuildError("empty or invalid reference vertices")
                    if not np.isfinite(vertices).all():
                        raise CalibrationBuildError("non-finite reference vertices")
                    reference_vertices += len(vertices)
                    if isinstance(loaded, trimesh.Trimesh):
                        faces = np.asarray(loaded.faces)
                        if manifest["reference"]["kind"] == "mesh" and not len(faces):
                            raise CalibrationBuildError("mesh reference has no faces")
                        reference_faces += len(faces)
                    elif manifest["reference"]["kind"] == "mesh":
                        raise CalibrationBuildError("mesh reference loaded as point cloud")
                    del loaded, vertices
                except Exception as exc:
                    errors.append(f"invalid reference for {row['unit_id']}: {path}: {exc}")
        elif manifest["status"] == "blocked-auth" and manifest["dataset"] != "omniobject3d":
            errors.append(f"unexpected auth blocker for {row['unit_id']}")
        for record in manifest.get("source", {}).get("records", []):
            path = Path(record["path"])
            if not path.is_file() or path.stat().st_size != record["size_bytes"]:
                errors.append(f"source record size mismatch for {row['unit_id']}: {path}")
                continue
            source_records += 1
            source_hash_bytes += path.stat().st_size
            if sha256_file(path) != record["sha256"]:
                errors.append(f"source record hash mismatch for {row['unit_id']}: {path}")
    strict_json, json_errors = _strict_json_files(output)
    errors.extend(json_errors)
    counts = {
        "units": len(registry.get("units", [])),
        "statuses": registry.get("summary", {}).get("statuses", {}),
        "reference_files": reference_files,
        "reference_vertices": reference_vertices,
        "reference_faces": reference_faces,
        "reference_bytes": reference_bytes,
        "source_records": source_records,
        "source_hash_bytes": source_hash_bytes,
        "conditioning_views": conditioning_views,
        "heldout_views": heldout_views,
        "decoded_images": decoded_images,
        "decoded_depths": decoded_depths,
        "prediction_files": prediction_files,
        "prediction_bytes": prediction_bytes,
        "strict_json_files": strict_json,
    }
    result = {
        "schema": "genrecon.gt-calibration-validation",
        "schema_version": 1,
        "result": "pass" if not errors else "fail",
        "counts": counts,
        "declared_blockers": [row for row in registry.get("units", []) if row["status"].startswith("blocked") or row["status"] == "source-prepared-unposed"],
        "errors": errors,
    }
    write_json(output / "validation.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "summarize", "validate", "all"))
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    plan_path = args.plan.resolve()
    output = args.output.resolve()
    source_root = output / "sources"
    plan = load_json(plan_path)
    if plan.get("schema") != PLAN_SCHEMA:
        raise SystemExit(f"Unexpected calibration plan schema in {plan_path}")
    selected = set(args.dataset or plan["datasets"])
    unknown = selected - set(plan["datasets"])
    if unknown:
        raise SystemExit(f"Unknown datasets: {sorted(unknown)}")
    output.mkdir(parents=True, exist_ok=True)
    if args.stage in {"prepare", "all"}:
        rows = []
        if "da3-scannetpp" in selected:
            rows.extend(prepare_scannetpp(plan, output))
        if "eth3d-indoor-training" in selected:
            rows.extend(prepare_eth3d(plan, output, source_root))
        if "seven-scenes" in selected:
            rows.extend(prepare_seven_scenes(plan, output, source_root, force=args.force))
        if "redwood-reconstruction" in selected:
            rows.extend(prepare_redwood(plan, output, source_root, force=args.force))
        if "dtu-mvs" in selected:
            rows.extend(prepare_dtu(plan, output, source_root, force=args.force))
        if "tanks-and-temples-training" in selected:
            rows.extend(prepare_tanks_and_temples(plan, output, source_root))
        if "omniobject3d" in selected:
            rows.extend(prepare_omni_blockers(plan, output))
        if selected != set(plan["datasets"]):
            existing = load_json(output / "registry.json")["units"] if (output / "registry.json").is_file() else []
            replaced = {row["dataset"] for row in rows}
            rows.extend(row for row in existing if row["dataset"] not in replaced)
        registry = build_registry(plan_path, output, rows)
        write_summary(output, registry)
    if args.stage == "summarize":
        registry = load_json(output / "registry.json")
        write_summary(output, registry)
    if args.stage in {"validate", "all"}:
        result = validate(output, plan)
        print(f"[gt-calibration-validation] {result['result']} {result['counts']}")
        if result["errors"]:
            for error in result["errors"][:20]:
                print(f"  - {error}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
