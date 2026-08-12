#!/usr/bin/env python3
"""Prepare and package GT representative GenRecon inference without GT leakage.

Only each unit's frozen conditioning RGB and conditioning camera metadata are
used to build VGGT pseudo geometry. Heldout RGB/depth and evaluation reference
geometry are explicitly excluded. GenRecon runs in a metric z-up work frame;
packaging maps its PLY and PBR GLB back to the declared calibration frame.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import struct
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.prepare_foundation_sfm import (  # noqa: E402
    VGGT_MODEL_SHA256,
    candidate_seed,
    cross_view_sparse_points,
    load_vggt_model,
    make_projection_plot,
    padded_shape,
    percentile_summary,
    resolve_vggt_checkpoint,
    run_genrecon_preflight,
    run_vggt_inference,
    save_predictions,
    sha256_file,
    write_binary_ply,
    write_colmap_text,
    write_json,
)

DEFAULT_CONFIG = ROOT / "configs" / "eval" / "gt_representative_inference_v1.json"
DEFAULT_UNITS_ROOT = ROOT / "data" / "gt-calibration-v1" / "units"
DEFAULT_INPUT_ROOT = ROOT / "data" / "gt-calibration-v1" / "genrecon-inputs-v1"
DEFAULT_PREDICTION_ROOT = ROOT / "outputs" / "gt-calibration-v1" / "representative-genrecon-v1"
DEFAULT_VGGT_ROOT = ROOT / "data" / "model-sources" / "vggt"
SCHEMA_VERSION = 1
GENRECON_INPUT_RELATIVE_PATHS = (
    "colmap_vggt/cameras.txt",
    "colmap_vggt/images.txt",
    "colmap_vggt/points3D.txt",
    *(f"rgb/{index:03d}.png" for index in range(8)),
)


class RepresentativeInferenceError(RuntimeError):
    pass


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            RepresentativeInferenceError(f"Non-finite JSON constant {value} in {path}")
        ),
    )


def conditioning_manifest_contract_sha256(document: dict[str, Any]) -> str:
    """Hash the frozen unit contract without prediction-registration outputs."""
    canonical = {
        key: value
        for key, value in document.items()
        if key not in {"prediction_mesh", "prediction_provenance"}
    }
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve(value: str, base: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _root_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _finite_matrix(value: Any, shape: tuple[int, int], label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise RepresentativeInferenceError(f"{label} must be a finite {shape} matrix")
    return matrix


def camera_to_world_from_record(
    record: dict[str, Any], *, return_diagnostic: bool = False
) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
    if "camera_to_world" in record:
        pose = _finite_matrix(record["camera_to_world"], (4, 4), "camera_to_world").copy()
    elif "world_to_camera" in record:
        pose = np.linalg.inv(
            _finite_matrix(record["world_to_camera"], (4, 4), "world_to_camera")
        )
    elif "qvec_wxyz" in record and "tvec_world_to_camera" in record:
        q = np.asarray(record["qvec_wxyz"], dtype=np.float64)
        q /= max(float(np.linalg.norm(q)), 1e-12)
        w, x, y, z = q
        rotation = np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        world_to_camera = np.eye(4, dtype=np.float64)
        world_to_camera[:3, :3] = rotation
        world_to_camera[:3, 3] = np.asarray(
            record["tvec_world_to_camera"], dtype=np.float64
        )
        pose = np.linalg.inv(world_to_camera)
    else:
        raise RepresentativeInferenceError("Conditioning camera has no supported pose")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise RepresentativeInferenceError("Camera pose has an invalid homogeneous row")
    raw_rotation = pose[:3, :3].copy()
    orthogonality_error = float(np.linalg.norm(raw_rotation.T @ raw_rotation - np.eye(3)))
    determinant = float(np.linalg.det(raw_rotation))
    if determinant <= 0.0 or orthogonality_error > 1e-3:
        raise RepresentativeInferenceError(
            "Camera rotation exceeds the finite-precision SO(3) normalization gate"
        )
    u, _, vt = np.linalg.svd(raw_rotation)
    normalized_rotation = u @ vt
    if np.linalg.det(normalized_rotation) < 0.0:
        u[:, -1] *= -1.0
        normalized_rotation = u @ vt
    pose[:3, :3] = normalized_rotation
    diagnostic = {
        "raw_orthogonality_error_frobenius": orthogonality_error,
        "raw_determinant": determinant,
        "nearest_so3_correction_frobenius": float(
            np.linalg.norm(normalized_rotation - raw_rotation)
        ),
    }
    return (pose, diagnostic) if return_diagnostic else pose


def intrinsics_from_record(
    record: dict[str, Any], image_size: tuple[int, int]
) -> np.ndarray:
    value = record.get("intrinsics")
    if isinstance(value, list):
        intrinsic = _finite_matrix(value, (3, 3), "intrinsics")
    elif isinstance(value, dict) and {"fx", "fy", "cx", "cy"} <= set(value):
        intrinsic = np.asarray(
            [
                [value["fx"], 0.0, value["cx"]],
                [0.0, value["fy"], value["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    elif isinstance(value, dict) and "params" in value:
        params = [float(item) for item in value["params"]]
        model = value.get("model")
        if model in {"PINHOLE", "OPENCV"} and len(params) >= 4:
            fx, fy, cx, cy = params[:4]
        elif model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"} and len(params) >= 3:
            fx = fy = params[0]
            cx, cy = params[1:3]
        else:
            raise RepresentativeInferenceError(f"Unsupported camera model {model!r}")
        intrinsic = np.asarray(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    else:
        raise RepresentativeInferenceError("Conditioning camera has no supported intrinsics")
    width, height = image_size
    if (
        intrinsic[0, 0] <= 0.0
        or intrinsic[1, 1] <= 0.0
        or not (0.0 <= intrinsic[0, 2] <= width)
        or not (0.0 <= intrinsic[1, 2] <= height)
    ):
        raise RepresentativeInferenceError("Conditioning intrinsics are implausible")
    return intrinsic


def load_conditioning_contract(unit_id: str, units_root: Path) -> dict[str, Any]:
    unit_dir = (units_root / unit_id).resolve()
    manifest_path = unit_dir / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("unit_id") != unit_id or manifest.get("status") != "prepared":
        raise RepresentativeInferenceError(f"Unit is not a prepared calibration package: {unit_id}")
    camera_path = _resolve(manifest["input"]["cameras"], unit_dir)
    camera_document = load_json(camera_path)
    records = camera_document.get("conditioning")
    if not isinstance(records, list) or len(records) != 8:
        raise RepresentativeInferenceError(f"{unit_id} must have exactly 8 conditioning cameras")
    declared = [_resolve(value, unit_dir) for value in manifest["input"]["conditioning_views"]]
    if len(declared) != 8 or len(set(declared)) != 8:
        raise RepresentativeInferenceError(f"{unit_id} conditioning RGB split is not 8 unique views")

    views = []
    for order, (record, declared_path) in enumerate(zip(records, declared)):
        if int(record.get("order", order)) != order:
            raise RepresentativeInferenceError(f"{unit_id} conditioning camera order is not frozen")
        record_path = _resolve(record["rgb"], unit_dir)
        if record_path != declared_path:
            raise RepresentativeInferenceError(
                f"{unit_id} camera/RGB manifest mismatch at conditioning view {order}"
            )
        if not record_path.is_file():
            raise RepresentativeInferenceError(f"Missing conditioning RGB: {record_path}")
        with Image.open(record_path) as opened:
            size = opened.size
            opened.verify()
        pose, pose_diagnostic = camera_to_world_from_record(
            record, return_diagnostic=True
        )
        views.append(
            {
                "order": order,
                "rgb": record_path,
                "image_size": size,
                "camera_to_world": pose,
                "pose_normalization": pose_diagnostic,
                "intrinsic": intrinsics_from_record(record, size),
                "source_label": str(
                    record.get(
                        "source_image",
                        record.get("source_view", record.get("source_frame", record_path.name)),
                    )
                ),
            }
        )

    return {
        "unit_id": unit_id,
        "unit_dir": unit_dir,
        "manifest_path": manifest_path,
        "manifest_conditioning_contract_sha256": conditioning_manifest_contract_sha256(
            manifest
        ),
        "camera_path": camera_path,
        "manifest": manifest,
        "views": views,
        "source_files_read": [manifest_path, camera_path, *declared],
        "declared_but_forbidden": {
            "heldout_rgb": list(manifest["input"].get("heldout_views", [])),
            "conditioning_depth": list(manifest["input"].get("conditioning_depths", [])),
            "heldout_depth": list(manifest["input"].get("heldout_depths", [])),
            "reference_geometry": list(manifest.get("reference", {}).get("paths", [])),
        },
    }


def umeyama_similarity(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Similarity inputs must have matching [N,3] shapes")
    if len(source) < 3 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Similarity alignment requires at least 3 finite correspondences")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    if np.linalg.matrix_rank(source_centered, tol=1e-8) < 2:
        raise ValueError("Similarity source camera trajectory is degenerate")
    u, singular, vt = np.linalg.svd(target_centered.T @ source_centered)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    denominator = float(np.sum(source_centered**2))
    scale = float(np.sum(singular * np.diag(correction)) / max(denominator, 1e-12))
    translation = target_mean - scale * rotation @ source_mean
    if scale <= 0.0 or not np.isfinite(scale):
        raise ValueError("Similarity alignment produced a non-positive scale")
    aligned = scale * (source @ rotation.T) + translation
    return {
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
        "aligned": aligned,
        "errors": np.linalg.norm(aligned - target, axis=1),
    }


def transform_world_similarity(
    points: np.ndarray,
    world_to_camera: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map world coordinates while preserving every pinhole projection."""
    points_new = scale * (np.asarray(points, dtype=np.float64) @ rotation.T) + translation
    extrinsic = np.asarray(world_to_camera, dtype=np.float64)
    output = np.empty_like(extrinsic)
    for index, old in enumerate(extrinsic):
        new_rotation = old[:3, :3] @ rotation.T
        output[index, :3, :3] = new_rotation
        output[index, :3, 3] = scale * old[:3, 3] - new_rotation @ translation
    return points_new, output


def trajectory_alignment(
    predicted_extrinsic: np.ndarray, official_c2w: np.ndarray
) -> dict[str, Any]:
    predicted_rotations = np.transpose(predicted_extrinsic[:, :3, :3], (0, 2, 1))
    predicted_centers = -np.einsum(
        "sji,sj->si", predicted_extrinsic[:, :3, :3], predicted_extrinsic[:, :3, 3]
    )
    target_centers = official_c2w[:, :3, 3]
    aligned = umeyama_similarity(predicted_centers, target_centers)
    rotation_errors = []
    for predicted, target in zip(predicted_rotations, official_c2w[:, :3, :3]):
        residual = (aligned["rotation"] @ predicted).T @ target
        cosine = np.clip((np.trace(residual) - 1.0) / 2.0, -1.0, 1.0)
        rotation_errors.append(float(np.degrees(np.arccos(cosine))))
    baseline = float(np.linalg.norm(np.ptp(target_centers, axis=0)))
    return {
        **aligned,
        "target_baseline_diagonal_m": baseline,
        "center_error_m": percentile_summary(aligned["errors"]),
        "center_error_normalized_by_baseline": percentile_summary(
            aligned["errors"] / max(baseline, 1e-12)
        ),
        "rotation_error_deg": percentile_summary(rotation_errors),
    }


def derive_official_to_work(
    official_points: np.ndarray, official_c2w: np.ndarray
) -> dict[str, Any]:
    points = np.asarray(official_points, dtype=np.float64)
    poses = np.asarray(official_c2w, dtype=np.float64)
    centers = poses[:, :3, 3]
    up = -np.mean(poses[:, :3, 1], axis=0)
    up /= max(float(np.linalg.norm(up)), 1e-12)
    forward = np.mean(poses[:, :3, 2], axis=0)
    forward -= up * float(np.dot(forward, up))
    if np.linalg.norm(forward) < 1e-6:
        _, _, axes = np.linalg.svd(centers - centers.mean(axis=0), full_matrices=False)
        forward = axes[0] - up * float(np.dot(axes[0], up))
    forward /= max(float(np.linalg.norm(forward)), 1e-12)
    right = np.cross(forward, up)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    forward = np.cross(up, right)
    rotation = np.stack((right, forward, up))
    if np.linalg.det(rotation) < 0.999:
        raise RepresentativeInferenceError("Official-to-work rotation is not right handed")
    rotated_points = points @ rotation.T
    rotated_centers = centers @ rotation.T
    floor = float(np.percentile(rotated_points[:, 2], 2.0))
    origin_rotated = np.asarray(
        [np.median(rotated_centers[:, 0]), np.median(rotated_centers[:, 1]), floor],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = -origin_rotated
    inverse = np.linalg.inv(matrix)
    return {
        "official_to_work": matrix,
        "work_to_official": inverse,
        "gravity_method": "negative_mean_conditioning_camera_y_axis",
        "translation_method": "conditioning_camera_xy_median_and_pseudo_point_z_p02",
        "floor_z_rotated": floor,
        "origin_rotated": origin_rotated,
    }


def transform_c2w(poses: np.ndarray, world_transform: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    transform = np.asarray(world_transform, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    output = np.repeat(np.eye(4, dtype=np.float64)[None], len(poses), axis=0)
    output[:, :3, :3] = np.einsum("ij,sjk->sik", rotation, poses[:, :3, :3])
    output[:, :3, 3] = poses[:, :3, 3] @ rotation.T + translation
    return output


def c2w_to_extrinsic(poses: np.ndarray) -> np.ndarray:
    output = np.empty((len(poses), 3, 4), dtype=np.float64)
    output[:, :3, :3] = np.transpose(poses[:, :3, :3], (0, 2, 1))
    output[:, :3, 3] = -np.einsum(
        "sij,sj->si", output[:, :3, :3], poses[:, :3, 3]
    )
    return output


def representative_quality_gate(
    point_metrics: dict[str, Any],
    trajectory: dict[str, Any],
    visible_fraction: float,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    center_p90 = trajectory["center_error_normalized_by_baseline"]["p90"]
    rotation_p90 = trajectory["rotation_error_deg"]["p90"]
    reasons = []
    if center_p90 is None or center_p90 > float(protocol["max_center_error_p90_baseline"]):
        reasons.append("conditioning_camera_center_alignment_failed")
    if rotation_p90 is None or rotation_p90 > float(protocol["max_rotation_error_p90_deg"]):
        reasons.append("conditioning_camera_rotation_alignment_failed")
    if visible_fraction < float(protocol["minimum_official_camera_visible_fraction"]):
        reasons.append("too_few_pseudo_points_visible_in_conditioning_cameras")
    verified_fraction = float(point_metrics["cross_view_verified_fraction"])
    opportunity_fraction = float(
        point_metrics["cross_view_overlap_opportunity_fraction"]
    )
    verified_given_overlap = float(
        point_metrics["cross_view_verified_fraction_given_overlap"]
    )
    verified_points = int(point_metrics["cross_view_verified_points"])
    low_overlap = opportunity_fraction <= float(
        protocol["low_overlap_max_opportunity_fraction"]
    )
    low_overlap_consistent = (
        low_overlap
        and verified_points >= int(protocol["low_overlap_minimum_verified_points"])
        and verified_given_overlap
        >= float(protocol["low_overlap_minimum_verified_fraction_given_overlap"])
    )
    if (
        verified_fraction < float(protocol["minimum_cross_view_verified_fraction"])
        and not low_overlap_consistent
    ):
        reasons.append("cross_view_verified_fraction_too_low")
    warnings = (
        ["sparse_conditioning_overlap_uses_conditional_consistency_gate"]
        if low_overlap_consistent
        else []
    )
    if reasons:
        grade, decision = "P-F", "fail"
    elif low_overlap_consistent:
        grade, decision = "P-C", "marginal"
    else:
        grade, decision = "P-B", "foundation_pass"
    return {
        "grade": grade,
        "decision": decision,
        "reasons": reasons,
        "warnings": warnings,
        "low_overlap_conditioning": low_overlap,
        "low_overlap_conditional_consistency_pass": low_overlap_consistent,
        "scope": "conditioning-only pseudo geometry, not GT",
    }


def _prepare_model_inputs(
    contract: dict[str, Any], scene_dir: Path, target: int = 518
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    output_rgb = scene_dir / "rgb"
    output_rgb.mkdir(parents=True, exist_ok=True)
    tensors: list[torch.Tensor] = []
    valid_masks = []
    dynamic_masks = []
    records = []
    for view in contract["views"]:
        with Image.open(view["rgb"]) as opened:
            rgb = opened.convert("RGB")
        width, height = rgb.size
        shape = padded_shape(width, height, target=target)
        output_name = f"{view['order']:03d}.png"
        rgba = rgb.convert("RGBA")
        rgba.putalpha(255)
        rgba.save(output_rgb / output_name, compress_level=3)
        resized = np.asarray(
            rgb.resize(
                (shape["resized_width"], shape["resized_height"]),
                Image.Resampling.BICUBIC,
            ),
            dtype=np.uint8,
        )
        canvas = np.full((target, target, 3), 255, dtype=np.uint8)
        valid = np.zeros((target, target), dtype=bool)
        left, top = shape["pad_left"], shape["pad_top"]
        right, bottom = left + shape["resized_width"], top + shape["resized_height"]
        canvas[top:bottom, left:right] = resized
        valid[top:bottom, left:right] = True
        tensors.append(torch.from_numpy(canvas.copy()).permute(2, 0, 1).float() / 255.0)
        valid_masks.append(valid)
        dynamic_masks.append(np.zeros_like(valid))
        records.append(
            {
                **shape,
                "source_name": output_name,
                "output_name": output_name,
                "source_label": view["source_label"],
                "source_rgb": _root_relative(view["rgb"]),
                "source_rgb_sha256": sha256_file(view["rgb"]),
                "dynamic_fraction": 0.0,
            }
        )
    return (
        torch.stack(tensors),
        np.stack(valid_masks),
        np.stack(dynamic_masks),
        records,
    )


def assign_visible_observations(
    points: np.ndarray,
    intrinsics: np.ndarray,
    extrinsic: np.ndarray,
    image_paths: list[Path],
) -> dict[str, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    count = len(points)
    best_score = np.full(count, -np.inf, dtype=np.float64)
    source = np.full(count, -1, dtype=np.int32)
    output_x = np.zeros(count, dtype=np.float32)
    output_y = np.zeros(count, dtype=np.float32)
    images = []
    sizes = []
    for path in image_paths:
        with Image.open(path) as opened:
            image = np.asarray(opened.convert("RGB"))
        images.append(image)
        sizes.append((image.shape[1], image.shape[0]))
    for camera_index, (intrinsic, pose, size) in enumerate(
        zip(intrinsics, extrinsic, sizes)
    ):
        camera = points @ pose[:3, :3].T + pose[:3, 3]
        z = camera[:, 2]
        u = intrinsic[0, 0] * camera[:, 0] / np.maximum(z, 1e-12) + intrinsic[0, 2]
        v = intrinsic[1, 1] * camera[:, 1] / np.maximum(z, 1e-12) + intrinsic[1, 2]
        width, height = size
        valid = (
            np.isfinite(u)
            & np.isfinite(v)
            & (z > 1e-5)
            & (u >= 0.0)
            & (u <= width - 1.0)
            & (v >= 0.0)
            & (v <= height - 1.0)
        )
        margin = np.minimum.reduce(
            (u / width, (width - 1.0 - u) / width, v / height, (height - 1.0 - v) / height)
        )
        score = np.where(valid, margin, -np.inf)
        update = score > best_score
        source[update] = camera_index
        output_x[update] = u[update].astype(np.float32)
        output_y[update] = v[update].astype(np.float32)
        best_score[update] = score[update]
    keep = source >= 0
    colors = np.zeros((int(np.count_nonzero(keep)), 3), dtype=np.uint8)
    kept_source = source[keep]
    kept_x = output_x[keep]
    kept_y = output_y[keep]
    for camera_index, image in enumerate(images):
        positions = np.flatnonzero(kept_source == camera_index)
        if len(positions):
            colors[positions] = image[
                np.rint(kept_y[positions]).astype(np.int32),
                np.rint(kept_x[positions]).astype(np.int32),
            ]
    return {
        "keep": keep,
        "source_frame": kept_source,
        "x_original": kept_x,
        "y_original": kept_y,
        "colors": colors,
        "visible_fraction": np.asarray([float(np.mean(keep))], dtype=np.float64),
    }


def _strict_source_audit(contract: dict[str, Any]) -> dict[str, Any]:
    records = []
    for path in contract["source_files_read"]:
        role = (
            "unit-manifest"
            if path == contract["manifest_path"]
            else "conditioning-camera-metadata"
            if path == contract["camera_path"]
            else "conditioning-rgb"
        )
        record = {
            "path": _root_relative(path),
            "size_bytes_at_read": path.stat().st_size,
            "sha256_at_read": sha256_file(path),
            "role": role,
        }
        if role == "unit-manifest":
            record["conditioning_contract_sha256"] = contract[
                "manifest_conditioning_contract_sha256"
            ]
        records.append(record)
    return {
        "policy": "conditioning-rgb-and-conditioning-camera-records-only",
        "source_files_read": records,
        "heldout_rgb_paths_read": [],
        "depth_paths_read": [],
        "reference_paths_read": [],
        "conditioning_camera_records_used": 8,
        "heldout_camera_records_used": 0,
        "camera_metadata_container_contains_heldout_records": True,
        "declared_but_not_read": contract["declared_but_forbidden"],
    }


def genrecon_input_asset_records(scene_dir: Path) -> list[dict[str, Any]]:
    records = []
    for relative in GENRECON_INPUT_RELATIVE_PATHS:
        path = scene_dir / relative
        if not path.is_file():
            raise RepresentativeInferenceError(f"Missing GenRecon input asset: {path}")
        records.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def validate_genrecon_input_assets(
    scene_dir: Path, input_manifest: dict[str, Any]
) -> None:
    records = input_manifest.get("genrecon_input_assets")
    if not isinstance(records, list) or [item.get("path") for item in records] != list(
        GENRECON_INPUT_RELATIVE_PATHS
    ):
        raise RepresentativeInferenceError("GenRecon input asset contract is missing or unordered")
    for expected in records:
        path = scene_dir / expected["path"]
        if (
            not path.is_file()
            or path.stat().st_size != expected.get("size_bytes")
            or sha256_file(path) != expected.get("sha256")
        ):
            raise RepresentativeInferenceError(f"GenRecon input asset changed: {path}")


def _trajectory_json(alignment: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": "Umeyama Sim(3) from 8 conditioning camera centers only",
        "matched_conditioning_cameras": 8,
        "scale_predicted_to_official": alignment["scale"],
        "rotation_predicted_to_official": alignment["rotation"],
        "translation_predicted_to_official": alignment["translation"],
        "target_baseline_diagonal_m": alignment["target_baseline_diagonal_m"],
        "center_error_m": alignment["center_error_m"],
        "center_error_normalized_by_baseline": alignment[
            "center_error_normalized_by_baseline"
        ],
        "rotation_error_deg": alignment["rotation_error_deg"],
        "gt_geometry_icp_used": False,
        "heldout_cameras_used": False,
    }


def _unit_reusable(
    scene_dir: Path, config_path: Path, contract: dict[str, Any]
) -> bool:
    status_path = scene_dir / "status.json"
    manifest_path = scene_dir / "manifest.json"
    if not status_path.is_file() or not manifest_path.is_file():
        return False
    try:
        status = load_json(status_path)
        manifest = load_json(manifest_path)
        if (
            status.get("status") != "completed"
            or manifest.get("build_contract", {}).get("tool_sha256")
            != sha256_file(Path(__file__))
            or manifest.get("build_contract", {}).get("config_sha256")
            != sha256_file(config_path)
            or _resolve(manifest.get("source_manifest", ""), ROOT)
            != contract["manifest_path"]
            or manifest.get("source_manifest_conditioning_contract_sha256")
            != contract["manifest_conditioning_contract_sha256"]
        ):
            return False
        validate_source_audit(manifest, manifest["source_audit"])
        validate_genrecon_input_assets(scene_dir, manifest)
        return True
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    except RepresentativeInferenceError:
        return False


def prepare_unit(
    contract: dict[str, Any],
    model: Any,
    model_provenance: dict[str, Any],
    input_root: Path,
    config_path: Path,
    protocol: dict[str, Any],
    *,
    force: bool,
) -> dict[str, Any]:
    unit_id = contract["unit_id"]
    scene_dir = input_root / "candidates" / unit_id
    if _unit_reusable(scene_dir, config_path, contract) and not force:
        print(f"[gt-input] {unit_id}: reuse completed foundation input", flush=True)
        return load_json(scene_dir / "status.json")
    if force and scene_dir.exists():
        shutil.rmtree(scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    write_json(scene_dir / "status.json", {"unit_id": unit_id, "status": "running"})
    try:
        images, valid, dynamic, records = _prepare_model_inputs(contract, scene_dir)
        prediction = run_vggt_inference(model, images)
        save_predictions(scene_dir / "predictions.npz", prediction, records, valid, dynamic)
        point_data, point_metrics = cross_view_sparse_points(
            prediction,
            valid,
            dynamic,
            records,
            scene_dir,
            confidence_percentile=float(protocol["confidence_percentile"]),
            relative_depth_tolerance=float(protocol["relative_depth_tolerance"]),
            minimum_support=int(protocol["minimum_support"]),
            prefilter_budget=int(protocol["prefilter_budget"]),
            maximum_points=int(protocol["maximum_points"]),
            seed=candidate_seed(int(protocol["seed"]), unit_id),
        )
        official_c2w = np.stack([item["camera_to_world"] for item in contract["views"]])
        alignment = trajectory_alignment(prediction["extrinsic"], official_c2w)
        pseudo_official, _ = transform_world_similarity(
            point_data["points_world"],
            prediction["extrinsic"],
            alignment["scale"],
            alignment["rotation"],
            alignment["translation"],
        )
        work_frame = derive_official_to_work(pseudo_official, official_c2w)
        official_to_work = work_frame["official_to_work"]
        points_work = (
            pseudo_official @ official_to_work[:3, :3].T + official_to_work[:3, 3]
        )
        c2w_work = transform_c2w(official_c2w, official_to_work)
        extrinsic_work = c2w_to_extrinsic(c2w_work)
        intrinsics = np.stack([item["intrinsic"] for item in contract["views"]])
        output_paths = [scene_dir / "rgb" / record["output_name"] for record in records]
        observations = assign_visible_observations(
            points_work, intrinsics, extrinsic_work, output_paths
        )
        keep = observations["keep"]
        points_work = points_work[keep]
        if len(points_work) < 500:
            raise RepresentativeInferenceError(
                f"Only {len(points_work)} pseudo points project into conditioning cameras"
            )
        point_data_work = {
            "source_frame": observations["source_frame"],
            "x_original": observations["x_original"],
            "y_original": observations["y_original"],
            "colors": observations["colors"],
            "confidence": point_data["confidence"][keep],
            "support": point_data["support"][keep],
        }
        colmap_metrics = write_colmap_text(
            scene_dir / "colmap_vggt",
            points_work,
            point_data_work,
            extrinsic_work,
            intrinsics,
            records,
        )
        write_binary_ply(
            scene_dir / "foundation_points.ply",
            points_work,
            point_data_work["colors"],
            point_data_work["confidence"],
            point_data_work["support"],
            point_data_work["source_frame"],
        )
        make_projection_plot(
            points_work,
            point_data_work["colors"],
            c2w_work[:, :3, 3],
            scene_dir / "foundation_projections.png",
            candidate_seed(int(protocol["seed"]), unit_id),
        )
        center_p90 = alignment["center_error_normalized_by_baseline"]["p90"]
        rotation_p90 = alignment["rotation_error_deg"]["p90"]
        visible_fraction = float(observations["visible_fraction"][0])
        quality_gate = representative_quality_gate(
            point_metrics, alignment, visible_fraction, protocol
        )
        gate_reasons = quality_gate["reasons"]
        grade = quality_gate["grade"]
        manifest = {
            "schema": "genrecon.gt-representative-foundation-input",
            "schema_version": SCHEMA_VERSION,
            "unit_id": unit_id,
            "title": unit_id,
            "track": "GT-pose-foundation-pseudo-geometry",
            "build_contract": {
                "tool_sha256": sha256_file(Path(__file__)),
                "config_sha256": sha256_file(config_path),
            },
            "source_manifest": _root_relative(contract["manifest_path"]),
            "source_manifest_sha256_at_read": sha256_file(contract["manifest_path"]),
            "source_manifest_conditioning_contract_sha256": contract[
                "manifest_conditioning_contract_sha256"
            ],
            "source_audit": _strict_source_audit(contract),
            "model": model_provenance,
            "inference": {
                "input_views": 8,
                "input_resolution": 518,
                "preprocess": "preserve-aspect-pad-to-518-patch14",
                "elapsed_seconds": prediction["elapsed_seconds"],
                "peak_memory_mib": prediction["peak_memory_mib"],
            },
            "views": records,
            "conditioning_pose_normalization": [
                item["pose_normalization"] for item in contract["views"]
            ],
            "point_filter": {
                **point_metrics,
                "points_visible_in_official_conditioning_cameras": len(points_work),
                "official_conditioning_camera_visible_fraction": visible_fraction,
            },
            "trajectory_alignment": _trajectory_json(alignment),
            "work_frame": {
                "official_to_work": work_frame["official_to_work"],
                "work_to_official": work_frame["work_to_official"],
                "gravity_method": work_frame["gravity_method"],
                "translation_method": work_frame["translation_method"],
                "scale": "official metric scale preserved",
            },
            "colmap_export": colmap_metrics,
            "genrecon_input_assets": genrecon_input_asset_records(scene_dir),
            "quality_gate": quality_gate,
            "limitations": [
                "VGGT pseudo geometry is not an observed SfM reconstruction or ground truth.",
                "Only 8 conditioning RGB images and their conditioning camera metadata are read.",
                "Official conditioning poses define a GT-pose generation track, not end-to-end estimated pose.",
                "Heldout RGB/depth and evaluation reference geometry are excluded from conditioning.",
                "Synthetic point observations and ERROR=0 exist only for GenRecon interface compatibility.",
            ],
        }
        write_json(scene_dir / "manifest.json", manifest)
        status = {
            "unit_id": unit_id,
            "status": "completed" if not gate_reasons else "failed-gate",
            "grade": grade,
            "point_count": len(points_work),
            "view_count": 8,
            "elapsed_seconds": time.monotonic() - start,
            "gate_reasons": gate_reasons,
        }
        write_json(scene_dir / "status.json", status)
        print(
            f"[gt-input] {unit_id}: status={status['status']} points={len(points_work)} "
            f"center_p90={center_p90:.4f} rot_p90={rotation_p90:.2f} "
            f"xview={point_metrics['cross_view_verified_fraction']:.3f}",
            flush=True,
        )
        return status
    except Exception as error:
        status = {
            "unit_id": unit_id,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_seconds": time.monotonic() - start,
        }
        write_json(scene_dir / "status.json", status)
        print(f"[gt-input] {unit_id}: FAILED {type(error).__name__}: {error}", flush=True)
        return status


def selected_unit_ids(config: dict[str, Any], requested: list[str]) -> list[str]:
    available = list(config["unit_ids"])
    if not requested:
        return available
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise RepresentativeInferenceError(f"Unknown representative units: {unknown}")
    return list(dict.fromkeys(requested))


def refresh_index(unit_ids: list[str], input_root: Path) -> dict[str, Any]:
    rows = []
    for unit_id in unit_ids:
        scene = input_root / "candidates" / unit_id
        status = load_json(scene / "status.json") if (scene / "status.json").is_file() else {}
        manifest = load_json(scene / "manifest.json") if (scene / "manifest.json").is_file() else {}
        preflight = (
            load_json(scene / "genrecon_preflight" / "preflight.json")
            if (scene / "genrecon_preflight" / "preflight.json").is_file()
            else {}
        )
        fallback = preflight.get("closest_camera_fallback_count")
        preflight_status = preflight.get("status", "missing")
        if preflight_status == "passed" and fallback == 0:
            routed_preflight = "passed"
        elif preflight_status == "marginal":
            routed_preflight = "blocked-zero-fallback"
        else:
            routed_preflight = preflight_status
        rows.append(
            {
                "candidate_id": unit_id,
                "title": unit_id,
                "track": "GT-pose-foundation-pseudo-geometry",
                "status": "completed" if status.get("status") == "completed" else status.get("status", "missing"),
                "grade": manifest.get("quality_gate", {}).get("grade"),
                "preflight": routed_preflight,
                "chunks": preflight.get("chunk_count"),
                "fallback": fallback,
                "visual_disposition": "gt_calibration_representative",
                "points": status.get("point_count"),
            }
        )
    document = {
        "schema": "genrecon.gt-representative-foundation-index",
        "schema_version": SCHEMA_VERSION,
        "summary": {
            "candidate_count": len(rows),
            "completed": sum(row["status"] == "completed" for row in rows),
            "zero_fallback_preflight": sum(row["preflight"] == "passed" for row in rows),
            "points": sum(row.get("points") or 0 for row in rows),
            "chunks": sum(row.get("chunks") or 0 for row in rows),
        },
        "candidates": rows,
    }
    write_json(input_root / "index.json", document)
    return document


def run_preflights(unit_ids: list[str], input_root: Path, *, force: bool) -> None:
    args = SimpleNamespace(
        force_preflight=force,
        genrecon_views=8,
        seed=42,
        fail_fast=False,
    )
    for unit_id in unit_ids:
        run_genrecon_preflight({"candidate_id": unit_id}, input_root, args)
    refresh_index(unit_ids, input_root)


_PLY_TYPES = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "float": "<f4",
    "double": "<f8",
}


def transform_binary_ply(source: Path, destination: Path, matrix: np.ndarray) -> dict[str, Any]:
    matrix = _finite_matrix(matrix, (4, 4), "work_to_official")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with source.open("rb") as input_handle:
        header_lines = []
        vertex_count = None
        current_element = None
        properties: list[tuple[str, str]] = []
        while True:
            line = input_handle.readline()
            if not line:
                raise RepresentativeInferenceError(f"Truncated PLY header: {source}")
            header_lines.append(line)
            text = line.decode("ascii").strip()
            values = text.split()
            if values[:2] == ["format", "binary_little_endian"]:
                pass
            elif values and values[0] == "format":
                raise RepresentativeInferenceError(f"Unsupported PLY format: {text}")
            elif len(values) == 3 and values[0] == "element":
                current_element = values[1]
                if current_element == "vertex":
                    vertex_count = int(values[2])
            elif values and values[0] == "property" and current_element == "vertex":
                if len(values) != 3 or values[1] == "list" or values[1] not in _PLY_TYPES:
                    raise RepresentativeInferenceError(f"Unsupported PLY vertex property: {text}")
                properties.append((values[2], _PLY_TYPES[values[1]]))
            if text == "end_header":
                break
        if vertex_count is None or not {"x", "y", "z"} <= {item[0] for item in properties}:
            raise RepresentativeInferenceError("PLY has no transformable XYZ vertices")
        dtype = np.dtype(properties, align=False)
        with temporary.open("wb") as output_handle:
            output_handle.write(b"".join(header_lines))
            remaining = vertex_count
            while remaining:
                count = min(remaining, 250_000)
                payload = input_handle.read(count * dtype.itemsize)
                if len(payload) != count * dtype.itemsize:
                    raise RepresentativeInferenceError("Truncated PLY vertex payload")
                values = np.frombuffer(payload, dtype=dtype, count=count).copy()
                xyz = np.column_stack((values["x"], values["y"], values["z"])).astype(
                    np.float64
                )
                transformed = xyz @ matrix[:3, :3].T + matrix[:3, 3]
                values["x"], values["y"], values["z"] = transformed.T
                output_handle.write(values.tobytes())
                remaining -= count
            shutil.copyfileobj(input_handle, output_handle, length=8 * 1024 * 1024)
    temporary.replace(destination)
    if destination.stat().st_size != source.stat().st_size:
        raise RepresentativeInferenceError("Transformed PLY size differs from source")
    return {
        "vertices": vertex_count,
        "size_bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def wrap_glb_with_transform(source: Path, destination: Path, matrix: np.ndarray) -> dict[str, Any]:
    matrix = _finite_matrix(matrix, (4, 4), "work_to_official")
    with source.open("rb") as handle:
        magic, version, total = struct.unpack("<4sII", handle.read(12))
        if magic != b"glTF" or version != 2 or total != source.stat().st_size:
            raise RepresentativeInferenceError(f"Invalid source GLB: {source}")
        chunks = []
        while handle.tell() < total:
            length, kind = struct.unpack("<I4s", handle.read(8))
            chunks.append((kind, handle.read(length)))
    if not chunks or chunks[0][0] != b"JSON":
        raise RepresentativeInferenceError("GLB first chunk is not JSON")
    document = json.loads(chunks[0][1].rstrip(b" \x00"))
    nodes = document.setdefault("nodes", [])
    for scene_index, scene in enumerate(document.get("scenes", [])):
        roots = list(scene.get("nodes", []))
        parent = len(nodes)
        nodes.append(
            {
                "name": f"work_to_official_scene_{scene_index}",
                "matrix": matrix.T.reshape(-1).tolist(),
                "children": roots,
            }
        )
        scene["nodes"] = [parent]
    json_payload = json.dumps(document, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    json_payload += b" " * ((-len(json_payload)) % 4)
    output_chunks = [(b"JSON", json_payload), *chunks[1:]]
    total_length = 12 + sum(8 + len(payload) for _, payload in output_chunks)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(struct.pack("<4sII", b"glTF", 2, total_length))
        for kind, payload in output_chunks:
            handle.write(struct.pack("<I4s", len(payload), kind))
            handle.write(payload)
    temporary.replace(destination)
    return {
        "size_bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "scene_count": len(document.get("scenes", [])),
        "mesh_count": len(document.get("meshes", [])),
        "material_count": len(document.get("materials", [])),
        "texture_count": len(document.get("textures", [])),
    }


def representative_input_contract_sha256(document: dict[str, Any]) -> str:
    canonical = json.loads(json.dumps(document, allow_nan=False))
    canonical.pop("source_manifest_sha256_at_read", None)
    inference = canonical.get("inference", {})
    inference.pop("elapsed_seconds", None)
    inference.pop("peak_memory_mib", None)
    model = canonical.get("model", {})
    model.pop("checkpoint_path", None)
    for record in canonical.get("source_audit", {}).get("source_files_read", []):
        if record.get("role") == "unit-manifest":
            record.pop("size_bytes_at_read", None)
            record.pop("sha256_at_read", None)
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def package_prediction(
    unit_id: str, input_root: Path, prediction_root: Path, *, force: bool
) -> dict[str, Any]:
    input_manifest_path = input_root / "candidates" / unit_id / "manifest.json"
    input_manifest = load_json(input_manifest_path)
    reconstruction = prediction_root / "candidates" / unit_id / "reconstruction"
    source_mesh = reconstruction / "mesh.ply"
    source_glb = reconstruction / "scene.glb"
    if not source_mesh.is_file() or not source_glb.is_file():
        raise RepresentativeInferenceError(f"GenRecon reconstruction is incomplete for {unit_id}")
    package = prediction_root / "candidates" / unit_id / "prediction_official"
    package_manifest_path = package / "manifest.json"
    source_hashes = {"mesh": sha256_file(source_mesh), "glb": sha256_file(source_glb)}
    matrix = _finite_matrix(
        input_manifest["work_frame"]["work_to_official"],
        (4, 4),
        "work_to_official",
    )
    input_contract = representative_input_contract_sha256(input_manifest)
    if package_manifest_path.is_file() and not force:
        previous = load_json(package_manifest_path)
        mesh_path = package / "mesh.ply"
        glb_path = package / "scene.glb"
        previous_matrix = np.asarray(
            previous.get("alignment", {}).get("work_to_official", []), dtype=np.float64
        )
        if (
            previous.get("source", {}).get("sha256") == source_hashes
            and previous.get("source", {}).get("input_manifest_contract_sha256")
            == input_contract
            and previous_matrix.shape == (4, 4)
            and np.array_equal(previous_matrix, matrix)
            and mesh_path.is_file()
            and glb_path.is_file()
            and previous.get("outputs", {}).get("mesh", {}).get("sha256")
            == sha256_file(mesh_path)
            and previous.get("outputs", {}).get("glb", {}).get("sha256")
            == sha256_file(glb_path)
        ):
            print(f"[gt-package] {unit_id}: reuse", flush=True)
            return previous
    mesh_record = transform_binary_ply(source_mesh, package / "mesh.ply", matrix)
    glb_record = wrap_glb_with_transform(source_glb, package / "scene.glb", matrix)
    document = {
        "schema": "genrecon.gt-representative-prediction-package",
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "unit_id": unit_id,
        "track": "GT-pose-foundation-pseudo-geometry",
        "coordinate_frame": "declared calibration/reference frame",
        "alignment": {
            "method": "conditioning-camera-only Sim(3), followed by exact work-to-official inverse",
            "work_to_official": matrix,
            "gt_geometry_icp_used": False,
            "reference_geometry_used": False,
            "heldout_views_used": False,
        },
        "source": {
            "reconstruction": _root_relative(reconstruction),
            "sha256": source_hashes,
            "input_manifest": _root_relative(input_manifest_path),
            "input_manifest_contract_sha256": input_contract,
        },
        "outputs": {
            "mesh": {"path": "mesh.ply", **mesh_record},
            "glb": {"path": "scene.glb", **glb_record},
        },
        "limitations": input_manifest["limitations"],
    }
    write_json(package_manifest_path, document)
    print(
        f"[gt-package] {unit_id}: vertices={mesh_record['vertices']} "
        f"mesh={mesh_record['size_bytes']} glb={glb_record['size_bytes']}",
        flush=True,
    )
    return document


def _resolve_recorded_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def validate_source_audit(
    input_manifest: dict[str, Any], audit: dict[str, Any]
) -> None:
    records = audit.get("source_files_read")
    if not isinstance(records, list) or len(records) != 10:
        raise RepresentativeInferenceError(
            "source audit must contain one manifest, one camera file, and 8 RGBs"
        )
    role_counts = {
        role: sum(item.get("role") == role for item in records)
        for role in ("unit-manifest", "conditioning-camera-metadata", "conditioning-rgb")
    }
    if role_counts != {
        "unit-manifest": 1,
        "conditioning-camera-metadata": 1,
        "conditioning-rgb": 8,
    }:
        raise RepresentativeInferenceError(f"invalid source audit roles: {role_counts}")

    manifest_records = [item for item in records if item["role"] == "unit-manifest"]
    unit_manifest_path = _resolve_recorded_path(input_manifest.get("source_manifest", ""))
    if _resolve_recorded_path(manifest_records[0]["path"]) != unit_manifest_path:
        raise RepresentativeInferenceError("source audit unit manifest path is inconsistent")
    unit_manifest = load_json(unit_manifest_path)
    unit_dir = unit_manifest_path.parent
    expected_camera = _resolve(unit_manifest["input"]["cameras"], unit_dir)
    expected_rgb = {
        _resolve(value, unit_dir)
        for value in unit_manifest["input"]["conditioning_views"]
    }
    recorded_camera = {
        _resolve_recorded_path(item["path"])
        for item in records
        if item["role"] == "conditioning-camera-metadata"
    }
    recorded_rgb = {
        _resolve_recorded_path(item["path"])
        for item in records
        if item["role"] == "conditioning-rgb"
    }
    if recorded_camera != {expected_camera} or recorded_rgb != expected_rgb or len(expected_rgb) != 8:
        raise RepresentativeInferenceError("source audit paths do not match the conditioning split")

    for record in records:
        path = _resolve_recorded_path(record["path"])
        if not path.is_file():
            raise RepresentativeInferenceError(f"audited source is missing: {path}")
        role = record["role"]
        if role == "unit-manifest":
            current_contract = conditioning_manifest_contract_sha256(unit_manifest)
            if current_contract != record.get("conditioning_contract_sha256"):
                raise RepresentativeInferenceError(
                    f"unit manifest conditioning contract changed: {path}"
                )
            if current_contract != input_manifest.get(
                "source_manifest_conditioning_contract_sha256"
            ):
                raise RepresentativeInferenceError(
                    "unit manifest conditioning contract does not match input manifest"
                )
        elif (
            path.stat().st_size != record.get("size_bytes_at_read")
            or sha256_file(path) != record.get("sha256_at_read")
        ):
            raise RepresentativeInferenceError(f"audited conditioning source changed: {path}")


def validate_prediction_package(package: Path, unit_id: str) -> Path:
    manifest_path = package / "manifest.json"
    document = load_json(manifest_path)
    alignment = document.get("alignment", {})
    if (
        document.get("schema") != "genrecon.gt-representative-prediction-package"
        or document.get("schema_version") != SCHEMA_VERSION
        or document.get("status") != "pass"
        or document.get("unit_id") != unit_id
        or document.get("track") != "GT-pose-foundation-pseudo-geometry"
        or document.get("coordinate_frame") != "declared calibration/reference frame"
        or alignment.get("gt_geometry_icp_used") is not False
        or alignment.get("reference_geometry_used") is not False
        or alignment.get("heldout_views_used") is not False
    ):
        raise RepresentativeInferenceError(f"Invalid prediction package contract: {package}")
    _finite_matrix(alignment.get("work_to_official"), (4, 4), "work_to_official")
    source = document.get("source", {})
    reconstruction = _resolve_recorded_path(source.get("reconstruction", ""))
    input_manifest = _resolve_recorded_path(source.get("input_manifest", ""))
    if (
        not input_manifest.is_file()
        or representative_input_contract_sha256(load_json(input_manifest))
        != source.get("input_manifest_contract_sha256")
    ):
        raise RepresentativeInferenceError(
            f"Prediction package input manifest changed: {input_manifest}"
        )
    for key, name in (("mesh", "mesh.ply"), ("glb", "scene.glb")):
        source_path = reconstruction / name
        if (
            not source_path.is_file()
            or sha256_file(source_path) != source.get("sha256", {}).get(key)
        ):
            raise RepresentativeInferenceError(
                f"Prediction package source {key} changed: {source_path}"
            )
    mesh = package / document["outputs"]["mesh"]["path"]
    glb = package / document["outputs"]["glb"]["path"]
    for key, path in (("mesh", mesh), ("glb", glb)):
        if not path.is_file() or path.stat().st_size != document["outputs"][key]["size_bytes"]:
            raise RepresentativeInferenceError(f"Missing prediction package {key}: {path}")
        if sha256_file(path) != document["outputs"][key]["sha256"]:
            raise RepresentativeInferenceError(f"Prediction package {key} hash mismatch: {path}")
    return mesh


def validate_all(
    unit_ids: list[str], input_root: Path, prediction_root: Path, *, require_packages: bool
) -> dict[str, Any]:
    errors = []
    counts = {
        "units": len(unit_ids),
        "completed_inputs": 0,
        "zero_fallback_preflights": 0,
        "conditioning_rgb": 0,
        "pseudo_points": 0,
        "chunks": 0,
        "prediction_packages": 0,
        "prediction_vertices": 0,
        "prediction_mesh_bytes": 0,
        "prediction_glb_bytes": 0,
        "strict_json_files": 0,
    }
    for unit_id in unit_ids:
        scene = input_root / "candidates" / unit_id
        try:
            status = load_json(scene / "status.json")
            manifest = load_json(scene / "manifest.json")
            preflight = load_json(scene / "genrecon_preflight" / "preflight.json")
            if (
                status.get("status") != "completed"
                or manifest["quality_gate"]["decision"]
                not in {"foundation_pass", "marginal"}
            ):
                raise RepresentativeInferenceError("foundation input gate did not pass")
            audit = manifest["source_audit"]
            if (
                audit["heldout_rgb_paths_read"]
                or audit["depth_paths_read"]
                or audit["reference_paths_read"]
                or audit["heldout_camera_records_used"] != 0
                or audit["conditioning_camera_records_used"] != 8
            ):
                raise RepresentativeInferenceError("forbidden GT/heldout inputs appear in source audit")
            validate_source_audit(manifest, audit)
            validate_genrecon_input_assets(scene, manifest)
            if preflight.get("status") != "passed" or preflight.get("closest_camera_fallback_count") != 0:
                raise RepresentativeInferenceError("preflight is not a zero-fallback pass")
            counts["completed_inputs"] += 1
            counts["zero_fallback_preflights"] += 1
            counts["conditioning_rgb"] += 8
            counts["pseudo_points"] += int(status["point_count"])
            counts["chunks"] += int(preflight["chunk_count"])
            package = prediction_root / "candidates" / unit_id / "prediction_official"
            if package.is_dir():
                validate_prediction_package(package, unit_id)
                package_manifest = load_json(package / "manifest.json")
                counts["prediction_packages"] += 1
                counts["prediction_vertices"] += int(
                    package_manifest["outputs"]["mesh"]["vertices"]
                )
                counts["prediction_mesh_bytes"] += int(
                    package_manifest["outputs"]["mesh"]["size_bytes"]
                )
                counts["prediction_glb_bytes"] += int(
                    package_manifest["outputs"]["glb"]["size_bytes"]
                )
            elif require_packages:
                raise RepresentativeInferenceError("prediction package is missing")
        except Exception as error:
            errors.append(f"{unit_id}: {type(error).__name__}: {error}")
    for root in (input_root, prediction_root):
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            try:
                load_json(path)
                counts["strict_json_files"] += 1
            except Exception as error:
                errors.append(f"strict JSON {path}: {error}")
    result = {
        "schema": "genrecon.gt-representative-inference-validation",
        "schema_version": SCHEMA_VERSION,
        "result": "pass" if not errors else "fail",
        "counts": counts,
        "checks": {
            "conditioning_only_source_audit": not any("forbidden" in item or "source audit" in item for item in errors),
            "all_input_quality_gates_pass": counts["completed_inputs"] == len(unit_ids),
            "all_preflights_zero_fallback": counts["zero_fallback_preflights"] == len(unit_ids),
            "all_prediction_packages_valid": (
                counts["prediction_packages"] == len(unit_ids) if require_packages else True
            ),
            "strict_json": not any(item.startswith("strict JSON") for item in errors),
        },
        "errors": errors,
    }
    write_json(prediction_root / "inference_validation.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "preflight", "package", "summarize", "validate-inputs", "validate")
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--units-root", type=Path, default=DEFAULT_UNITS_ROOT)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument("--vggt-root", type=Path, default=DEFAULT_VGGT_ROOT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--unit", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.config = args.config.resolve()
    args.units_root = args.units_root.resolve()
    args.input_root = args.input_root.resolve()
    args.prediction_root = args.prediction_root.resolve()
    args.vggt_root = args.vggt_root.resolve()
    config = load_json(args.config)
    if config.get("schema") != "genrecon.gt-representative-inference-plan":
        raise SystemExit(f"Unexpected inference config: {args.config}")
    unit_ids = selected_unit_ids(config, args.unit)
    args.input_root.mkdir(parents=True, exist_ok=True)
    args.prediction_root.mkdir(parents=True, exist_ok=True)

    if args.stage == "prepare":
        if not torch.cuda.is_available():
            raise RepresentativeInferenceError("VGGT preparation requires CUDA")
        checkpoint = resolve_vggt_checkpoint(args.checkpoint)
        if sha256_file(checkpoint) != VGGT_MODEL_SHA256:
            raise RepresentativeInferenceError("VGGT checkpoint SHA256 mismatch")
        random.seed(config["protocol"]["seed"])
        np.random.seed(config["protocol"]["seed"])
        torch.manual_seed(config["protocol"]["seed"])
        torch.cuda.manual_seed_all(config["protocol"]["seed"])
        model, provenance = load_vggt_model(args.vggt_root, checkpoint)
        write_json(
            args.input_root / "config.json",
            {
                "schema": "genrecon.gt-representative-foundation-config",
                "schema_version": SCHEMA_VERSION,
                "plan": _root_relative(args.config),
                "plan_sha256": sha256_file(args.config),
                "unit_ids": list(config["unit_ids"]),
                "requested_unit_ids": unit_ids,
                "model": provenance,
                "tool_sha256": sha256_file(Path(__file__)),
            },
        )
        for unit_id in unit_ids:
            contract = load_conditioning_contract(unit_id, args.units_root)
            prepare_unit(
                contract,
                model,
                provenance,
                args.input_root,
                args.config,
                config["protocol"],
                force=args.force,
            )
        del model
        torch.cuda.empty_cache()
        refresh_index(unit_ids, args.input_root)
    elif args.stage == "preflight":
        run_preflights(unit_ids, args.input_root, force=args.force)
    elif args.stage == "package":
        for unit_id in unit_ids:
            package_prediction(
                unit_id, args.input_root, args.prediction_root, force=args.force
            )
    elif args.stage == "summarize":
        print(json.dumps(refresh_index(unit_ids, args.input_root)["summary"], indent=2))
    else:
        result = validate_all(
            unit_ids,
            args.input_root,
            args.prediction_root,
            require_packages=args.stage == "validate",
        )
        print(f"[gt-representative-validation] {result['result']} {result['counts']}")
        for error in result["errors"][:20]:
            print(f"  - {error}")
        return 0 if result["result"] == "pass" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
