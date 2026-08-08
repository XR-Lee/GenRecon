#!/usr/bin/env python3
"""Build VGGT foundation-SfM fallback point clouds for internet video shots.

The tool consumes outputs from ``prepare_internet_sfm_preproducts.py`` and emits
z-up, proxy-scaled, confidence-filtered point clouds plus a COLMAP text layout
that GenRecon's ``Iphone`` mode can consume. Foundation predictions are kept
separate from observed COLMAP geometry and are never labelled as ground truth.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import html
import io
import json
import math
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_PREPRODUCTS_ROOT = ROOT / "data" / "internet-zero-shot" / "sfm-preproducts-v1"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "internet-zero-shot" / "foundation-sfm-v1"
DEFAULT_VGGT_ROOT = ROOT / "data" / "model-sources" / "vggt"
DEFAULT_VISUAL_REVIEW = ROOT / "configs" / "eval" / "foundation_sfm_visual_review_v1.json"
VGGT_CODE_REVISION = "a288dd0f14786c93483e45524328726ab7b1b4ce"
VGGT_MODEL_REPO = "facebook/VGGT-1B"
VGGT_MODEL_REVISION = "860abec7937da0a4c03c41d3c269c366e82abdf9"
VGGT_MODEL_FILENAME = "model.pt"
VGGT_MODEL_SHA256 = "d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0"
VGGT_MODEL_SIZE = 5_026_874_952
VGGT_MODEL_LICENSE = "CC-BY-NC-4.0"
VGGT_OMEGA_CODE_REVISION = "39a0cb8af88554f15ddcb5354cd52bde588fa014"
SCHEMA_VERSION = 1
COLMAP_PAIR_ID_BASE = 2_147_483_647


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(value), indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def percentile_summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def numeric_frame_key(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)(?=\.[^.]+$)", Path(name).name)
    return (int(match.group(1)) if match else 0, Path(name).name)


def candidate_seed(base_seed: int, candidate_id: str) -> int:
    digest = hashlib.sha256(candidate_id.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:4], "little")) % (2**32)


def load_preproducts(preproducts_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    index_path = preproducts_root / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing preproducts index: {index_path}")
    document = json.loads(index_path.read_text())
    rows = []
    for candidate in document["candidates"]:
        identifier = candidate["candidate_id"]
        directory = preproducts_root / "candidates" / identifier
        quality = json.loads((directory / "quality.json").read_text())
        rows.append(
            {
                "candidate_id": identifier,
                "title": candidate.get("title", identifier),
                "preproducts_dir": directory,
                "preproducts_quality": quality,
                "preproducts_grade": quality.get("quality_gate", {}).get("grade", "F"),
            }
        )
    return document, rows


def dynamic_fraction_table(candidate_dir: Path) -> dict[str, float]:
    document = json.loads((candidate_dir / "mask_summary.json").read_text())
    return {
        frame["image"]: float(frame.get("dynamic_fraction", 0.0))
        for frame in document["frames"]
    }


def database_pair_rows(database_path: Path) -> dict[tuple[str, str], int]:
    if not database_path.is_file():
        return {}
    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    try:
        image_names = {
            int(image_id): str(name)
            for image_id, name in connection.execute("SELECT image_id, name FROM images")
        }
        result = {}
        for pair_id, rows in connection.execute(
            "SELECT pair_id, rows FROM two_view_geometries WHERE rows > 0"
        ):
            image_id1 = int(pair_id) // COLMAP_PAIR_ID_BASE
            image_id2 = int(pair_id) % COLMAP_PAIR_ID_BASE
            if image_id1 not in image_names or image_id2 not in image_names:
                continue
            pair = tuple(sorted((image_names[image_id1], image_names[image_id2])))
            result[pair] = int(rows)
        return result
    finally:
        connection.close()


def evenly_select_with_dynamic(
    names: list[str], dynamic: dict[str, float], maximum: int
) -> list[str]:
    names = sorted(dict.fromkeys(names), key=numeric_frame_key)
    if len(names) <= maximum:
        return names
    edges = np.linspace(0, len(names), maximum + 1)
    selected = []
    for index in range(maximum):
        start = int(math.floor(edges[index]))
        end = max(start + 1, int(math.floor(edges[index + 1])))
        target = 0.5 * (edges[index] + edges[index + 1] - 1)
        choices = list(range(start, min(end, len(names))))
        best = min(
            choices,
            key=lambda position: (
                dynamic.get(names[position], 1.0)
                + 0.03 * abs(position - target) / max(end - start, 1),
                abs(position - target),
                numeric_frame_key(names[position]),
            ),
        )
        selected.append(names[best])
    return selected


def select_foundation_views(candidate: dict[str, Any], maximum: int) -> dict[str, Any]:
    candidate_dir = candidate["preproducts_dir"]
    quality = candidate["preproducts_quality"]
    dynamic = dynamic_fraction_table(candidate_dir)
    all_names = sorted((path.name for path in (candidate_dir / "rgb").glob("*")), key=numeric_frame_key)
    largest = quality.get("largest_model") or {}
    registered = [name for name in largest.get("registered_image_names", []) if name in dynamic]
    if len(registered) >= 4:
        selected = evenly_select_with_dynamic(registered, dynamic, maximum)
        return {
            "policy": "largest_colmap_fragment_even_bins_dynamic_tiebreak",
            "source_pool_count": len(registered),
            "selected": selected,
            "selected_dynamic_fraction": [dynamic[name] for name in selected],
            "verified_pair_window_score": None,
        }

    pair_rows = database_pair_rows(candidate_dir / "database.db")
    window_size = min(maximum, len(all_names))
    if window_size == 0:
        raise ValueError(f"No RGB frames for {candidate['candidate_id']}")
    best_score = -float("inf")
    best_start = 0
    best_pair_sum = 0
    for start in range(len(all_names) - window_size + 1):
        window = all_names[start : start + window_size]
        pair_score = 0.0
        pair_sum = 0
        for gap in (1, 2):
            for left, right in zip(window, window[gap:]):
                rows = pair_rows.get(tuple(sorted((left, right))), 0)
                pair_sum += rows
                pair_score += math.log1p(rows)
        dynamic_mean = float(np.mean([dynamic.get(name, 1.0) for name in window]))
        score = pair_score / max(window_size, 1) - 2.0 * dynamic_mean
        if score > best_score:
            best_score = score
            best_start = start
            best_pair_sum = pair_sum
    selected = all_names[best_start : best_start + window_size]
    return {
        "policy": "best_contiguous_verified_pair_window",
        "source_pool_count": len(all_names),
        "selected": selected,
        "selected_dynamic_fraction": [dynamic.get(name, 0.0) for name in selected],
        "verified_pair_window_score": best_score,
        "verified_pair_window_inliers": best_pair_sum,
    }


def padded_shape(width: int, height: int, target: int = 518, patch: int = 14) -> dict[str, int]:
    if width >= height:
        resized_width = target
        resized_height = max(patch, round(height * target / width / patch) * patch)
    else:
        resized_height = target
        resized_width = max(patch, round(width * target / height / patch) * patch)
    resized_width = min(target, resized_width)
    resized_height = min(target, resized_height)
    pad_left = (target - resized_width) // 2
    pad_top = (target - resized_height) // 2
    return {
        "original_width": int(width),
        "original_height": int(height),
        "resized_width": int(resized_width),
        "resized_height": int(resized_height),
        "pad_left": int(pad_left),
        "pad_top": int(pad_top),
        "target": int(target),
    }


def model_xy_to_original(
    x: np.ndarray, y: np.ndarray, record: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    original_x = (x - record["pad_left"]) * (
        record["original_width"] / record["resized_width"]
    )
    original_y = (y - record["pad_top"]) * (
        record["original_height"] / record["resized_height"]
    )
    return original_x, original_y


def model_intrinsics_to_original(intrinsic: np.ndarray, record: dict[str, Any]) -> np.ndarray:
    result = np.eye(3, dtype=np.float64)
    sx = record["original_width"] / record["resized_width"]
    sy = record["original_height"] / record["resized_height"]
    result[0, 0] = float(intrinsic[0, 0]) * sx
    result[1, 1] = float(intrinsic[1, 1]) * sy
    result[0, 2] = (float(intrinsic[0, 2]) - record["pad_left"]) * sx
    result[1, 2] = (float(intrinsic[1, 2]) - record["pad_top"]) * sy
    return result


def build_model_inputs(
    candidate_dir: Path,
    selected_names: list[str],
    output_rgb: Path,
    mask_mode: str,
    target: int = 518,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    tensors = []
    valid_masks = []
    dynamic_masks = []
    records = []
    output_rgb.mkdir(parents=True, exist_ok=True)
    for name in selected_names:
        rgb_path = candidate_dir / "rgb" / name
        dynamic_path = candidate_dir / "masks_dynamic" / f"{Path(name).stem}.png"
        with Image.open(rgb_path) as opened:
            rgb = opened.convert("RGB")
        with Image.open(dynamic_path) as opened:
            dynamic_original = opened.convert("L")
        if rgb.size != dynamic_original.size:
            raise ValueError(f"RGB/mask dimensions differ for {name}")
        width, height = rgb.size
        record = padded_shape(width, height, target=target)
        record.update(
            {
                "source_name": name,
                "output_name": f"{Path(name).stem}.png",
                "dynamic_fraction": float(np.mean(np.asarray(dynamic_original) > 0)),
            }
        )
        rgba = rgb.convert("RGBA")
        rgba.putalpha(Image.eval(dynamic_original, lambda value: 255 - value))
        rgba.save(output_rgb / record["output_name"], compress_level=3)

        inference_rgb = np.array(rgb, dtype=np.uint8, copy=True)
        original_dynamic_array = np.asarray(dynamic_original, dtype=np.uint8) > 0
        if mask_mode == "white":
            inference_rgb[original_dynamic_array] = 255
        elif mask_mode == "black":
            inference_rgb[original_dynamic_array] = 0
        elif mask_mode != "none":
            raise ValueError(f"Unsupported inference mask mode: {mask_mode}")
        resized_rgb = np.array(
            Image.fromarray(inference_rgb).resize(
                (record["resized_width"], record["resized_height"]), Image.Resampling.BICUBIC
            ),
            copy=True,
        )
        resized_dynamic = np.asarray(
            dynamic_original.resize(
                (record["resized_width"], record["resized_height"]), Image.Resampling.NEAREST
            )
        )
        if mask_mode == "white":
            resized_rgb[resized_dynamic > 0] = 255
        elif mask_mode == "black":
            resized_rgb[resized_dynamic > 0] = 0
        canvas = np.full((target, target, 3), 255, dtype=np.uint8)
        valid = np.zeros((target, target), dtype=bool)
        dynamic = np.zeros((target, target), dtype=bool)
        left = record["pad_left"]
        top = record["pad_top"]
        right = left + record["resized_width"]
        bottom = top + record["resized_height"]
        canvas[top:bottom, left:right] = resized_rgb
        valid[top:bottom, left:right] = True
        dynamic[top:bottom, left:right] = resized_dynamic > 0
        tensors.append(torch.from_numpy(canvas.copy()).permute(2, 0, 1).float() / 255.0)
        valid_masks.append(valid)
        dynamic_masks.append(dynamic)
        records.append(record)
    return (
        torch.stack(tensors),
        np.stack(valid_masks),
        np.stack(dynamic_masks),
        records,
    )


def resolve_vggt_checkpoint(checkpoint: Path | None) -> Path:
    if checkpoint is None:
        from huggingface_hub import hf_hub_download

        checkpoint = Path(
            hf_hub_download(
                repo_id=VGGT_MODEL_REPO,
                filename=VGGT_MODEL_FILENAME,
                revision=VGGT_MODEL_REVISION,
            )
        )
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"VGGT checkpoint does not exist: {checkpoint}")
    if checkpoint.stat().st_size != VGGT_MODEL_SIZE:
        raise ValueError(
            f"VGGT checkpoint has {checkpoint.stat().st_size} bytes, expected {VGGT_MODEL_SIZE}"
        )
    return checkpoint


def load_vggt_model(vggt_root: Path, checkpoint: Path) -> tuple[Any, dict[str, Any]]:
    if not vggt_root.is_dir():
        vggt_root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "https://github.com/facebookresearch/vggt.git",
                str(vggt_root),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(vggt_root), "checkout", VGGT_CODE_REVISION], check=True
        )
    revision = subprocess.check_output(
        ["git", "-C", str(vggt_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != VGGT_CODE_REVISION:
        raise ValueError(f"VGGT code revision is {revision}, expected {VGGT_CODE_REVISION}")
    sys.path.insert(0, str(vggt_root))
    from vggt.models.vggt import VGGT

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = VGGT(enable_point=False, enable_track=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    del state
    if missing:
        raise ValueError(f"VGGT state dict missing keys: {missing[:10]}")
    allowed_prefixes = ("point_head.", "track_head.")
    invalid_unexpected = [key for key in unexpected if not key.startswith(allowed_prefixes)]
    if invalid_unexpected:
        raise ValueError(f"Unexpected VGGT state keys: {invalid_unexpected[:10]}")
    model.eval().to("cuda")
    return model, {
        "architecture": "VGGT-1B camera+depth heads",
        "repository": "https://github.com/facebookresearch/vggt",
        "code_revision": revision,
        "checkpoint_repository": VGGT_MODEL_REPO,
        "checkpoint_revision": VGGT_MODEL_REVISION,
        "checkpoint_filename": VGGT_MODEL_FILENAME,
        "checkpoint_path": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": VGGT_MODEL_SHA256,
        "checkpoint_license": VGGT_MODEL_LICENSE,
        "omega_upgrade": {
            "repository": "https://github.com/facebookresearch/vggt-omega",
            "code_revision_reviewed": VGGT_OMEGA_CODE_REVISION,
            "checkpoint_status": "gated_not_available_without_approved_huggingface_access",
        },
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "dtype": "bfloat16",
    }


def run_vggt_inference(model: Any, images: torch.Tensor) -> dict[str, Any]:
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    images_cuda = images.to("cuda", non_blocking=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        predictions = model(images_cuda)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], predictions["images"].shape[-2:]
    )
    depth = predictions["depth"].squeeze(0).float().cpu().numpy()
    confidence = predictions["depth_conf"].squeeze(0).float().cpu().numpy()
    extrinsic = extrinsic.squeeze(0).float().cpu().numpy()
    intrinsic = intrinsic.squeeze(0).float().cpu().numpy()
    points = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)
    peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    del predictions, images_cuda
    torch.cuda.empty_cache()
    return {
        "depth": depth,
        "confidence": confidence,
        "extrinsic": extrinsic,
        "intrinsic": intrinsic,
        "world_points": points,
        "elapsed_seconds": elapsed,
        "peak_memory_mib": peak,
    }


def cross_view_sparse_points(
    prediction: dict[str, Any],
    valid_masks: np.ndarray,
    dynamic_masks: np.ndarray,
    records: list[dict[str, Any]],
    candidate_dir: Path,
    confidence_percentile: float,
    relative_depth_tolerance: float,
    minimum_support: int,
    prefilter_budget: int,
    maximum_points: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    depth = prediction["depth"][..., 0]
    confidence = prediction["confidence"]
    extrinsic = prediction["extrinsic"]
    intrinsic = prediction["intrinsic"]
    world_points = prediction["world_points"]
    scene_count, height, width = depth.shape
    base_mask = (
        valid_masks
        & ~dynamic_masks
        & np.isfinite(depth)
        & np.isfinite(confidence)
        & (depth > 0)
    )
    confidence_values = confidence[base_mask]
    if confidence_values.size == 0:
        raise ValueError("No finite static depth predictions")
    threshold = float(np.percentile(confidence_values, confidence_percentile))
    candidate_mask = base_mask & (confidence >= threshold)
    flat_indices = np.flatnonzero(candidate_mask)
    candidate_count_before_budget = int(flat_indices.size)
    rng = np.random.default_rng(seed)
    if flat_indices.size > prefilter_budget:
        flat_indices = rng.choice(flat_indices, prefilter_budget, replace=False)
    source, y_model, x_model = np.unravel_index(flat_indices, candidate_mask.shape)
    points = world_points[source, y_model, x_model].astype(np.float64, copy=False)
    confidence_selected = confidence.ravel()[flat_indices].astype(np.float32, copy=False)
    support = np.zeros(len(points), dtype=np.int16)
    best_relative_residual = np.full(len(points), np.inf, dtype=np.float32)
    target_confidence_threshold = float(np.percentile(confidence_values, 50))

    for target in range(scene_count):
        camera_points = points @ extrinsic[target, :3, :3].T + extrinsic[target, :3, 3]
        z_camera = camera_points[:, 2]
        denominator = np.maximum(z_camera, 1e-8)
        projected_x = intrinsic[target, 0, 0] * camera_points[:, 0] / denominator + intrinsic[target, 0, 2]
        projected_y = intrinsic[target, 1, 1] * camera_points[:, 1] / denominator + intrinsic[target, 1, 2]
        finite_projection = np.isfinite(projected_x) & np.isfinite(projected_y)
        x_nearest = np.rint(
            np.where(finite_projection, np.clip(projected_x, -1, width), -1)
        ).astype(np.int32)
        y_nearest = np.rint(
            np.where(finite_projection, np.clip(projected_y, -1, height), -1)
        ).astype(np.int32)
        inside = (
            finite_projection
            & (z_camera > 0)
            & (x_nearest >= 0)
            & (x_nearest < width)
            & (y_nearest >= 0)
            & (y_nearest < height)
            & (source != target)
        )
        indices = np.flatnonzero(inside)
        if indices.size == 0:
            continue
        target_static = (
            valid_masks[target, y_nearest[indices], x_nearest[indices]]
            & ~dynamic_masks[target, y_nearest[indices], x_nearest[indices]]
            & (
                confidence[target, y_nearest[indices], x_nearest[indices]]
                >= target_confidence_threshold
            )
        )
        indices = indices[target_static]
        if indices.size == 0:
            continue
        target_depth = depth[target, y_nearest[indices], x_nearest[indices]]
        relative = np.abs(z_camera[indices] - target_depth) / np.maximum(
            np.maximum(z_camera[indices], target_depth), 1e-6
        )
        consistent = relative <= relative_depth_tolerance
        support[indices[consistent]] += 1
        best_relative_residual[indices] = np.minimum(
            best_relative_residual[indices], relative.astype(np.float32)
        )

    verified = support >= minimum_support
    verified_count = int(np.count_nonzero(verified))
    fallback_to_unverified = verified_count < min(5_000, maximum_points)
    eligible = verified if not fallback_to_unverified else np.ones(len(points), dtype=bool)
    eligible_indices = np.flatnonzero(eligible)
    quality_order = eligible_indices[
        np.lexsort(
            (
                eligible_indices,
                -confidence_selected[eligible_indices],
                -support[eligible_indices],
            )
        )
    ]
    grid_width = math.ceil(width / 4)
    grid_keys = (
        (source[quality_order].astype(np.int64) * math.ceil(height / 4) + y_model[quality_order] // 4)
        * grid_width
        + x_model[quality_order] // 4
    )
    _, unique_positions = np.unique(grid_keys, return_index=True)
    spatial_order = quality_order[np.sort(unique_positions)]
    per_frame_budget = max(1, maximum_points // scene_count)
    selected_parts = []
    for frame_index in range(scene_count):
        frame_choices = spatial_order[source[spatial_order] == frame_index]
        selected_parts.append(frame_choices[:per_frame_budget])
    selected_indices = np.concatenate(selected_parts) if selected_parts else np.empty(0, dtype=np.int64)
    if selected_indices.size < maximum_points:
        already = np.zeros(len(points), dtype=bool)
        already[selected_indices] = True
        fill = quality_order[~already[quality_order]][: maximum_points - selected_indices.size]
        selected_indices = np.concatenate([selected_indices, fill])
    if selected_indices.size > maximum_points:
        selected_indices = selected_indices[:maximum_points]
    selected_indices = selected_indices.astype(np.int64, copy=False)

    final_source = source[selected_indices].astype(np.int32)
    final_x_model = x_model[selected_indices].astype(np.float32)
    final_y_model = y_model[selected_indices].astype(np.float32)
    final_x_original = np.empty(len(selected_indices), dtype=np.float32)
    final_y_original = np.empty(len(selected_indices), dtype=np.float32)
    colors = np.empty((len(selected_indices), 3), dtype=np.uint8)
    for frame_index, record in enumerate(records):
        frame_positions = np.flatnonzero(final_source == frame_index)
        if frame_positions.size == 0:
            continue
        original_x, original_y = model_xy_to_original(
            final_x_model[frame_positions], final_y_model[frame_positions], record
        )
        original_x = np.clip(original_x, 0, record["original_width"] - 1)
        original_y = np.clip(original_y, 0, record["original_height"] - 1)
        final_x_original[frame_positions] = original_x
        final_y_original[frame_positions] = original_y
        with Image.open(candidate_dir / "rgb" / record["source_name"]) as opened:
            original_rgb = np.asarray(opened.convert("RGB"))
        colors[frame_positions] = original_rgb[
            np.rint(original_y).astype(np.int32), np.rint(original_x).astype(np.int32)
        ]

    selected_verified = verified[selected_indices]
    selected_best = best_relative_residual[selected_indices]
    selected_best[~np.isfinite(selected_best)] = np.nan
    result = {
        "points_world": points[selected_indices].astype(np.float64),
        "colors": colors,
        "source_frame": final_source,
        "x_original": final_x_original,
        "y_original": final_y_original,
        "x_model": final_x_model,
        "y_model": final_y_model,
        "confidence": confidence_selected[selected_indices],
        "support": support[selected_indices].astype(np.uint16),
        "best_relative_depth_residual": selected_best,
        "verified": selected_verified,
    }
    metrics = {
        "confidence_percentile": confidence_percentile,
        "confidence_threshold": threshold,
        "confidence": percentile_summary(confidence_values),
        "static_valid_pixels": int(np.count_nonzero(base_mask)),
        "confidence_candidate_pixels": candidate_count_before_budget,
        "prefiltered_points": int(len(points)),
        "cross_view_verified_points": verified_count,
        "cross_view_verified_fraction": verified_count / max(len(points), 1),
        "relative_depth_tolerance": relative_depth_tolerance,
        "minimum_other_view_support": minimum_support,
        "fallback_includes_unverified_points": fallback_to_unverified,
        "exported_points": int(len(selected_indices)),
        "exported_verified_points": int(np.count_nonzero(selected_verified)),
        "exported_support": percentile_summary(result["support"]),
        "exported_best_relative_depth_residual": percentile_summary(
            result["best_relative_depth_residual"]
        ),
        "exported_points_per_source_frame": [
            int(np.count_nonzero(final_source == frame_index)) for frame_index in range(scene_count)
        ],
    }
    return result, metrics


def align_z_up_and_proxy_scale(
    points: np.ndarray,
    extrinsic: np.ndarray,
    proxy_camera_height_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rotations = extrinsic[:, :3, :3].astype(np.float64)
    translations = extrinsic[:, :3, 3].astype(np.float64)
    camera_centers = -np.einsum("sji,sj->si", rotations, translations)
    camera_to_world_rotations = np.transpose(rotations, (0, 2, 1))
    up = -np.mean(camera_to_world_rotations[:, :, 1], axis=0)
    up /= max(np.linalg.norm(up), 1e-12)
    forward = np.mean(camera_to_world_rotations[:, :, 2], axis=0)
    forward = forward - up * np.dot(forward, up)
    if np.linalg.norm(forward) < 1e-6:
        centered = camera_centers - np.mean(camera_centers, axis=0)
        _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
        forward = right_vectors[0] - up * np.dot(right_vectors[0], up)
    forward /= max(np.linalg.norm(forward), 1e-12)
    right = np.cross(forward, up)
    right /= max(np.linalg.norm(right), 1e-12)
    forward = np.cross(up, right)
    forward /= max(np.linalg.norm(forward), 1e-12)
    world_to_aligned_rotation = np.stack([right, forward, up])

    rotated_points = points @ world_to_aligned_rotation.T
    rotated_centers = camera_centers @ world_to_aligned_rotation.T
    robust_point_z = np.percentile(rotated_points[:, 2], [1, 2, 50, 98, 99])
    floor_z = float(robust_point_z[1])
    unscaled_camera_height = float(np.median(rotated_centers[:, 2]) - floor_z)
    robust_vertical_span = float(robust_point_z[3] - robust_point_z[1])
    if (
        unscaled_camera_height > max(1e-6, 0.03 * robust_vertical_span)
        and unscaled_camera_height < max(1e-6, 2.0 * robust_vertical_span)
    ):
        scale = proxy_camera_height_m / unscaled_camera_height
        scale_method = "median_camera_height_above_point_z_p02"
    else:
        scale = 2.7 / max(robust_vertical_span, 1e-6)
        scale_method = "fallback_robust_vertical_span_to_2.7m"
    origin_rotated = np.array(
        [np.median(rotated_centers[:, 0]), np.median(rotated_centers[:, 1]), floor_z],
        dtype=np.float64,
    )
    aligned_points = scale * (rotated_points - origin_rotated)

    aligned_extrinsic = np.empty_like(extrinsic, dtype=np.float64)
    for index, (rotation, translation) in enumerate(zip(rotations, translations)):
        aligned_extrinsic[index, :3, :3] = rotation @ world_to_aligned_rotation.T
        old_origin = world_to_aligned_rotation.T @ origin_rotated
        aligned_extrinsic[index, :3, 3] = scale * (rotation @ old_origin + translation)
    aligned_centers = -np.einsum(
        "sji,sj->si", aligned_extrinsic[:, :3, :3], aligned_extrinsic[:, :3, 3]
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * world_to_aligned_rotation
    transform[:3, 3] = -scale * origin_rotated
    return aligned_points, aligned_extrinsic, {
        "gravity_method": "negative_mean_predicted_camera_y_axis",
        "world_to_z_up_rotation": world_to_aligned_rotation,
        "proxy_scale_method": scale_method,
        "proxy_camera_height_m": proxy_camera_height_m,
        "unscaled_camera_height": unscaled_camera_height,
        "scale_factor": scale,
        "floor_z_before_scale": floor_z,
        "origin_in_rotated_frame": origin_rotated,
        "old_world_to_proxy_metric_z_up": transform,
        "camera_centers_proxy_m": aligned_centers,
        "point_bounds_proxy_m": [aligned_points.min(axis=0), aligned_points.max(axis=0)],
        "point_percentiles_proxy_m": {
            "p01": np.percentile(aligned_points, 1, axis=0),
            "p50": np.percentile(aligned_points, 50, axis=0),
            "p99": np.percentile(aligned_points, 99, axis=0),
        },
    }


def rotation_to_qvec(rotation: np.ndarray) -> np.ndarray:
    quaternion_xyzw = Rotation.from_matrix(rotation).as_quat()
    qvec = np.array(
        [quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1], quaternion_xyzw[2]],
        dtype=np.float64,
    )
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def write_binary_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    support: np.ndarray,
    source_frame: np.ndarray,
) -> None:
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("confidence", "<f4"),
            ("support", "<u2"),
            ("source_frame", "<u2"),
        ]
    )
    vertices = np.empty(len(points), dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = points.T.astype(np.float32)
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    vertices["confidence"] = confidence.astype(np.float32)
    vertices["support"] = support.astype(np.uint16)
    vertices["source_frame"] = source_frame.astype(np.uint16)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property float confidence\nproperty ushort support\nproperty ushort source_frame\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        vertices.tofile(handle)


def write_colmap_text(
    directory: Path,
    points: np.ndarray,
    point_data: dict[str, np.ndarray],
    extrinsic: np.ndarray,
    intrinsics_original: np.ndarray,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    source_frame = point_data["source_frame"]
    observations: list[list[int]] = [[] for _ in records]
    for point_index, frame_index in enumerate(source_frame):
        observations[int(frame_index)].append(point_index)
    point2d_index = np.empty(len(points), dtype=np.int64)
    for frame_index, point_indices in enumerate(observations):
        for index_in_image, point_index in enumerate(point_indices):
            point2d_index[point_index] = index_in_image

    camera_lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(records)}",
    ]
    image_lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(records)}, mean observations per image: {len(points) / max(len(records), 1):.6f}",
    ]
    for frame_index, record in enumerate(records):
        camera_id = frame_index + 1
        image_id = frame_index + 1
        intrinsic = intrinsics_original[frame_index]
        camera_lines.append(
            f"{camera_id} PINHOLE {record['original_width']} {record['original_height']} "
            f"{intrinsic[0, 0]:.17g} {intrinsic[1, 1]:.17g} "
            f"{intrinsic[0, 2]:.17g} {intrinsic[1, 2]:.17g}"
        )
        qvec = rotation_to_qvec(extrinsic[frame_index, :3, :3])
        translation = extrinsic[frame_index, :3, 3]
        image_lines.append(
            f"{image_id} "
            + " ".join(f"{value:.17g}" for value in qvec)
            + " "
            + " ".join(f"{value:.17g}" for value in translation)
            + f" {camera_id} {record['output_name']}"
        )
        image_lines.append(
            " ".join(
                f"{point_data['x_original'][point_index]:.8f} "
                f"{point_data['y_original'][point_index]:.8f} {point_index + 1}"
                for point_index in observations[frame_index]
            )
        )

    point_lines = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
        f"# Number of points: {len(points)}, mean track length: 1",
    ]
    for point_index, point in enumerate(points):
        red, green, blue = point_data["colors"][point_index]
        image_id = int(source_frame[point_index]) + 1
        point_lines.append(
            f"{point_index + 1} {point[0]:.17g} {point[1]:.17g} {point[2]:.17g} "
            f"{int(red)} {int(green)} {int(blue)} 0 {image_id} {int(point2d_index[point_index])}"
        )
    (directory / "cameras.txt").write_text("\n".join(camera_lines) + "\n")
    (directory / "images.txt").write_text("\n".join(image_lines) + "\n")
    (directory / "points3D.txt").write_text("\n".join(point_lines) + "\n")

    reprojection_errors = []
    for point_index, point in enumerate(points):
        frame_index = int(source_frame[point_index])
        camera_point = extrinsic[frame_index, :3, :3] @ point + extrinsic[frame_index, :3, 3]
        intrinsic = intrinsics_original[frame_index]
        projected = np.array(
            [
                intrinsic[0, 0] * camera_point[0] / camera_point[2] + intrinsic[0, 2],
                intrinsic[1, 1] * camera_point[1] / camera_point[2] + intrinsic[1, 2],
            ]
        )
        observed = np.array(
            [point_data["x_original"][point_index], point_data["y_original"][point_index]]
        )
        reprojection_errors.append(float(np.linalg.norm(projected - observed)))
    return {
        "camera_count": len(records),
        "registered_image_count": len(records),
        "point_count": len(points),
        "synthetic_track_length": 1,
        "error_field_semantics": "zero_placeholder_not_measured_COLMAP_reprojection_error",
        "source_observation_reprojection_error_px": percentile_summary(reprojection_errors),
    }


def parse_colmap_reference_images(path: Path) -> dict[str, dict[str, np.ndarray]]:
    if not path.is_file():
        return {}
    result = {}
    lines = path.read_text().splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line or line.startswith("#"):
            index += 1
            continue
        parts = line.split(maxsplit=9)
        if len(parts) < 10:
            raise ValueError(f"Malformed COLMAP image header in {path}: {line}")
        qvec = np.array([float(value) for value in parts[1:5]])
        translation = np.array([float(value) for value in parts[5:8]])
        rotation = Rotation.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()
        result[parts[9]] = {
            "center": -rotation.T @ translation,
            "camera_to_world_rotation": rotation.T,
        }
        index += 2
    return result


def compare_camera_trajectory_to_colmap(
    predicted_extrinsic: np.ndarray,
    records: list[dict[str, Any]],
    reference_images_path: Path,
) -> dict[str, Any] | None:
    references = parse_colmap_reference_images(reference_images_path)
    matched = [
        (index, references[record["source_name"]])
        for index, record in enumerate(records)
        if record["source_name"] in references
    ]
    if len(matched) < 3:
        return None
    indices = [item[0] for item in matched]
    target_centers = np.stack([item[1]["center"] for item in matched])
    target_rotations = np.stack([item[1]["camera_to_world_rotation"] for item in matched])
    rotations = predicted_extrinsic[indices, :3, :3]
    translations = predicted_extrinsic[indices, :3, 3]
    predicted_centers = -np.einsum("sji,sj->si", rotations, translations)
    predicted_rotations = np.transpose(rotations, (0, 2, 1))

    source_mean = predicted_centers.mean(axis=0)
    target_mean = target_centers.mean(axis=0)
    source_centered = predicted_centers - source_mean
    target_centered = target_centers - target_mean
    u, singular, vt = np.linalg.svd(target_centered.T @ source_centered)
    diagonal = np.eye(3)
    diagonal[-1, -1] = np.sign(np.linalg.det(u @ vt))
    alignment_rotation = u @ diagonal @ vt
    alignment_scale = float(
        np.sum(singular * np.diag(diagonal)) / max(np.sum(source_centered**2), 1e-12)
    )
    alignment_translation = target_mean - alignment_scale * alignment_rotation @ source_mean
    aligned_centers = (
        alignment_scale * (alignment_rotation @ predicted_centers.T).T + alignment_translation
    )
    position_errors = np.linalg.norm(aligned_centers - target_centers, axis=1)
    target_baseline = float(np.linalg.norm(np.ptp(target_centers, axis=0)))
    rotation_errors = []
    for predicted_rotation, target_rotation in zip(predicted_rotations, target_rotations):
        residual = (alignment_rotation @ predicted_rotation).T @ target_rotation
        cosine = np.clip((np.trace(residual) - 1) / 2, -1, 1)
        rotation_errors.append(float(np.degrees(np.arccos(cosine))))
    return {
        "reference": "largest masked-COLMAP fragment",
        "matched_camera_count": len(matched),
        "sim3_scale_foundation_to_colmap": alignment_scale,
        "colmap_camera_baseline_diagonal": target_baseline,
        "camera_center_error_colmap_units": percentile_summary(position_errors),
        "camera_center_error_normalized_by_baseline": percentile_summary(
            position_errors / max(target_baseline, 1e-12)
        ),
        "camera_rotation_error_deg": percentile_summary(rotation_errors),
        "alignment_rotation": alignment_rotation,
        "alignment_translation": alignment_translation,
    }


def foundation_quality_gate(
    point_metrics: dict[str, Any],
    geometry: dict[str, Any],
    colmap_agreement: dict[str, Any] | None,
    intrinsics: np.ndarray,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    reasons = []
    warnings = []
    point_count = int(point_metrics["exported_points"])
    verified_fraction = float(point_metrics["cross_view_verified_fraction"])
    centers = np.asarray(geometry["camera_centers_proxy_m"])
    camera_baseline = float(np.linalg.norm(np.ptp(centers, axis=0)))
    point_depth_proxy = float(
        np.median(
            np.linalg.norm(
                np.asarray(geometry["point_percentiles_proxy_m"]["p50"])[None] - centers,
                axis=1,
            )
        )
    )
    baseline_depth = camera_baseline / max(point_depth_proxy, 1e-12)
    invalid_intrinsics = 0
    for intrinsic, record in zip(intrinsics, records):
        width, height = record["original_width"], record["original_height"]
        if (
            not np.isfinite(intrinsic).all()
            or intrinsic[0, 0] <= 0.2 * width
            or intrinsic[0, 0] >= 5.0 * width
            or intrinsic[1, 1] <= 0.2 * height
            or intrinsic[1, 1] >= 5.0 * height
            or not (0 <= intrinsic[0, 2] <= width)
            or not (0 <= intrinsic[1, 2] <= height)
        ):
            invalid_intrinsics += 1
    reference_p90 = None
    reference_rotation_p90 = None
    reference_camera_count = 0
    if colmap_agreement is not None:
        reference_p90 = colmap_agreement["camera_center_error_normalized_by_baseline"]["p90"]
        reference_rotation_p90 = colmap_agreement["camera_rotation_error_deg"]["p90"]
        reference_camera_count = int(colmap_agreement["matched_camera_count"])
    if point_count < 500:
        reasons.append("fewer_than_500_foundation_points")
    if invalid_intrinsics:
        reasons.append("implausible_predicted_intrinsics")
    if verified_fraction < 0.1:
        reasons.append("cross_view_verified_fraction_below_0.10")
    if baseline_depth < 0.03:
        reasons.append("predicted_camera_baseline_too_small")
    if point_metrics["fallback_includes_unverified_points"]:
        warnings.append("export_contains_cross_view_unverified_points")
    if verified_fraction < 0.5:
        warnings.append("cross_view_verified_fraction_below_0.50")
    if baseline_depth < 0.1:
        warnings.append("weak_predicted_viewpoint_baseline")
    if colmap_agreement is not None and reference_camera_count < 8:
        warnings.append("fewer_than_8_colmap_anchor_cameras")
    if reference_p90 is not None and reference_p90 > 0.25:
        warnings.append("foundation_camera_centers_disagree_with_colmap_fragment")
    if reference_rotation_p90 is not None and reference_rotation_p90 > 25:
        warnings.append("foundation_camera_rotations_disagree_with_colmap_fragment")

    strong_reference = (
        reference_camera_count >= 8
        and reference_p90 is not None
        and reference_p90 <= 0.25
        and reference_rotation_p90 is not None
        and reference_rotation_p90 <= 25
    )
    usable_reference_for_b = (
        colmap_agreement is None
        or reference_camera_count < 5
        or (
            reference_p90 is not None
            and reference_p90 <= 0.4
            and reference_rotation_p90 is not None
            and reference_rotation_p90 <= 45
        )
    )
    if reasons:
        grade = "P-F"
        decision = "fail"
    elif (
        point_count >= 20_000
        and verified_fraction >= 0.7
        and baseline_depth >= 0.15
        and strong_reference
        and not point_metrics["fallback_includes_unverified_points"]
    ):
        grade = "P-A"
        decision = "foundation_pass"
    elif (
        point_count >= 5_000
        and verified_fraction >= 0.4
        and baseline_depth >= 0.08
        and usable_reference_for_b
    ):
        grade = "P-B"
        decision = "foundation_pass"
    else:
        grade = "P-C"
        decision = "marginal"
    return {
        "protocol": "foundation-pseudo-geometry-v2",
        "grade": grade,
        "decision": decision,
        "reasons": reasons,
        "warnings": warnings,
        "camera_baseline_proxy_m": camera_baseline,
        "baseline_to_proxy_scene_depth": baseline_depth,
        "invalid_intrinsics_count": invalid_intrinsics,
        "scope": "pseudo_geometry_for_initialization_and_diagnostics_not_ground_truth",
    }


def make_selected_contact(
    candidate_dir: Path, records: list[dict[str, Any]], output_path: Path
) -> None:
    columns = 4
    tile_width, tile_height = 320, 200
    rows = math.ceil(len(records) / columns)
    canvas = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, record in enumerate(records):
        with Image.open(candidate_dir / "rgb" / record["source_name"]) as opened:
            image = opened.convert("RGB")
        with Image.open(
            candidate_dir / "masks_dynamic" / f"{Path(record['source_name']).stem}.png"
        ) as opened:
            dynamic = opened.convert("L")
        overlay = Image.new("RGB", image.size, (192, 45, 52))
        image = Image.composite(overlay, image, dynamic.point(lambda value: min(150, value)))
        image.thumbnail((tile_width, tile_height - 20), Image.Resampling.LANCZOS)
        left = (index % columns) * tile_width
        top = (index // columns) * tile_height
        canvas.paste(image, (left + (tile_width - image.width) // 2, top + 20))
        draw.rectangle((left, top, left + tile_width - 1, top + tile_height - 1), outline=(180, 185, 190))
        draw.text(
            (left + 6, top + 5),
            f"{index:02d} {record['source_name']} dyn={100 * record['dynamic_fraction']:.1f}%",
            fill=(25, 28, 31),
            font=font,
        )
    canvas.save(output_path, quality=90)


def make_projection_plot(
    points: np.ndarray,
    colors: np.ndarray,
    camera_centers: np.ndarray,
    output_path: Path,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(points))
    if len(indices) > 50_000:
        indices = rng.choice(indices, 50_000, replace=False)
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, (first, second, title) in zip(
        axes, ((0, 1, "top x-y"), (0, 2, "front x-z"), (1, 2, "side y-z"))
    ):
        axis.scatter(
            points[indices, first],
            points[indices, second],
            s=0.25,
            c=colors[indices].astype(np.float32) / 255.0,
            linewidths=0,
        )
        axis.plot(camera_centers[:, first], camera_centers[:, second], "o-", color="#c0392b", ms=3, lw=1)
        for index, center in enumerate(camera_centers):
            axis.text(center[first], center[second], str(index), fontsize=6, color="#8e241b")
        axis.set_aspect("equal")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_predictions(
    path: Path,
    prediction: dict[str, Any],
    records: list[dict[str, Any]],
    valid_masks: np.ndarray,
    dynamic_masks: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        names=np.asarray([record["source_name"] for record in records]),
        depth=prediction["depth"].astype(np.float16),
        confidence=prediction["confidence"].astype(np.float16),
        extrinsic=prediction["extrinsic"].astype(np.float32),
        intrinsic=prediction["intrinsic"].astype(np.float32),
        valid_mask=np.packbits(valid_masks.reshape(len(records), -1), axis=1),
        dynamic_mask=np.packbits(dynamic_masks.reshape(len(records), -1), axis=1),
        mask_shape=np.asarray(valid_masks.shape, dtype=np.int32),
    )


def run_candidate(
    candidate: dict[str, Any],
    model: Any,
    model_provenance: dict[str, Any],
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    identifier = candidate["candidate_id"]
    candidate_output = output_root / "candidates" / identifier
    status_path = candidate_output / "status.json"
    if status_path.is_file() and not args.force:
        previous = json.loads(status_path.read_text())
        if previous.get("status") == "completed":
            print(f"[{identifier}] foundation output already completed", flush=True)
            return previous
    if args.force and candidate_output.exists():
        shutil.rmtree(candidate_output)
    candidate_output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    write_json(
        status_path,
        {
            "candidate_id": identifier,
            "status": "running",
            "started_utc": utc_now(),
        },
    )
    try:
        selection = select_foundation_views(candidate, args.max_views)
        selected_names = selection["selected"]
        images, valid_masks, dynamic_masks, records = build_model_inputs(
            candidate["preproducts_dir"],
            selected_names,
            candidate_output / "rgb",
            args.inference_mask_mode,
        )
        make_selected_contact(
            candidate["preproducts_dir"], records, candidate_output / "selected_contact.jpg"
        )
        prediction = run_vggt_inference(model, images)
        save_predictions(
            candidate_output / "predictions.npz",
            prediction,
            records,
            valid_masks,
            dynamic_masks,
        )
        point_data, point_metrics = cross_view_sparse_points(
            prediction,
            valid_masks,
            dynamic_masks,
            records,
            candidate["preproducts_dir"],
            confidence_percentile=args.confidence_percentile,
            relative_depth_tolerance=args.relative_depth_tolerance,
            minimum_support=args.minimum_support,
            prefilter_budget=args.prefilter_budget,
            maximum_points=args.max_points,
            seed=candidate_seed(args.seed, identifier),
        )
        aligned_points, aligned_extrinsic, alignment = align_z_up_and_proxy_scale(
            point_data["points_world"],
            prediction["extrinsic"],
            args.proxy_camera_height,
        )
        intrinsics_original = np.stack(
            [
                model_intrinsics_to_original(intrinsic, record)
                for intrinsic, record in zip(prediction["intrinsic"], records)
            ]
        )
        colmap_metrics = write_colmap_text(
            candidate_output / "colmap_vggt",
            aligned_points,
            point_data,
            aligned_extrinsic,
            intrinsics_original,
            records,
        )
        write_binary_ply(
            candidate_output / "foundation_points.ply",
            aligned_points,
            point_data["colors"],
            point_data["confidence"],
            point_data["support"],
            point_data["source_frame"],
        )
        camera_centers = np.asarray(alignment["camera_centers_proxy_m"])
        make_projection_plot(
            aligned_points,
            point_data["colors"],
            camera_centers,
            candidate_output / "foundation_projections.png",
            candidate_seed(args.seed, identifier),
        )
        colmap_agreement = compare_camera_trajectory_to_colmap(
            prediction["extrinsic"],
            records,
            candidate["preproducts_dir"] / "colmap" / "images.txt",
        )
        quality_gate = foundation_quality_gate(
            point_metrics, alignment, colmap_agreement, intrinsics_original, records
        )
        manifest = {
            "schema": "genrecon.foundation-sfm-scene",
            "schema_version": SCHEMA_VERSION,
            "created_utc": utc_now(),
            "candidate_id": identifier,
            "title": candidate["title"],
            "source_preproducts": str(candidate["preproducts_dir"]),
            "source_preproducts_grade": candidate["preproducts_grade"],
            "model": model_provenance,
            "selection": selection,
            "inference": {
                "input_resolution": 518,
                "preprocess_mode": "preserve_aspect_pad_to_518_patch14",
                "dynamic_mask_mode": args.inference_mask_mode,
                "dynamic_points_exported": False,
                "elapsed_seconds": prediction["elapsed_seconds"],
                "peak_memory_mib": prediction["peak_memory_mib"],
            },
            "views": records,
            "point_filter": point_metrics,
            "alignment": alignment,
            "colmap_export": colmap_metrics,
            "colmap_fragment_agreement": colmap_agreement,
            "quality_gate": quality_gate,
            "limitations": [
                "Foundation depth and poses are predictions, not triangulated observations.",
                "Proxy metric scale uses assumed handheld camera height and is not auditable metric ground truth.",
                "ERROR=0 in points3D.txt is a compatibility placeholder, not measured reprojection error.",
                "Each exported point has one synthetic source observation; cross-view support is stored in PLY and manifest.",
            ],
        }
        write_json(candidate_output / "manifest.json", manifest)
        completed = {
            "candidate_id": identifier,
            "status": "completed",
            "completed_utc": utc_now(),
            "elapsed_seconds": time.monotonic() - start,
            "grade": quality_gate["grade"],
            "point_count": len(aligned_points),
            "view_count": len(records),
        }
        write_json(status_path, completed)
        print(
            f"[{identifier}] {quality_gate['grade']} views={len(records)} "
            f"points={len(aligned_points)} infer={prediction['elapsed_seconds']:.2f}s "
            f"peak={prediction['peak_memory_mib']:.0f}MiB",
            flush=True,
        )
        return completed
    except Exception as error:
        failed = {
            "candidate_id": identifier,
            "status": "failed",
            "failed_utc": utc_now(),
            "elapsed_seconds": time.monotonic() - start,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        write_json(status_path, failed)
        print(f"[{identifier}] FAILED: {type(error).__name__}: {error}", flush=True)
        if args.fail_fast:
            raise
        return failed


def read_exported_intrinsics(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[1] != "PINHOLE":
            raise ValueError(f"Expected PINHOLE export, found {parts[1]}")
        camera_id = int(parts[0])
        fx, fy, cx, cy = (float(value) for value in parts[4:8])
        intrinsic = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        rows.append((camera_id, intrinsic))
    rows.sort(key=lambda item: item[0])
    return np.stack([item[1] for item in rows])


def refresh_foundation_quality(
    candidate: dict[str, Any], output_root: Path
) -> dict[str, Any] | None:
    directory = output_root / "candidates" / candidate["candidate_id"]
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    intrinsics = read_exported_intrinsics(directory / "colmap_vggt" / "cameras.txt")
    gate = foundation_quality_gate(
        manifest["point_filter"],
        manifest["alignment"],
        manifest.get("colmap_fragment_agreement"),
        intrinsics,
        manifest["views"],
    )
    manifest["quality_gate"] = gate
    manifest["quality_refreshed_utc"] = utc_now()
    write_json(manifest_path, manifest)
    status_path = directory / "status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text())
        status["grade"] = gate["grade"]
        write_json(status_path, status)
    return gate


def run_genrecon_preflight(
    candidate: dict[str, Any], output_root: Path, args: argparse.Namespace
) -> dict[str, Any]:
    identifier = candidate["candidate_id"]
    scene_dir = output_root / "candidates" / identifier
    manifest_path = scene_dir / "manifest.json"
    if not manifest_path.is_file():
        return {"candidate_id": identifier, "status": "missing_foundation_output"}
    preflight_dir = scene_dir / "genrecon_preflight"
    status_path = preflight_dir / "preflight.json"
    if status_path.is_file() and not args.force_preflight:
        previous = json.loads(status_path.read_text())
        if previous.get("status") in {"passed", "marginal"}:
            print(
                f"[{identifier}] GenRecon preflight already {previous['status']}", flush=True
            )
            return previous
    if args.force_preflight and preflight_dir.exists():
        shutil.rmtree(preflight_dir)
    preflight_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    captured = io.StringIO()
    try:
        from inference.get_chunks import IphoneChunker
        from inference.get_images import IphoneImageSelecter

        with contextlib.redirect_stdout(captured):
            centers, original_to_chunks, _, _ = IphoneChunker(
                colmap_subdir="colmap_vggt"
            ).get_chunks(scene_dir, preflight_dir)
            selected = IphoneImageSelecter(center_crop=False).get_images(
                original_to_chunks,
                scene_dir / "colmap_vggt" / "cameras.txt",
                min(args.genrecon_views, len(json.loads(manifest_path.read_text())["views"])),
                preflight_dir,
                seed=args.seed,
            )
        log = captured.getvalue()
        (preflight_dir / "preflight.log").write_text(log)
        fallback_count = log.count("falling back to closest camera")
        import trimesh

        clean_cloud = trimesh.load(preflight_dir / "clean_points.ply", process=False)
        clean_point_count = int(len(clean_cloud.vertices))
        usable = bool(centers) and clean_point_count >= 500
        preflight_status = (
            "passed" if usable and fallback_count == 0 else "marginal" if usable else "failed"
        )
        result = {
            "schema": "genrecon.foundation-sfm-preflight",
            "schema_version": SCHEMA_VERSION,
            "candidate_id": identifier,
            "status": preflight_status,
            "created_utc": utc_now(),
            "elapsed_seconds": time.monotonic() - start,
            "colmap_subdir": "colmap_vggt",
            "clean_point_count": clean_point_count,
            "chunk_count": len(centers),
            "scene_image_crop_count": int(selected.scene_images_512.shape[0]),
            "condition_view_count": len(selected.cond2d_images_512),
            "condition_chunk_indices": selected.chunk_indices,
            "closest_camera_fallback_count": fallback_count,
            "alpha_mask_zero_fraction_512": float(
                np.mean(selected.scene_images_512.numpy() == 0)
            ),
        }
        if result["status"] == "marginal":
            result["warning"] = "One or more chunks used closest-camera fallback"
        elif result["status"] == "failed":
            result["error"] = "No usable chunks or fewer than 500 clean points"
        write_json(status_path, result)
        print(
            f"[{identifier}] preflight={result['status']} clean={clean_point_count} "
            f"chunks={len(centers)} fallback={fallback_count}",
            flush=True,
        )
        return result
    except Exception as error:
        log = captured.getvalue()
        (preflight_dir / "preflight.log").write_text(log)
        result = {
            "schema": "genrecon.foundation-sfm-preflight",
            "schema_version": SCHEMA_VERSION,
            "candidate_id": identifier,
            "status": "failed",
            "created_utc": utc_now(),
            "elapsed_seconds": time.monotonic() - start,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        write_json(status_path, result)
        print(f"[{identifier}] preflight FAILED: {type(error).__name__}: {error}", flush=True)
        if args.fail_fast:
            raise
        return result


def summary_rows(candidates: list[dict[str, Any]], output_root: Path) -> list[dict[str, Any]]:
    review_path = output_root / "visual_review.json"
    review_document = json.loads(review_path.read_text()) if review_path.is_file() else {}
    review_by_id = {
        item["candidate_id"]: item for item in review_document.get("candidates", [])
    }
    rows = []
    for candidate in candidates:
        directory = output_root / "candidates" / candidate["candidate_id"]
        status = json.loads((directory / "status.json").read_text()) if (directory / "status.json").is_file() else {}
        manifest = json.loads((directory / "manifest.json").read_text()) if (directory / "manifest.json").is_file() else {}
        preflight = (
            json.loads((directory / "genrecon_preflight" / "preflight.json").read_text())
            if (directory / "genrecon_preflight" / "preflight.json").is_file()
            else {}
        )
        gate = manifest.get("quality_gate", {})
        agreement = manifest.get("colmap_fragment_agreement") or {}
        review = review_by_id.get(candidate["candidate_id"], {})
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "title": candidate["title"],
                "source_grade": candidate["preproducts_grade"],
                "status": status.get("status", "missing"),
                "foundation_grade": gate.get("grade", "-"),
                "views": status.get("view_count"),
                "points": status.get("point_count"),
                "verified_fraction": (manifest.get("point_filter") or {}).get("cross_view_verified_fraction"),
                "baseline_depth": gate.get("baseline_to_proxy_scene_depth"),
                "colmap_pose_p90": (agreement.get("camera_center_error_normalized_by_baseline") or {}).get("p90"),
                "preflight": preflight.get("status", "missing"),
                "clean_points": preflight.get("clean_point_count"),
                "chunks": preflight.get("chunk_count"),
                "fallback": preflight.get("closest_camera_fallback_count"),
                "visual_disposition": review.get("disposition", "not_reviewed"),
                "visual_note": review.get("reason", ""),
                "warnings": ", ".join(gate.get("reasons", []) + gate.get("warnings", [])),
            }
        )
    return rows


def write_overview(rows: list[dict[str, Any]], output_root: Path) -> None:
    columns = 4
    tile_width, tile_height = 360, 360
    row_count = math.ceil(len(rows) / columns)
    canvas = Image.new(
        "RGB", (columns * tile_width, row_count * tile_height), (238, 241, 242)
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    grade_colors = {
        "P-A": (24, 130, 92),
        "P-B": (24, 130, 92),
        "P-C": (173, 106, 18),
        "P-F": (186, 61, 66),
    }
    for index, row in enumerate(rows):
        left = (index % columns) * tile_width
        top = (index // columns) * tile_height
        draw.rectangle(
            (left, top, left + tile_width - 1, top + tile_height - 1),
            fill="white",
            outline=(204, 210, 213),
        )
        draw.rectangle(
            (left, top, left + tile_width - 1, top + 5),
            fill=grade_colors.get(row["foundation_grade"], (90, 90, 90)),
        )
        draw.text(
            (left + 8, top + 10),
            f"{row['candidate_id']}  {row['foundation_grade']}  pre={row['preflight']}",
            fill=(32, 37, 42),
            font=font,
        )
        draw.text(
            (left + 8, top + 26), row["title"][:50], fill=(32, 37, 42), font=font
        )
        for image_index, (filename, maximum_height) in enumerate(
            (("selected_contact.jpg", 180), ("foundation_projections.png", 125))
        ):
            path = output_root / "candidates" / row["candidate_id"] / filename
            with Image.open(path) as opened:
                preview = opened.convert("RGB")
            preview.thumbnail((tile_width - 12, maximum_height), Image.Resampling.LANCZOS)
            preview_top = top + 45 + (0 if image_index == 0 else 185)
            canvas.paste(preview, (left + (tile_width - preview.width) // 2, preview_top))
        verified = row["verified_fraction"]
        draw.text(
            (left + 8, top + 338),
            f"pts={row['points']} xview={verified:.2f} chunks={row['chunks']}/{row['fallback']}",
            fill=(95, 104, 112),
            font=font,
        )
    canvas.save(output_root / "overview.jpg", quality=90)


def capture_review_screenshots(output_root: Path) -> None:
    browser = next(
        (
            path
            for executable in ("google-chrome", "chromium", "chromium-browser")
            if (path := shutil.which(executable)) is not None
        ),
        None,
    )
    if browser is None:
        raise RuntimeError("No headless Chrome/Chromium executable found for review screenshots")
    page_url = (output_root / "index.html").resolve().as_uri()
    for filename, width, height in (
        ("index_desktop.png", 1440, 1200),
        ("index_mobile.png", 390, 844),
    ):
        subprocess.run(
            [
                browser,
                "--headless",
                "--no-sandbox",
                "--disable-gpu",
                "--allow-file-access-from-files",
                "--hide-scrollbars",
                f"--window-size={width},{height}",
                "--virtual-time-budget=8000",
                f"--screenshot={output_root / filename}",
                page_url,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def stage_visual_review(
    candidates: list[dict[str, Any]], output_root: Path, source_path: Path
) -> None:
    if not source_path.is_file():
        raise FileNotFoundError(f"Visual review source does not exist: {source_path}")
    source = json.loads(source_path.read_text())
    requested = {candidate["candidate_id"] for candidate in candidates}
    entries = [
        entry for entry in source.get("candidates", []) if entry["candidate_id"] in requested
    ]
    if {entry["candidate_id"] for entry in entries} != requested:
        missing = sorted(requested - {entry["candidate_id"] for entry in entries})
        raise ValueError(f"Visual review is missing candidates: {missing}")
    staged = dict(source)
    staged["candidates"] = entries
    staged["summary"] = dict(sorted(Counter(entry["disposition"] for entry in entries).items()))
    write_json(output_root / "visual_review.json", staged)


def write_summary(
    candidates: list[dict[str, Any]], output_root: Path, visual_review_path: Path
) -> dict[str, Any]:
    stage_visual_review(candidates, output_root, visual_review_path)
    rows = summary_rows(candidates, output_root)
    grades = Counter(row["foundation_grade"] for row in rows)
    preflight = Counter(row["preflight"] for row in rows)
    summary = {
        "schema": "genrecon.foundation-sfm-index",
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "summary": {
            "candidate_count": len(rows),
            "completed_count": sum(row["status"] == "completed" for row in rows),
            "foundation_grades": dict(sorted(grades.items())),
            "preflight": dict(sorted(preflight.items())),
        },
        "candidates": rows,
    }
    write_json(output_root / "index.json", summary)
    fields = list(rows[0].keys()) if rows else []
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def metric(value: Any, suffix: str = "", digits: int = 2) -> str:
        if value is None:
            return "-"
        return f"{float(value):.{digits}f}{suffix}"

    cards = []
    for row in rows:
        identifier = row["candidate_id"]
        directory = f"candidates/{identifier}"
        cards.append(
            f'''<article class="card grade-{html.escape(row['foundation_grade'])}">
<header><div><code>{html.escape(identifier)}</code><h2>{html.escape(row['title'])}</h2></div><b>{html.escape(row['foundation_grade'])}</b></header>
<img src="{directory}/selected_contact.jpg" alt="selected masked views" loading="lazy">
<img src="{directory}/foundation_projections.png" alt="foundation point projections" loading="lazy">
<div class="metrics"><span>source<b>{html.escape(row['source_grade'])}</b></span><span>views<b>{row['views'] or '-'}</b></span><span>points<b>{row['points'] or '-'}</b></span><span>xview<b>{metric(None if row['verified_fraction'] is None else 100*row['verified_fraction'], '%', 1)}</b></span><span>baseline/depth<b>{metric(row['baseline_depth'])}</b></span><span>COLMAP pose p90<b>{metric(row['colmap_pose_p90'])}</b></span><span>preflight<b>{html.escape(row['preflight'])}</b></span><span>clean points<b>{row['clean_points'] or '-'}</b></span><span>chunks/fallback<b>{row['chunks'] or '-'}/{row['fallback'] if row['fallback'] is not None else '-'}</b></span></div>
<p><b>visual: {html.escape(row['visual_disposition'])}</b><br>{html.escape(row['visual_note'])}<br>{html.escape(row['warnings'] or 'No foundation gate warning')}</p>
<nav><a href="{directory}/manifest.json">manifest</a><a href="{directory}/foundation_points.ply">PLY</a><a href="{directory}/colmap_vggt/">COLMAP</a><a href="{directory}/genrecon_preflight/cameras.json">GenRecon cameras</a></nav>
</article>'''
        )
    page = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Foundation SfM fallback</title><style>
:root{{--ink:#252a30;--muted:#68717b;--line:#d5dadd;--bg:#f2f4f4;--pass:#18825c;--marginal:#ad6a12;--fail:#ba3d42}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px Arial,sans-serif;letter-spacing:0}}body>header{{padding:22px;background:white;border-bottom:1px solid var(--line)}}h1{{margin:0 0 8px;font-size:25px}}body>header p{{margin:4px 0;color:var(--muted)}}main{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;padding:18px}}.card{{background:white;border:1px solid var(--line);border-top:4px solid var(--fail);border-radius:6px;overflow:hidden}}.grade-P-A,.grade-P-B{{border-top-color:var(--pass)}}.grade-P-C{{border-top-color:var(--marginal)}}.card header{{display:flex;justify-content:space-between;gap:12px;padding:13px}}code{{color:var(--muted)}}h2{{font-size:17px;margin:5px 0 0}}.card header>b{{flex:none;font-size:22px;white-space:nowrap}}.card>img{{display:block;width:100%;height:250px;object-fit:contain;background:#111;border-top:1px solid var(--line)}}.metrics{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));border-top:1px solid var(--line)}}.metrics span{{min-width:0;padding:8px;color:var(--muted);font-size:11px;border-right:1px solid var(--line);border-bottom:1px solid var(--line)}}.metrics b{{display:block;color:var(--ink);font-size:13px;overflow-wrap:anywhere}}.card p{{padding:10px 14px;margin:0;color:var(--muted);overflow-wrap:anywhere}}nav{{display:flex;gap:18px;flex-wrap:wrap;padding:0 14px 14px}}a{{color:#096b5a}}@media(max-width:800px){{main{{grid-template-columns:1fr;padding:10px}}.card>img{{height:210px}}.metrics{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
</style></head><body><header><h1>Foundation SfM fallback</h1><p>VGGT-1B pseudo geometry. P-A/P-B are initialization passes, not ground truth.</p><p>Completed {summary['summary']['completed_count']}/{len(rows)} | Grades {html.escape(str(dict(sorted(grades.items()))))} | Preflight {html.escape(str(dict(sorted(preflight.items()))))}</p></header><main>{''.join(cards)}</main></body></html>'''
    (output_root / "index.html").write_text(page)
    write_overview(rows, output_root)
    capture_review_screenshots(output_root)
    return summary


def validate_outputs(candidates: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    errors = []
    counts = {
        "candidates": 0,
        "rgba_images": 0,
        "alpha_pixels_checked": 0,
        "points": 0,
        "colmap_cameras": 0,
        "colmap_images": 0,
        "strict_json_files": 0,
        "html_cards": 0,
        "html_local_references": 0,
    }
    try:
        import pycolmap
    except ImportError:
        pycolmap = None
        errors.append("pycolmap unavailable; cannot independently load COLMAP text")

    for candidate in candidates:
        identifier = candidate["candidate_id"]
        directory = output_root / "candidates" / identifier
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            errors.append(f"{identifier}: missing manifest.json")
            continue
        counts["candidates"] += 1
        try:
            manifest = json.loads(manifest_path.read_text())
            expected_views = len(manifest["views"])
            rgba_paths = sorted((directory / "rgb").glob("*.png"))
            if len(rgba_paths) != expected_views:
                raise ValueError("RGBA view count differs from manifest")
            for view in manifest["views"]:
                rgba_path = directory / "rgb" / view["output_name"]
                dynamic_path = (
                    candidate["preproducts_dir"]
                    / "masks_dynamic"
                    / f"{Path(view['source_name']).stem}.png"
                )
                with Image.open(rgba_path) as opened:
                    if opened.mode != "RGBA":
                        raise ValueError(f"GenRecon RGB is not RGBA: {rgba_path}")
                    rgba = np.asarray(opened)
                with Image.open(dynamic_path) as opened:
                    dynamic = np.asarray(opened.convert("L"))
                if rgba.shape[:2] != dynamic.shape:
                    raise ValueError(f"RGBA/mask dimensions differ: {rgba_path}")
                if not np.array_equal(rgba[..., 3], 255 - dynamic):
                    raise ValueError(f"RGBA alpha is not inverse dynamic mask: {rgba_path}")
                counts["alpha_pixels_checked"] += int(dynamic.size)
            counts["rgba_images"] += len(rgba_paths)

            ply_path = directory / "foundation_points.ply"
            with ply_path.open("rb") as handle:
                header = b""
                while b"end_header\n" not in header:
                    block = handle.readline()
                    if not block:
                        raise ValueError("Truncated PLY header")
                    header += block
            match = re.search(rb"element vertex (\d+)", header)
            point_count = int(match.group(1)) if match else -1
            if point_count != manifest["colmap_export"]["point_count"]:
                raise ValueError("PLY point count differs from manifest")
            expected_payload_bytes = point_count * 23
            actual_payload_bytes = ply_path.stat().st_size - len(header)
            if actual_payload_bytes != expected_payload_bytes:
                raise ValueError(
                    f"PLY payload is {actual_payload_bytes} bytes, expected {expected_payload_bytes}"
                )
            counts["points"] += point_count

            if pycolmap is not None:
                reconstruction = pycolmap.Reconstruction(str(directory / "colmap_vggt"))
                if reconstruction.num_reg_images() != expected_views:
                    raise ValueError("COLMAP registered image count differs")
                if reconstruction.num_points3D() != point_count:
                    raise ValueError("COLMAP point count differs")
                for point in reconstruction.points3D.values():
                    if not np.isfinite(point.xyz).all():
                        raise ValueError("COLMAP contains non-finite point coordinates")
                for camera in reconstruction.cameras.values():
                    if camera.has_bogus_params(0.1, 10.0, 1.0) or not np.isfinite(camera.params).all():
                        raise ValueError("COLMAP contains invalid camera parameters")
                output_names = {view["output_name"] for view in manifest["views"]}
                if {image.name for image in reconstruction.images.values()} != output_names:
                    raise ValueError("COLMAP image names differ from manifest")
                counts["colmap_cameras"] += reconstruction.num_cameras()
                counts["colmap_images"] += reconstruction.num_reg_images()
            if manifest["colmap_export"]["source_observation_reprojection_error_px"]["max"] > 0.2:
                raise ValueError("source observation numerical reprojection residual exceeds 0.2 px")

            preflight_dir = directory / "genrecon_preflight"
            preflight = json.loads((preflight_dir / "preflight.json").read_text())
            preflight_status = preflight.get("status")
            if preflight_status not in {"passed", "marginal"}:
                raise ValueError("GenRecon consumer preflight did not produce usable chunks")
            if (
                manifest["quality_gate"]["decision"] == "foundation_pass"
                and preflight_status != "passed"
            ):
                raise ValueError("foundation pass has a non-passing GenRecon preflight")
            for artifact in (
                "clean_points.ply",
                "chunk_layout.png",
                "chunk_transforms.json",
                "cameras.json",
            ):
                if not (preflight_dir / artifact).is_file():
                    raise ValueError(f"GenRecon preflight artifact is missing: {artifact}")
        except Exception as error:
            errors.append(f"{identifier}: {type(error).__name__}: {error}")

    review_path = output_root / "visual_review.json"
    if not review_path.is_file():
        errors.append("visual_review.json is missing")
    else:
        review = json.loads(review_path.read_text())
        if len(review.get("candidates", [])) != len(candidates):
            errors.append("visual review candidate count differs")

    index_path = output_root / "index.html"
    if not index_path.is_file():
        errors.append("review index.html is missing")
    else:
        page = index_path.read_text()
        counts["html_cards"] = page.count('<article class="card')
        if counts["html_cards"] != len(candidates):
            errors.append("review index card count differs")
        for reference in re.findall(r'(?:href|src)="([^"]+)"', page):
            if reference.startswith(("http://", "https://", "#")):
                continue
            counts["html_local_references"] += 1
            if not (output_root / reference).exists():
                errors.append(f"review index local reference is missing: {reference}")
    screenshot_dimensions = {}
    for filename, expected_size in (
        ("index_desktop.png", (1440, 1200)),
        ("index_mobile.png", (390, 844)),
        ("overview.jpg", (1440, math.ceil(len(candidates) / 4) * 360)),
    ):
        path = output_root / filename
        if not path.is_file():
            errors.append(f"review image is missing: {filename}")
            continue
        with Image.open(path) as opened:
            screenshot_dimensions[filename] = list(opened.size)
            if opened.size != expected_size:
                errors.append(f"review image has wrong dimensions: {filename}")

    config = json.loads((output_root / "config.json").read_text())
    model_config = config.get("model") or {}
    checkpoint_path = Path(model_config.get("checkpoint_path", ""))
    checkpoint_hash = sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
    if checkpoint_hash != VGGT_MODEL_SHA256:
        errors.append("VGGT checkpoint SHA256 mismatch")
    if model_config.get("code_revision") != VGGT_CODE_REVISION:
        errors.append("VGGT code revision mismatch")

    for path in output_root.rglob("*.json"):
        try:
            json.loads(
                path.read_text(),
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite constant {value}")
                ),
            )
            counts["strict_json_files"] += 1
        except Exception as error:
            errors.append(f"{path}: strict JSON failure: {error}")
    result = {
        "schema": "genrecon.foundation-sfm-validation",
        "schema_version": SCHEMA_VERSION,
        "validated_utc": utc_now(),
        "result": "pass" if not errors else "fail",
        "counts": counts,
        "model_checkpoint": {
            "path": checkpoint_path,
            "sha256": checkpoint_hash,
            "matches_expected": checkpoint_hash == VGGT_MODEL_SHA256,
        },
        "checks": {
            "rgba_alpha_matches_dynamic_masks": not any(
                "RGBA" in error or "alpha" in error for error in errors
            ),
            "ply_payload_colmap_manifest_counts_match": not any(
                "PLY" in error or "point count" in error for error in errors
            ),
            "colmap_text_loads_and_is_finite": pycolmap is not None
            and not any("COLMAP" in error for error in errors),
            "genrecon_consumer_preflight": not any("preflight" in error for error in errors),
            "visual_review_complete": not any("visual review" in error for error in errors),
            "review_index_and_images": not any("review " in error for error in errors),
            "review_image_dimensions": screenshot_dimensions,
            "model_provenance": not any("VGGT" in error for error in errors),
            "strict_json": not any("strict JSON" in error for error in errors),
        },
        "errors": errors,
    }
    write_json(output_root / "validation.json", result)
    if errors:
        raise RuntimeError(f"Foundation output validation failed with {len(errors)} error(s)")
    return result


def selected_candidates(all_candidates: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.candidate:
        requested = set(args.candidate)
        selected = [item for item in all_candidates if item["candidate_id"] in requested]
        missing = sorted(requested - {item["candidate_id"] for item in selected})
        if missing:
            raise ValueError(f"Unknown candidate IDs: {missing}")
        return selected
    allowed = set(args.source_grades)
    return [item for item in all_candidates if item["preproducts_grade"] in allowed]


def write_run_config(
    output_root: Path,
    args: argparse.Namespace,
    candidates: list[dict[str, Any]],
    model_provenance: dict[str, Any] | None,
) -> None:
    write_json(
        output_root / "config.json",
        {
            "schema": "genrecon.foundation-sfm-run-config",
            "schema_version": SCHEMA_VERSION,
            "created_utc": utc_now(),
            "arguments": vars(args),
            "candidate_ids": [item["candidate_id"] for item in candidates],
            "model": model_provenance,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("run", "quality", "preflight", "summarize", "validate", "all")
    )
    parser.add_argument("--preproducts-root", type=Path, default=DEFAULT_PREPRODUCTS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--vggt-root", type=Path, default=DEFAULT_VGGT_ROOT)
    parser.add_argument("--visual-review", type=Path, default=DEFAULT_VISUAL_REVIEW)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--source-grades", nargs="+", default=["C", "F"])
    parser.add_argument("--max-views", type=int, default=16)
    parser.add_argument("--max-points", type=int, default=100_000)
    parser.add_argument("--prefilter-budget", type=int, default=250_000)
    parser.add_argument("--confidence-percentile", type=float, default=70.0)
    parser.add_argument("--relative-depth-tolerance", type=float, default=0.15)
    parser.add_argument("--minimum-support", type=int, default=1)
    parser.add_argument("--proxy-camera-height", type=float, default=1.6)
    parser.add_argument("--inference-mask-mode", choices=("white", "black", "none"), default="white")
    parser.add_argument("--genrecon-views", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-preflight", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.stage in {"run", "all"} and not torch.cuda.is_available():
        raise RuntimeError("VGGT inference requires CUDA")
    if args.max_views < 2:
        raise ValueError("--max-views must be at least 2")
    if args.max_points < 500:
        raise ValueError("--max-points must be at least 500")
    if args.prefilter_budget < args.max_points:
        raise ValueError("--prefilter-budget must be at least --max-points")
    if not 0 <= args.confidence_percentile < 100:
        raise ValueError("--confidence-percentile must be in [0, 100)")
    if not 0 < args.relative_depth_tolerance < 1:
        raise ValueError("--relative-depth-tolerance must be in (0, 1)")
    if args.minimum_support < 1:
        raise ValueError("--minimum-support must be positive")
    if args.proxy_camera_height <= 0:
        raise ValueError("--proxy-camera-height must be positive")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.preproducts_root = args.preproducts_root.resolve()
    args.output_root = args.output_root.resolve()
    args.vggt_root = args.vggt_root.resolve()
    args.visual_review = args.visual_review.resolve()
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.resolve()
    validate_arguments(args)
    _, all_candidates = load_preproducts(args.preproducts_root)
    candidates = selected_candidates(all_candidates, args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.stage in {"run", "all"}:
        checkpoint = resolve_vggt_checkpoint(args.checkpoint)
        actual_hash = sha256_file(checkpoint)
        if actual_hash != VGGT_MODEL_SHA256:
            raise ValueError(f"VGGT checkpoint SHA256 is {actual_hash}, expected {VGGT_MODEL_SHA256}")
        model, model_provenance = load_vggt_model(args.vggt_root, checkpoint)
        write_run_config(args.output_root, args, candidates, model_provenance)
        for candidate in candidates:
            run_candidate(candidate, model, model_provenance, args.output_root, args)
        del model
        torch.cuda.empty_cache()
    elif not (args.output_root / "config.json").is_file():
        write_run_config(args.output_root, args, candidates, None)

    if args.stage in {"quality", "all"}:
        for candidate in candidates:
            gate = refresh_foundation_quality(candidate, args.output_root)
            if gate is not None:
                print(
                    f"[{candidate['candidate_id']}] quality={gate['grade']} "
                    f"warnings={len(gate['warnings'])}",
                    flush=True,
                )
    if args.stage in {"preflight", "all"}:
        for candidate in candidates:
            run_genrecon_preflight(candidate, args.output_root, args)
    if args.stage in {"summarize", "all"}:
        summary = write_summary(candidates, args.output_root, args.visual_review)
        print(f"[summary] {summary['summary']}", flush=True)
    if args.stage in {"validate", "all"}:
        result = validate_outputs(candidates, args.output_root)
        print(f"[validation] {result['result']} {result['counts']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
