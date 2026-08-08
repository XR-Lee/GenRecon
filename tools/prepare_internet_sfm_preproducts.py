#!/usr/bin/env python3
"""Prepare masked internet-video shots, run COLMAP, and report SfM quality.

The processing unit is one automatically selected continuous shot per raw video.
Dynamic masks follow COLMAP's convention: white pixels are eligible for feature
extraction and black pixels are ignored. Source RGB frames are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

try:
    from tools.collect_internet_video_candidates import (
        ffmpeg_executable,
        sha256_file,
        write_json,
    )
except ModuleNotFoundError:  # Direct execution puts tools/ on sys.path.
    from collect_internet_video_candidates import ffmpeg_executable, sha256_file, write_json

SCHEMA_VERSION = 1
DEFAULT_RAW_INDEX = Path("data/internet-zero-shot/raw-candidates-v1/index.json")
DEFAULT_OUTPUT = Path("data/internet-zero-shot/sfm-preproducts-v1")
DYNAMIC_CATEGORY_NAMES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "tv",
    "laptop",
    "cell phone",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def percentile_summary(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def parse_scene_metadata(text: str) -> list[dict[str, float]]:
    samples: list[dict[str, float]] = []
    timestamp: float | None = None
    for line in text.splitlines():
        match = re.search(r"pts_time:([0-9.eE+-]+)", line)
        if match:
            timestamp = float(match.group(1))
            continue
        match = re.search(r"lavfi\.scene_score=([0-9.eE+-]+)", line)
        if match and timestamp is not None:
            samples.append({"time_s": timestamp, "score": float(match.group(1))})
            timestamp = None
    return samples


def detect_scene_boundaries(
    samples: Sequence[dict[str, float]],
    *,
    threshold: float = 0.35,
    minimum_spacing_s: float = 1.5,
) -> list[dict[str, float]]:
    peaks: list[dict[str, float]] = []
    for sample in samples:
        if sample["score"] < threshold or sample["time_s"] <= 0:
            continue
        if peaks and sample["time_s"] - peaks[-1]["time_s"] < minimum_spacing_s:
            if sample["score"] > peaks[-1]["score"]:
                peaks[-1] = dict(sample)
        else:
            peaks.append(dict(sample))
    return peaks


def _window_scene_metrics(
    samples: Sequence[dict[str, float]], start_s: float, end_s: float
) -> dict[str, Any]:
    values = [
        sample["score"]
        for sample in samples
        if start_s + 0.5 <= sample["time_s"] <= end_s - 0.5
    ]
    summary = percentile_summary(values)
    median = float(summary["median"] or 0.0)
    p90 = float(summary["p90"] or 0.0)
    activity = min(1.0, median / 0.04) * 0.55 + min(1.0, p90 / 0.15) * 0.45
    return {"scene_score": summary, "activity_score": float(activity)}


def generate_shot_windows(
    duration_s: float,
    samples: Sequence[dict[str, float]],
    boundaries: Sequence[dict[str, float]],
    *,
    minimum_duration_s: float = 8.0,
    maximum_duration_s: float = 90.0,
    long_window_stride_s: float = 45.0,
    maximum_candidates: int = 14,
) -> list[dict[str, Any]]:
    cuts = [0.0]
    cuts.extend(float(item["time_s"]) for item in boundaries if 0 < item["time_s"] < duration_s)
    cuts.append(float(duration_s))
    cuts = sorted(set(cuts))
    windows: list[dict[str, Any]] = []
    for segment_index, (raw_start, raw_end) in enumerate(zip(cuts, cuts[1:])):
        raw_duration = raw_end - raw_start
        trim = min(1.0, raw_duration * 0.05)
        start, end = raw_start + trim, raw_end - trim
        available = end - start
        if available < minimum_duration_s:
            continue
        starts: list[float]
        if available <= maximum_duration_s:
            starts = [start]
        else:
            starts = list(
                np.arange(start, end - maximum_duration_s + 1e-6, long_window_stride_s)
            )
            final_start = end - maximum_duration_s
            if not starts or final_start - starts[-1] > 1.0:
                starts.append(final_start)
        for window_index, window_start in enumerate(starts):
            window_end = min(end, window_start + maximum_duration_s)
            scene_metrics = _window_scene_metrics(samples, window_start, window_end)
            duration = window_end - window_start
            duration_score = min(1.0, duration / 30.0)
            windows.append(
                {
                    "segment_index": segment_index,
                    "window_index": window_index,
                    "start_s": round(float(window_start), 3),
                    "end_s": round(float(window_end), 3),
                    "duration_s": round(float(duration), 3),
                    "duration_score": duration_score,
                    **scene_metrics,
                }
            )
    if not windows:
        return []

    # Preserve long intervals, active intervals, and temporal coverage. This is
    # intentionally deterministic and prevents a long interview from consuming
    # every slot before visual diagnostics are evaluated.
    selected_indices: set[int] = set()
    by_duration = sorted(
        range(len(windows)),
        key=lambda index: (windows[index]["duration_s"], windows[index]["activity_score"]),
        reverse=True,
    )
    by_activity = sorted(
        range(len(windows)),
        key=lambda index: (windows[index]["activity_score"], windows[index]["duration_s"]),
        reverse=True,
    )
    selected_indices.update(by_duration[:5])
    selected_indices.update(by_activity[:7])
    coverage_count = min(4, len(windows))
    for value in np.linspace(0, len(windows) - 1, coverage_count):
        selected_indices.add(int(round(float(value))))
    ranked = sorted(
        selected_indices,
        key=lambda index: (
            0.55 * windows[index]["activity_score"]
            + 0.45 * windows[index]["duration_score"],
            windows[index]["duration_s"],
            -windows[index]["start_s"],
        ),
        reverse=True,
    )[:maximum_candidates]
    result = []
    for rank, index in enumerate(ranked, start=1):
        item = dict(windows[index])
        item["proposal_rank"] = rank
        result.append(item)
    return result


def sample_video_frames(path: Path, timestamps: Sequence[float]) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frames: list[np.ndarray] = []
    try:
        for timestamp in timestamps:
            capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp)) * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Could not decode {path} at {timestamp:.3f}s")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return frames


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not mask.any():
        return mask.astype(bool)
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def dynamic_mask_from_prediction(
    prediction: dict[str, Any],
    categories: Sequence[str],
    *,
    score_threshold: float,
    dilation_fraction: float,
    shape: tuple[int, int],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    height, width = shape
    union = np.zeros((height, width), dtype=bool)
    instances: list[dict[str, Any]] = []
    scores = prediction["scores"].detach().cpu().numpy()
    labels = prediction["labels"].detach().cpu().numpy()
    masks = prediction["masks"].detach().cpu().numpy()[:, 0]
    boxes = prediction["boxes"].detach().cpu().numpy()
    for score, label, mask_probability, box in zip(scores, labels, masks, boxes):
        if float(score) < score_threshold:
            continue
        name = categories[int(label)]
        if name not in DYNAMIC_CATEGORY_NAMES:
            continue
        instance_mask = mask_probability >= 0.5
        union |= instance_mask
        instances.append(
            {
                "category": name,
                "score": float(score),
                "box_xyxy": [float(value) for value in box],
                "mask_fraction_before_dilation": float(instance_mask.mean()),
            }
        )
    radius = (
        max(2, int(round(max(height, width) * dilation_fraction)))
        if dilation_fraction > 0
        else 0
    )
    union = _dilate_mask(union, radius)
    return union, instances


def load_dynamic_model(device: str) -> tuple[Any, Any, list[str], dict[str, Any]]:
    import torch
    import torchvision
    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_V2_Weights,
        maskrcnn_resnet50_fpn_v2,
    )

    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights).eval().to(device)
    checkpoint = Path.home() / ".cache/torch/hub/checkpoints" / Path(weights.url).name
    provenance = {
        "architecture": "maskrcnn_resnet50_fpn_v2",
        "weights": str(weights),
        "weights_url": weights.url,
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "device": device,
        "dynamic_categories": sorted(DYNAMIC_CATEGORY_NAMES),
    }
    return model, torch, list(weights.meta["categories"]), provenance


def infer_dynamic_masks(
    model: Any,
    torch_module: Any,
    categories: Sequence[str],
    images: Sequence[np.ndarray],
    *,
    device: str,
    score_threshold: float,
    dilation_fraction: float,
    batch_size: int,
) -> list[tuple[np.ndarray, list[dict[str, Any]]]]:
    results: list[tuple[np.ndarray, list[dict[str, Any]]]] = []
    with torch_module.inference_mode():
        for start in range(0, len(images), batch_size):
            batch_images = images[start : start + batch_size]
            tensors = [
                torch_module.from_numpy(np.array(image, copy=True, order="C"))
                .permute(2, 0, 1)
                .float()
                .div_(255.0)
                .to(device)
                for image in batch_images
            ]
            predictions = model(tensors)
            for image, prediction in zip(batch_images, predictions):
                results.append(
                    dynamic_mask_from_prediction(
                        prediction,
                        categories,
                        score_threshold=score_threshold,
                        dilation_fraction=dilation_fraction,
                        shape=image.shape[:2],
                    )
                )
            del tensors, predictions
    return results


def ratio_test_matches(pairs: Iterable[Sequence[Any]], ratio: float = 0.75) -> list[Any]:
    return [
        pair[0]
        for pair in pairs
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
    ]


def _orb_window_metrics(
    frames: Sequence[np.ndarray], dynamic_masks: Sequence[np.ndarray]
) -> dict[str, Any]:
    orb = cv2.ORB_create(nfeatures=2500, fastThreshold=10)
    features: list[tuple[list[Any], np.ndarray | None]] = []
    counts: list[int] = []
    brightness: list[float] = []
    for frame, dynamic in zip(frames, dynamic_masks):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        static_mask = np.where(dynamic, 0, 255).astype(np.uint8)
        keypoints, descriptors = orb.detectAndCompute(gray, static_mask)
        features.append((keypoints, descriptors))
        counts.append(len(keypoints))
        brightness.append(float(gray.mean()))

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    displacements: list[float] = []
    inlier_ratios: list[float] = []
    successful_pairs = 0
    for frame_a, frame_b, feature_a, feature_b in zip(
        frames, frames[1:], features, features[1:]
    ):
        keypoints_a, descriptors_a = feature_a
        keypoints_b, descriptors_b = feature_b
        if descriptors_a is None or descriptors_b is None:
            continue
        pairs = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
        good = ratio_test_matches(pairs)
        if len(good) < 20:
            continue
        source = np.asarray([keypoints_a[item.queryIdx].pt for item in good], dtype=np.float32)
        target = np.asarray([keypoints_b[item.trainIdx].pt for item in good], dtype=np.float32)
        homography, inliers = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
        if homography is None or inliers is None:
            continue
        height, width = frame_a.shape[:2]
        probes = np.asarray(
            [[[0, 0]], [[width - 1, 0]], [[0, height - 1]], [[width - 1, height - 1]], [[width / 2, height / 2]]],
            dtype=np.float32,
        )
        transformed = cv2.perspectiveTransform(probes, homography)
        displacement = np.linalg.norm(transformed[:, 0] - probes[:, 0], axis=1)
        displacements.append(float(np.median(displacement) / math.hypot(width, height)))
        inlier_ratios.append(float(inliers.mean()))
        successful_pairs += 1
    pair_count = max(0, len(frames) - 1)
    return {
        "static_orb_keypoints": percentile_summary(counts),
        "background_motion_normalized": percentile_summary(displacements),
        "homography_inlier_ratio": percentile_summary(inlier_ratios),
        "matched_pair_fraction": successful_pairs / pair_count if pair_count else 0.0,
        "brightness": percentile_summary(brightness),
    }


def score_window(
    window: dict[str, Any],
    frames: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    instances: Sequence[Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    orb_metrics = _orb_window_metrics(frames, masks)
    dynamic_fractions = [float(mask.mean()) for mask in masks]
    person_presence = [
        any(instance["category"] == "person" for instance in frame_instances)
        for frame_instances in instances
    ]
    keypoints = float(orb_metrics["static_orb_keypoints"]["median"] or 0.0)
    motion = float(orb_metrics["background_motion_normalized"]["median"] or 0.0)
    pair_fraction = float(orb_metrics["matched_pair_fraction"])
    mean_dynamic = float(np.mean(dynamic_fractions)) if dynamic_fractions else 0.0
    mean_brightness = float(orb_metrics["brightness"]["mean"] or 0.0)

    feature_score = min(1.0, keypoints / 800.0)
    motion_score = min(1.0, motion / 0.02)
    if motion > 0.30:
        motion_score *= max(0.0, (0.50 - motion) / 0.20)
    dynamic_score = max(0.0, 1.0 - mean_dynamic / 0.50)
    brightness_score = min(1.0, mean_brightness / 45.0) * min(
        1.0, max(0.0, 255.0 - mean_brightness) / 45.0
    )
    score = (
        0.16 * float(window["duration_score"])
        + 0.12 * float(window["activity_score"])
        + 0.24 * feature_score
        + 0.26 * motion_score
        + 0.12 * pair_fraction
        + 0.07 * dynamic_score
        + 0.03 * brightness_score
    )
    person_fraction = float(np.mean(person_presence)) if person_presence else 0.0
    flags: list[str] = []
    if motion < 0.002:
        score *= 0.65
        flags.append("near_static_camera")
    if person_fraction >= 0.8 and motion < 0.01:
        score *= 0.65
        flags.append("likely_static_interview")
    if pair_fraction < 0.4:
        score *= 0.75
        flags.append("weak_background_continuity")
    if mean_dynamic > 0.45:
        score *= 0.65
        flags.append("high_dynamic_coverage")
    return {
        **window,
        "sample_timestamps_s": [],
        "sample_dynamic_fraction": percentile_summary(dynamic_fractions),
        "sample_person_frame_fraction": person_fraction,
        "sample_instance_category_counts": dict(
            sorted(Counter(instance["category"] for group in instances for instance in group).items())
        ),
        "visual_metrics": orb_metrics,
        "score_components": {
            "feature": feature_score,
            "motion": motion_score,
            "pair_continuity": pair_fraction,
            "dynamic_static_area": dynamic_score,
            "brightness": brightness_score,
        },
        "selection_score": float(score),
        "selection_flags": flags,
    }


def analyze_candidate_shots(
    raw_root: Path,
    candidate: dict[str, Any],
    candidate_output: Path,
    ffmpeg: str,
    model: Any,
    torch_module: Any,
    categories: Sequence[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    candidate_output.mkdir(parents=True, exist_ok=True)
    preview_path = raw_root / candidate["preview"]["local_path"]
    metadata_path = candidate_output / "scene_scores.txt"
    if not metadata_path.is_file() or args.force:
        command = [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(preview_path),
            "-vf",
            f"fps={args.scene_analysis_fps},select='gte(scene,0)',metadata=print:file={metadata_path.resolve()}",
            "-an",
            "-f",
            "null",
            "-",
        ]
        subprocess.run(command, check=True)
    samples = parse_scene_metadata(metadata_path.read_text())
    boundaries = detect_scene_boundaries(
        samples,
        threshold=args.scene_cut_threshold,
        minimum_spacing_s=args.scene_cut_spacing,
    )
    windows = generate_shot_windows(
        float(candidate["duration_s"]),
        samples,
        boundaries,
        minimum_duration_s=args.minimum_shot_duration,
        maximum_duration_s=args.maximum_shot_duration,
        long_window_stride_s=args.window_stride,
        maximum_candidates=args.maximum_window_candidates,
    )
    evaluated: list[dict[str, Any]] = []
    for window in windows:
        sample_count = min(args.selection_samples, max(3, int(window["duration_s"] // 2)))
        margin = min(1.0, window["duration_s"] * 0.08)
        timestamps = np.linspace(
            window["start_s"] + margin,
            window["end_s"] - margin,
            sample_count,
        ).tolist()
        frames = sample_video_frames(preview_path, timestamps)
        predictions = infer_dynamic_masks(
            model,
            torch_module,
            categories,
            frames,
            device=args.device,
            score_threshold=args.mask_score_threshold,
            dilation_fraction=args.mask_dilation_fraction,
            batch_size=args.mask_batch_size,
        )
        masks = [item[0] for item in predictions]
        instances = [item[1] for item in predictions]
        record = score_window(window, frames, masks, instances)
        record["sample_timestamps_s"] = [round(float(value), 3) for value in timestamps]
        evaluated.append(record)
    evaluated.sort(key=lambda item: (item["selection_score"], item["duration_s"]), reverse=True)
    for rank, item in enumerate(evaluated, start=1):
        item["evaluated_rank"] = rank
    selection = {
        "schema": "genrecon.internet-video-shot-selection",
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate["candidate_id"],
        "source_preview": str(preview_path),
        "scene_analysis": {
            "fps": args.scene_analysis_fps,
            "cut_threshold": args.scene_cut_threshold,
            "cut_minimum_spacing_s": args.scene_cut_spacing,
            "sample_count": len(samples),
            "score": percentile_summary([item["score"] for item in samples]),
            "boundaries": boundaries,
        },
        "policy": {
            "minimum_shot_duration_s": args.minimum_shot_duration,
            "maximum_shot_duration_s": args.maximum_shot_duration,
            "maximum_window_candidates": args.maximum_window_candidates,
            "selection_samples_per_window": args.selection_samples,
            "automatic_selection": True,
            "manual_review_required": True,
        },
        "windows": evaluated,
        "selected": evaluated[0] if evaluated else None,
    }
    write_json(candidate_output / "selection.json", selection)
    return selection


def extract_selected_frames(
    raw_root: Path,
    candidate: dict[str, Any],
    candidate_output: Path,
    selected: dict[str, Any],
    ffmpeg: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rgb_path = candidate_output / "rgb"
    if args.force and rgb_path.exists():
        shutil.rmtree(rgb_path)
    rgb_path.mkdir(parents=True, exist_ok=True)
    existing = sorted(rgb_path.glob("frame_*.jpg"))
    source_path = raw_root / candidate["source_video"]["local_path"]
    if not existing:
        scale = (
            f"scale='min({args.max_image_size},iw)':'min({args.max_image_size},ih)'"
            ":force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1"
        )
        command = [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{selected['start_s']:.3f}",
            "-t",
            f"{selected['duration_s']:.3f}",
            "-i",
            str(source_path),
            "-map",
            "0:v:0",
            "-vf",
            f"fps={args.frame_fps},{scale}",
            "-frames:v",
            str(args.maximum_frames),
            "-q:v",
            "2",
            "-threads",
            str(args.ffmpeg_threads),
            str(rgb_path / "frame_%06d.jpg"),
        ]
        subprocess.run(command, check=True)
        existing = sorted(rgb_path.glob("frame_*.jpg"))
    if len(existing) < 3:
        raise RuntimeError(f"Frame extraction produced only {len(existing)} frames")
    records = []
    for index, path in enumerate(existing):
        with Image.open(path) as image:
            width, height = image.size
        records.append(
            {
                "index": index,
                "name": path.name,
                "timestamp_s": round(float(selected["start_s"] + index / args.frame_fps), 6),
                "width": width,
                "height": height,
                "size_bytes": path.stat().st_size,
            }
        )
    document = {
        "schema": "genrecon.internet-video-extracted-frames",
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate["candidate_id"],
        "source_video": str(source_path),
        "source_sha256": candidate["source_video"]["sha256"],
        "selected_start_s": selected["start_s"],
        "selected_end_s": selected["end_s"],
        "fps": args.frame_fps,
        "max_image_size": args.max_image_size,
        "maximum_frames": args.maximum_frames,
        "frame_count": len(records),
        "frames": records,
    }
    write_json(candidate_output / "frames.json", document)
    return document


def _save_mask_contact(
    rgb_paths: Sequence[Path], dynamic_paths: Sequence[Path], output_path: Path
) -> None:
    if not rgb_paths:
        return
    count = min(12, len(rgb_paths))
    indices = np.linspace(0, len(rgb_paths) - 1, count).round().astype(int)
    tile_width, tile_height, gap = 320, 180, 6
    columns, rows = 4, 3
    canvas = Image.new(
        "RGB",
        (columns * tile_width + (columns + 1) * gap, rows * tile_height + (rows + 1) * gap),
        "#111418",
    )
    for tile_index, frame_index in enumerate(indices):
        with Image.open(rgb_paths[frame_index]) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(dynamic_paths[frame_index]) as image:
            mask = np.asarray(image.convert("L"), dtype=np.uint8) > 0
        overlay = rgb.astype(np.float32)
        overlay[mask] = 0.35 * overlay[mask] + 0.65 * np.asarray([230, 45, 92], dtype=np.float32)
        rendered = Image.fromarray(np.rint(overlay).astype(np.uint8))
        rendered.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_width, tile_height), "#000000")
        left = (tile_width - rendered.width) // 2
        top = (tile_height - rendered.height) // 2
        tile.paste(rendered, (left, top))
        draw = ImageDraw.Draw(tile)
        draw.text(
            (6, 5),
            f"{frame_index + 1:03d}  dynamic {float(mask.mean()) * 100:.1f}%",
            fill="#ffffff",
            stroke_width=2,
            stroke_fill="#111111",
        )
        x = gap + (tile_index % columns) * (tile_width + gap)
        y = gap + (tile_index // columns) * (tile_height + gap)
        canvas.paste(tile, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92, subsampling=0)


def generate_frame_masks(
    candidate: dict[str, Any],
    candidate_output: Path,
    frames_document: dict[str, Any],
    model: Any,
    torch_module: Any,
    categories: Sequence[str],
    model_provenance: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    dynamic_path = candidate_output / "masks_dynamic"
    colmap_path = candidate_output / "masks_colmap"
    if args.force:
        for path in (dynamic_path, colmap_path):
            if path.exists():
                shutil.rmtree(path)
    dynamic_path.mkdir(parents=True, exist_ok=True)
    colmap_path.mkdir(parents=True, exist_ok=True)
    rgb_paths = [candidate_output / "rgb" / item["name"] for item in frames_document["frames"]]
    records: list[dict[str, Any]] = []
    for start in range(0, len(rgb_paths), args.mask_batch_size):
        batch_paths = rgb_paths[start : start + args.mask_batch_size]
        images = []
        for path in batch_paths:
            with Image.open(path) as image:
                images.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
        predictions = infer_dynamic_masks(
            model,
            torch_module,
            categories,
            images,
            device=args.device,
            score_threshold=args.mask_score_threshold,
            dilation_fraction=args.mask_dilation_fraction,
            batch_size=args.mask_batch_size,
        )
        for offset, (path, image, prediction) in enumerate(zip(batch_paths, images, predictions)):
            dynamic, instances = prediction
            dynamic_file = dynamic_path / f"{path.stem}.png"
            colmap_file = colmap_path / f"{path.name}.png"
            Image.fromarray(np.where(dynamic, 255, 0).astype(np.uint8), mode="L").save(dynamic_file)
            Image.fromarray(np.where(dynamic, 0, 255).astype(np.uint8), mode="L").save(colmap_file)
            records.append(
                {
                    "index": start + offset,
                    "image": path.name,
                    "dynamic_mask": str(dynamic_file.relative_to(candidate_output)),
                    "colmap_mask": str(colmap_file.relative_to(candidate_output)),
                    "dynamic_fraction": float(dynamic.mean()),
                    "static_fraction": float(1.0 - dynamic.mean()),
                    "instance_count": len(instances),
                    "instances": instances,
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                }
            )
        print(
            f"[{candidate['candidate_id']}] masks {min(start + len(batch_paths), len(rgb_paths))}/{len(rgb_paths)}",
            flush=True,
        )
    dynamic_paths = [candidate_output / item["dynamic_mask"] for item in records]
    _save_mask_contact(rgb_paths, dynamic_paths, candidate_output / "mask_contact.jpg")
    document = {
        "schema": "genrecon.internet-video-dynamic-masks",
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate["candidate_id"],
        "created_utc": utc_now(),
        "mask_semantics": {
            "masks_dynamic": "255 means a detected dynamic pixel",
            "masks_colmap": "255 means SIFT features are allowed; 0 means ignored",
            "source_rgb_modified": False,
        },
        "policy": {
            "score_threshold": args.mask_score_threshold,
            "mask_probability_threshold": 0.5,
            "dilation_fraction_of_long_side": args.mask_dilation_fraction,
        },
        "model": model_provenance,
        "summary": {
            "frame_count": len(records),
            "dynamic_fraction": percentile_summary([item["dynamic_fraction"] for item in records]),
            "frames_with_dynamic_instances": sum(item["instance_count"] > 0 for item in records),
            "category_counts": dict(
                sorted(
                    Counter(
                        instance["category"]
                        for item in records
                        for instance in item["instances"]
                    ).items()
                )
            ),
        },
        "frames": records,
    }
    write_json(candidate_output / "mask_summary.json", document)
    return document


def prepare_candidate(
    raw_root: Path,
    candidate: dict[str, Any],
    output_root: Path,
    ffmpeg: str,
    model: Any,
    torch_module: Any,
    categories: Sequence[str],
    model_provenance: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    candidate_output = output_root / "candidates" / candidate["candidate_id"]
    status_path = candidate_output / "preparation_status.json"
    if status_path.is_file() and not args.force:
        status = json.loads(status_path.read_text())
        if status.get("status") == "prepared":
            print(f"[{candidate['candidate_id']}] preparation already complete", flush=True)
            return status
    started = time.monotonic()
    try:
        selection = analyze_candidate_shots(
            raw_root,
            candidate,
            candidate_output,
            ffmpeg,
            model,
            torch_module,
            categories,
            args,
        )
        if selection["selected"] is None:
            raise RuntimeError("No continuous shot meets the minimum duration")
        frames = extract_selected_frames(
            raw_root,
            candidate,
            candidate_output,
            selection["selected"],
            ffmpeg,
            args,
        )
        masks = generate_frame_masks(
            candidate,
            candidate_output,
            frames,
            model,
            torch_module,
            categories,
            model_provenance,
            args,
        )
        status = {
            "candidate_id": candidate["candidate_id"],
            "status": "prepared",
            "elapsed_s": time.monotonic() - started,
            "selected_start_s": selection["selected"]["start_s"],
            "selected_end_s": selection["selected"]["end_s"],
            "selection_score": selection["selected"]["selection_score"],
            "frame_count": frames["frame_count"],
            "mean_dynamic_fraction": masks["summary"]["dynamic_fraction"]["mean"],
        }
    except Exception as error:
        status = {
            "candidate_id": candidate["candidate_id"],
            "status": "failed",
            "elapsed_s": time.monotonic() - started,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    write_json(status_path, status)
    print(f"[{candidate['candidate_id']}] preparation {status['status']}", flush=True)
    return status


def _import_pycolmap() -> Any:
    try:
        import pycolmap
    except ImportError as error:
        raise RuntimeError(
            "pycolmap is unavailable. Run with PYTHONPATH=/tmp/pycolmap-wheel or install pycolmap."
        ) from error
    return pycolmap


def verify_colmap_keypoint_masks(
    pycolmap: Any, database_path: Path, masks_path: Path
) -> dict[str, Any]:
    total = 0
    blocked = 0
    rounded_blocked = 0
    per_image = []
    with pycolmap.Database.open(str(database_path)) as database:
        images = database.read_all_images()
        for image in images:
            keypoints = database.read_keypoints(image.image_id)
            mask_path = masks_path / f"{image.name}.png"
            with Image.open(mask_path) as opened:
                mask = np.asarray(opened.convert("L"), dtype=np.uint8)
            if len(keypoints):
                # COLMAP samples masks using integer truncation, equivalent to
                # floor for non-negative image coordinates. Rounded centers are
                # reported separately as an edge diagnostic but are not leakage.
                columns = np.clip(np.floor(keypoints[:, 0]).astype(int), 0, mask.shape[1] - 1)
                rows = np.clip(np.floor(keypoints[:, 1]).astype(int), 0, mask.shape[0] - 1)
                count = int((mask[rows, columns] == 0).sum())
                rounded_columns = np.clip(
                    np.rint(keypoints[:, 0]).astype(int), 0, mask.shape[1] - 1
                )
                rounded_rows = np.clip(
                    np.rint(keypoints[:, 1]).astype(int), 0, mask.shape[0] - 1
                )
                rounded_count = int((mask[rounded_rows, rounded_columns] == 0).sum())
            else:
                count = 0
                rounded_count = 0
            total += len(keypoints)
            blocked += count
            rounded_blocked += rounded_count
            per_image.append(
                {
                    "image": image.name,
                    "keypoints": int(len(keypoints)),
                    "blocked_centers": count,
                    "rounded_centers_on_blocked_edge": rounded_count,
                }
            )
        pair_ids, pair_inliers = database.read_two_view_geometry_num_inliers()
        pair_counts = [int(value) for value in pair_inliers]
        database_metrics = {
            "image_count": database.num_images(),
            "camera_count": database.num_cameras(),
            "keypoint_count": database.num_keypoints(),
            "matched_pair_count": database.num_matched_image_pairs(),
            "verified_pair_count": database.num_verified_image_pairs(),
            "inlier_match_count": database.num_inlier_matches(),
            "verified_pair_inliers": percentile_summary(pair_counts),
        }
    return {
        "total_keypoint_centers": int(total),
        "keypoint_centers_on_blocked_pixels": int(blocked),
        "rounded_centers_on_blocked_edge": int(rounded_blocked),
        "blocked_fraction": blocked / total if total else 0.0,
        "per_image": per_image,
        "database": database_metrics,
    }


def reconstruction_metrics(reconstruction: Any, total_images: int) -> dict[str, Any]:
    points = list(reconstruction.points3D.values())
    errors = np.asarray([point.error for point in points], dtype=np.float64)
    tracks = np.asarray([point.track.length() for point in points], dtype=np.int64)
    quality = (errors <= 2.0) & (tracks >= 3) if len(points) else np.zeros(0, dtype=bool)
    registered_ids = reconstruction.reg_image_ids()
    center_by_image = {
        image_id: np.asarray(reconstruction.images[image_id].projection_center(), dtype=np.float64)
        for image_id in registered_ids
    }
    centers = np.asarray(list(center_by_image.values()), dtype=np.float64).reshape(-1, 3)
    if len(centers):
        center_span = np.ptp(centers, axis=0)
        centered = centers - centers.mean(axis=0)
        singular_values = np.linalg.svd(centered, compute_uv=False) if len(centers) >= 2 else np.zeros(3)
    else:
        center_span = np.zeros(3)
        singular_values = np.zeros(3)

    point_xyz = np.asarray([point.xyz for point in points], dtype=np.float64).reshape(-1, 3)
    if len(point_xyz):
        robust_point_span = np.percentile(point_xyz, 95, axis=0) - np.percentile(point_xyz, 5, axis=0)
    else:
        robust_point_span = np.zeros(3)
    observed_depths: list[float] = []
    triangulation_angles: list[float] = []
    for point in points:
        observing_centers = np.asarray(
            [
                center_by_image[element.image_id]
                for element in point.track.elements
                if element.image_id in center_by_image
            ],
            dtype=np.float64,
        ).reshape(-1, 3)
        if len(observing_centers) < 2:
            continue
        rays = observing_centers - np.asarray(point.xyz, dtype=np.float64)
        distances = np.linalg.norm(rays, axis=1)
        valid = distances > 1e-12
        if int(valid.sum()) < 2:
            continue
        rays = rays[valid] / distances[valid, None]
        observed_depths.append(float(distances[valid].min()))
        minimum_dot = float(np.min(np.clip(rays @ rays.T, -1.0, 1.0)))
        triangulation_angles.append(float(np.degrees(np.arccos(minimum_dot))))
    baseline = float(np.linalg.norm(center_span))
    scene_depth = float(np.median(observed_depths)) if observed_depths else None
    viewpoint_geometry = {
        "camera_baseline_diagonal_sfm_units": baseline,
        "robust_point_span_5_95_sfm_units": [float(value) for value in robust_point_span],
        "robust_point_span_diagonal_sfm_units": float(np.linalg.norm(robust_point_span)),
        "median_min_observing_camera_depth_sfm_units": scene_depth,
        "baseline_to_observed_depth": (
            baseline / scene_depth if scene_depth is not None and scene_depth > 0 else None
        ),
        "triangulation_angle_deg": percentile_summary(triangulation_angles),
    }
    cameras = []
    for camera in reconstruction.cameras.values():
        cameras.append(
            {
                "camera_id": int(camera.camera_id),
                "model": str(camera.model.name),
                "width": int(camera.width),
                "height": int(camera.height),
                "params": [float(value) for value in camera.params],
                "mean_focal_length_px": float(camera.mean_focal_length()),
                "focal_to_long_side": float(camera.mean_focal_length() / max(camera.width, camera.height)),
                "has_bogus_params": bool(camera.has_bogus_params(0.2, 5.0, 1.0)),
            }
        )
    return {
        "registered_images": int(reconstruction.num_reg_images()),
        "registered_fraction": reconstruction.num_reg_images() / total_images if total_images else 0.0,
        "registered_image_names": sorted(
            reconstruction.images[image_id].name for image_id in reconstruction.reg_image_ids()
        ),
        "points3D": int(reconstruction.num_points3D()),
        "observations": int(reconstruction.compute_num_observations()),
        "mean_observations_per_registered_image": finite_float(
            reconstruction.compute_mean_observations_per_reg_image()
        ),
        "mean_reprojection_error_px": finite_float(reconstruction.compute_mean_reprojection_error()),
        "reprojection_error_px": percentile_summary(errors),
        "track_length": percentile_summary(tracks),
        "quality_points_error_le_2px_track_ge_3": int(quality.sum()),
        "quality_point_fraction": float(quality.mean()) if len(quality) else 0.0,
        "camera_center_span_sfm_units": [float(value) for value in center_span],
        "camera_center_singular_values": [float(value) for value in singular_values],
        "viewpoint_geometry": viewpoint_geometry,
        "cameras": cameras,
    }


def grade_sfm_quality(metrics: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    warnings: list[str] = []
    largest = metrics.get("largest_model")
    if largest is None:
        return {"grade": "F", "gate": "fail", "reasons": ["no_reconstruction_model"], "warnings": []}
    registered = float(largest["registered_fraction"])
    points = int(largest["points3D"])
    quality_points = int(largest["quality_points_error_le_2px_track_ge_3"])
    error = largest["mean_reprojection_error_px"]
    track = largest["track_length"]["mean"]
    blocked = int(metrics["mask_application"]["keypoint_centers_on_blocked_pixels"])
    dynamic = float(metrics["dynamic_masks"]["dynamic_fraction"]["mean"] or 0.0)
    dominance = float(metrics.get("largest_model_registered_dominance", 0.0))
    geometry = largest.get("viewpoint_geometry") or {}
    baseline_to_depth = geometry.get("baseline_to_observed_depth")
    triangulation_angle = (geometry.get("triangulation_angle_deg") or {}).get("median")

    if blocked > 0:
        reasons.append("keypoints_found_on_blocked_mask_pixels")
    if registered < 0.4:
        reasons.append("registered_fraction_below_0.40")
    if points < 500:
        reasons.append("fewer_than_500_sparse_points")
    if error is None or error > 4.0:
        reasons.append("mean_reprojection_error_above_4px_or_missing")
    if dynamic > 0.35:
        warnings.append("mean_dynamic_mask_fraction_above_0.35")
    elif dynamic > 0.25:
        warnings.append("mean_dynamic_mask_fraction_above_0.25")
    if baseline_to_depth is None or baseline_to_depth < 0.10:
        reasons.append("insufficient_viewpoint_baseline")
    elif baseline_to_depth < 0.50:
        warnings.append("low_viewpoint_baseline")
    if triangulation_angle is None or triangulation_angle < 2.0:
        reasons.append("insufficient_triangulation_angle")
    elif triangulation_angle < 4.0:
        warnings.append("weak_triangulation_angle")
    if dominance < 0.9:
        warnings.append("fragmented_reconstruction_models")
    if any(camera["has_bogus_params"] for camera in largest["cameras"]):
        warnings.append("implausible_camera_intrinsics")

    if reasons:
        return {"grade": "F", "gate": "fail", "reasons": reasons, "warnings": warnings}
    if (
        registered >= 0.90
        and points >= 5000
        and quality_points >= 3000
        and error is not None
        and error <= 1.5
        and track is not None
        and track >= 4.0
        and dominance >= 0.9
        and baseline_to_depth is not None
        and baseline_to_depth >= 0.50
        and triangulation_angle is not None
        and triangulation_angle >= 4.0
        and dynamic <= 0.25
    ):
        return {"grade": "A", "gate": "pass", "reasons": [], "warnings": warnings}
    if (
        registered >= 0.75
        and points >= 2000
        and quality_points >= 1000
        and error is not None
        and error <= 2.0
        and track is not None
        and track >= 3.0
        and dominance >= 0.8
        and baseline_to_depth is not None
        and baseline_to_depth >= 0.50
        and triangulation_angle is not None
        and triangulation_angle >= 4.0
        and dynamic <= 0.25
    ):
        return {"grade": "B", "gate": "pass", "reasons": [], "warnings": warnings}
    warnings.extend(
        item
        for item, condition in [
            ("registered_fraction_below_0.75", registered < 0.75),
            ("fewer_than_2000_sparse_points", points < 2000),
            ("fewer_than_1000_quality_points", quality_points < 1000),
            ("mean_reprojection_error_above_2px", error is None or error > 2.0),
            ("mean_track_length_below_3", track is None or track < 3.0),
        ]
        if condition and item not in warnings
    )
    return {"grade": "C", "gate": "marginal", "reasons": [], "warnings": warnings}


def _save_qc_contact(
    candidate_output: Path, registered_names: set[str], output_path: Path
) -> None:
    rgb_paths = sorted((candidate_output / "rgb").glob("frame_*.jpg"))
    if not rgb_paths:
        return
    count = min(20, len(rgb_paths))
    indices = np.linspace(0, len(rgb_paths) - 1, count).round().astype(int)
    tile_width, tile_height, gap = 256, 144, 6
    columns, rows = 5, 4
    canvas = Image.new(
        "RGB",
        (columns * tile_width + (columns + 1) * gap, rows * tile_height + (rows + 1) * gap),
        "#111418",
    )
    for tile_index, frame_index in enumerate(indices):
        path = rgb_paths[frame_index]
        dynamic_path = candidate_output / "masks_dynamic" / f"{path.stem}.png"
        with Image.open(path) as opened:
            rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        with Image.open(dynamic_path) as opened:
            dynamic = np.asarray(opened.convert("L"), dtype=np.uint8) > 0
        overlay = rgb.astype(np.float32)
        overlay[dynamic] = 0.55 * overlay[dynamic] + 0.45 * np.asarray([230, 45, 92])
        image = Image.fromarray(np.rint(overlay).astype(np.uint8))
        image.thumbnail((tile_width - 6, tile_height - 6), Image.Resampling.LANCZOS)
        registered = path.name in registered_names
        border = "#187a5a" if registered else "#b43b3b"
        tile = Image.new("RGB", (tile_width, tile_height), border)
        tile.paste(image, ((tile_width - image.width) // 2, (tile_height - image.height) // 2))
        draw = ImageDraw.Draw(tile)
        draw.text(
            (7, 6),
            f"{frame_index + 1:03d} {'REG' if registered else 'MISS'}",
            fill="#ffffff",
            stroke_width=2,
            stroke_fill="#111111",
        )
        x = gap + (tile_index % columns) * (tile_width + gap)
        y = gap + (tile_index // columns) * (tile_height + gap)
        canvas.paste(tile, (x, y))
    canvas.save(output_path, quality=92, subsampling=0)


def _save_trajectory_plot(reconstruction: Any, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = np.asarray(
        [reconstruction.images[index].projection_center() for index in reconstruction.reg_image_ids()],
        dtype=np.float64,
    ).reshape(-1, 3)
    points = np.asarray([point.xyz for point in reconstruction.points3D.values()], dtype=np.float64).reshape(-1, 3)
    if len(centers) < 2 or len(points) < 3:
        return
    combined = np.concatenate([centers, points], axis=0)
    centered = combined - np.median(combined, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    projected_centers = (centers - np.median(combined, axis=0)) @ vh[:2].T
    if len(points) > 30000:
        rng = np.random.default_rng(42)
        points = points[rng.choice(len(points), 30000, replace=False)]
    projected_points = (points - np.median(combined, axis=0)) @ vh[:2].T
    figure, axis = plt.subplots(figsize=(8, 6), dpi=130)
    axis.scatter(projected_points[:, 0], projected_points[:, 1], s=0.8, c="#77818a", alpha=0.28)
    axis.plot(projected_centers[:, 0], projected_centers[:, 1], "-o", ms=2.5, lw=1.0, c="#14735b")
    axis.scatter(projected_centers[0, 0], projected_centers[0, 1], s=55, c="#2368a2", label="first")
    axis.scatter(projected_centers[-1, 0], projected_centers[-1, 1], s=55, c="#b04a39", label="last")
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_title("Sparse points and camera trajectory (PCA projection, arbitrary scale)")
    axis.set_xlabel("principal axis 1")
    axis.set_ylabel("principal axis 2")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def run_colmap_candidate(
    candidate: dict[str, Any], output_root: Path, args: argparse.Namespace
) -> dict[str, Any]:
    pycolmap = _import_pycolmap()
    candidate_output = output_root / "candidates" / candidate["candidate_id"]
    status_path = candidate_output / "preparation_status.json"
    quality_path = candidate_output / "quality.json"
    if quality_path.is_file() and not args.force_colmap:
        existing = json.loads(quality_path.read_text())
        print(f"[{candidate['candidate_id']}] COLMAP already complete", flush=True)
        return existing
    if not status_path.is_file() or json.loads(status_path.read_text()).get("status") != "prepared":
        result = {
            "candidate_id": candidate["candidate_id"],
            "status": "skipped",
            "reason": "preparation_not_complete",
            "quality_gate": {"grade": "F", "gate": "fail", "reasons": ["preparation_not_complete"], "warnings": []},
        }
        write_json(quality_path, result)
        return result
    rgb_path = candidate_output / "rgb"
    masks_path = candidate_output / "masks_colmap"
    database_path = candidate_output / "database.db"
    sparse_path = candidate_output / "sparse"
    text_path = candidate_output / "colmap"
    best_binary_path = candidate_output / "sparse_best"
    if args.force_colmap or not quality_path.is_file():
        for path in (database_path, sparse_path, text_path, best_binary_path):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
    sparse_path.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    frame_count = len(list(rgb_path.glob("frame_*.jpg")))
    mask_summary = json.loads((candidate_output / "mask_summary.json").read_text())
    try:
        reader = pycolmap.ImageReaderOptions()
        reader.mask_path = str(masks_path.resolve())
        reader.camera_model = "SIMPLE_RADIAL"
        extraction = pycolmap.FeatureExtractionOptions()
        extraction.max_image_size = args.colmap_max_image_size
        extraction.num_threads = args.colmap_threads
        extraction.sift.max_num_features = args.colmap_max_features
        pycolmap.extract_features(
            database_path=str(database_path),
            image_path=str(rgb_path),
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader,
            extraction_options=extraction,
            device=pycolmap.Device.cpu,
        )
        matching = pycolmap.FeatureMatchingOptions()
        matching.num_threads = args.colmap_threads
        matching.guided_matching = True
        verification = pycolmap.TwoViewGeometryOptions()
        verification.ransac.max_error = 4.0
        if frame_count <= args.exhaustive_max_images:
            pairing = pycolmap.ExhaustivePairingOptions()
            pairing.block_size = 50
            pycolmap.match_exhaustive(
                str(database_path),
                matching_options=matching,
                pairing_options=pairing,
                verification_options=verification,
                device=pycolmap.Device.cpu,
            )
            matcher_name = "exhaustive"
        else:
            pairing = pycolmap.SequentialPairingOptions()
            pairing.overlap = args.sequential_overlap
            pairing.quadratic_overlap = True
            pairing.loop_detection = False
            pycolmap.match_sequential(
                str(database_path),
                matching_options=matching,
                pairing_options=pairing,
                verification_options=verification,
                device=pycolmap.Device.cpu,
            )
            matcher_name = "sequential_quadratic"
        mapper = pycolmap.IncrementalPipelineOptions()
        mapper.multiple_models = True
        mapper.max_num_models = 10
        mapper.min_model_size = min(8, frame_count)
        mapper.random_seed = 42
        mapper.num_threads = args.colmap_threads
        mapper.max_runtime_seconds = int(args.colmap_max_runtime)
        mapper.mapper.random_seed = 42
        mapper.mapper.init_min_num_inliers = 50
        mapper.mapper.abs_pose_min_num_inliers = 30
        mapper.mapper.filter_max_reproj_error = 4.0
        mapper.triangulation.random_seed = 42
        models = pycolmap.incremental_mapping(
            database_path=str(database_path),
            image_path=str(rgb_path),
            output_path=str(sparse_path),
            options=mapper,
        )
        mask_application = verify_colmap_keypoint_masks(pycolmap, database_path, masks_path)
        model_metrics = []
        model_items = sorted(models.items(), key=lambda item: int(item[0]))
        for model_id, reconstruction in model_items:
            metrics = reconstruction_metrics(reconstruction, frame_count)
            metrics["model_id"] = int(model_id)
            model_metrics.append(metrics)
        model_metrics.sort(
            key=lambda item: (item["registered_images"], item["points3D"]), reverse=True
        )
        largest_model = model_metrics[0] if model_metrics else None
        union_registered = set(
            name for model in model_metrics for name in model["registered_image_names"]
        )
        if largest_model is not None:
            best_model_id = int(largest_model["model_id"])
            best_reconstruction = models[best_model_id]
            text_path.mkdir(parents=True, exist_ok=True)
            best_binary_path.mkdir(parents=True, exist_ok=True)
            best_reconstruction.write_text(str(text_path))
            best_reconstruction.write(str(best_binary_path))
            best_reconstruction.export_PLY(str(candidate_output / "sparse_points.ply"))
            _save_qc_contact(
                candidate_output,
                set(largest_model["registered_image_names"]),
                candidate_output / "qc_contact.jpg",
            )
            _save_trajectory_plot(best_reconstruction, candidate_output / "trajectory.png")
        result = {
            "schema": "genrecon.internet-video-colmap-quality",
            "schema_version": SCHEMA_VERSION,
            "candidate_id": candidate["candidate_id"],
            "status": "completed",
            "completed_utc": utc_now(),
            "elapsed_s": time.monotonic() - started,
            "input": {
                "frame_count": frame_count,
                "image_path": str(rgb_path),
                "mask_path": str(masks_path),
                "camera_mode": "SINGLE",
                "camera_model": "SIMPLE_RADIAL",
            },
            "configuration": {
                "pycolmap_version": pycolmap.__version__,
                "pycolmap_has_cuda": bool(pycolmap.has_cuda),
                "max_image_size": args.colmap_max_image_size,
                "max_num_features": args.colmap_max_features,
                "matcher": matcher_name,
                "sequential_overlap": args.sequential_overlap,
                "guided_matching": True,
                "random_seed": 42,
                "scale_status": "arbitrary_monocular_sfm_scale",
                "gravity_status": "not_aligned",
            },
            "dynamic_masks": mask_summary["summary"],
            "mask_application": mask_application,
            "model_count": len(model_metrics),
            "registered_union_count": len(union_registered),
            "largest_model_registered_dominance": (
                largest_model["registered_images"] / len(union_registered)
                if largest_model is not None and union_registered
                else 0.0
            ),
            "largest_model": largest_model,
            "models": model_metrics,
        }
        result["quality_gate"] = grade_sfm_quality(result)
    except Exception as error:
        result = {
            "schema": "genrecon.internet-video-colmap-quality",
            "schema_version": SCHEMA_VERSION,
            "candidate_id": candidate["candidate_id"],
            "status": "failed",
            "completed_utc": utc_now(),
            "elapsed_s": time.monotonic() - started,
            "error_type": type(error).__name__,
            "error": str(error),
            "quality_gate": {
                "grade": "F",
                "gate": "fail",
                "reasons": ["colmap_exception"],
                "warnings": [],
            },
        }
    write_json(quality_path, result)
    print(
        f"[{candidate['candidate_id']}] COLMAP {result['status']} "
        f"grade={result['quality_gate']['grade']} elapsed={result['elapsed_s']:.1f}s",
        flush=True,
    )
    return result


def refresh_colmap_quality(
    candidate: dict[str, Any], output_root: Path
) -> dict[str, Any] | None:
    pycolmap = _import_pycolmap()
    candidate_output = output_root / "candidates" / candidate["candidate_id"]
    quality_path = candidate_output / "quality.json"
    if not quality_path.is_file():
        return None
    quality = json.loads(quality_path.read_text())
    quality["quality_protocol_version"] = 2
    quality["quality_refreshed_utc"] = utc_now()
    if quality.get("status") != "completed":
        write_json(quality_path, quality)
        return quality
    frame_count = int((quality.get("input") or {}).get("frame_count", 0))
    sparse_path = candidate_output / "sparse"
    model_metrics = []
    if sparse_path.is_dir():
        model_directories = sorted(
            (path for path in sparse_path.iterdir() if path.is_dir() and path.name.isdigit()),
            key=lambda path: int(path.name),
        )
        for model_path in model_directories:
            reconstruction = pycolmap.Reconstruction(str(model_path))
            metrics = reconstruction_metrics(reconstruction, frame_count)
            metrics["model_id"] = int(model_path.name)
            model_metrics.append(metrics)
    model_metrics.sort(
        key=lambda item: (item["registered_images"], item["points3D"]), reverse=True
    )
    largest_model = model_metrics[0] if model_metrics else None
    union_registered = {
        name for model in model_metrics for name in model["registered_image_names"]
    }
    quality["model_count"] = len(model_metrics)
    quality["registered_union_count"] = len(union_registered)
    quality["largest_model_registered_dominance"] = (
        largest_model["registered_images"] / len(union_registered)
        if largest_model is not None and union_registered
        else 0.0
    )
    quality["largest_model"] = largest_model
    quality["models"] = model_metrics
    quality["quality_gate"] = grade_sfm_quality(quality)
    write_json(quality_path, quality)
    return quality


def _relative_link(target: Path, root: Path) -> str:
    return os.path.relpath(target, root).replace(os.sep, "/")


def write_summary_index(
    raw_index_path: Path,
    raw_document: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    candidates = []
    for raw in raw_document["candidates"]:
        directory = output_root / "candidates" / raw["candidate_id"]
        preparation_path = directory / "preparation_status.json"
        quality_path = directory / "quality.json"
        preparation = json.loads(preparation_path.read_text()) if preparation_path.is_file() else None
        quality = json.loads(quality_path.read_text()) if quality_path.is_file() else None
        selection_path = directory / "selection.json"
        selection = json.loads(selection_path.read_text()) if selection_path.is_file() else None
        candidates.append(
            {
                "candidate_id": raw["candidate_id"],
                "title": raw["title"],
                "category": raw["category"],
                "raw_source_page": raw["source_page"],
                "raw_poster": _relative_link(
                    raw_index_path.parent / raw["poster"], output_root
                ),
                "preparation": preparation,
                "selection": selection["selected"] if selection else None,
                "quality": quality,
                "artifacts": {
                    "selection": _relative_link(selection_path, output_root) if selection_path.is_file() else None,
                    "mask_contact": _relative_link(directory / "mask_contact.jpg", output_root) if (directory / "mask_contact.jpg").is_file() else None,
                    "qc_contact": _relative_link(directory / "qc_contact.jpg", output_root) if (directory / "qc_contact.jpg").is_file() else None,
                    "trajectory": _relative_link(directory / "trajectory.png", output_root) if (directory / "trajectory.png").is_file() else None,
                    "quality": _relative_link(quality_path, output_root) if quality_path.is_file() else None,
                    "colmap": _relative_link(directory / "colmap", output_root) if (directory / "colmap").is_dir() else None,
                },
            }
        )
    grades = Counter(
        item["quality"]["quality_gate"]["grade"]
        for item in candidates
        if item["quality"] is not None
    )
    summary = {
        "schema": "genrecon.internet-video-sfm-preproducts-index",
        "schema_version": SCHEMA_VERSION,
        "generated_utc": utc_now(),
        "source_index": str(raw_index_path),
        "summary": {
            "candidate_count": len(candidates),
            "prepared_count": sum(
                item["preparation"] is not None and item["preparation"].get("status") == "prepared"
                for item in candidates
            ),
            "colmap_completed_count": sum(
                item["quality"] is not None and item["quality"].get("status") == "completed"
                for item in candidates
            ),
            "grades": dict(sorted(grades.items())),
        },
        "candidates": candidates,
    }
    write_json(output_root / "index.json", summary)

    rows = []
    for item in candidates:
        quality = item["quality"] or {}
        gate = quality.get("quality_gate", {"grade": "-", "gate": "pending"})
        largest = quality.get("largest_model") or {}
        selection = item["selection"] or {}
        dynamic = (quality.get("dynamic_masks") or {}).get("dynamic_fraction", {})
        rows.append(
            {
                "id": item["candidate_id"],
                "title": item["title"],
                "grade": gate.get("grade", "-"),
                "gate": gate.get("gate", "pending"),
                "shot": (
                    f"{selection.get('start_s', 0):.1f}-{selection.get('end_s', 0):.1f}s"
                    if selection
                    else "-"
                ),
                "frames": (item["preparation"] or {}).get("frame_count", 0),
                "dynamic": dynamic.get("mean"),
                "registered": largest.get("registered_fraction"),
                "points": largest.get("points3D"),
                "error": largest.get("mean_reprojection_error_px"),
                "track": (largest.get("track_length") or {}).get("mean"),
                "baseline_depth": (largest.get("viewpoint_geometry") or {}).get(
                    "baseline_to_observed_depth"
                ),
                "triangulation_angle": (
                    (largest.get("viewpoint_geometry") or {}).get("triangulation_angle_deg") or {}
                ).get("median"),
                "warnings": ", ".join(gate.get("reasons", []) + gate.get("warnings", [])),
                "artifacts": item["artifacts"],
                "poster": item["raw_poster"],
            }
        )
    with (output_root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "title",
                "grade",
                "gate",
                "shot",
                "frames",
                "dynamic",
                "registered",
                "points",
                "error",
                "track",
                "baseline_depth",
                "triangulation_angle",
                "warnings",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    def metric(value: Any, suffix: str = "", digits: int = 2) -> str:
        number = finite_float(value)
        return "-" if number is None else f"{number:.{digits}f}{suffix}"

    cards = []
    for row in rows:
        artifacts = row["artifacts"]
        image = artifacts["qc_contact"] or artifacts["mask_contact"] or row["poster"]
        links = []
        for label, key in [("selection", "selection"), ("masks", "mask_contact"), ("quality", "quality"), ("trajectory", "trajectory"), ("COLMAP", "colmap")]:
            if artifacts.get(key):
                links.append(f'<a href="{html.escape(artifacts[key])}">{label}</a>')
        cards.append(
            f"""<article class="card grade-{html.escape(row['grade'])}">
<div class="head"><div><span>{html.escape(row['id'])}</span><h2>{html.escape(row['title'])}</h2></div><strong>{html.escape(row['grade'])}</strong></div>
<a href="{html.escape(image)}"><img src="{html.escape(image)}" loading="lazy" alt="QC contact"></a>
<div class="metrics"><span>shot<b>{html.escape(row['shot'])}</b></span><span>frames<b>{row['frames']}</b></span><span>dynamic<b>{metric(None if row['dynamic'] is None else 100*row['dynamic'], '%', 1)}</b></span><span>registered<b>{metric(None if row['registered'] is None else 100*row['registered'], '%', 1)}</b></span><span>points<b>{row['points'] or '-'}</b></span><span>error<b>{metric(row['error'], ' px')}</b></span><span>track<b>{metric(row['track'])}</b></span><span>baseline/depth<b>{metric(row['baseline_depth'])}</b></span><span>tri-angle<b>{metric(row['triangulation_angle'], ' deg')}</b></span></div>
<p>{html.escape(row['warnings'] or 'No gate warning')}</p><nav>{' '.join(links)}</nav></article>"""
        )
    html_page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Internet SfM preproducts</title><style>
:root{{--ink:#20252b;--muted:#687078;--line:#d8dde1;--bg:#f2f4f5;--green:#187a5a;--amber:#a36016;--red:#aa3838}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif;letter-spacing:0}}header{{background:#fff;border-bottom:1px solid var(--line);padding:18px 22px}}h1{{font-size:23px;margin:0 0 4px}}header p{{margin:0;color:var(--muted)}}main{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;padding:18px;max-width:1500px;margin:auto}}.card{{background:#fff;border:1px solid var(--line);border-top:4px solid var(--muted);border-radius:6px;overflow:hidden}}.grade-A,.grade-B{{border-top-color:var(--green)}}.grade-C{{border-top-color:var(--amber)}}.grade-F{{border-top-color:var(--red)}}.head{{display:flex;justify-content:space-between;gap:12px;padding:12px 14px}}.head span{{font:12px ui-monospace,monospace;color:var(--muted)}}h2{{font-size:16px;line-height:1.25;margin:3px 0 0}}.head strong{{font-size:22px}}img{{display:block;width:100%;aspect-ratio:5/2;object-fit:contain;background:#111}}.metrics{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));border-top:1px solid var(--line);border-bottom:1px solid var(--line)}}.metrics span{{min-width:0;padding:8px;border-right:1px solid var(--line);border-bottom:1px solid var(--line);font-size:11px;color:var(--muted)}}.metrics b{{display:block;color:var(--ink);font-size:13px;white-space:normal;overflow-wrap:anywhere}}.card p{{margin:0;padding:10px 14px;color:var(--muted);min-height:42px;overflow-wrap:anywhere}}nav{{padding:0 14px 12px}}nav a{{margin-right:13px;color:#176b57}}@media(max-width:800px){{main{{grid-template-columns:1fr;padding:10px}}.metrics{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
</style></head><body><header><h1>Internet indoor SfM preproducts</h1><p>Masked COLMAP QC: A/B pass, C marginal, F fail. Scale is arbitrary and gravity is not aligned.</p><p>Prepared {summary['summary']['prepared_count']}/{len(candidates)} | COLMAP {summary['summary']['colmap_completed_count']}/{len(candidates)} | Grades {html.escape(str(summary['summary']['grades']))}</p></header><main>{''.join(cards)}</main></body></html>"""
    (output_root / "index.html").write_text(html_page)
    return summary


def validate_preproducts(
    raw_document: dict[str, Any], output_root: Path
) -> dict[str, Any]:
    pycolmap = _import_pycolmap()
    errors: list[str] = []
    counts = {
        "candidates": 0,
        "frames": 0,
        "dynamic_masks": 0,
        "colmap_masks": 0,
        "database_keypoints": 0,
        "database_blocked_keypoints": 0,
        "completed_colmap": 0,
        "timeout_colmap": 0,
        "best_models": 0,
        "best_model_points3D": 0,
    }
    for candidate in raw_document["candidates"]:
        identifier = candidate["candidate_id"]
        directory = output_root / "candidates" / identifier
        counts["candidates"] += 1
        try:
            preparation = json.loads((directory / "preparation_status.json").read_text())
            if preparation.get("status") != "prepared":
                raise ValueError(f"preparation status is {preparation.get('status')}")
            frames = json.loads((directory / "frames.json").read_text())
            masks = json.loads((directory / "mask_summary.json").read_text())
            rgb_paths = sorted((directory / "rgb").glob("frame_*.jpg"))
            dynamic_paths = sorted((directory / "masks_dynamic").glob("frame_*.png"))
            colmap_paths = sorted((directory / "masks_colmap").glob("frame_*.jpg.png"))
            expected = int(frames["frame_count"])
            if not (expected == len(rgb_paths) == len(dynamic_paths) == len(colmap_paths)):
                raise ValueError("frame/mask file counts differ")
            if expected != int(masks["summary"]["frame_count"]):
                raise ValueError("mask summary frame count differs")
            for rgb_path, dynamic_path, colmap_path in zip(
                rgb_paths, dynamic_paths, colmap_paths
            ):
                with Image.open(rgb_path) as opened:
                    rgb_size = opened.size
                    opened.verify()
                with Image.open(dynamic_path) as opened:
                    dynamic = np.asarray(opened.convert("L"), dtype=np.uint8)
                    dynamic_size = opened.size
                with Image.open(colmap_path) as opened:
                    colmap_mask = np.asarray(opened.convert("L"), dtype=np.uint8)
                    colmap_size = opened.size
                if rgb_size != dynamic_size or rgb_size != colmap_size:
                    raise ValueError(f"mask dimensions differ for {rgb_path.name}")
                if not np.all((dynamic == 0) | (dynamic == 255)):
                    raise ValueError(f"dynamic mask is not binary for {rgb_path.name}")
                if not np.array_equal(255 - dynamic, colmap_mask):
                    raise ValueError(f"COLMAP mask is not inverse for {rgb_path.name}")
            counts["frames"] += expected
            counts["dynamic_masks"] += len(dynamic_paths)
            counts["colmap_masks"] += len(colmap_paths)

            quality = json.loads((directory / "quality.json").read_text())
            if int(quality.get("quality_protocol_version", 0)) != 2:
                raise ValueError("quality protocol is not v2")
            if quality.get("status") == "completed":
                counts["completed_colmap"] += 1
            elif quality.get("status") == "timeout":
                counts["timeout_colmap"] += 1
            else:
                raise ValueError(f"unexpected COLMAP status {quality.get('status')}")
            application = verify_colmap_keypoint_masks(
                pycolmap, directory / "database.db", directory / "masks_colmap"
            )
            counts["database_keypoints"] += application["total_keypoint_centers"]
            counts["database_blocked_keypoints"] += application[
                "keypoint_centers_on_blocked_pixels"
            ]
            if application["keypoint_centers_on_blocked_pixels"] != 0:
                raise ValueError("database contains keypoints on blocked mask pixels")
            recorded = quality.get("mask_application") or {}
            if application["total_keypoint_centers"] != recorded.get(
                "total_keypoint_centers"
            ):
                raise ValueError("recorded keypoint count differs from database")

            largest = quality.get("largest_model")
            if largest is not None:
                reconstruction = pycolmap.Reconstruction(str(directory / "sparse_best"))
                if reconstruction.num_reg_images() != largest["registered_images"]:
                    raise ValueError("best-model registered image count differs")
                if reconstruction.num_points3D() != largest["points3D"]:
                    raise ValueError("best-model point count differs")
                for filename in ("cameras.txt", "images.txt", "points3D.txt"):
                    if not (directory / "colmap" / filename).is_file():
                        raise ValueError(f"missing COLMAP text artifact {filename}")
                counts["best_models"] += 1
                counts["best_model_points3D"] += reconstruction.num_points3D()
        except Exception as error:
            errors.append(f"{identifier}: {type(error).__name__}: {error}")

    strict_json_count = 0
    for path in output_root.rglob("*.json"):
        try:
            json.loads(
                path.read_text(),
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite constant {value}")
                ),
            )
            strict_json_count += 1
        except Exception as error:
            errors.append(f"{path}: strict JSON failure: {error}")
    counts["strict_json_files"] = strict_json_count

    checkpoint = Path(
        json.loads((output_root / "config.json").read_text())["mask_model"]["checkpoint"]
    )
    expected_checkpoint_hash = json.loads((output_root / "config.json").read_text())[
        "mask_model"
    ]["checkpoint_sha256"]
    checkpoint_hash = sha256_file(checkpoint) if checkpoint.is_file() else None
    if checkpoint_hash != expected_checkpoint_hash:
        errors.append("dynamic-mask checkpoint SHA256 mismatch")

    review_index = output_root / "index.html"
    html_card_count = 0
    local_reference_count = 0
    if not review_index.is_file():
        errors.append("review index.html is missing")
    else:
        page = review_index.read_text()
        html_card_count = page.count('<article class="card')
        if html_card_count != counts["candidates"]:
            errors.append(f"review index card count is {html_card_count}")
        for reference in re.findall(r'(?:href|src)="([^"]+)"', page):
            if reference.startswith(("http://", "https://", "#")):
                continue
            local_reference_count += 1
            if not (output_root / reference).exists():
                errors.append(f"review index local reference is missing: {reference}")
    screenshot_dimensions = {}
    for filename, expected_size in (
        ("index_desktop.png", (1440, 1200)),
        ("index_mobile.png", (390, 844)),
    ):
        path = output_root / filename
        if not path.is_file():
            errors.append(f"review screenshot is missing: {filename}")
            continue
        with Image.open(path) as opened:
            screenshot_dimensions[filename] = list(opened.size)
            if opened.size != expected_size:
                errors.append(f"review screenshot has wrong dimensions: {filename}")
    counts["html_cards"] = html_card_count
    counts["html_local_references"] = local_reference_count

    result = {
        "schema": "genrecon.internet-video-sfm-preproducts-validation",
        "schema_version": SCHEMA_VERSION,
        "validated_utc": utc_now(),
        "result": "pass" if not errors else "fail",
        "counts": counts,
        "mask_model_checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_hash,
            "matches_manifest": checkpoint_hash == expected_checkpoint_hash,
        },
        "checks": {
            "all_rgb_decodable": not any("image" in error.lower() for error in errors),
            "all_masks_binary_inverse_and_dimension_matched": not any(
                "mask" in error.lower() and "checkpoint" not in error.lower()
                for error in errors
            ),
            "all_database_keypoints_respect_masks": counts[
                "database_blocked_keypoints"
            ]
            == 0,
            "all_best_models_match_quality_json": not any(
                "best-model" in error for error in errors
            ),
            "strict_json": not any("strict JSON" in error for error in errors),
            "review_index_has_all_candidates_and_local_artifacts": not any(
                "review index" in error for error in errors
            ),
            "review_screenshot_dimensions": screenshot_dimensions,
        },
        "errors": errors,
    }
    write_json(output_root / "validation.json", result)
    if errors:
        raise RuntimeError(f"Preproduct validation failed with {len(errors)} error(s)")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "colmap", "quality", "summarize", "validate", "all")
    )
    parser.add_argument("--raw-index", type=Path, default=DEFAULT_RAW_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--ffmpeg")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-colmap", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scene-analysis-fps", type=float, default=2.0)
    parser.add_argument("--scene-cut-threshold", type=float, default=0.35)
    parser.add_argument("--scene-cut-spacing", type=float, default=1.5)
    parser.add_argument("--minimum-shot-duration", type=float, default=8.0)
    parser.add_argument("--maximum-shot-duration", type=float, default=90.0)
    parser.add_argument("--window-stride", type=float, default=45.0)
    parser.add_argument("--maximum-window-candidates", type=int, default=14)
    parser.add_argument("--selection-samples", type=int, default=5)
    parser.add_argument("--frame-fps", type=float, default=2.0)
    parser.add_argument("--maximum-frames", type=int, default=180)
    parser.add_argument("--max-image-size", type=int, default=1600)
    parser.add_argument("--ffmpeg-threads", type=int, default=4)
    parser.add_argument("--mask-score-threshold", type=float, default=0.35)
    parser.add_argument("--mask-dilation-fraction", type=float, default=0.008)
    parser.add_argument("--mask-batch-size", type=int, default=2)
    parser.add_argument("--colmap-max-image-size", type=int, default=1280)
    parser.add_argument("--colmap-max-features", type=int, default=4096)
    parser.add_argument("--colmap-threads", type=int, default=min(16, os.cpu_count() or 8))
    parser.add_argument("--colmap-max-runtime", type=int, default=300)
    parser.add_argument("--exhaustive-max-images", type=int, default=60)
    parser.add_argument("--sequential-overlap", type=int, default=10)
    return parser


def selected_candidates(document: dict[str, Any], identifiers: Sequence[str]) -> list[dict[str, Any]]:
    candidates = document["candidates"]
    if not identifiers:
        return candidates
    requested = set(identifiers)
    chosen = [item for item in candidates if item["candidate_id"] in requested]
    missing = requested - {item["candidate_id"] for item in chosen}
    if missing:
        raise ValueError(f"Unknown candidate IDs: {sorted(missing)}")
    return chosen


def write_run_config(
    output_root: Path,
    args: argparse.Namespace,
    raw_index: Path,
    model_provenance: dict[str, Any] | None,
) -> None:
    existing_path = output_root / "config.json"
    if model_provenance is None and existing_path.is_file():
        model_provenance = json.loads(existing_path.read_text()).get("mask_model")
    payload = {
        "schema": "genrecon.internet-video-sfm-preproducts-config",
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "raw_index": str(raw_index),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "mask_model": model_provenance,
        "limitations": [
            "Automatic shot selection is a preflight heuristic and requires human review.",
            "Semantic masks cannot identify every moving object or distinguish a moving chair from a static chair.",
            "Monocular COLMAP scale is arbitrary and gravity is not aligned.",
        ],
    }
    write_json(output_root / "config.json", payload)


def main() -> None:
    args = build_parser().parse_args()
    raw_index = args.raw_index.resolve()
    output_root = args.output.resolve()
    raw_document = json.loads(raw_index.read_text())
    raw_root = raw_index.parent
    candidates = selected_candidates(raw_document, args.candidate)
    output_root.mkdir(parents=True, exist_ok=True)
    model_provenance = None

    if args.stage in {"prepare", "all"}:
        ffmpeg = ffmpeg_executable(args.ffmpeg)
        model, torch_module, categories, model_provenance = load_dynamic_model(args.device)
        write_run_config(output_root, args, raw_index, model_provenance)
        for candidate in candidates:
            prepare_candidate(
                raw_root,
                candidate,
                output_root,
                ffmpeg,
                model,
                torch_module,
                categories,
                model_provenance,
                args,
            )
    if args.stage in {"colmap", "all"}:
        write_run_config(output_root, args, raw_index, model_provenance)
        for candidate in candidates:
            run_colmap_candidate(candidate, output_root, args)
    if args.stage in {"quality", "all"}:
        for candidate in candidates:
            result = refresh_colmap_quality(candidate, output_root)
            if result is not None:
                print(
                    f"[{candidate['candidate_id']}] quality v2 "
                    f"grade={result['quality_gate']['grade']}",
                    flush=True,
                )
    if args.stage in {"summarize", "all"}:
        write_summary_index(raw_index, raw_document, output_root)
        print(f"[summary] {output_root / 'index.html'}", flush=True)
    if args.stage in {"validate", "all"}:
        result = validate_preproducts(raw_document, output_root)
        print(
            f"[validation] {result['result']} frames={result['counts']['frames']} "
            f"keypoints={result['counts']['database_keypoints']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
