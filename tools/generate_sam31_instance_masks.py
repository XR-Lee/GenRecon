#!/usr/bin/env python3
"""Generate crop-aligned SAM3.1 masks from COLMAP track point prompts.

The evaluation protocol matches the existing SAM2 fallback experiment where
possible: the same scene crops, 70/30 point-ID split, part ROIs, and farthest
positive prompts are used. SAM3.1 multiplex currently exposes point-only
interactive prompts, so the SAM2 point-plus-box result is retained as a
reference rather than treated as a strictly identical prompt protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

try:
    from tools.evaluate_view_fidelity import read_colmap_images, read_colmap_points
    from tools.generate_sam2_instance_masks import (
        _overlay,
        _part_index,
        farthest_points_2d,
        project_world_points,
    )
except ModuleNotFoundError:
    from evaluate_view_fidelity import read_colmap_images, read_colmap_points
    from generate_sam2_instance_masks import (
        _overlay,
        _part_index,
        farthest_points_2d,
        project_world_points,
    )


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def collect_output_masks(outputs: dict[str, Any], shape: tuple[int, int]) -> dict[int, np.ndarray]:
    """Return nonempty SAM masks keyed by object ID."""
    ids = np.asarray(outputs.get("out_obj_ids", []), dtype=np.int64).reshape(-1)
    masks = np.asarray(outputs.get("out_binary_masks", []), dtype=bool)
    if masks.size == 0:
        return {}
    masks = masks.reshape(-1, *shape)
    if len(ids) != len(masks):
        raise ValueError(f"SAM3.1 returned {len(ids)} IDs but {len(masks)} masks")
    return {int(object_id): mask for object_id, mask in zip(ids, masks) if mask.any()}


def binary_mask_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    if candidate.shape != reference.shape:
        raise ValueError(f"mask shape mismatch: {candidate.shape} != {reference.shape}")
    candidate = candidate.astype(bool)
    reference = reference.astype(bool)
    intersection = int(np.logical_and(candidate, reference).sum())
    union = int(np.logical_or(candidate, reference).sum())
    return {
        "intersection": intersection,
        "union": union,
        "iou": intersection / union if union else 1.0,
        "candidate_recall_of_reference": intersection / int(reference.sum()) if reference.any() else 1.0,
        "reference_recall_of_candidate": intersection / int(candidate.sum()) if candidate.any() else 1.0,
    }


def _start_session_compat(predictor: Any, image_path: Path) -> str:
    """Start an image session while filtering an upstream SAM3.1 API mismatch."""
    session_id = str(uuid.uuid4())
    state = predictor.model.init_state(
        resource_path=str(image_path.resolve()),
        offload_video_to_cpu=True,
        async_loading_frames=False,
    )
    now = time.time()
    predictor._all_inference_states[session_id] = {
        "state": state,
        "session_id": session_id,
        "start_time": now,
        "last_use_time": now,
    }
    return session_id


def _inclusion(mask: np.ndarray, point_sets: list[np.ndarray]) -> tuple[int, int]:
    available = [points for points in point_sets if len(points)]
    if not available:
        return 0, 0
    all_points = np.concatenate(available, axis=0)
    height, width = mask.shape
    pixels = np.rint(all_points).astype(int)
    pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
    return int(mask[pixels[:, 1], pixels[:, 0]].sum()), len(pixels)


def _contact_sheet(items: list[tuple[int, np.ndarray]], path: Path, columns: int = 4) -> None:
    if not items:
        return
    thumb_w, thumb_h = 384, 384
    label_h = 28
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for position, (view_index, raster) in enumerate(items):
        row, column = divmod(position, columns)
        thumb = Image.fromarray(raster).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x, y = column * thumb_w, row * (thumb_h + label_h)
        sheet.paste(thumb, (x, y + label_h))
        draw.text((x + 8, y + 6), f"view {view_index:02d}", fill="black")
    sheet.save(path, quality=90)


def _weighted_recall(records: list[dict[str, Any]], prefix: str) -> float:
    total = sum(int(record[f"{prefix}_points"]) for record in records)
    inside = sum(int(record[f"{prefix}_inside"]) for record in records)
    return inside / total if total else 0.0


def _repo_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam3-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--cameras-json", type=Path, required=True)
    parser.add_argument("--chunk-layout-json", type=Path, required=True)
    parser.add_argument("--anchors-json", type=Path, required=True)
    parser.add_argument("--colmap-images", type=Path, required=True)
    parser.add_argument("--colmap-points", type=Path, required=True)
    parser.add_argument("--part-box", type=float, nargs=6, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-mask-dir", type=Path)
    parser.add_argument("--max-prompts-per-part", type=int, default=8)
    parser.add_argument("--view-indices", type=int, nargs="+")
    parser.add_argument("--output-prob-thresh", type=float, default=0.0)
    parser.add_argument("--max-prompt-attempts", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-num-objects", type=int, default=4)
    parser.add_argument("--multiplex-count", type=int, default=16)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SAM3.1 multiplex evaluation requires CUDA")
    if args.multiplex_count != 16:
        raise ValueError("The supplied SAM3.1 checkpoint is fixed to multiplex_count=16")
    args.output.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.sam3_repo.resolve()))
    from sam3.model_builder import build_sam3_predictor

    camera_document = json.loads(args.cameras_json.read_text())
    layout = json.loads(args.chunk_layout_json.read_text())
    anchors = json.loads(args.anchors_json.read_text())
    images = read_colmap_images(args.colmap_images)
    points = read_colmap_points(args.colmap_points)
    by_name = {Path(record.name).name: record for record in images.values()}
    train_ids = set(anchors["training_point_ids"])
    holdout_ids = set(anchors["holdout_point_ids"])

    size = float(layout["chunk_size_m"])
    center0 = np.asarray(layout["centers"][0], dtype=np.float64)
    world_to_chunk0 = np.eye(4)
    world_to_chunk0[:3, :3] *= 1.0 / size
    world_to_chunk0[:3, 3] = -center0 / size

    requested = set(args.view_indices) if args.view_indices else set(range(len(camera_document["scene"])))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    build_started = time.perf_counter()
    predictor = build_sam3_predictor(
        checkpoint_path=str(args.checkpoint.resolve()),
        version="sam3.1",
        compile=False,
        warm_up=False,
        max_num_objects=max(args.max_num_objects, len(args.part_box)),
        multiplex_count=args.multiplex_count,
        use_fa3=False,
        use_rope_real=False,
        async_loading_frames=False,
        default_output_prob_thresh=args.output_prob_thresh,
    )
    build_seconds = time.perf_counter() - build_started
    parameter_count = sum(parameter.numel() for parameter in predictor.model.parameters())

    records: list[dict[str, Any]] = []
    overview_items: list[tuple[int, np.ndarray]] = []
    total_intersection = 0
    total_union = 0
    evaluation_started = time.perf_counter()

    for view_index, camera in enumerate(camera_document["scene"]):
        if view_index not in requested:
            continue
        view_started = time.perf_counter()
        image_path = args.scene_dir / f"view_{view_index:03d}.png"
        raster = np.asarray(Image.open(image_path).convert("RGB"))
        height, width = raster.shape[:2]
        image_name = Path(camera["img_path"]).name
        colmap_image = by_name[image_name]
        observed_ids = {
            int(point_id)
            for _, _, point_id in colmap_image.observations
            if int(point_id) in points
        }

        part_train_world: list[list[np.ndarray]] = [[] for _ in args.part_box]
        part_holdout_world: list[list[np.ndarray]] = [[] for _ in args.part_box]
        for point_id in observed_ids:
            point = points[point_id]
            part = _part_index(point.xyz, args.part_box)
            if part is None:
                continue
            if point_id in train_ids:
                part_train_world[part].append(point.xyz)
            elif point_id in holdout_ids:
                part_holdout_world[part].append(point.xyz)

        extrinsics = np.asarray(camera["extrinsics_c0"], dtype=np.float64)
        intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
        prompt_parts: list[np.ndarray] = []
        part_projected_train: list[np.ndarray] = []
        part_projected_holdout: list[np.ndarray] = []
        active_parts: list[int] = []
        for part_index in range(len(args.part_box)):
            train_world = np.asarray(part_train_world[part_index], dtype=np.float64).reshape(-1, 3)
            holdout_world = np.asarray(part_holdout_world[part_index], dtype=np.float64).reshape(-1, 3)
            train_uv, train_valid = project_world_points(
                train_world, world_to_chunk0, extrinsics, intrinsics
            )
            holdout_uv, holdout_valid = project_world_points(
                holdout_world, world_to_chunk0, extrinsics, intrinsics
            )
            train_px = train_uv[train_valid] * np.asarray([width, height])
            holdout_px = holdout_uv[holdout_valid] * np.asarray([width, height])
            if not len(train_px):
                continue
            prompt_parts.append(farthest_points_2d(train_px, args.max_prompts_per_part))
            part_projected_train.append(train_px)
            part_projected_holdout.append(holdout_px)
            active_parts.append(part_index)

        object_masks: dict[int, np.ndarray] = {}
        prompt_seconds = 0.0
        prompt_attempts: dict[str, int] = {}
        missing_parts: list[int] = []
        if active_parts:
            session_id = _start_session_compat(predictor, image_path)
            try:
                for part_index, prompts in zip(active_parts, prompt_parts):
                    object_id = part_index + 1
                    relative_points = prompts / np.asarray([width, height])
                    attempts = 0
                    while object_id not in object_masks and attempts < args.max_prompt_attempts:
                        attempts += 1
                        prompt_started = time.perf_counter()
                        response = predictor.handle_request(
                            {
                                "type": "add_prompt",
                                "session_id": session_id,
                                "frame_index": 0,
                                "points": torch.as_tensor(relative_points, dtype=torch.float32),
                                "point_labels": torch.ones(len(prompts), dtype=torch.int32),
                                "obj_id": object_id,
                                "clear_old_points": True,
                                "rel_coordinates": True,
                                "output_prob_thresh": args.output_prob_thresh,
                            }
                        )
                        prompt_seconds += time.perf_counter() - prompt_started
                        object_masks.update(
                            collect_output_masks(response["outputs"], (height, width))
                        )
                    prompt_attempts[str(part_index)] = attempts
                    if object_id not in object_masks:
                        missing_parts.append(part_index)
            finally:
                predictor.handle_request(
                    {
                        "type": "close_session",
                        "session_id": session_id,
                        "run_gc_collect": True,
                        "clear_cache_threshold": 100,
                    }
                )

        union_mask = np.zeros((height, width), dtype=bool)
        for mask in object_masks.values():
            union_mask |= mask
        train_inside, train_total = _inclusion(union_mask, part_projected_train)
        holdout_inside, holdout_total = _inclusion(union_mask, part_projected_holdout)

        comparison = None
        if args.reference_mask_dir is not None:
            reference_path = args.reference_mask_dir / f"view_{view_index:03d}.png"
            reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
            if reference is None:
                raise FileNotFoundError(reference_path)
            comparison = binary_mask_metrics(union_mask, reference > 0)
            total_intersection += int(comparison["intersection"])
            total_union += int(comparison["union"])

        Image.fromarray(union_mask.astype(np.uint8) * 255).save(
            args.output / f"view_{view_index:03d}.png"
        )
        for part_index in range(len(args.part_box)):
            part_mask = object_masks.get(part_index + 1, np.zeros_like(union_mask))
            Image.fromarray(part_mask.astype(np.uint8) * 255).save(
                args.output / f"view_{view_index:03d}_part_{part_index}.png"
            )
        overlay = _overlay(raster, union_mask)
        Image.fromarray(overlay).save(
            args.output / f"view_{view_index:03d}_overlay.jpg", quality=90
        )
        overview_items.append((view_index, overlay))
        record = {
            "view_index": view_index,
            "image": image_name,
            "active_parts": active_parts,
            "missing_parts": missing_parts,
            "prompt_counts": [len(prompt) for prompt in prompt_parts],
            "prompt_attempts": prompt_attempts,
            "mask_fraction": float(union_mask.mean()),
            "part_mask_fractions": {
                str(part_index): float(object_masks.get(part_index + 1, np.zeros_like(union_mask)).mean())
                for part_index in active_parts
            },
            "training_inside": train_inside,
            "training_points": train_total,
            "training_recall": train_inside / train_total if train_total else None,
            "holdout_inside": holdout_inside,
            "holdout_points": holdout_total,
            "holdout_recall": holdout_inside / holdout_total if holdout_total else None,
            "prompt_seconds": prompt_seconds,
            "view_seconds": time.perf_counter() - view_started,
            "sam2_reference": comparison,
        }
        records.append(record)
        print(
            f"[sam3.1] view={view_index:02d} parts={active_parts} missing={missing_parts} "
            f"area={union_mask.mean():.3f} train={train_inside}/{train_total} "
            f"holdout={holdout_inside}/{holdout_total} prompt={prompt_seconds:.3f}s"
        )

    evaluation_seconds = time.perf_counter() - evaluation_started
    reference_records = [record for record in records if record["sam2_reference"] is not None]
    summary = {
        "schema": "genrecon.sam31-instance-masks",
        "protocol": {
            "prompt_type": "positive COLMAP track points only",
            "max_prompts_per_part": args.max_prompts_per_part,
            "max_identical_prompt_attempts": args.max_prompt_attempts,
            "identical_retry_reason": "SAM3.1 multiplex can return an empty first-object output; retry adds no prompt information",
            "part_boxes_world": args.part_box,
            "point_id_split": "same fixed 70/30 train/holdout split as SAM2 fallback",
            "sam2_reference_prompt_type": "positive points plus track-extent boxes",
            "strictly_prompt_matched_to_sam2": False,
            "part_masks_persisted": True,
            "part_mask_filename": "view_NNN_part_P.png",
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "sam3_repo": str(args.sam3_repo),
            "sam3_repo_commit": _repo_commit(args.sam3_repo),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "parameter_count": parameter_count,
            "build_seconds": build_seconds,
            "evaluation_seconds": evaluation_seconds,
            "prompt_seconds": float(sum(record["prompt_seconds"] for record in records)),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "compile": False,
            "flash_attention_3": False,
            "autocast": "bfloat16",
            "model_parameter_dtype": "float32",
            "multiplex_count": args.multiplex_count,
        },
        "views": records,
        "aggregate": {
            "view_count": len(records),
            "active_view_count": sum(bool(record["active_parts"]) for record in records),
            "missing_part_predictions": sum(len(record["missing_parts"]) for record in records),
            "mean_mask_fraction": float(np.mean([record["mask_fraction"] for record in records])),
            "training_recall": _weighted_recall(records, "training"),
            "holdout_recall": _weighted_recall(records, "holdout"),
            "mean_prompt_seconds_per_active_view": float(
                np.mean([record["prompt_seconds"] for record in records if record["active_parts"]])
            ) if any(record["active_parts"] for record in records) else 0.0,
            "sam2_reference_mean_view_iou": float(
                np.mean([record["sam2_reference"]["iou"] for record in reference_records])
            ) if reference_records else None,
            "sam2_reference_global_pixel_iou": total_intersection / total_union if total_union else None,
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _contact_sheet(overview_items, args.output / "overview.jpg")
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
