#!/usr/bin/env python3
"""Generate crop-aligned SAM2 instance masks from COLMAP track prompts.

COLMAP supplies cross-view identity and visible prompt points. SAM2 is used only
to densify those sparse observations into per-crop masks; no heldout track is
used as a prompt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor

try:
    from tools.evaluate_view_fidelity import read_colmap_images, read_colmap_points
except ModuleNotFoundError:
    from evaluate_view_fidelity import read_colmap_images, read_colmap_points


def farthest_points_2d(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) <= count:
        return points
    first = int(np.argmax(np.linalg.norm(points - points.mean(axis=0), axis=1)))
    chosen = [first]
    min_distance = np.sum((points - points[first]) ** 2, axis=1)
    while len(chosen) < count:
        index = int(np.argmax(min_distance))
        chosen.append(index)
        min_distance = np.minimum(min_distance, np.sum((points - points[index]) ** 2, axis=1))
    return points[np.asarray(chosen)]


def project_world_points(
    xyz_world: np.ndarray,
    world_to_chunk0: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    world_h = np.concatenate([xyz_world, np.ones((len(xyz_world), 1))], axis=1)
    chunk0 = world_h @ world_to_chunk0.T
    camera = chunk0 @ extrinsics.T
    depth = camera[:, 2]
    normalized = camera[:, :3] / np.where(depth[:, None] > 1e-8, depth[:, None], 1.0)
    uv = normalized @ intrinsics.T
    valid = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < 1) & (uv[:, 1] >= 0) & (uv[:, 1] < 1)
    return uv[:, :2], valid


def choose_mask(masks: torch.Tensor, scores: torch.Tensor, prompt_points: np.ndarray) -> tuple[np.ndarray, int]:
    candidates = masks.numpy().astype(bool)
    height, width = candidates.shape[-2:]
    pixels = np.rint(prompt_points).astype(int)
    pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
    ranking = []
    for index, candidate in enumerate(candidates):
        recall = float(candidate[pixels[:, 1], pixels[:, 0]].mean()) if len(pixels) else 0.0
        ranking.append((recall, float(scores[index]), -float(candidate.mean()), index))
    selected = max(ranking)[-1]
    return candidates[selected], int(selected)


def _part_index(xyz: np.ndarray, boxes: list[list[float]]) -> int | None:
    for index, (x0, x1, y0, y1, z0, z1) in enumerate(boxes):
        if x0 <= xyz[0] <= x1 and y0 <= xyz[1] <= y1 and z0 <= xyz[2] <= z1:
            return index
    return None


def _padded_prompts(part_points: list[np.ndarray]) -> tuple[list[list[list[list[float]]]], list[list[list[int]]]]:
    longest = max(len(points) for points in part_points)
    points_out = []
    labels_out = []
    for points in part_points:
        padded = points.tolist() + [[-10.0, -10.0]] * (longest - len(points))
        labels = [1] * len(points) + [-10] * (longest - len(points))
        points_out.append(padded)
        labels_out.append(labels)
    return [points_out], [labels_out]


def _overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = image.astype(np.float32).copy()
    tint = np.zeros_like(result)
    tint[..., 1] = 255
    result[mask] = 0.55 * result[mask] + 0.45 * tint[mask]
    edges = cv2.Canny((mask.astype(np.uint8) * 255), 50, 150) > 0
    result[edges] = np.asarray([255, 255, 255])
    return np.clip(result, 0, 255).astype(np.uint8)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--cameras-json", type=Path, required=True)
    parser.add_argument("--chunk-layout-json", type=Path, required=True)
    parser.add_argument("--anchors-json", type=Path, required=True)
    parser.add_argument("--colmap-images", type=Path, required=True)
    parser.add_argument("--colmap-points", type=Path, required=True)
    parser.add_argument("--part-box", type=float, nargs=6, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-prompts-per-part", type=int, default=8)
    parser.add_argument("--box-padding-px", type=float, default=20.0)
    parser.add_argument("--point-only", action="store_true")
    parser.add_argument("--single-mask", action="store_true")
    parser.add_argument("--view-indices", type=int, nargs="+")
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
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

    device = torch.device(args.device)
    processor = Sam2Processor.from_pretrained(args.model, local_files_only=True)
    model = Sam2Model.from_pretrained(args.model, local_files_only=True, dtype=torch.bfloat16).to(device)
    model.eval()

    requested = set(args.view_indices) if args.view_indices else set(range(len(camera_document["scene"])))
    records: list[dict[str, Any]] = []
    overview_items = []
    for view_index, camera in enumerate(camera_document["scene"]):
        if view_index not in requested:
            continue
        image_path = args.scene_dir / f"view_{view_index:03d}.png"
        image = Image.open(image_path).convert("RGB")
        raster = np.asarray(image)
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
        prompt_parts = []
        boxes = []
        part_projected_train = []
        part_projected_holdout = []
        active_parts = []
        for part_index in range(len(args.part_box)):
            train_world = np.asarray(part_train_world[part_index], dtype=np.float64).reshape(-1, 3)
            holdout_world = np.asarray(part_holdout_world[part_index], dtype=np.float64).reshape(-1, 3)
            train_uv, train_valid = project_world_points(train_world, world_to_chunk0, extrinsics, intrinsics)
            holdout_uv, holdout_valid = project_world_points(holdout_world, world_to_chunk0, extrinsics, intrinsics)
            train_px = train_uv[train_valid] * np.asarray([width, height])
            holdout_px = holdout_uv[holdout_valid] * np.asarray([width, height])
            if not len(train_px):
                continue
            prompts = farthest_points_2d(train_px, args.max_prompts_per_part)
            lower = np.maximum(train_px.min(axis=0) - args.box_padding_px, 0)
            upper = np.minimum(train_px.max(axis=0) + args.box_padding_px, [width - 1, height - 1])
            prompt_parts.append(prompts)
            boxes.append([float(lower[0]), float(lower[1]), float(upper[0]), float(upper[1])])
            part_projected_train.append(train_px)
            part_projected_holdout.append(holdout_px)
            active_parts.append(part_index)

        union = np.zeros((height, width), dtype=bool)
        selected_candidates = []
        if prompt_parts:
            input_points, input_labels = _padded_prompts(prompt_parts)
            processor_kwargs = {
                "images": image,
                "input_points": input_points,
                "input_labels": input_labels,
                "return_tensors": "pt",
            }
            if not args.point_only:
                processor_kwargs["input_boxes"] = [boxes]
            inputs = processor(**processor_kwargs).to(device)
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model(**inputs, multimask_output=not args.single_mask)
            masks = processor.post_process_masks(outputs.pred_masks.float().cpu(), inputs["original_sizes"].cpu())[0]
            scores = outputs.iou_scores.float().cpu()[0]
            for object_index, train_px in enumerate(part_projected_train):
                selected_mask, candidate_index = choose_mask(masks[object_index], scores[object_index], train_px)
                union |= selected_mask
                selected_candidates.append(candidate_index)
            del inputs, outputs, masks

        def inclusion(point_sets: list[np.ndarray]) -> tuple[int, int]:
            all_points = np.concatenate(point_sets, axis=0) if point_sets and any(len(item) for item in point_sets) else np.empty((0, 2))
            if not len(all_points):
                return 0, 0
            pixels = np.rint(all_points).astype(int)
            pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
            pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
            return int(union[pixels[:, 1], pixels[:, 0]].sum()), len(pixels)

        train_inside, train_total = inclusion(part_projected_train)
        holdout_inside, holdout_total = inclusion(part_projected_holdout)
        mask_u8 = union.astype(np.uint8) * 255
        Image.fromarray(mask_u8).save(args.output / f"view_{view_index:03d}.png")
        overlay = _overlay(raster, union)
        Image.fromarray(overlay).save(args.output / f"view_{view_index:03d}_overlay.jpg", quality=90)
        record = {
            "view_index": view_index,
            "image": image_name,
            "active_parts": active_parts,
            "selected_candidates": selected_candidates,
            "mask_fraction": float(union.mean()),
            "training_points": train_total,
            "training_recall": train_inside / train_total if train_total else None,
            "holdout_points": holdout_total,
            "holdout_recall": holdout_inside / holdout_total if holdout_total else None,
        }
        records.append(record)
        overview_items.append((view_index, overlay))
        print(
            f"[sam2] view={view_index:02d} parts={active_parts} area={union.mean():.3f} "
            f"train={train_inside}/{train_total} holdout={holdout_inside}/{holdout_total}"
        )

    summary = {
        "schema": "genrecon.sam2-instance-masks",
        "model": str(args.model),
        "protocol": {
            "prompt_type": "positive COLMAP track points only" if args.point_only else "positive COLMAP track points plus track-extent boxes",
            "multimask_output": not args.single_mask,
            "max_prompts_per_part": args.max_prompts_per_part,
        },
        "views": records,
        "aggregate": {
            "view_count": len(records),
            "mean_mask_fraction": float(np.mean([record["mask_fraction"] for record in records])),
            "training_recall": float(
                sum((record["training_recall"] or 0) * record["training_points"] for record in records)
                / max(1, sum(record["training_points"] for record in records))
            ),
            "holdout_recall": float(
                sum((record["holdout_recall"] or 0) * record["holdout_points"] for record in records)
                / max(1, sum(record["holdout_points"] for record in records))
            ),
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if overview_items:
        columns = 4
        rows = int(np.ceil(len(overview_items) / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(16, 4 * rows), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (view_index, overlay) in zip(axes.ravel(), overview_items):
            ax.imshow(overlay)
            ax.set_title(f"view {view_index:02d}")
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(args.output / "overview.jpg", dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    main()
