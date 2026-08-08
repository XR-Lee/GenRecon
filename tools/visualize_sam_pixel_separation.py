#!/usr/bin/env python3
"""Visualize pixel- and patch-level separation differences between SAM2 and SAM3.1."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

SAM2_COLOR = np.asarray([0, 188, 212], dtype=np.float32)
SAM3_COLOR = np.asarray([255, 145, 0], dtype=np.float32)
AGREE_COLOR = np.asarray([55, 200, 95], dtype=np.float32)
PART0_COLOR = np.asarray([0, 190, 220], dtype=np.float32)
PART1_COLOR = np.asarray([255, 190, 45], dtype=np.float32)
OVERLAP_COLOR = np.asarray([225, 45, 170], dtype=np.float32)
UNEXPLAINED_COLOR = np.asarray([230, 55, 55], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--sam2", type=Path, required=True)
    parser.add_argument("--sam31", type=Path, required=True)
    parser.add_argument("--sam2-summary", type=Path, required=True)
    parser.add_argument("--sam31-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-views", default="1,24,12,9,30,14")
    parser.add_argument("--patch-resolution", type=int, default=32)
    parser.add_argument("--patch-threshold", type=float, default=0.25)
    return parser.parse_args()


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0


def downsample_mask(path: Path, patch_resolution: int, threshold: float) -> np.ndarray:
    image = Image.open(path).convert("L")
    reduced = np.asarray(
        image.resize((patch_resolution, patch_resolution), Image.Resampling.BOX),
        dtype=np.float32,
    ) / 255.0
    return reduced >= threshold


def binary_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    return intersection / union if union else 1.0


def mask_boundary(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask)
    structure = ndimage.generate_binary_structure(2, 1)
    dilated = ndimage.binary_dilation(mask, structure=structure, iterations=radius)
    eroded = ndimage.binary_erosion(mask, structure=structure, iterations=radius)
    return dilated ^ eroded


def fragment_metrics(mask: np.ndarray, small_component_pixels: int = 64) -> dict[str, float | int]:
    labels, count = ndimage.label(mask)
    if not count:
        return {
            "components": 0,
            "significant_components": 0,
            "largest_component_fraction": 0.0,
            "small_component_pixel_fraction": 0.0,
        }
    areas = np.bincount(labels.ravel())[1:]
    total = int(areas.sum())
    return {
        "components": int(count),
        "significant_components": int(np.count_nonzero(areas >= small_component_pixels)),
        "largest_component_fraction": float(areas.max() / total),
        "small_component_pixel_fraction": float(areas[areas < small_component_pixels].sum() / total),
    }


def mask_metrics(
    sam2: np.ndarray,
    sam31: np.ndarray,
    sam2_patch: np.ndarray,
    sam31_patch: np.ndarray,
) -> dict[str, Any]:
    both = sam2 & sam31
    union = sam2 | sam31
    only2 = sam2 & ~sam31
    only3 = sam31 & ~sam2
    xor = only2 | only3
    boundary = mask_boundary(sam2) | mask_boundary(sam31)
    distance = ndimage.distance_transform_edt(~boundary) if boundary.any() else np.full(sam2.shape, np.inf)
    xor_count = int(xor.sum())
    union_count = int(union.sum())

    patch_union = sam2_patch | sam31_patch
    patch_only2 = sam2_patch & ~sam31_patch
    patch_only3 = sam31_patch & ~sam2_patch
    return {
        "pixels": int(sam2.size),
        "sam2_pixels": int(sam2.sum()),
        "sam31_pixels": int(sam31.sum()),
        "intersection_pixels": int(both.sum()),
        "union_pixels": union_count,
        "sam2_only_pixels": int(only2.sum()),
        "sam31_only_pixels": int(only3.sum()),
        "xor_pixels": xor_count,
        "iou": float(binary_iou(sam2, sam31)),
        "xor_fraction_of_union": float(xor_count / union_count) if union_count else 0.0,
        "xor_within_3px_boundary": float(np.count_nonzero(xor & (distance <= 3)) / xor_count)
        if xor_count
        else 1.0,
        "xor_within_10px_boundary": float(np.count_nonzero(xor & (distance <= 10)) / xor_count)
        if xor_count
        else 1.0,
        "patch_sam2": int(sam2_patch.sum()),
        "patch_sam31": int(sam31_patch.sum()),
        "patch_union": int(patch_union.sum()),
        "patch_sam2_only": int(patch_only2.sum()),
        "patch_sam31_only": int(patch_only3.sum()),
        "patch_iou": float(binary_iou(sam2_patch, sam31_patch)),
    }


def part_metrics(union: np.ndarray, parts: list[np.ndarray]) -> dict[str, Any]:
    if not parts:
        return {}
    parts_or = np.logical_or.reduce(parts)
    overlap = np.zeros_like(union)
    for left_index, left in enumerate(parts):
        for right in parts[left_index + 1 :]:
            overlap |= left & right
    unexplained = union & ~parts_or
    outside_union = parts_or & ~union
    return {
        "part_union_iou": float(binary_iou(union, parts_or)),
        "overlap_pixels": int(overlap.sum()),
        "unexplained_union_pixels": int(unexplained.sum()),
        "part_pixels_outside_union": int(outside_union.sum()),
        "parts": [fragment_metrics(part) for part in parts],
    }


def blend_mask(rgb: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float = 0.48) -> np.ndarray:
    result = rgb.astype(np.float32).copy()
    result[mask] = result[mask] * (1.0 - alpha) + color * alpha
    result[mask_boundary(mask, 2)] = color
    return np.clip(result, 0, 255).astype(np.uint8)


def difference_image(rgb: np.ndarray, sam2: np.ndarray, sam31: np.ndarray) -> np.ndarray:
    result = rgb.astype(np.float32) * 0.28
    for mask, color in (
        (sam2 & sam31, AGREE_COLOR),
        (sam2 & ~sam31, SAM2_COLOR),
        (sam31 & ~sam2, SAM3_COLOR),
    ):
        result[mask] = result[mask] * 0.25 + color * 0.75
    return np.clip(result, 0, 255).astype(np.uint8)


def parts_image(rgb: np.ndarray, union: np.ndarray, parts: list[np.ndarray]) -> np.ndarray:
    result = rgb.astype(np.float32) * 0.30
    if not parts:
        return np.clip(result, 0, 255).astype(np.uint8)
    part0 = parts[0]
    part1 = parts[1] if len(parts) > 1 else np.zeros_like(part0)
    overlap = part0 & part1
    unexplained = union & ~(part0 | part1)
    for mask, color in (
        (part0 & ~overlap, PART0_COLOR),
        (part1 & ~overlap, PART1_COLOR),
        (overlap, OVERLAP_COLOR),
        (unexplained, UNEXPLAINED_COLOR),
    ):
        result[mask] = result[mask] * 0.20 + color * 0.80
    return np.clip(result, 0, 255).astype(np.uint8)


def boundary_image(rgb: np.ndarray, sam2: np.ndarray, sam31: np.ndarray) -> np.ndarray:
    result = rgb.copy()
    result[mask_boundary(sam2, 2)] = SAM2_COLOR.astype(np.uint8)
    result[mask_boundary(sam31, 2)] = SAM3_COLOR.astype(np.uint8)
    return result


def patch_difference(sam2_patch: np.ndarray, sam31_patch: np.ndarray, scale: int = 16) -> np.ndarray:
    result = np.full((*sam2_patch.shape, 3), 22, dtype=np.uint8)
    result[sam2_patch & sam31_patch] = AGREE_COLOR.astype(np.uint8)
    result[sam2_patch & ~sam31_patch] = SAM2_COLOR.astype(np.uint8)
    result[sam31_patch & ~sam2_patch] = SAM3_COLOR.astype(np.uint8)
    return np.repeat(np.repeat(result, scale, axis=0), scale, axis=1)


def crop_bounds(mask: np.ndarray, margin: int = 48) -> tuple[slice, slice]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return slice(0, mask.shape[0]), slice(0, mask.shape[1])
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(mask.shape[1], int(xs.max()) + margin + 1)
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(mask.shape[0], int(ys.max()) + margin + 1)
    return slice(y0, y1), slice(x0, x1)


def save_view_panel(
    output: Path,
    index: int,
    image_name: str,
    rgb: np.ndarray,
    sam2: np.ndarray,
    sam31: np.ndarray,
    parts: list[np.ndarray],
    metrics: dict[str, Any],
) -> None:
    panels = [
        rgb,
        blend_mask(rgb, sam2, SAM2_COLOR),
        blend_mask(rgb, sam31, SAM3_COLOR),
        difference_image(rgb, sam2, sam31),
        parts_image(rgb, sam31, parts),
        boundary_image(rgb, sam2, sam31),
    ]
    titles = [
        f"view {index:02d} | {image_name}",
        f"SAM2 matched | area {metrics['sam2_pixels'] / metrics['pixels']:.1%}",
        f"SAM3.1 | area {metrics['sam31_pixels'] / metrics['pixels']:.1%}",
        f"agreement green | S2 cyan | S3 orange\nIoU {metrics['iou']:.3f}",
        "SAM3.1 parts | tabletop cyan | cabinet gold",
        f"boundaries | S2 cyan | S3 orange\npatch IoU {metrics['patch_iou']:.3f}",
    ]
    figure, axes = plt.subplots(2, 3, figsize=(16, 10))
    for axis, panel, title in zip(axes.ravel(), panels, titles):
        axis.imshow(panel)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    figure.tight_layout(pad=0.8)
    figure.savefig(output, dpi=140, pil_kwargs={"quality": 92})
    plt.close(figure)


def save_key_grid(records: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(len(records), 6, figsize=(19, 3.25 * len(records)), squeeze=False)
    titles = ["Original", "SAM2 matched", "SAM3.1", "Pixel agreement/difference", "SAM3.1 parts", "Boundaries"]
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=11)
    for row, record in enumerate(records):
        rgb = record["rgb"]
        sam2 = record["sam2"]
        sam31 = record["sam31"]
        panels = [
            rgb,
            blend_mask(rgb, sam2, SAM2_COLOR),
            blend_mask(rgb, sam31, SAM3_COLOR),
            difference_image(rgb, sam2, sam31),
            parts_image(rgb, sam31, record["parts"]),
            boundary_image(rgb, sam2, sam31),
        ]
        for column, panel in enumerate(panels):
            axes[row, column].imshow(panel)
            axes[row, column].axis("off")
        metrics = record["metrics"]
        axes[row, 0].set_ylabel(
            f"view {record['view_index']:02d}\n{record['image']}\npixel {metrics['iou']:.3f}\npatch {metrics['patch_iou']:.3f}",
            fontsize=9,
        )
    figure.tight_layout(pad=0.5)
    figure.savefig(path, dpi=130, pil_kwargs={"quality": 92})
    plt.close(figure)


def save_boundary_zooms(records: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(len(records), 3, figsize=(15, 4.0 * len(records)), squeeze=False)
    for row, record in enumerate(records):
        bounds = crop_bounds(record["sam2"] | record["sam31"])
        rgb = record["rgb"]
        panels = [
            rgb[bounds],
            boundary_image(rgb, record["sam2"], record["sam31"])[bounds],
            difference_image(rgb, record["sam2"], record["sam31"])[bounds],
        ]
        for column, panel in enumerate(panels):
            axes[row, column].imshow(panel)
            axes[row, column].axis("off")
        axes[row, 0].set_ylabel(f"view {record['view_index']:02d}\n{record['image']}", fontsize=9)
    for axis, title in zip(axes[0], ("Object crop", "S2 cyan / S3 orange boundaries", "Agreement and pixel differences")):
        axis.set_title(title, fontsize=11)
    figure.tight_layout(pad=0.6)
    figure.savefig(path, dpi=140, pil_kwargs={"quality": 92})
    plt.close(figure)


def save_patch_grid(records: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(len(records), 3, figsize=(10, 3.1 * len(records)), squeeze=False)
    for row, record in enumerate(records):
        sam2 = record["sam2_patch"]
        sam31 = record["sam31_patch"]
        axes[row, 0].imshow(sam2, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[row, 1].imshow(sam31, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[row, 2].imshow(patch_difference(sam2, sam31))
        for axis in axes[row]:
            axis.axis("off")
        axes[row, 0].set_ylabel(
            f"view {record['view_index']:02d}\npatch IoU {record['metrics']['patch_iou']:.3f}", fontsize=9
        )
    for axis, title in zip(axes[0], ("SAM2 32x32", "SAM3.1 32x32", "Both green / S2 cyan / S3 orange")):
        axis.set_title(title, fontsize=11)
    figure.tight_layout(pad=0.7)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_active_contact(records: list[dict[str, Any]], path: Path) -> None:
    columns = 4
    rows = int(np.ceil(len(records) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(16, 3.8 * rows), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, record in zip(axes.ravel(), records):
        axis.imshow(boundary_image(record["rgb"], record["sam2"], record["sam31"]))
        axis.set_title(
            f"view {record['view_index']:02d} {record['image']} | pixel {record['metrics']['iou']:.3f} | patch {record['metrics']['patch_iou']:.3f}",
            fontsize=9,
        )
        axis.axis("off")
    figure.tight_layout(pad=0.6)
    figure.savefig(path, dpi=130, pil_kwargs={"quality": 90})
    plt.close(figure)


def weighted_recall(views: list[dict[str, Any]], field: str) -> float | None:
    points_field = field.replace("_recall", "_points")
    total_points = sum(int(view.get(points_field, 0)) for view in views)
    if not total_points:
        return None
    inside = sum(
        float(view[field]) * int(view[points_field])
        for view in views
        if view.get(field) is not None
    )
    return inside / total_points


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    views_output = args.output / "views"
    views_output.mkdir(exist_ok=True)

    sam2_document = json.loads(args.sam2_summary.read_text())
    sam31_document = json.loads(args.sam31_summary.read_text())
    sam2_views = {int(view["view_index"]): view for view in sam2_document["views"]}
    sam31_views = {int(view["view_index"]): view for view in sam31_document["views"]}
    indices = sorted(set(sam2_views) | set(sam31_views))

    records: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    aggregate_counts = {
        key: 0
        for key in (
            "intersection_pixels",
            "union_pixels",
            "sam2_only_pixels",
            "sam31_only_pixels",
            "patch_intersection",
            "patch_union",
            "patch_sam2_only",
            "patch_sam31_only",
        )
    }

    for index in indices:
        rgb = np.asarray(Image.open(args.images / f"view_{index:03d}.png").convert("RGB"))
        sam2_path = args.sam2 / f"view_{index:03d}.png"
        sam31_path = args.sam31 / f"view_{index:03d}.png"
        sam2 = load_mask(sam2_path)
        sam31 = load_mask(sam31_path)
        if sam2.shape != sam31.shape or sam2.shape != rgb.shape[:2]:
            raise ValueError(f"shape mismatch for view {index}: {rgb.shape}, {sam2.shape}, {sam31.shape}")
        sam2_patch = downsample_mask(sam2_path, args.patch_resolution, args.patch_threshold)
        sam31_patch = downsample_mask(sam31_path, args.patch_resolution, args.patch_threshold)
        parts = []
        part_index = 0
        while (args.sam31 / f"view_{index:03d}_part_{part_index}.png").is_file():
            parts.append(load_mask(args.sam31 / f"view_{index:03d}_part_{part_index}.png"))
            part_index += 1

        metrics = mask_metrics(sam2, sam31, sam2_patch, sam31_patch)
        metrics["parts"] = part_metrics(sam31, parts)
        metadata2 = sam2_views[index]
        metadata31 = sam31_views[index]
        image_name = metadata31["image"]
        active = bool(metrics["union_pixels"])
        row = {
            "view_index": index,
            "image": image_name,
            "active": active,
            "active_parts": metadata31.get("active_parts", []),
            **metrics,
            "sam2_training_recall": metadata2.get("training_recall"),
            "sam2_holdout_recall": metadata2.get("holdout_recall"),
            "sam31_training_recall": metadata31.get("training_recall"),
            "sam31_holdout_recall": metadata31.get("holdout_recall"),
        }
        summary_rows.append(row)
        record = {
            "view_index": index,
            "image": image_name,
            "rgb": rgb,
            "sam2": sam2,
            "sam31": sam31,
            "sam2_patch": sam2_patch,
            "sam31_patch": sam31_patch,
            "parts": parts,
            "metrics": metrics,
        }
        records.append(record)
        if active:
            save_view_panel(
                views_output / f"view_{index:03d}_pixel_separation.jpg",
                index,
                image_name,
                rgb,
                sam2,
                sam31,
                parts,
                metrics,
            )
            aggregate_counts["intersection_pixels"] += metrics["intersection_pixels"]
            aggregate_counts["union_pixels"] += metrics["union_pixels"]
            aggregate_counts["sam2_only_pixels"] += metrics["sam2_only_pixels"]
            aggregate_counts["sam31_only_pixels"] += metrics["sam31_only_pixels"]
            aggregate_counts["patch_intersection"] += int((sam2_patch & sam31_patch).sum())
            aggregate_counts["patch_union"] += metrics["patch_union"]
            aggregate_counts["patch_sam2_only"] += metrics["patch_sam2_only"]
            aggregate_counts["patch_sam31_only"] += metrics["patch_sam31_only"]

    active_records = [record for record in records if record["metrics"]["union_pixels"]]
    key_indices = [int(value) for value in args.key_views.split(",") if value.strip()]
    key_records = [next(record for record in records if record["view_index"] == index) for index in key_indices]
    save_key_grid(key_records, args.output / "key_views_pixel_separation.jpg")
    save_boundary_zooms(key_records, args.output / "key_view_boundary_zooms.jpg")
    save_patch_grid(key_records, args.output / "key_views_patch_masks.png")
    save_active_contact(active_records, args.output / "all_active_boundaries.jpg")

    active_rows = [row for row in summary_rows if row["active"]]
    pixel_ious = np.asarray([row["iou"] for row in active_rows], dtype=np.float64)
    patch_ious = np.asarray([row["patch_iou"] for row in active_rows], dtype=np.float64)
    global_summary = {
        "active_views": len(active_rows),
        "pixel_global_iou": aggregate_counts["intersection_pixels"] / aggregate_counts["union_pixels"],
        "pixel_mean_active_iou": float(pixel_ious.mean()),
        "pixel_median_active_iou": float(np.median(pixel_ious)),
        "pixel_views_below_0_5_iou": [row["view_index"] for row in active_rows if row["iou"] < 0.5],
        "pixel_views_at_least_0_9_iou": int(np.count_nonzero(pixel_ious >= 0.9)),
        "patch_global_iou": aggregate_counts["patch_intersection"] / aggregate_counts["patch_union"],
        "patch_mean_active_iou": float(patch_ious.mean()),
        "patch_median_active_iou": float(np.median(patch_ious)),
        "patch_views_below_0_5_iou": [row["view_index"] for row in active_rows if row["patch_iou"] < 0.5],
        "sam2_only_fraction_of_pixel_union": aggregate_counts["sam2_only_pixels"] / aggregate_counts["union_pixels"],
        "sam31_only_fraction_of_pixel_union": aggregate_counts["sam31_only_pixels"] / aggregate_counts["union_pixels"],
        "sam2_only_fraction_of_patch_union": aggregate_counts["patch_sam2_only"] / aggregate_counts["patch_union"],
        "sam31_only_fraction_of_patch_union": aggregate_counts["patch_sam31_only"] / aggregate_counts["patch_union"],
        "sam2_training_track_recall": weighted_recall(list(sam2_views.values()), "training_recall"),
        "sam2_holdout_track_recall": weighted_recall(list(sam2_views.values()), "holdout_recall"),
        "sam31_training_track_recall": weighted_recall(list(sam31_views.values()), "training_recall"),
        "sam31_holdout_track_recall": weighted_recall(list(sam31_views.values()), "holdout_recall"),
        "key_views": key_indices,
    }
    document = {
        "schema": "genrecon.sam-pixel-separation",
        "protocol": {
            "sam2": "SAM2.1 point-only single-mask",
            "sam31": "SAM3.1 point-only multiplex single-mask",
            "dense_instance_gt_available": False,
            "patch_resolution": args.patch_resolution,
            "patch_threshold": args.patch_threshold,
            "difference_colors": {
                "agreement": "green",
                "sam2_only": "cyan",
                "sam31_only": "orange",
                "sam31_tabletop": "cyan",
                "sam31_cabinet_front": "gold",
            },
        },
        "global": global_summary,
        "views": summary_rows,
    }
    (args.output / "summary.json").write_text(json.dumps(document, indent=2) + "\n")

    with (args.output / "summary.csv").open("w", newline="") as handle:
        fields = [
            "view_index",
            "image",
            "active",
            "iou",
            "patch_iou",
            "sam2_pixels",
            "sam31_pixels",
            "sam2_only_pixels",
            "sam31_only_pixels",
            "xor_within_3px_boundary",
            "xor_within_10px_boundary",
            "sam2_holdout_recall",
            "sam31_holdout_recall",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({field: row.get(field) for field in fields})

    table_rows = "\n".join(
        "<tr>"
        f"<td>{row['view_index']:02d}</td><td>{row['image']}</td>"
        f"<td>{row['iou']:.3f}</td><td>{row['patch_iou']:.3f}</td>"
        f"<td>{row['sam2_only_pixels']:,}</td><td>{row['sam31_only_pixels']:,}</td>"
        f"<td><a href='views/view_{row['view_index']:03d}_pixel_separation.jpg'>panel</a></td>"
        "</tr>"
        for row in sorted(active_rows, key=lambda item: item["iou"])
    )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>SAM pixel separation</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;max-width:1500px}}img{{max-width:100%;border:1px solid #ccc}}table{{border-collapse:collapse;width:100%}}th,td{{padding:6px 9px;border-bottom:1px solid #ddd;text-align:right}}th:nth-child(2),td:nth-child(2){{text-align:left}}h1,h2{{letter-spacing:0}}</style></head>
<body><h1>SAM2.1 vs SAM3.1 pixel separation</h1>
<p>Active views: {global_summary['active_views']}; global pixel IoU: {global_summary['pixel_global_iou']:.3f}; global 32x32 patch IoU: {global_summary['patch_global_iou']:.3f}.</p>
<h2>Key views</h2><img src="key_views_pixel_separation.jpg">
<h2>Boundary zooms</h2><img src="key_view_boundary_zooms.jpg">
<h2>Actual 32x32 ownership masks</h2><img src="key_views_patch_masks.png">
<h2>All active boundaries</h2><img src="all_active_boundaries.jpg">
<h2>Per-view metrics</h2><table><thead><tr><th>View</th><th>Image</th><th>Pixel IoU</th><th>Patch IoU</th><th>SAM2-only px</th><th>SAM3-only px</th><th>Detail</th></tr></thead><tbody>{table_rows}</tbody></table>
</body></html>"""
    (args.output / "index.html").write_text(html)


if __name__ == "__main__":
    main()
