#!/usr/bin/env python3
"""Project SfM-to-GLB geometric errors onto the original registered RGB views."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

try:
    from tools.evaluate_view_fidelity import (
        ColmapCamera,
        ColmapImage,
        ColmapPoint,
        read_colmap_cameras,
        read_colmap_images,
        read_colmap_points,
    )
except ModuleNotFoundError:  # Direct execution puts tools/ on sys.path.
    from evaluate_view_fidelity import (
        ColmapCamera,
        ColmapImage,
        ColmapPoint,
        read_colmap_cameras,
        read_colmap_images,
        read_colmap_points,
    )


class OverlayError(RuntimeError):
    pass


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n")


def load_point_surface_metrics(path: str | Path) -> dict[int, dict[str, float | int]]:
    metrics: dict[int, dict[str, float | int]] = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            point_id = int(row["point_id"])
            if point_id < 0:
                continue
            metrics[point_id] = {
                "distance_m": float(row["distance_m"]),
                "inside_glb_aabb": int(row["inside_glb_aabb"]),
            }
    if not metrics:
        raise OverlayError(f"No point metrics found in {path}")
    return metrics


def _load_depth(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image, dtype=np.uint16)
    return values.astype(np.float32) / 1000.0


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _load_render(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    return rgba[:, :, :3], rgba[:, :, 3] > 127


def blend_rgb_glb(
    original: np.ndarray,
    glb_render: np.ndarray,
    glb_mask: np.ndarray,
    *,
    glb_weight: float = 0.5,
) -> np.ndarray:
    if original.shape != glb_render.shape or original.shape[:2] != glb_mask.shape:
        raise ValueError("Original, GLB render, and GLB mask dimensions must match")
    if not 0 <= glb_weight <= 1:
        raise ValueError("GLB blend weight must be in [0, 1]")
    result = original.astype(np.float32).copy()
    result[glb_mask] = (
        (1 - glb_weight) * original[glb_mask].astype(np.float32)
        + glb_weight * glb_render[glb_mask].astype(np.float32)
    )
    return np.rint(result).astype(np.uint8)


def collect_view_errors(
    record: ColmapImage,
    camera: ColmapCamera,
    points: dict[int, ColmapPoint],
    direct_metrics: dict[int, dict[str, float | int]],
    glb_depth: np.ndarray,
) -> dict[str, np.ndarray]:
    height, width = glb_depth.shape
    sx, sy = width / camera.width, height / camera.height
    pixels: list[tuple[int, int]] = []
    point_ids: list[int] = []
    direct_distance: list[float] = []
    visibility_absolute: list[float] = []
    visibility_signed: list[float] = []
    depth_valid: list[bool] = []

    for source_x, source_y, point_id_value in record.observations:
        point_id = int(point_id_value)
        metric = direct_metrics.get(point_id)
        point = points.get(point_id)
        if point is None or metric is None or int(metric["inside_glb_aabb"]) != 1:
            continue
        camera_point = record.world_to_camera[:3, :3] @ point.xyz + record.world_to_camera[:3, 3]
        if camera_point[2] <= 0:
            continue
        column = int(np.clip(round(source_x * sx), 0, width - 1))
        row = int(np.clip(round(source_y * sy), 0, height - 1))
        rendered_depth = float(glb_depth[row, column])
        valid = rendered_depth > 0
        signed = rendered_depth - float(camera_point[2]) if valid else float("nan")
        pixels.append((column, row))
        point_ids.append(point_id)
        direct_distance.append(float(metric["distance_m"]))
        visibility_signed.append(signed)
        visibility_absolute.append(abs(signed) if valid else float("nan"))
        depth_valid.append(valid)

    return {
        "depth_valid": np.asarray(depth_valid, dtype=bool),
        "direct_distance_m": np.asarray(direct_distance, dtype=np.float32),
        "pixels_xy": np.asarray(pixels, dtype=np.int32).reshape(-1, 2),
        "point_id": np.asarray(point_ids, dtype=np.int64),
        "visibility_absolute_m": np.asarray(visibility_absolute, dtype=np.float32),
        "visibility_signed_m": np.asarray(visibility_signed, dtype=np.float32),
    }


def _colors(values: np.ndarray, *, maximum: float, cmap_name: str, signed: bool = False) -> np.ndarray:
    from matplotlib import colormaps

    if signed:
        normalized = (np.clip(values, -maximum, maximum) + maximum) / (2 * maximum)
    else:
        normalized = np.clip(values, 0, maximum) / maximum
    return np.rint(colormaps[cmap_name](normalized)[:, :3] * 255).astype(np.uint8)


def draw_point_overlay(
    original: np.ndarray,
    pixels: np.ndarray,
    values: np.ndarray,
    *,
    maximum: float,
    cmap_name: str,
    signed: bool = False,
    missing: np.ndarray | None = None,
    radius: int = 4,
) -> np.ndarray:
    if len(pixels) != len(values):
        raise ValueError("pixels and values must have equal length")
    result = original.astype(np.float32).copy()
    color_layer = np.zeros_like(original)
    alpha = np.zeros(original.shape[:2], dtype=np.uint8)
    finite = np.isfinite(values)
    colors = _colors(values[finite], maximum=maximum, cmap_name=cmap_name, signed=signed)

    for (column, row), color in zip(pixels[finite], colors):
        center = (int(column), int(row))
        cv2.circle(color_layer, center, radius + 1, (12, 12, 12), -1, lineType=cv2.LINE_AA)
        cv2.circle(alpha, center, radius + 1, 205, -1, lineType=cv2.LINE_AA)
        cv2.circle(color_layer, center, radius, tuple(int(value) for value in color), -1, lineType=cv2.LINE_AA)
        cv2.circle(alpha, center, radius, 235, -1, lineType=cv2.LINE_AA)

    if missing is not None:
        for column, row in pixels[np.asarray(missing, dtype=bool)]:
            center = (int(column), int(row))
            cv2.drawMarker(
                color_layer,
                center,
                (185, 185, 185),
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=9,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
            cv2.drawMarker(
                alpha,
                center,
                230,
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=9,
                thickness=2,
                line_type=cv2.LINE_AA,
            )

    weight = alpha.astype(np.float32)[:, :, None] / 255.0
    result = result * (1 - weight) + color_layer.astype(np.float32) * weight
    return np.rint(np.clip(result, 0, 255)).astype(np.uint8)


def _panel(
    image: np.ndarray,
    *,
    title: str,
    legend_left: str = "",
    legend_center: str = "",
    legend_right: str = "",
    cmap_name: str | None = None,
    signed: bool = False,
) -> np.ndarray:
    height, width = image.shape[:2]
    top_height, bottom_height = 34, 48
    canvas = Image.new("RGB", (width, height + top_height + bottom_height), (24, 24, 24))
    canvas.paste(Image.fromarray(image), (0, top_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 9), title, fill=(245, 245, 245))
    if cmap_name is not None:
        gradient_width = min(420, width - 80)
        gradient = np.linspace(-1 if signed else 0, 1, gradient_width, dtype=np.float32)
        maximum = 1.0
        gradient_colors = _colors(
            gradient,
            maximum=maximum,
            cmap_name=cmap_name,
            signed=signed,
        )[None, :, :]
        gradient_image = Image.fromarray(np.repeat(gradient_colors, 12, axis=0))
        left = (width - gradient_width) // 2
        top = top_height + height + 5
        canvas.paste(gradient_image, (left, top))
        text_y = top + 16
        draw.text((left, text_y), legend_left, fill=(230, 230, 230))
        center_box = draw.textbbox((0, 0), legend_center)
        center_width = center_box[2] - center_box[0]
        draw.text((width // 2 - center_width // 2, text_y), legend_center, fill=(230, 230, 230))
        right_box = draw.textbbox((0, 0), legend_right)
        right_width = right_box[2] - right_box[0]
        draw.text((left + gradient_width - right_width, text_y), legend_right, fill=(230, 230, 230))
    return np.asarray(canvas)


def _distance_stats(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0}
    return {
        "count": int(len(finite)),
        "mean_m": float(np.mean(finite)),
        "median_m": float(np.median(finite)),
        "p90_m": float(np.quantile(finite, 0.90)),
        "within_0.05m": float(np.mean(finite <= 0.05)),
        "within_0.10m": float(np.mean(finite <= 0.10)),
        "within_0.20m": float(np.mean(finite <= 0.20)),
    }


def _view_summary(samples: dict[str, np.ndarray]) -> dict[str, Any]:
    valid = samples["depth_valid"]
    signed = samples["visibility_signed_m"][valid]
    return {
        "direct_surface": _distance_stats(samples["direct_distance_m"]),
        "eligible_observations": int(len(valid)),
        "visibility_absolute": _distance_stats(samples["visibility_absolute_m"][valid]),
        "visibility_depth_coverage": float(valid.mean()) if len(valid) else float("nan"),
        "visibility_missing": int((~valid).sum()),
        "visibility_signed": {
            "count": int(len(signed)),
            "glb_behind_over_0.10m": float(np.mean(signed > 0.10)) if len(signed) else float("nan"),
            "glb_in_front_over_0.10m": float(np.mean(signed < -0.10)) if len(signed) else float("nan"),
            "mean_m": float(np.mean(signed)) if len(signed) else float("nan"),
            "median_m": float(np.median(signed)) if len(signed) else float("nan"),
        },
    }


def _write_view(
    output: Path,
    original: np.ndarray,
    glb_render: np.ndarray,
    glb_mask: np.ndarray,
    samples: dict[str, np.ndarray],
    *,
    image_name: str,
    group: str,
    direct_max_m: float,
    depth_max_m: float,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    pixels = samples["pixels_xy"]
    rgb_glb_blend = blend_rgb_glb(original, glb_render, glb_mask, glb_weight=0.5)
    glb_display = glb_render.copy()
    glb_display[~glb_mask] = 0
    direct = draw_point_overlay(
        rgb_glb_blend,
        pixels,
        samples["direct_distance_m"],
        maximum=direct_max_m,
        cmap_name="turbo",
    )
    missing = ~samples["depth_valid"]
    absolute = draw_point_overlay(
        rgb_glb_blend,
        pixels,
        samples["visibility_absolute_m"],
        maximum=depth_max_m,
        cmap_name="turbo",
        missing=missing,
    )
    signed = draw_point_overlay(
        rgb_glb_blend,
        pixels,
        samples["visibility_signed_m"],
        maximum=depth_max_m,
        cmap_name="coolwarm",
        signed=True,
        missing=missing,
    )

    original_panel = _panel(original, title=f"1. Original RGB | {image_name} | {group}")
    glb_panel = _panel(glb_display, title="2. Same-camera GLB albedo render | black = missing")
    blend_panel = _panel(rgb_glb_blend, title="3. Original RGB / GLB render | 50:50 on GLB mask")
    direct_panel = _panel(
        direct,
        title=f"4. Exact SfM-to-surface error on RGB/GLB blend | {len(pixels)} observations",
        legend_left="0",
        legend_center=f"{direct_max_m * 50:.0f}cm",
        legend_right=f">={direct_max_m * 100:.0f}cm",
        cmap_name="turbo",
    )
    absolute_panel = _panel(
        absolute,
        title="5. Visibility-aware absolute depth error | gray X = no GLB depth",
        legend_left="0",
        legend_center=f"{depth_max_m * 50:.0f}cm",
        legend_right=f">={depth_max_m * 100:.0f}cm",
        cmap_name="turbo",
    )
    signed_panel = _panel(
        signed,
        title="6. Signed GLB depth - SfM depth on RGB/GLB blend",
        legend_left=f"-{depth_max_m * 100:.0f}cm GLB in front",
        legend_center="0",
        legend_right=f"+{depth_max_m * 100:.0f}cm GLB behind",
        cmap_name="coolwarm",
        signed=True,
    )
    Image.fromarray(blend_panel).save(output / "rgb_glb_50_50_overlay.jpg", quality=94)
    Image.fromarray(direct_panel).save(output / "surface_distance_overlay.jpg", quality=94)
    Image.fromarray(absolute_panel).save(output / "visibility_absolute_overlay.jpg", quality=94)
    Image.fromarray(signed_panel).save(output / "visibility_signed_overlay.jpg", quality=94)
    comparison = np.concatenate(
        [
            np.concatenate([original_panel, glb_panel, blend_panel], axis=1),
            np.concatenate([direct_panel, absolute_panel, signed_panel], axis=1),
        ]
    )
    Image.fromarray(comparison).save(output / "comparison.jpg", quality=92, subsampling=0)
    summary = _view_summary(samples)
    summary.update({"group": group, "image": image_name})
    _write_json(output / "metrics.json", summary)
    return summary


def _aggregate_samples(samples_by_group: dict[str, list[dict[str, np.ndarray]]], group: str) -> dict[str, Any]:
    selected = (
        samples_by_group["input"] + samples_by_group["heldout"]
        if group == "all"
        else samples_by_group[group]
    )
    if not selected:
        return {"views": 0}
    combined: dict[str, np.ndarray] = {}
    for key in selected[0]:
        combined[key] = np.concatenate([item[key] for item in selected], axis=0)
    summary = _view_summary(combined)
    summary["views"] = len(selected)
    return summary


def _write_contact_sheets(output: Path) -> None:
    contact_root = output / "contact_sheets"
    contact_root.mkdir(parents=True, exist_ok=True)
    representatives: list[Path] = []
    for group in ("input", "heldout"):
        paths = sorted((output / group).glob("*/comparison.jpg"))
        if paths:
            representatives.extend(
                paths[index] for index in np.linspace(0, len(paths) - 1, min(4, len(paths)), dtype=int)
            )
        for page, start in enumerate(range(0, len(paths), 8), start=1):
            images = []
            for path in paths[start : start + 8]:
                image = Image.open(path).convert("RGB")
                image.thumbnail((1200, 880), Image.Resampling.LANCZOS)
                images.append(image.copy())
            if not images:
                continue
            canvas = Image.new("RGB", (max(item.width for item in images), sum(item.height for item in images)))
            top = 0
            for image in images:
                canvas.paste(image, (0, top))
                top += image.height
            canvas.save(contact_root / f"{group}_{page:02d}.jpg", quality=90)

    overview_images = []
    for path in representatives:
        image = Image.open(path).convert("RGB")
        image.thumbnail((1100, 800), Image.Resampling.LANCZOS)
        overview_images.append(image.copy())
    if overview_images:
        canvas = Image.new(
            "RGB", (max(item.width for item in overview_images), sum(item.height for item in overview_images))
        )
        top = 0
        for image in overview_images:
            canvas.paste(image, (0, top))
            top += image.height
        canvas.save(output / "overview.jpg", quality=92)


def _pct(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.1%}"


def _cm(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value * 100:.2f}cm"


def _write_report(output: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# 原始 RGB 视角下的 SfM→GLB 几何误差 Overlay",
        "",
        "每个彩色点对应 COLMAP `images.txt` 中的真实二维观测。只使用 `error≤2px、track≥3` 且位于 GLB AABB 内的三维点。",
        "没有对稀疏点做全图插值，因此黑白 RGB 区域不代表零误差，而是没有可用 SfM 几何观测。",
        "",
        "## 汇总（按观测计数）",
        "",
        "| 子集 | 视角 | 观测 | 直接最近面中位误差 | 直接≤10cm | GLB深度命中 | 可见深度中位误差 | 可见≤10cm | 错误前表面>10cm | 后表面>10cm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("input", "heldout", "all"):
        item = summary["subsets"][group]
        direct = item["direct_surface"]
        visibility = item["visibility_absolute"]
        signed = item["visibility_signed"]
        lines.append(
            f"| {group} | {item['views']} | {item['eligible_observations']:,} | "
            f"{_cm(direct['median_m'])} | {_pct(direct['within_0.10m'])} | "
            f"{_pct(item['visibility_depth_coverage'])} | {_cm(visibility['median_m'])} | "
            f"{_pct(visibility['within_0.10m'])} | {_pct(signed['glb_in_front_over_0.10m'])} | "
            f"{_pct(signed['glb_behind_over_0.10m'])} |"
        )
    lines.extend(
        [
            "",
            "## 六联图与色标",
            "",
            "第一行依次是原始 RGB、同位姿 GLB albedo 渲染、以及仅在 GLB mask 内做的 50:50 RGB/GLB overlay。",
            "第二行的三种几何误差都叠加在该 50:50 overlay 上，而不是只叠加在原图上。",
            "",
            "- `rgb_glb_50_50_overlay.jpg`：原图与 GLB 渲染的配准 overlay。",
            "- `surface_distance_overlay.jpg`：SfM 点到任意 GLB 三角面的精确3D距离，0–30cm。",
            "- `visibility_absolute_overlay.jpg`：原相机射线上 SfM 深度与首个 GLB 表面的绝对误差，0–50cm；灰叉表示无 GLB 深度。",
            "- `visibility_signed_overlay.jpg`：`GLB depth - SfM depth`；蓝色负值表示 GLB 错误地挡在 SfM 点前，红色正值表示 GLB 在点后。",
            "- `comparison.jpg`：上述内容的六联图。",
            "",
            "## 目录",
            "",
            "- `input/`、`heldout/`：逐相机 overlay、六联图和指标。",
            "- `contact_sheets/`：全部44个视角的分页总览。",
            "- `overview.jpg`：8个代表视角。",
            "- `summary.json`：按输入/留出/全部观测聚合的数值。",
            "",
            "这些 overlay 是稀疏 SfM 几何诊断，不是稠密深度误差图。要得到逐像素几何误差，需要 COLMAP dense depth/MVS 或对齐激光深度。",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines))


def render(args: argparse.Namespace) -> dict[str, Any]:
    cameras = read_colmap_cameras(args.cameras)
    images = read_colmap_images(args.images)
    points = read_colmap_points(args.points3d)
    direct_metrics = load_point_surface_metrics(args.point_metrics)
    samples_by_group: dict[str, list[dict[str, np.ndarray]]] = defaultdict(list)
    view_rows = []

    for index, record in enumerate(sorted(images.values(), key=lambda item: item.name), start=1):
        image_name = Path(record.name).name
        stem = Path(image_name).stem
        input_source = args.view_root / "input" / stem
        heldout_source = args.view_root / "heldout" / stem
        if input_source.is_dir():
            group, source = "input", input_source
        elif heldout_source.is_dir():
            group, source = "heldout", heldout_source
        else:
            raise OverlayError(f"No rendered view directory found for {image_name}")
        original = _load_rgb(source / "original.jpg")
        glb_render, glb_mask = _load_render(source / "glb_render.png")
        depth = _load_depth(source / "glb_depth_mm.png")
        camera = cameras[record.camera_id]
        samples = collect_view_errors(record, camera, points, direct_metrics, depth)
        destination = args.output / group / stem
        row = _write_view(
            destination,
            original,
            glb_render,
            glb_mask,
            samples,
            image_name=image_name,
            group=group,
            direct_max_m=args.direct_max_error,
            depth_max_m=args.depth_max_error,
        )
        samples_by_group[group].append(samples)
        view_rows.append(row)
        print(
            f"[sfm-overlay] {index:02d}/{len(images)} {image_name}: "
            f"points={len(samples['point_id'])}, depth={samples['depth_valid'].mean() if len(samples['point_id']) else 0:.1%}"
        )

    summary = {
        "inputs": {
            "cameras": args.cameras,
            "images": args.images,
            "point_metrics": args.point_metrics,
            "points3d": args.points3d,
            "view_root": args.view_root,
        },
        "scales": {
            "direct_surface_max_m": args.direct_max_error,
            "visibility_depth_max_m": args.depth_max_error,
        },
        "schema": "genrecon.sfm-error-rgb-overlays",
        "schema_version": 1,
        "subsets": {
            group: _aggregate_samples(samples_by_group, group) for group in ("input", "heldout", "all")
        },
        "views": view_rows,
    }
    _write_json(args.output / "summary.json", summary)
    _write_contact_sheets(args.output)
    _write_report(args.output, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--points3d", type=Path, required=True)
    parser.add_argument("--point-metrics", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--direct-max-error", type=float, default=0.30)
    parser.add_argument("--depth-max-error", type=float, default=0.50)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for path in (args.cameras, args.images, args.points3d, args.point_metrics, args.view_root):
        if not path.exists():
            raise SystemExit(f"Required input does not exist: {path}")
    if args.direct_max_error <= 0 or args.depth_max_error <= 0:
        raise SystemExit("Error-map scales must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    summary = render(args)
    print(json.dumps(summary["subsets"]["all"], indent=2))


if __name__ == "__main__":
    main()
