#!/usr/bin/env python3
"""Create final camera, depth, metric, and chunk-layout visualizations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def find_view(root: Path, stem: str) -> Path:
    matches = list((root / "fidelity" / "views").glob(f"*/{stem}"))
    if len(matches) != 1:
        raise ValueError(f"expected one fidelity view for {stem} under {root}, found {len(matches)}")
    return matches[0]


def load_rgb(path: Path, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize(size, Image.Resampling.LANCZOS))


def label_image(image: np.ndarray, label: str, label_height: int = 32) -> Image.Image:
    raster = Image.fromarray(image)
    canvas = Image.new("RGB", (raster.width, raster.height + label_height), "white")
    canvas.paste(raster, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), label, fill="black", font=font(16))
    return canvas


def overlay(original: np.ndarray, render: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = original.copy()
    valid = mask > 0
    result[valid] = np.clip(
        0.5 * original[valid].astype(np.float32) + 0.5 * render[valid].astype(np.float32),
        0,
        255,
    ).astype(np.uint8)
    return result


def key_view_sheet(
    baseline: Path,
    fused: Path,
    final: Path,
    safeguarded: Path,
    stems: list[str],
    output: Path,
) -> None:
    tile_size = (320, 214)
    columns = [
        "Original",
        "A baseline",
        "N2b fused",
        "Full obj-center direct",
        "Full safeguarded",
        "RGB/safeguarded 50:50",
    ]
    rows = []
    for stem in stems:
        base_view, fused_view, final_view, safeguarded_view = (
            find_view(root, stem) for root in (baseline, fused, final, safeguarded)
        )
        original = load_rgb(base_view / "original.jpg", tile_size)
        base_render = load_rgb(base_view / "ply_render.png", tile_size)
        fused_render = load_rgb(fused_view / "ply_render.png", tile_size)
        final_render = load_rgb(final_view / "ply_render.png", tile_size)
        safeguarded_render = load_rgb(safeguarded_view / "ply_render.png", tile_size)
        safeguarded_mask = np.asarray(
            Image.open(safeguarded_view / "ply_mask.png").convert("L").resize(
                tile_size, Image.Resampling.NEAREST
            )
        )
        images = [
            original,
            base_render,
            fused_render,
            final_render,
            safeguarded_render,
            overlay(original, safeguarded_render, safeguarded_mask),
        ]
        labelled = [label_image(image, f"{stem} | {column}") for image, column in zip(images, columns)]
        row = Image.new("RGB", (sum(item.width for item in labelled), labelled[0].height), "white")
        x = 0
        for item in labelled:
            row.paste(item, (x, 0))
            x += item.width
        rows.append(row)
    sheet = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet.save(output, quality=92, subsampling=0)


def all_view_contact(baseline: Path, final: Path, output: Path) -> None:
    stems = sorted(path.name for path in (baseline / "fidelity" / "views").glob("*/*"))
    tile_w, tile_h = 640, 160
    columns = 2
    rows = (len(stems) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_w, rows * (tile_h + 28)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, stem in enumerate(stems):
        base_view = find_view(baseline, stem)
        final_view = find_view(final, stem)
        original = load_rgb(base_view / "original.jpg", (tile_w // 3, tile_h))
        base_render = load_rgb(base_view / "ply_render.png", (tile_w // 3, tile_h))
        final_render = load_rgb(final_view / "ply_render.png", (tile_w - 2 * (tile_w // 3), tile_h))
        tile = np.concatenate([original, base_render, final_render], axis=1)
        row, column = divmod(index, columns)
        x, y = column * tile_w, row * (tile_h + 28)
        sheet.paste(Image.fromarray(tile), (x, y + 28))
        draw.text((x + 7, y + 6), f"{stem} | RGB / baseline / full obj-center", fill="black", font=font(15))
    sheet.save(output, quality=90, subsampling=0)


def depth_change_sheet(baseline: Path, final: Path, stems: list[str], output: Path) -> None:
    cmap = plt.get_cmap("coolwarm")
    tile_size = (480, 320)
    rows = []
    for stem in stems:
        base_view, final_view = find_view(baseline, stem), find_view(final, stem)
        original = load_rgb(base_view / "original.jpg", tile_size)
        base = cv2.imread(str(base_view / "ply_depth_mm.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
        final_depth = cv2.imread(str(final_view / "ply_depth_mm.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
        common = (base > 0) & (final_depth > 0)
        delta = np.zeros_like(base)
        delta[common] = final_depth[common] - base[common]
        normalized = np.clip((delta + 0.10) / 0.20, 0, 1)
        colored = (cmap(normalized)[..., :3] * 255).astype(np.uint8)
        colored[~common] = 0
        colored = np.asarray(Image.fromarray(colored).resize(tile_size, Image.Resampling.NEAREST))
        pair = [
            label_image(original, f"{stem} | Original"),
            label_image(colored, f"{stem} | final - baseline depth, clipped +/-10cm"),
        ]
        row = Image.new("RGB", (pair[0].width + pair[1].width, pair[0].height), "white")
        row.paste(pair[0], (0, 0))
        row.paste(pair[1], (pair[0].width, 0))
        rows.append(row)
    sheet = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet.save(output, quality=92, subsampling=0)


def layout_figure(current_path: Path, adaptive_path: Path, output: Path) -> None:
    current = json.loads(current_path.read_text())
    adaptive = json.loads(adaptive_path.read_text())
    size = float(current["chunk_size_m"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for axis, document, title in zip(axes, (current, adaptive), ("A baseline layout", "Full object-centered layout")):
        centers = np.asarray(document["centers"])
        for index, center in enumerate(centers):
            is_object = index in {3, 4, 5}
            color = "#e45756" if is_object else "#4c78a8"
            axis.add_patch(
                patches.Rectangle(
                    (center[0] - size / 2, center[1] - size / 2),
                    size,
                    size,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.20 if not is_object else 0.35,
                    linewidth=2,
                )
            )
            axis.text(center[0], center[1], str(index), ha="center", va="center", fontsize=10)
        axis.add_patch(patches.Rectangle((-2, -2.15), 4.25, 1.15, fill=False, edgecolor="black", linewidth=2))
        axis.set_aspect("equal")
        axis.set_xlim(-6.2, 6.3)
        axis.set_ylim(-4.2, 1.2)
        axis.set_title(title)
        axis.set_xlabel("world x (m)")
        axis.set_ylabel("world y (m)")
        axis.grid(alpha=0.2)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def focused_metrics(summary_path: Path, output: Path) -> None:
    variants = json.loads(summary_path.read_text())["variants"]
    labels = [
        "A baseline",
        "A0 obj-center raw",
        "N2b fused",
        "Full obj-center N2b",
        "Full safeguarded",
    ]
    x = np.arange(len(labels))
    worktop = [variants[label]["regions"]["worktop"]["heldout"] for label in labels]
    cabinet = [variants[label]["regions"]["cabinet_front"]["heldout"] for label in labels]
    global_stats = [variants[label]["global_sfm"]["heldout"] for label in labels]
    latent = [variants[label]["latent"] for label in labels]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes[0, 0].bar(x - 0.18, [item["median_m"] * 100 for item in worktop], 0.36, label="median")
    axes[0, 0].bar(x + 0.18, [item["p90_m"] * 100 for item in worktop], 0.36, label="P90")
    axes[0, 0].set_title("Heldout full-worktop depth error")
    axes[0, 0].set_ylabel("cm")
    axes[0, 0].legend()
    axes[0, 1].bar(x - 0.18, [item["median_m"] * 100 for item in cabinet], 0.36, label="median")
    axes[0, 1].bar(x + 0.18, [item["p90_m"] * 100 for item in cabinet], 0.36, label="P90")
    axes[0, 1].set_title("Heldout cabinet-front depth error")
    axes[0, 1].set_ylabel("cm")
    axes[0, 1].legend()
    axes[1, 0].bar(x - 0.18, [item["cross_view_median_m"] * 100 for item in global_stats], 0.36, label="median cm")
    axes[1, 0].bar(x + 0.18, [item["sfm_depth_coverage_view_mean"] * 10 for item in global_stats], 0.36, label="coverage x10")
    axes[1, 0].set_title("Global heldout guardrails")
    axes[1, 0].legend()
    stage_names = ["shape_slat", "texture_slat"]
    for offset, stage in zip((-0.18, 0.18), stage_names):
        values = [item.get(stage, {}).get("auc_mean", np.nan) for item in latent]
        axes[1, 1].bar(x + offset, values, 0.36, label=stage)
    axes[1, 1].set_title("ROI disagreement AUC")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.set_xticks(x, labels, rotation=14)
        axis.grid(axis="y", alpha=0.2)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--safeguarded", type=Path, required=True)
    parser.add_argument("--current-layout", type=Path, required=True)
    parser.add_argument("--adaptive-layout", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--key-view",
        action="append",
        default=["DSC_0903", "DSC_0906", "DSC_0918", "DSC_0924", "DSC_0904", "DSC_0925"],
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    key_view_sheet(
        args.baseline,
        args.fused,
        args.final,
        args.safeguarded,
        args.key_view,
        args.output / "key_views_comparison.jpg",
    )
    all_view_contact(args.baseline, args.safeguarded, args.output / "all_views_contact.jpg")
    depth_change_sheet(
        args.baseline,
        args.safeguarded,
        args.key_view,
        args.output / "depth_change_key_views.jpg",
    )
    layout_figure(args.current_layout, args.adaptive_layout, args.output / "chunk_layout_comparison.png")
    focused_metrics(args.summary, args.output / "final_metrics.png")
    html = """<!doctype html><html><head><meta charset='utf-8'><title>Object-centered chunk experiment</title>
<style>body{font-family:Arial,sans-serif;margin:24px;color:#202124;background:#fff}h1{font-size:24px}img{display:block;max-width:100%;height:auto;margin:12px 0 28px;border:1px solid #d5d7da}code{background:#f3f4f6;padding:2px 4px}</style></head><body>
<h1>lecture_room object-centered chunk experiment</h1>
<h2>Final metrics</h2><img src='final_metrics.png'>
<h2>Chunk layouts</h2><img src='chunk_layout_comparison.png'>
<h2>Key camera views: direct and safeguarded outputs</h2><img src='key_views_comparison.jpg'>
<h2>Depth changes</h2><img src='depth_change_key_views.jpg'>
<h2>All 23 COLMAP views</h2><img src='all_views_contact.jpg'>
</body></html>"""
    (args.output / "index.html").write_text(html)


if __name__ == "__main__":
    main()
