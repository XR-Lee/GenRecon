#!/usr/bin/env python3
"""Build visual comparisons for strict single-instance object chunk experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contextual", type=Path, required=True)
    parser.add_argument("--strict", type=Path, required=True)
    parser.add_argument("--context-encoded", type=Path)
    parser.add_argument("--patch-only", type=Path)
    parser.add_argument("--big", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-images", default="DSC_0903,DSC_0904,DSC_0906,DSC_0918,DSC_0924,DSC_0925")
    return parser.parse_args()


def find_view(run: Path, image_stem: str) -> Path:
    matches = list((run / "fidelity" / "views").glob(f"*/{image_stem}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one fidelity view for {image_stem} under {run}, got {matches}")
    return matches[0]


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def strict_overlay(view: Path) -> np.ndarray:
    original = read_rgb(view / "original.jpg").astype(np.float32)
    rendered = read_rgb(view / "ply_render.png").astype(np.float32)
    mask = np.asarray(Image.open(view / "ply_mask.png").convert("L")) > 0
    output = original * 0.35
    output[mask] = original[mask] * 0.5 + rendered[mask] * 0.5
    return np.clip(output, 0, 255).astype(np.uint8)


def save_key_comparison(args: argparse.Namespace, image_stems: list[str]) -> None:
    if args.context_encoded is None:
        runs = [args.contextual, args.strict, args.big]
        titles = ["Original", "N2c contextual", "O3c strict object", "O1b 4.5m seeded", "RGB/O3 overlay"]
    elif args.patch_only is None:
        runs = [args.contextual, args.strict, args.context_encoded]
        titles = ["Original", "N2c contextual", "O3c pre-mask", "O4 context+global", "RGB/O3 overlay", "RGB/O4 overlay"]
    else:
        runs = [args.contextual, args.strict, args.context_encoded, args.patch_only]
        titles = ["Original", "N2c", "O3c pre-mask", "O4 context+global", "O4p patches only", "RGB/O4 overlay", "RGB/O4p overlay"]
    figure, axes = plt.subplots(
        len(image_stems),
        len(titles),
        figsize=(4 * len(titles), 3.0 * len(image_stems)),
        squeeze=False,
    )
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=11)
    for row, stem in enumerate(image_stems):
        strict_view = find_view(args.strict, stem)
        panels = [read_rgb(strict_view / "original.jpg")]
        panels.extend(read_rgb(find_view(run, stem) / "ply_render.png") for run in runs)
        if args.patch_only is None:
            panels.append(strict_overlay(strict_view))
        if args.context_encoded is not None:
            panels.append(strict_overlay(find_view(args.context_encoded, stem)))
        if args.patch_only is not None:
            panels.append(strict_overlay(find_view(args.patch_only, stem)))
        for column, panel in enumerate(panels):
            axes[row, column].imshow(panel)
            axes[row, column].axis("off")
        group = strict_view.parent.name
        axes[row, 0].set_ylabel(f"{stem}\n{group}", fontsize=9)
    figure.tight_layout(pad=0.5)
    figure.savefig(args.output / "key_views_comparison.jpg", dpi=135, pil_kwargs={"quality": 92})
    plt.close(figure)


def save_all_views(args: argparse.Namespace) -> None:
    strict_dirs = sorted((args.strict / "fidelity" / "views").glob("*/*"), key=lambda path: path.name)
    if args.patch_only is not None:
        titles = ["Original", "N2c contextual", "O3c pre-mask", "O4 context+global", "O4p patches only"]
    elif args.context_encoded is not None:
        titles = ["Original", "N2c contextual", "O3c pre-mask", "O4 context+global"]
    else:
        titles = ["Original", "N2c contextual", "O3c strict object", "O1b 4.5m seeded"]
    figure, axes = plt.subplots(
        len(strict_dirs),
        len(titles),
        figsize=(3.5 * len(titles), 2.25 * len(strict_dirs)),
        squeeze=False,
    )
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=10)
    for row, strict_view in enumerate(strict_dirs):
        stem = strict_view.name
        fourth_run = args.context_encoded if args.context_encoded is not None else args.big
        panels = [
            read_rgb(strict_view / "original.jpg"),
            read_rgb(find_view(args.contextual, stem) / "ply_render.png"),
            read_rgb(strict_view / "ply_render.png"),
            read_rgb(find_view(fourth_run, stem) / "ply_render.png"),
        ]
        if args.patch_only is not None:
            panels.append(read_rgb(find_view(args.patch_only, stem) / "ply_render.png"))
        for column, panel in enumerate(panels):
            axes[row, column].imshow(panel)
            axes[row, column].axis("off")
        axes[row, 0].set_ylabel(f"{stem}\n{strict_view.parent.name}", fontsize=8)
    figure.tight_layout(pad=0.35)
    figure.savefig(args.output / "all_23_views.jpg", dpi=105, pil_kwargs={"quality": 88})
    plt.close(figure)


def save_conditions(args: argparse.Namespace) -> None:
    view_indices = [1, 24, 10]
    rows = 3 if args.context_encoded is not None else 2
    figure, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows), squeeze=False)
    for column, view_index in enumerate(view_indices):
        axes[0, column].imshow(read_rgb(args.strict / "scene" / f"view_{view_index:03d}.png"))
        axes[0, column].set_title(f"O3 pre-masked scene {view_index}")
        axes[0, column].axis("off")
        axes[1, column].imshow(read_rgb(args.strict / f"chunk_{column:03d}" / "cond2d.png"))
        axes[1, column].set_title(f"O3 chunk {column} cond2D")
        axes[1, column].axis("off")
        if args.context_encoded is not None:
            axes[2, column].imshow(read_rgb(args.context_encoded / f"chunk_{column:03d}" / "cond2d.png"))
            axes[2, column].set_title(f"O4 full-context cond2D {column}")
            axes[2, column].axis("off")
    figure.tight_layout(pad=0.7)
    figure.savefig(args.output / "strict_object_conditions.jpg", dpi=150, pil_kwargs={"quality": 92})
    plt.close(figure)


def save_metrics(args: argparse.Namespace) -> None:
    variants = json.loads(args.summary.read_text())["variants"]
    if "O4p context-encoded patches" in variants:
        labels = [
            "A baseline",
            "N2c contextual",
            "O3c strict pre-mask",
            "O4 context-encoded globals",
            "O4p context-encoded patches",
            "Full contextual valid",
        ]
        short = ["Baseline", "N2c", "O3c", "O4", "O4p", "Full valid"]
    elif "O4 context-encoded tokens" in variants:
        labels = ["A baseline", "N2c contextual", "O3c strict pre-mask", "O4 context-encoded tokens", "Full contextual valid"]
        short = ["Baseline", "N2c", "O3c", "O4", "Full valid"]
    else:
        labels = ["A baseline", "N0 raw", "N2c contextual", "O3c strict object", "Full contextual valid", "O1 big seeded"]
        short = ["Baseline", "N0 raw", "N2c", "O3c strict", "Full valid", "O1 big"]
    worktop = [variants[label]["regions"]["worktop"]["heldout"] for label in labels]
    cabinet = [variants[label]["regions"]["cabinet_front"]["heldout"] for label in labels]
    plane = [variants[label]["plane"]["fit_p90_m"] * 1000 for label in labels]
    vertices = [variants[label]["mesh_vertices"] / 1e6 for label in labels]
    x = np.arange(len(labels))

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
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

    width = 0.36
    axes[1, 0].bar(x - width / 2, [item["coverage"] * 100 for item in worktop], width, label="worktop")
    axes[1, 0].bar(x + width / 2, [item["coverage"] * 100 for item in cabinet], width, label="cabinet")
    axes[1, 0].set_title("Heldout ROI depth coverage")
    axes[1, 0].set_ylabel("percent")
    axes[1, 0].set_ylim(0, 105)
    axes[1, 0].legend()

    axis2 = axes[1, 1].twinx()
    axes[1, 1].bar(x - width / 2, plane, width, label="plane P90 mm", color="#2878b5")
    axis2.bar(x + width / 2, vertices, width, label="mesh M vertices", color="#e87500")
    axes[1, 1].set_title("Plane residual and mesh size")
    axes[1, 1].set_ylabel("mm")
    axis2.set_ylabel("million vertices")
    lines1, names1 = axes[1, 1].get_legend_handles_labels()
    lines2, names2 = axis2.get_legend_handles_labels()
    axes[1, 1].legend(lines1 + lines2, names1 + names2)

    for axis in axes.ravel():
        axis.set_xticks(x, short, rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(args.output / "strict_object_metrics.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    image_stems = [value.strip() for value in args.key_images.split(",") if value.strip()]
    save_key_comparison(args, image_stems)
    save_all_views(args)
    save_conditions(args)
    save_metrics(args)
    html = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Strict object chunks</title>
<style>body{font-family:system-ui,sans-serif;margin:24px;max-width:1500px}img{max-width:100%;border:1px solid #ccc}h1,h2{letter-spacing:0}</style></head><body>
<h1>Object-token context-encoding comparison</h1>
<p><a href="README.md">Result summary</a></p>
<h2>Metrics</h2><img src="strict_object_metrics.png">
<h2>Key camera views</h2><img src="key_views_comparison.jpg">
<h2>DINO encoder image conditions</h2><img src="strict_object_conditions.jpg">
<h2>All 23 COLMAP cameras</h2><img src="all_23_views.jpg">
</body></html>"""
    (args.output / "index.html").write_text(html)


if __name__ == "__main__":
    main()
