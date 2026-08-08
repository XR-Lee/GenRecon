#!/usr/bin/env python3
"""Visualize which saved scene views are eligible feature sources per chunk.

This reconstructs the exact in-frustum validity test used by GenRecon's
2D-to-3D projection on the saved shape-SLat occupancy coordinates. It does not
reconstruct the learned per-view softmax weights because those tensors are
collapsed by the aggregator and are not persisted by inference.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import trimesh
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3], got {points.shape}")
    if transform.shape != (4, 4):
        raise ValueError(f"transform must have shape [4, 4], got {transform.shape}")
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return (homogeneous @ transform.T)[:, :3]


def project_points(
    points: np.ndarray,
    world_to_camera: np.ndarray,
    intrinsic: np.ndarray,
    *,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror project_features_on_points: positive camera depth and UV in [0, 1)."""
    camera_points = transform_points(points, world_to_camera)
    depth = camera_points[:, 2]
    safe_depth = np.where(depth > eps, depth, 1.0)
    normalized = camera_points / safe_depth[:, None]
    uvw = normalized @ np.asarray(intrinsic, dtype=np.float64).T
    uv = uvw[:, :2]
    valid = (
        (depth > eps)
        & np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < 1.0)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < 1.0)
    )
    return uv, depth, valid


def infer_crop_side(intrinsic: np.ndarray, tolerance: float = 0.05) -> str:
    cx = float(np.asarray(intrinsic)[0, 2])
    if cx > 0.5 + tolerance:
        return "left"
    if cx < 0.5 - tolerance:
        return "right"
    return "center"


def _load_points(path: Path) -> np.ndarray:
    loaded = trimesh.load(str(path), process=False)
    points = np.asarray(loaded.vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError(f"No XYZ points found in {path}")
    return points


def _primary_view_index(scene: list[dict[str, Any]], cond: dict[str, Any]) -> int | None:
    cond_path = Path(cond["img_path"]).name
    cond_intrinsic = np.asarray(cond["intrinsics"], dtype=np.float64)
    cond_extrinsic = np.asarray(cond["extrinsics_c0"], dtype=np.float64)
    for index, source in enumerate(scene):
        if Path(source["img_path"]).name != cond_path:
            continue
        if not np.allclose(source["intrinsics"], cond_intrinsic, atol=1e-6):
            continue
        if np.allclose(source["extrinsics_c0"], cond_extrinsic, atol=1e-6):
            return index
    return None


def _source_metrics(
    points_c0: np.ndarray,
    source: dict[str, Any],
    *,
    patch_resolution: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    uv, _, valid = project_points(
        points_c0,
        np.asarray(source["extrinsics_c0"], dtype=np.float64),
        np.asarray(source["intrinsics"], dtype=np.float64),
    )
    valid_uv = uv[valid]
    if len(valid_uv):
        patch_xy = np.floor(valid_uv * patch_resolution).astype(np.int64)
        patch_xy = np.clip(patch_xy, 0, patch_resolution - 1)
        patch_ids = patch_xy[:, 1] * patch_resolution + patch_xy[:, 0]
        unique_patches = int(np.unique(patch_ids).size)
    else:
        unique_patches = 0
    metrics = {
        "image": Path(source["img_path"]).name,
        "crop": infer_crop_side(np.asarray(source["intrinsics"])),
        "valid_voxels": int(valid.sum()),
        "valid_fraction": float(valid.mean()),
        "unique_dino_patches": unique_patches,
        "dino_patch_fraction": float(unique_patches / (patch_resolution**2)),
    }
    return metrics, uv, valid


def _projection_thumbnail(
    source_image: Path,
    uv: np.ndarray,
    valid: np.ndarray,
    label: str,
    *,
    primary: bool,
    tile_size: int = 256,
) -> Image.Image:
    with Image.open(source_image) as opened:
        image = np.asarray(opened.convert("RGB").resize((tile_size, tile_size), Image.Resampling.LANCZOS)).copy()

    valid_uv = uv[valid]
    point_mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
    if len(valid_uv):
        pixels = np.floor(valid_uv * tile_size).astype(np.int64)
        pixels = np.clip(pixels, 0, tile_size - 1)
        point_mask[pixels[:, 1], pixels[:, 0]] = 255
        point_mask = cv2.dilate(point_mask, np.ones((3, 3), np.uint8), iterations=1)
        overlay = image.astype(np.float32)
        selected = point_mask > 0
        overlay[selected] = 0.35 * overlay[selected] + 0.65 * np.array([20, 240, 110], dtype=np.float32)
        image = np.rint(overlay).astype(np.uint8)

    label_height = 42
    border = 4 if primary else 1
    border_color = (255, 196, 40) if primary else (70, 70, 70)
    canvas = Image.new("RGB", (tile_size, tile_size + label_height), (18, 18, 18))
    canvas.paste(Image.fromarray(image), (0, 0))
    draw = ImageDraw.Draw(canvas)
    for inset in range(border):
        draw.rectangle(
            (inset, inset, tile_size - 1 - inset, tile_size - 1 - inset),
            outline=border_color,
        )
    draw.text((7, tile_size + 4), label, fill=(245, 245, 245))
    if primary:
        draw.text((tile_size - 63, tile_size + 22), "COND2D", fill=(255, 196, 40))
    return canvas


def _contact_sheet(tiles: list[Image.Image], output: Path, columns: int = 4) -> None:
    if not tiles:
        return
    tile_width = max(tile.width for tile in tiles)
    tile_height = max(tile.height for tile in tiles)
    rows = (len(tiles) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_width, rows * tile_height), (8, 8, 8))
    for index, tile in enumerate(tiles):
        canvas.paste(tile, ((index % columns) * tile_width, (index // columns) * tile_height))
    canvas.save(output, quality=91)


def _coverage_plot(
    points_world: np.ndarray,
    valid_stack: np.ndarray,
    sources: list[dict[str, Any]],
    primary_index: int | None,
    output: Path,
    chunk_index: int,
) -> dict[str, Any]:
    source_count = valid_stack.sum(axis=0)
    fractions = valid_stack.mean(axis=1)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    vmax = max(1, valid_stack.shape[0])
    scatter = axes[0].scatter(
        points_world[:, 0],
        points_world[:, 1],
        c=source_count,
        s=5,
        cmap="viridis",
        vmin=0,
        vmax=vmax,
        linewidths=0,
    )
    axes[0].set_title("Eligible scene crops per occupied voxel: XY")
    axes[0].set_xlabel("world x (m)")
    axes[0].set_ylabel("world y (m)")
    axes[0].set_aspect("equal", adjustable="box")

    axes[1].scatter(
        points_world[:, 0],
        points_world[:, 2],
        c=source_count,
        s=5,
        cmap="viridis",
        vmin=0,
        vmax=vmax,
        linewidths=0,
    )
    axes[1].set_title("Eligible scene crops per occupied voxel: XZ")
    axes[1].set_xlabel("world x (m)")
    axes[1].set_ylabel("world z (m)")
    axes[1].set_aspect("equal", adjustable="box")
    figure.colorbar(scatter, ax=axes[:2], label="in-frustum source crops")

    colors = ["#4c78a8"] * len(fractions)
    if primary_index is not None:
        colors[primary_index] = "#f2b134"
    axes[2].bar(np.arange(len(fractions)), fractions * 100.0, color=colors)
    axes[2].set_title("In-frustum fraction of saved occupied voxels")
    axes[2].set_xlabel("scene crop index")
    axes[2].set_ylabel("eligible voxels (%)")
    axes[2].set_xticks(np.arange(len(fractions)))
    axes[2].set_xticklabels([f"{i:02d}" for i in range(len(fractions))], rotation=90, fontsize=7)
    axes[2].set_ylim(0, 105)
    axes[2].grid(axis="y", alpha=0.25)
    figure.suptitle(
        f"Chunk {chunk_index:03d}: geometric feature-source eligibility (not learned weights)",
        fontsize=14,
    )
    figure.savefig(output, dpi=150)
    plt.close(figure)

    quantiles = np.quantile(source_count, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "eligible_source_count_min": int(quantiles[0]),
        "eligible_source_count_q25": float(quantiles[1]),
        "eligible_source_count_median": float(quantiles[2]),
        "eligible_source_count_q75": float(quantiles[3]),
        "eligible_source_count_max": int(quantiles[4]),
        "zero_source_voxels": int((source_count == 0).sum()),
    }


def _overview(cards: list[tuple[int, Path, str]], output: Path) -> None:
    columns = 4
    card_width, image_height, label_height = 320, 240, 52
    rows = (len(cards) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * card_width, rows * (image_height + label_height)), (10, 10, 10))
    draw = ImageDraw.Draw(canvas)
    for position, (chunk_index, image_path, label) in enumerate(cards):
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            image.thumbnail((card_width, image_height), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (card_width, image_height), (0, 0, 0))
            tile.paste(image, ((card_width - image.width) // 2, (image_height - image.height) // 2))
        x = (position % columns) * card_width
        y = (position // columns) * (image_height + label_height)
        canvas.paste(tile, (x, y))
        draw.text((x + 8, y + image_height + 5), f"chunk {chunk_index:03d}", fill=(255, 255, 255))
        draw.text((x + 8, y + image_height + 25), label, fill=(180, 180, 180))
    canvas.save(output, quality=91)


def _write_html(output: Path, chunks: list[dict[str, Any]]) -> None:
    cards = []
    for chunk in chunks:
        index = int(chunk["chunk_index"])
        rel = f"chunk_{index:03d}"
        primary = html.escape(str(chunk.get("primary_source_label", "not matched")))
        cards.append(
            f'<section><h2>Chunk {index:03d}</h2>'
            f'<p>{chunk["point_count"]:,} occupied voxels; median eligible crops '
            f'{chunk["eligible_source_count_median"]:.1f}; cond2D: {primary}</p>'
            f'<div class="images"><a href="{rel}/cond2d.png"><img src="{rel}/cond2d.png" alt="chunk {index:03d} cond2D"></a>'
            f'<a href="{rel}/coverage.png"><img src="{rel}/coverage.png" alt="chunk {index:03d} source coverage"></a></div>'
            f'<p><a href="{rel}/sources.jpg">All scene-crop projection overlays</a> | '
            f'<a href="{rel}/metrics.json">metrics.json</a></p></section>'
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chunk feature sources</title><style>
body{{margin:0;background:#111;color:#eee;font:14px system-ui,sans-serif}}header{{padding:20px 28px;border-bottom:1px solid #333}}main{{padding:18px 28px}}section{{padding:18px 0;border-bottom:1px solid #333}}h1,h2{{margin:0 0 8px}}p{{color:#bbb}}a{{color:#8fc7ff}}.images{{display:grid;grid-template-columns:minmax(220px,360px) minmax(480px,1fr);gap:12px;align-items:start}}img{{display:block;width:100%;height:auto;background:#000}}@media(max-width:850px){{.images{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>Chunk feature-source eligibility</h1>
<p>Green pixels are saved occupied voxels projected into each scene crop. This reproduces the in-frustum validity mask, not occlusion and not learned aggregator weights.</p>
<p><a href="README.md">Method and storage notes</a> | <a href="overview.jpg">cond2D overview</a> | <a href="source_views.csv">CSV</a></p></header><main>{''.join(cards)}</main></body></html>
"""
    (output / "index.html").write_text(document)


def _write_readme(output: Path, reconstruction: Path, chunks: list[dict[str, Any]]) -> None:
    point_count = sum(int(chunk["point_count"]) for chunk in chunks)
    lines = [
        "# Chunk feature-source eligibility",
        "",
        "This directory reconstructs which saved scene crops pass GenRecon's positive-depth and in-image projection test for each saved occupied shape-SLat voxel.",
        "It does not show occlusion-aware visibility or learned aggregator weights; neither was persisted by the original inference run.",
        "",
        "## What is shown",
        "",
        "- `overview.jpg`: the one per-chunk `cond_2D` crop used for transformer cross-attention.",
        "- `chunk_NNN/sources.jpg`: all scene crops; green pixels are occupied voxels that project in-frustum. The gold border marks `cond_2D`.",
        "- `chunk_NNN/coverage.png`: XY/XZ eligible-source counts and per-view coverage.",
        "- `source_views.csv`: machine-readable source eligibility per chunk and scene crop.",
        "",
        "## Storage and use",
        "",
        f"- Reconstruction source: `{reconstruction}`.",
        f"- Saved occupied coordinates analyzed: {point_count:,} rows across {len(chunks)} chunks.",
        "- `scene/view_NNN.png` stores debug copies of the scene crops; `cameras.json` stores their intrinsics/extrinsics and the per-chunk cond2D selection.",
        "- During inference, DINO tokens are projected to `[voxel, view, feature]`, masked in-frustum, and collapsed across the view dimension by a learned mean/variance + softmax aggregator.",
        "- The projected per-view features, validity mask, softmax weights, and aggregated `cond_3D` were transient and are absent from the final PLY/GLB caches.",
        "- `to_glb_inputs.pt/attr_volume` stores decoded PBR attributes, not camera-source provenance.",
        "",
        "## Interpretation",
        "",
        "Eligibility means only positive camera depth and normalized UV inside `[0,1)`. A green projection may be occluded by another surface and may receive a very small learned weight.",
        "Exact attribution requires a future inference dump of top-k view ids/weights at the sparse-structure, shape, and texture aggregation stages.",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines))


def visualize(reconstruction: Path, output: Path, patch_resolution: int = 32) -> dict[str, Any]:
    reconstruction = reconstruction.resolve()
    output.mkdir(parents=True, exist_ok=True)
    camera_document = json.loads((reconstruction / "cameras.json").read_text())
    transform_document = json.loads((reconstruction / "chunk_transforms.json").read_text())
    scene = camera_document["scene"]
    chunks_by_index = {int(item["chunk_index"]): item for item in camera_document["chunks"]}
    world_to_chunk0 = np.asarray(transform_document["chunks"][0]["M_original_to_chunk"], dtype=np.float64)

    source_rows: list[dict[str, Any]] = []
    chunk_summaries: list[dict[str, Any]] = []
    overview_cards: list[tuple[int, Path, str]] = []

    coord_paths = sorted(reconstruction.glob("coords_*.ply"))
    if not coord_paths:
        raise FileNotFoundError(f"No coords_NNN.ply files found in {reconstruction}")

    for coord_path in coord_paths:
        chunk_index = int(coord_path.stem.split("_")[-1])
        if chunk_index not in chunks_by_index:
            raise KeyError(f"Chunk {chunk_index} has coordinates but no cameras.json entry")
        chunk_output = output / f"chunk_{chunk_index:03d}"
        chunk_output.mkdir(parents=True, exist_ok=True)
        points_world = _load_points(coord_path)
        points_c0 = transform_points(points_world, world_to_chunk0)
        cond = chunks_by_index[chunk_index]["cond2d_view"]
        primary_index = _primary_view_index(scene, cond)

        tiles: list[Image.Image] = []
        valid_masks: list[np.ndarray] = []
        source_metrics: list[dict[str, Any]] = []
        for view_index, source in enumerate(scene):
            metrics, uv, valid = _source_metrics(
                points_c0,
                source,
                patch_resolution=patch_resolution,
            )
            metrics.update(
                {
                    "chunk_index": chunk_index,
                    "view_index": view_index,
                    "is_cond2d": view_index == primary_index,
                }
            )
            source_metrics.append(metrics)
            source_rows.append(metrics)
            valid_masks.append(valid)
            label = (
                f"#{view_index:02d} {Path(source['img_path']).stem} {metrics['crop'][0].upper()}\n"
                f"valid {metrics['valid_fraction'] * 100:.1f}% | patches {metrics['unique_dino_patches']}"
            )
            source_image = reconstruction / "scene" / f"view_{view_index:03d}.png"
            tiles.append(
                _projection_thumbnail(
                    source_image,
                    uv,
                    valid,
                    label,
                    primary=view_index == primary_index,
                )
            )

        _contact_sheet(tiles, chunk_output / "sources.jpg")
        valid_stack = np.stack(valid_masks, axis=0)
        coverage_summary = _coverage_plot(
            points_world,
            valid_stack,
            scene,
            primary_index,
            chunk_output / "coverage.png",
            chunk_index,
        )
        cond_path = reconstruction / f"chunk_{chunk_index:03d}" / "cond2d.png"
        with Image.open(cond_path) as image:
            image.convert("RGB").save(chunk_output / "cond2d.png")

        if primary_index is None:
            primary_label = "not matched"
        else:
            primary = source_metrics[primary_index]
            primary_label = f"#{primary_index:02d} {Path(primary['image']).stem} {primary['crop']}"
        chunk_summary = {
            "chunk_index": chunk_index,
            "point_count": int(len(points_world)),
            "primary_source_index": primary_index,
            "primary_source_label": primary_label,
            **coverage_summary,
            "sources": source_metrics,
        }
        (chunk_output / "metrics.json").write_text(json.dumps(chunk_summary, indent=2) + "\n")
        chunk_summaries.append(chunk_summary)
        overview_cards.append((chunk_index, cond_path, primary_label))

    with (output / "source_views.csv").open("w", newline="") as handle:
        fieldnames = [
            "chunk_index",
            "view_index",
            "image",
            "crop",
            "is_cond2d",
            "valid_voxels",
            "valid_fraction",
            "unique_dino_patches",
            "dino_patch_fraction",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(source_rows)

    summary = {
        "method": "saved-shape-occupancy in-frustum feature-source eligibility",
        "warning": "not occlusion-aware visibility and not learned aggregator weights",
        "reconstruction": str(reconstruction),
        "scene_view_count": len(scene),
        "chunk_count": len(chunk_summaries),
        "patch_resolution": patch_resolution,
        "chunks": chunk_summaries,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _overview(overview_cards, output / "overview.jpg")
    _write_readme(output, reconstruction, chunk_summaries)
    _write_html(output, chunk_summaries)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch-resolution", type=int, default=32)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name in ("cameras.json", "chunk_transforms.json"):
        if not (args.reconstruction / name).is_file():
            raise FileNotFoundError(args.reconstruction / name)
    if args.patch_resolution <= 0:
        raise ValueError("--patch-resolution must be positive")
    summary = visualize(args.reconstruction, args.output, args.patch_resolution)
    print(
        f"Wrote {summary['chunk_count']} chunks x {summary['scene_view_count']} scene views "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
