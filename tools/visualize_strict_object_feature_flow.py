#!/usr/bin/env python3
"""Audit strict object-only image, SAM, DINO-token, and cond3D source flow."""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.projection_ownership import build_projection_ownership, downsample_mask


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return (homogeneous @ np.asarray(transform, dtype=np.float64).T)[:, :3]


def transform_boxes(
    boxes: list[list[float]], transform: np.ndarray
) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    result = []
    for x0, x1, y0, y1, z0, z1 in boxes:
        corners = np.asarray(
            [[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)],
            dtype=np.float64,
        )
        converted = transform_points(corners, transform)
        result.append((tuple(converted.min(axis=0)), tuple(converted.max(axis=0))))
    return result


def infer_crop_side(intrinsic: np.ndarray, tolerance: float = 0.05) -> str:
    cx = float(np.asarray(intrinsic)[0, 2])
    if cx > 0.5 + tolerance:
        return "left"
    if cx < 0.5 - tolerance:
        return "right"
    return "center"


def load_exact_crop(view: dict[str, Any], size: int) -> np.ndarray:
    """Reproduce the iPhone selector's extreme square crop and direct resize."""
    with Image.open(view["img_path"]) as opened:
        image = opened.convert("RGB")
        width, height = image.size
        if width > height:
            crop_size = height
            left = 0 if float(view["intrinsics"][0][2]) >= 0.5 else width - crop_size
            image = image.crop((left, 0, left + crop_size, crop_size))
        elif height > width:
            crop_size = width
            top = 0 if float(view["intrinsics"][1][2]) >= 0.5 else height - crop_size
            image = image.crop((0, top, crop_size, top + crop_size))
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255.0


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(mask, dtype=np.float32))[None, None]
    return (
        F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
        .squeeze()
        .numpy()
    )


def load_view_mask(
    mask_dir: Path,
    mask_view_index: int | None,
    *,
    enabled: bool,
    size: int,
) -> np.ndarray:
    if mask_view_index is None or not enabled:
        return np.zeros((size, size), dtype=np.float32)
    source = np.asarray(
        Image.open(mask_dir / f"view_{mask_view_index:03d}.png").convert("L"),
        dtype=np.float32,
    ) / 255.0
    return resize_mask(source, size)


def feature_similarity(
    first: np.ndarray,
    second: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int | None]]:
    if first.shape != second.shape or first.ndim != 3:
        raise ValueError("feature maps must have matching [H,W,D] shapes")
    if mask.shape != first.shape[:2]:
        raise ValueError("feature mask must match feature-map spatial dimensions")
    first_norm = np.linalg.norm(first, axis=-1)
    second_norm = np.linalg.norm(second, axis=-1)
    cosine = np.sum(first * second, axis=-1) / np.maximum(first_norm * second_norm, 1e-8)
    relative_l2 = np.linalg.norm(first - second, axis=-1) / np.maximum(first_norm, 1e-8)
    selected_cosine = cosine[mask]
    selected_l2 = relative_l2[mask]
    if not len(selected_cosine):
        return cosine, {
            "patches": 0,
            "cosine_mean": None,
            "cosine_median": None,
            "cosine_p10": None,
            "relative_l2_mean": None,
            "relative_l2_median": None,
        }
    return cosine, {
        "patches": int(mask.sum()),
        "cosine_mean": float(selected_cosine.mean()),
        "cosine_median": float(np.median(selected_cosine)),
        "cosine_p10": float(np.percentile(selected_cosine, 10)),
        "relative_l2_mean": float(selected_l2.mean()),
        "relative_l2_median": float(np.median(selected_l2)),
    }


def classify_sources(
    points_chunk0: np.ndarray,
    cameras: list[dict[str, Any]],
    ownership: dict[str, Any],
    raw_patch_masks: np.ndarray,
) -> dict[str, np.ndarray]:
    """Reproduce strict projection eligibility for arbitrary chunk-0 points."""
    point_count = len(points_chunk0)
    view_count = len(cameras)
    shape = (view_count, point_count)
    in_frustum = np.zeros(shape, dtype=bool)
    raw_sam = np.zeros(shape, dtype=bool)
    active_sam = np.zeros(shape, dtype=bool)
    surface = np.zeros(shape, dtype=bool)
    unknown = np.zeros(shape, dtype=bool)
    free = np.zeros(shape, dtype=bool)
    occluded = np.zeros(shape, dtype=bool)
    final = np.zeros(shape, dtype=bool)
    uv_all = np.zeros((view_count, point_count, 2), dtype=np.float32)

    owned_masks = ownership["mask"].cpu().numpy().reshape(view_count, 32, 32)
    depth_known = ownership["depth_valid"].cpu().numpy().reshape(view_count, 32, 32)
    reference = ownership["surface_depth"].cpu().numpy().reshape(view_count, 32, 32)
    band = float(ownership["surface_band"])
    homogeneous = np.concatenate(
        [points_chunk0, np.ones((point_count, 1), dtype=np.float64)], axis=1
    )

    for view_index, camera in enumerate(cameras):
        camera_points = homogeneous @ np.asarray(camera["extrinsics_c0"], dtype=np.float64).T
        depth = camera_points[:, 2]
        safe = np.where(depth > 1e-6, depth, 1.0)
        uvw = (camera_points[:, :3] / safe[:, None]) @ np.asarray(
            camera["intrinsics"], dtype=np.float64
        ).T
        uv = uvw[:, :2]
        valid = (
            (depth > 1e-6)
            & (uv[:, 0] >= 0.0)
            & (uv[:, 0] < 1.0)
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] < 1.0)
        )
        patch_x = np.floor(uv[:, 0] * 32).astype(np.int64).clip(0, 31)
        patch_y = np.floor(uv[:, 1] * 32).astype(np.int64).clip(0, 31)
        raw_owned = raw_patch_masks[view_index, patch_y, patch_x]
        owned = owned_masks[view_index, patch_y, patch_x]
        known = depth_known[view_index, patch_y, patch_x]
        depth_delta = depth - reference[view_index, patch_y, patch_x]
        is_surface = valid & owned & known & (np.abs(depth_delta) <= band)
        is_unknown = valid & owned & ~known
        is_free = valid & owned & known & (depth_delta < -band)
        is_occluded = valid & owned & known & (depth_delta > band)

        in_frustum[view_index] = valid
        raw_sam[view_index] = valid & raw_owned
        active_sam[view_index] = valid & owned
        surface[view_index] = is_surface
        unknown[view_index] = is_unknown
        free[view_index] = is_free
        occluded[view_index] = is_occluded
        final[view_index] = is_surface | is_unknown
        uv_all[view_index] = uv.astype(np.float32)

    return {
        "in_frustum": in_frustum,
        "raw_sam": raw_sam,
        "active_sam": active_sam,
        "surface": surface,
        "unknown": unknown,
        "free": free,
        "occluded": occluded,
        "final": final,
        "uv": uv_all,
    }


def encode_images(
    tensors: list[torch.Tensor],
    model_path: str,
    batch_size: int,
) -> torch.Tensor:
    from genrecon.modules.image_feature_extractor import DinoV3FeatureExtractor

    if not torch.cuda.is_available():
        raise RuntimeError("DINO feature audit requires CUDA")
    extractor = DinoV3FeatureExtractor(model_path)
    extractor.cuda()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[start : start + batch_size]).unsqueeze(0)
            features = extractor(batch).squeeze(0).cpu()
            outputs.append(features)
            print(f"[DINO] encoded {min(start + batch_size, len(tensors))}/{len(tensors)}")
    extractor.cpu()
    del extractor
    torch.cuda.empty_cache()
    return torch.cat(outputs, dim=0)


def rgb_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.rint(np.clip(array, 0.0, 1.0) * 255).astype(np.uint8))


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(0, 220, 150)) -> Image.Image:
    result = image.copy()
    alpha = np.clip(mask, 0.0, 1.0)[..., None] * 0.48
    result = result * (1.0 - alpha) + np.asarray(color, dtype=np.float32)[None, None] / 255.0 * alpha
    return rgb_image(result)


def overlay_parts(image: np.ndarray, masks: list[np.ndarray]) -> Image.Image:
    result = image.copy()
    colors = np.asarray([[30, 210, 245], [255, 174, 44]], dtype=np.float32) / 255.0
    for index, mask in enumerate(masks[:2]):
        alpha = np.clip(mask, 0.0, 1.0)[..., None] * 0.5
        result = result * (1.0 - alpha) + colors[index][None, None] * alpha
    return rgb_image(result)


def grid_image(values: np.ndarray, cmap: str, low: float, high: float, mask: np.ndarray | None = None) -> Image.Image:
    scaled = np.clip((values - low) / max(high - low, 1e-8), 0.0, 1.0)
    colored = matplotlib.colormaps[cmap](scaled)[..., :3]
    if mask is not None:
        colored[~mask] = 0.0
    image = Image.fromarray(np.rint(colored * 255).astype(np.uint8))
    return image.resize((512, 512), Image.Resampling.NEAREST)


def ownership_grid(mask: np.ndarray, known: np.ndarray) -> Image.Image:
    colors = np.zeros((*mask.shape, 3), dtype=np.uint8)
    colors[mask & ~known] = (244, 178, 48)
    colors[mask & known] = (48, 201, 112)
    return Image.fromarray(colors).resize((512, 512), Image.Resampling.NEAREST)


def pca_image(
    features: np.ndarray,
    pca: PCA,
    lower: np.ndarray,
    upper: np.ndarray,
    mask: np.ndarray | None = None,
) -> Image.Image:
    projected = pca.transform(features.reshape(-1, features.shape[-1])).reshape(32, 32, 3)
    rgb = np.clip((projected - lower) / np.maximum(upper - lower, 1e-8), 0.0, 1.0)
    if mask is not None:
        rgb[~mask] = 0.0
    return rgb_image(rgb).resize((512, 512), Image.Resampling.NEAREST)


def labeled_tile(image: Image.Image, title: str, subtitle: str = "", width: int = 300) -> Image.Image:
    source = image.convert("RGB")
    source.thumbnail((width, width), Image.Resampling.LANCZOS)
    height = width + 54
    canvas = Image.new("RGB", (width, height), (12, 12, 12))
    canvas.paste(source, ((width - source.width) // 2, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((7, width + 6), title, fill=(245, 245, 245))
    draw.text((7, width + 28), subtitle, fill=(170, 170, 170))
    return canvas


def contact_sheet(tiles: list[Image.Image], output: Path, columns: int) -> None:
    width = max(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles)
    rows = (len(tiles) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * width, rows * height), (6, 6, 6))
    for index, tile in enumerate(tiles):
        canvas.paste(tile, ((index % columns) * width, (index // columns) * height))
    canvas.save(output, quality=92)


def detailed_panel(tiles: list[tuple[str, str, Image.Image]], output: Path) -> None:
    rendered = [labeled_tile(image, title, subtitle) for title, subtitle, image in tiles]
    contact_sheet(rendered, output, columns=6)


def summarize_sources(
    source: dict[str, np.ndarray],
    point_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    point_count = source["final"].shape[1]
    if point_mask is None:
        point_mask = np.ones(point_count, dtype=bool)
    point_mask = np.asarray(point_mask, dtype=bool)
    selected_count = int(point_mask.sum())
    selected = {key: value[:, point_mask] for key, value in source.items() if key != "uv"}
    final_count = selected["final"].sum(axis=0)
    raw_sam_count = selected["raw_sam"].sum(axis=0)
    active_sam_count = selected["active_sam"].sum(axis=0)
    surface_count = selected["surface"].sum(axis=0)
    final_pairs = int(selected["final"].sum())
    unknown_pairs = int(selected["unknown"].sum())
    if selected_count:
        quantiles = np.percentile(final_count, [0, 10, 50, 90, 100])
    else:
        quantiles = np.zeros(5)
    return {
        "point_count": selected_count,
        "in_frustum_pairs": int(selected["in_frustum"].sum()),
        "raw_sam_pairs": int(selected["raw_sam"].sum()),
        "active_sam_pairs": int(selected["active_sam"].sum()),
        "strict_eligible_pairs": final_pairs,
        "track_surface_pairs": int(selected["surface"].sum()),
        "unknown_accepted_pairs": unknown_pairs,
        "unknown_fraction_of_eligible": unknown_pairs / final_pairs if final_pairs else None,
        "free_rejected_pairs": int(selected["free"].sum()),
        "occluded_rejected_pairs": int(selected["occluded"].sum()),
        "eligible_sources_min": int(quantiles[0]),
        "eligible_sources_p10": float(quantiles[1]),
        "eligible_sources_median": float(quantiles[2]),
        "eligible_sources_p90": float(quantiles[3]),
        "eligible_sources_max": int(quantiles[4]),
        "zero_source_points": int((final_count == 0).sum()),
        "zero_source_fraction": float((final_count == 0).mean()) if selected_count else None,
        "sam_only_zero_source_fraction": float((raw_sam_count == 0).mean()) if selected_count else None,
        "active_sam_zero_source_fraction": float((active_sam_count == 0).mean()) if selected_count else None,
        "surface_only_zero_source_fraction": float((surface_count == 0).mean()) if selected_count else None,
    }


def source_plot(
    points_world: np.ndarray,
    source: dict[str, np.ndarray],
    view_labels: list[str],
    output: Path,
    chunk_index: int,
) -> dict[str, Any]:
    final_count = source["final"].sum(axis=0)
    unknown_count = source["unknown"].sum(axis=0)
    figure, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    scatter = axes[0, 0].scatter(
        points_world[:, 0], points_world[:, 1], c=final_count, s=9, cmap="viridis", linewidths=0
    )
    axes[0, 0].set_title("Strict eligible source count: XY")
    axes[0, 0].set_xlabel("world x (m)")
    axes[0, 0].set_ylabel("world y (m)")
    axes[0, 0].set_aspect("equal", adjustable="box")
    figure.colorbar(scatter, ax=axes[0, 0], label="eligible views")

    unknown_fraction = unknown_count / np.maximum(final_count, 1)
    scatter_unknown = axes[0, 1].scatter(
        points_world[:, 0],
        points_world[:, 2],
        c=unknown_fraction * 100.0,
        s=9,
        cmap="magma",
        vmin=0,
        vmax=100,
        linewidths=0,
    )
    axes[0, 1].set_title("Eligible sources without track depth: XZ")
    axes[0, 1].set_xlabel("world x (m)")
    axes[0, 1].set_ylabel("world z (m)")
    axes[0, 1].set_aspect("equal", adjustable="box")
    figure.colorbar(scatter_unknown, ax=axes[0, 1], label="unknown eligible sources (%)")

    x = np.arange(len(view_labels))
    denominator = max(1, len(points_world))
    axes[1, 0].bar(x, source["in_frustum"].sum(axis=1) / denominator * 100, label="in-frustum")
    axes[1, 0].bar(x, source["raw_sam"].sum(axis=1) / denominator * 100, label="SAM patch")
    axes[1, 0].bar(x, source["final"].sum(axis=1) / denominator * 100, label="strict eligible")
    axes[1, 0].set_title("Per-view candidate retention")
    axes[1, 0].set_ylabel("output occupancy points (%)")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(view_labels, rotation=90, fontsize=7)
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.25)

    axes[1, 1].bar(x, source["surface"].sum(axis=1), label="track-depth surface", color="#2ca25f")
    axes[1, 1].bar(
        x,
        source["unknown"].sum(axis=1),
        bottom=source["surface"].sum(axis=1),
        label="unknown accepted",
        color="#f0ad3d",
    )
    axes[1, 1].set_title("Exact strict eligible pair composition")
    axes[1, 1].set_ylabel("view-point pairs")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(view_labels, rotation=90, fontsize=7)
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", alpha=0.25)
    figure.suptitle(f"Chunk {chunk_index}: cond3D feature-source audit", fontsize=16)
    figure.savefig(output, dpi=150)
    plt.close(figure)

    return summarize_sources(source)


def projection_thumbnail(
    original: np.ndarray,
    pixel_mask: np.ndarray,
    uv: np.ndarray,
    eligible: np.ndarray,
    title: str,
    subtitle: str,
    *,
    active: bool,
) -> Image.Image:
    image = np.asarray(
        overlay_mask(original, pixel_mask).resize((256, 256)), dtype=np.uint8
    ).copy()
    selected_uv = uv[eligible]
    if len(selected_uv):
        pixels = np.floor(selected_uv * 256).astype(np.int64).clip(0, 255)
        image[pixels[:, 1], pixels[:, 0]] = (40, 255, 90)
    tile = labeled_tile(Image.fromarray(image), title, subtitle, width=256)
    draw = ImageDraw.Draw(tile)
    color = (68, 212, 118) if active else (225, 72, 72)
    for inset in range(3):
        draw.rectangle((inset, inset, tile.width - 1 - inset, tile.height - 1 - inset), outline=color)
    return tile


def fit_feature_pca(feature_sets: list[np.ndarray], seed: int = 42) -> tuple[PCA, np.ndarray, np.ndarray]:
    vectors = np.concatenate([features.reshape(-1, features.shape[-1]) for features in feature_sets], axis=0)
    rng = np.random.default_rng(seed)
    if len(vectors) > 12000:
        vectors = vectors[rng.choice(len(vectors), 12000, replace=False)]
    pca = PCA(n_components=3, svd_solver="randomized", random_state=seed)
    projected = pca.fit_transform(vectors)
    lower = np.percentile(projected, 1, axis=0)
    upper = np.percentile(projected, 99, axis=0)
    return pca, lower, upper


def overview_plot(
    chunks: list[dict[str, Any]],
    scene_dino_rows: list[dict[str, Any]],
    cond_indices: list[int],
    output: Path,
) -> None:
    labels = [f"chunk {chunk['chunk_index']}" for chunk in chunks]
    x = np.arange(len(chunks))
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    view_indices = [int(row["view_index"]) for row in scene_dino_rows]
    medians = [float(row["cosine_median"]) for row in scene_dino_rows]
    colors = ["#f0ad3d" if index in cond_indices else "#4c78a8" for index in view_indices]
    axes[0, 0].bar(np.arange(len(view_indices)), medians, color=colors)
    axes[0, 0].set_title("Foreground DINO: original vs strict-input")
    axes[0, 0].set_ylabel("cosine median")
    axes[0, 0].set_ylim(min(0.9, min(medians) - 0.01), 1.0)
    axes[0, 0].set_xticks(np.arange(len(view_indices)))
    axes[0, 0].set_xticklabels([f"{index:02d}" for index in view_indices], rotation=90)
    axes[0, 0].grid(axis="y", alpha=0.25)

    in_frustum = np.asarray([chunk["source"]["in_frustum_pairs"] for chunk in chunks], dtype=float)
    raw_sam = np.asarray([chunk["source"]["raw_sam_pairs"] for chunk in chunks], dtype=float)
    active_sam = np.asarray([chunk["source"]["active_sam_pairs"] for chunk in chunks], dtype=float)
    eligible = np.asarray([chunk["source"]["strict_eligible_pairs"] for chunk in chunks], dtype=float)
    width = 0.24
    axes[0, 1].bar(x - width, raw_sam / in_frustum * 100, width, label="SAM / in-frustum")
    axes[0, 1].bar(x, active_sam / in_frustum * 100, width, label="active SAM / in-frustum")
    axes[0, 1].bar(x + width, eligible / in_frustum * 100, width, label="strict eligible / in-frustum")
    axes[0, 1].set_title("cond3D candidate retention")
    axes[0, 1].set_ylabel("candidate pairs retained (%)")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels)
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", alpha=0.25)

    surface = np.asarray([chunk["source"]["track_surface_pairs"] for chunk in chunks], dtype=float)
    unknown = np.asarray([chunk["source"]["unknown_accepted_pairs"] for chunk in chunks], dtype=float)
    axes[1, 0].bar(x, surface / eligible * 100, label="track-depth surface", color="#2ca25f")
    axes[1, 0].bar(
        x,
        unknown / eligible * 100,
        bottom=surface / eligible * 100,
        label="unknown accepted",
        color="#f0ad3d",
    )
    axes[1, 0].set_title("Strict eligible source composition")
    axes[1, 0].set_ylabel("eligible pairs (%)")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels)
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.25)

    zero_source = np.asarray([chunk["source"]["zero_source_fraction"] for chunk in chunks])
    surface_only_zero = np.asarray(
        [chunk["source"]["surface_only_zero_source_fraction"] for chunk in chunks]
    )
    mixed = np.asarray([chunk["mixed_dino_patches"] / chunk["dino_mask_patches"] for chunk in chunks])
    depth_unknown = np.asarray(
        [chunk["depth_unknown_patches"] / chunk["dino_mask_patches"] for chunk in chunks]
    )
    gap_width = 0.19
    axes[1, 1].bar(x - 1.5 * gap_width, zero_source * 100, gap_width, label="final zero-source points")
    axes[1, 1].bar(
        x - 0.5 * gap_width,
        surface_only_zero * 100,
        gap_width,
        label="zero-source if unknown rejected",
    )
    axes[1, 1].bar(x + 0.5 * gap_width, mixed * 100, gap_width, label="mixed boundary DINO patches")
    axes[1, 1].bar(x + 1.5 * gap_width, depth_unknown * 100, gap_width, label="mask patches without depth")
    axes[1, 1].set_title("Information gaps after strict masking")
    axes[1, 1].set_ylabel("percent")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels)
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", alpha=0.25)

    figure.suptitle("Strict object-only conditioning: diagnostic summary", fontsize=16)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def region_source_plot(chunks: list[dict[str, Any]], output: Path) -> None:
    labels = [f"chunk {chunk['chunk_index']}" for chunk in chunks]
    regions = ["tabletop", "cabinet", "support_shell"]
    colors = ["#4c78a8", "#2ca25f", "#f0ad3d"]
    x = np.arange(len(chunks))
    width = 0.24
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    totals = np.asarray([chunk["source"]["point_count"] for chunk in chunks], dtype=float)
    bottom = np.zeros(len(chunks), dtype=float)
    for region, color in zip(regions, colors):
        values = np.asarray([chunk["source_regions"][region]["point_count"] for chunk in chunks]) / totals * 100
        axes[0].bar(x, values, bottom=bottom, label=region, color=color)
        bottom += values
    axes[0].set_title("Strict output occupancy by registered region")
    axes[0].set_ylabel("output points (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 100)
    axes[0].legend()

    for offset, (region, color) in enumerate(zip(regions, colors)):
        zero_values = [
            chunk["source_regions"][region]["zero_source_fraction"]
            if chunk["source_regions"][region]["zero_source_fraction"] is not None
            else np.nan
            for chunk in chunks
        ]
        axes[1].bar(x + (offset - 1) * width, np.asarray(zero_values) * 100, width, label=region, color=color)
    axes[1].set_title("Output points with zero cond3D sources")
    axes[1].set_ylabel("region points (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 100)
    axes[1].legend()

    for offset, (region, color) in enumerate(zip(regions, colors)):
        unknown_values = [
            chunk["source_regions"][region]["unknown_fraction_of_eligible"]
            if chunk["source_regions"][region]["unknown_fraction_of_eligible"] is not None
            else np.nan
            for chunk in chunks
        ]
        axes[2].bar(x + (offset - 1) * width, np.asarray(unknown_values) * 100, width, label=region, color=color)
    axes[2].set_title("Eligible sources accepted without track depth")
    axes[2].set_ylabel("eligible pairs (%)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_ylim(0, 100)
    axes[2].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def write_html(output: Path, chunks: list[dict[str, Any]]) -> None:
    def percent(value: float | None) -> str:
        return "n/a" if value is None else f"{value * 100:.1f}%"

    sections = []
    for chunk in chunks:
        index = int(chunk["chunk_index"])
        source = chunk["source"]
        dino = chunk["cond2d_dino_context_vs_strict"]
        tabletop = chunk["source_regions"]["tabletop"]
        cabinet = chunk["source_regions"]["cabinet"]
        sections.append(
            f'<section><h2>Chunk {index}</h2>'
            f'<p>cond2D local view {chunk["cond2d_view_index"]}, canonical mask {chunk["mask_view_index"]}; '
            f'{source["strict_eligible_pairs"]:,} eligible view-point pairs, '
            f'{source["unknown_fraction_of_eligible"] * 100:.1f}% accepted without track depth; '
            f'context-vs-strict foreground DINO cosine median {dino["cosine_median"]:.3f}. '
            f'Tabletop points {tabletop["point_count"]}, zero-source {percent(tabletop["zero_source_fraction"])} / unknown {percent(tabletop["unknown_fraction_of_eligible"])}; '
            f'cabinet points {cabinet["point_count"]}, zero-source {percent(cabinet["zero_source_fraction"])} / unknown {percent(cabinet["unknown_fraction_of_eligible"])}.</p>'
            f'<div class="wide"><a href="chunk_{index:03d}/condition_flow.jpg"><img src="chunk_{index:03d}/condition_flow.jpg" alt="chunk condition flow"></a></div>'
            f'<div class="wide"><a href="chunk_{index:03d}/source_eligibility.png"><img src="chunk_{index:03d}/source_eligibility.png" alt="source eligibility"></a></div>'
            f'<p><a href="chunk_{index:03d}/source_views.jpg">All active source crops</a> | '
            f'<a href="chunk_{index:03d}/metrics.json">metrics.json</a></p></section>'
        )
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Strict object feature-flow audit</title>
<style>body{{margin:0;background:#111;color:#eee;font:14px system-ui,sans-serif}}header,main{{padding:20px 28px}}header{{border-bottom:1px solid #333}}section{{padding:22px 0;border-bottom:1px solid #333}}h1,h2{{margin:0 0 8px}}p{{color:#bbb;max-width:1100px}}a{{color:#8fc7ff}}img{{display:block;width:100%;height:auto;background:#000}}.wide{{max-width:1800px;margin:12px 0}}</style></head>
<body><header><h1>Strict object-only feature-flow audit</h1>
<p>Pixel masks, actual 32x32 DINO ownership, recomputed DINO feature changes, and exact cond3D candidate eligibility for the three object chunks.</p>
<p>Eligible sources are not learned attribution weights. Aggregator softmax weights were not saved by inference.</p>
<p><a href="README.md">Method</a> | <a href="all_scene_view_gate.jpg">All 32 view gates</a> | <a href="active_scene_views.jpg">Active-view DINO audit</a> | <a href="summary.json">summary.json</a> | <a href="dino_feature_diagnostics_fp16.pt">DINO tensor dump</a></p></header><main><section><h2>Diagnostic summary</h2><div class="wide"><a href="overview.png"><img src="overview.png" alt="diagnostic summary"></a></div><div class="wide"><a href="region_sources.png"><img src="region_sources.png" alt="region source summary"></a></div></section>{''.join(sections)}</main></body></html>"""
    (output / "index.html").write_text(document)


def visualize(args: argparse.Namespace) -> dict[str, Any]:
    strict = args.strict.resolve()
    contextual = args.contextual.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    camera_document = json.loads((strict / "cameras.json").read_text())
    cameras = camera_document["scene"]
    chunks_json = {int(chunk["chunk_index"]): chunk for chunk in camera_document["chunks"]}
    run_args = json.loads((strict / "args.json").read_text())
    context_encoded = bool(run_args.get("object_feature_encode_context", False))
    ownership_json = json.loads((strict / "projection_ownership.json").read_text())
    transforms = json.loads((strict / "chunk_transforms.json").read_text())
    world_to_chunk0 = np.asarray(transforms["chunks"][0]["M_original_to_chunk"], dtype=np.float64)
    chunk_size_m = float(transforms["chunks"][0]["M_chunk_to_original"][0][0])
    mask_dir = (ROOT / run_args["projection_ownership_mask_dir"]).resolve()
    scene_root = (ROOT / run_args["path"]).resolve()
    enabled = set(int(value) for value in ownership_json["enabled_view_indices"])
    mask_mapping = ownership_json["mask_view_indices"]

    ownership, rebuilt_summary = build_projection_ownership(
        camera_document=camera_document,
        mask_dir=mask_dir,
        anchors_document=json.loads((ROOT / run_args["projection_ownership_anchors_json"]).read_text()),
        colmap_images_path=scene_root / run_args["colmap_subdir"] / "images.txt",
        colmap_points_path=scene_root / run_args["colmap_subdir"] / "points3D.txt",
        roi_boxes_world=run_args["projection_ownership_roi_box"],
        roi_bounds_chunk0=transform_boxes(run_args["projection_ownership_roi_box"], world_to_chunk0),
        world_to_chunk0=torch.tensor(world_to_chunk0, dtype=torch.float32),
        chunk_size_m=chunk_size_m,
        mode=run_args["projection_ownership_mode"],
        support_bounds_chunk0=transform_boxes(run_args["projection_ownership_support_box"], world_to_chunk0),
        enabled_view_indices=enabled,
        mask_view_indices=mask_mapping,
        global_feature_filter=run_args["projection_ownership_global_filter"],
        hard_support=run_args["projection_ownership_hard_support"],
        seed_surface=run_args["projection_ownership_seed_surface"],
        patch_resolution=32,
        mask_patch_threshold=run_args["projection_ownership_mask_patch_threshold"],
        depth_fill_radius_patches=run_args["projection_ownership_depth_fill_radius"],
        surface_band_m=run_args["projection_ownership_surface_band_m"],
        min_free_views=run_args["projection_ownership_min_free_views"],
        allow_unknown=run_args["projection_ownership_allow_unknown"],
        part_count=ownership_json["part_count"],
    )

    if rebuilt_summary["mask_patches"] != ownership_json["mask_patches"]:
        raise RuntimeError("rebuilt ownership mask differs from the strict inference diagnostics")
    if rebuilt_summary["depth_known_patches"] != ownership_json["depth_known_patches"]:
        raise RuntimeError("rebuilt ownership depth differs from the strict inference diagnostics")

    raw_patch_masks = []
    originals_512: dict[int, np.ndarray] = {}
    originals_1024: dict[int, np.ndarray] = {}
    strict_masks_512: dict[int, np.ndarray] = {}
    strict_masks_1024: dict[int, np.ndarray] = {}
    reproduction_diffs = []
    for view_index, camera in enumerate(cameras):
        mask_index = mask_mapping[view_index]
        if mask_index is None:
            raw_patch_masks.append(np.zeros((32, 32), dtype=bool))
        else:
            raw_patch_masks.append(
                downsample_mask(
                    mask_dir / f"view_{mask_index:03d}.png",
                    32,
                    run_args["projection_ownership_mask_patch_threshold"],
                )
            )
        original_1024 = load_exact_crop(camera, 1024)
        enabled_mask_1024 = load_view_mask(
            mask_dir, mask_index, enabled=view_index in enabled, size=1024
        )
        encoded_1024 = (
            original_1024 if context_encoded else original_1024 * enabled_mask_1024[..., None]
        )
        predicted = np.floor(np.clip(encoded_1024, 0, 1) * 255).astype(np.uint8)
        saved = np.asarray(Image.open(strict / "scene" / f"view_{view_index:03d}.png").convert("RGB"))
        difference = np.abs(predicted.astype(np.int16) - saved.astype(np.int16))
        reproduction_diffs.append(
            {
                "view_index": view_index,
                "different_values": int((difference > 0).sum()),
                "max_abs_u8": int(difference.max()),
                "mean_abs_u8": float(difference.mean()),
            }
        )
        if view_index in enabled:
            originals_1024[view_index] = original_1024
            originals_512[view_index] = load_exact_crop(camera, 512)
            strict_masks_1024[view_index] = enabled_mask_1024
            strict_masks_512[view_index] = load_view_mask(
                mask_dir, mask_index, enabled=True, size=512
            )
    raw_patch_masks_np = np.stack(raw_patch_masks)

    active_indices = sorted(enabled)
    cond_indices = [int(chunks_json[index]["cond2d_view"]["scene_view_index"]) for index in sorted(chunks_json)]
    pipeline_config = json.loads((ROOT / run_args["pipeline_config"]).read_text())
    model_path = pipeline_config["args"]["image_cond_model"]["args"]["model_name"]
    rgb_encoding_mode = (
        "full_context_before_token_gate" if context_encoded else "sam_masked_before_dino"
    )
    feature_dump_path = output / "dino_feature_diagnostics_fp16.pt"
    if feature_dump_path.is_file():
        cached = torch.load(feature_dump_path, map_location="cpu", weights_only=True)
        cache_matches = (
            cached.get("schema") == "genrecon.strict-object-dino-feature-audit"
            and cached.get("model_path") == model_path
            and cached.get("active_view_indices") == active_indices
            and cached.get("cond2d_view_indices") == cond_indices
            and cached.get("rgb_encoding_mode") == rgb_encoding_mode
        )
    else:
        cached = None
        cache_matches = False

    if cache_matches:
        print(f"[DINO] reusing {feature_dump_path}")
        scene_original_features = cached["scene_original_tokens"].float()
        scene_strict_features = cached["scene_strict_input_tokens"].float()
        cond_context_features = cached["cond2d_context15_tokens"].float()
        patch_masks_active = cached["active_patch_masks"].bool()
    else:
        input_tensors: list[torch.Tensor] = []
        input_labels: list[tuple[str, int]] = []
        for view_index in active_indices:
            original = originals_512[view_index]
            mask = strict_masks_512[view_index]
            input_tensors.append(torch.from_numpy(original).permute(2, 0, 1).float())
            input_labels.append(("scene_original", view_index))
            encoded_input = original if context_encoded else original * mask[..., None]
            input_tensors.append(torch.from_numpy(encoded_input).permute(2, 0, 1).float())
            input_labels.append(("scene_strict", view_index))
        for view_index in cond_indices:
            original = originals_512[view_index]
            mask = strict_masks_512[view_index]
            weight = 0.15 + 0.85 * mask
            input_tensors.append(torch.from_numpy(original * weight[..., None]).permute(2, 0, 1).float())
            input_labels.append(("cond_context15", view_index))

        features = encode_images(input_tensors, model_path, args.batch_size)
        feature_lookup = {label: features[index] for index, label in enumerate(input_labels)}
        scene_original_features = torch.stack(
            [feature_lookup[("scene_original", index)] for index in active_indices]
        )
        scene_strict_features = torch.stack(
            [feature_lookup[("scene_strict", index)] for index in active_indices]
        )
        cond_context_features = torch.stack(
            [feature_lookup[("cond_context15", index)] for index in cond_indices]
        )
        patch_masks_active = ownership["mask"][active_indices].reshape(len(active_indices), 32, 32)
        torch.save(
            {
                "schema": "genrecon.strict-object-dino-feature-audit",
                "model_path": model_path,
                "rgb_encoding_mode": rgb_encoding_mode,
                "active_view_indices": active_indices,
                "cond2d_view_indices": cond_indices,
                "scene_original_tokens": scene_original_features.half(),
                "scene_strict_input_tokens": scene_strict_features.half(),
                "cond2d_context15_tokens": cond_context_features.half(),
                "active_patch_masks": patch_masks_active,
                "note": "Strict final patch tokens equal scene_strict_input_tokens with non-mask patches zeroed.",
            },
            feature_dump_path,
        )

    original_patch_sets = [
        scene_original_features[index, 5:].float().numpy().reshape(32, 32, -1)
        for index in range(len(active_indices))
    ]
    strict_patch_sets = [
        scene_strict_features[index, 5:].float().numpy().reshape(32, 32, -1)
        for index in range(len(active_indices))
    ]
    context_patch_sets = [
        cond_context_features[index, 5:].float().numpy().reshape(32, 32, -1)
        for index in range(len(cond_indices))
    ]
    pca, pca_low, pca_high = fit_feature_pca(original_patch_sets + strict_patch_sets)

    active_position = {view_index: position for position, view_index in enumerate(active_indices)}
    scene_dino_rows = []
    cosine_maps: dict[int, np.ndarray] = {}
    all_selected_cosines = []
    for view_index in active_indices:
        position = active_position[view_index]
        patch_mask = patch_masks_active[position].numpy()
        cosine, metrics = feature_similarity(
            original_patch_sets[position], strict_patch_sets[position], patch_mask
        )
        cosine_maps[view_index] = cosine
        all_selected_cosines.append(cosine[patch_mask])
        scene_dino_rows.append({"view_index": view_index, **metrics})
    cosine_values = np.concatenate(all_selected_cosines)
    cosine_low = float(np.percentile(cosine_values, 1))
    cosine_high = float(np.percentile(cosine_values, 99))

    active_tiles = []
    ownership_masks_np = ownership["mask"].numpy().reshape(len(cameras), 32, 32)
    depth_known_np = ownership["depth_valid"].numpy().reshape(len(cameras), 32, 32)
    view_diag = {int(row["view_index"]): row for row in ownership_json["views"]}
    view_gate_rows = []
    view_gate_cards = []
    for view_index, camera in enumerate(cameras):
        mask_index = mask_mapping[view_index]
        is_active = view_index in enabled
        original = originals_512.get(view_index)
        if original is None:
            original = load_exact_crop(camera, 512)
        raw_pixel_mask = load_view_mask(
            mask_dir, mask_index, enabled=mask_index is not None, size=512
        )
        training_points = int(camera.get("object_feature_training_points", 0))
        if is_active:
            reason = "enabled"
        elif mask_index is None:
            reason = "disabled: no canonical crop match"
        else:
            reason = f"disabled: training points {training_points} < {run_args['object_feature_min_training_points']}"
        panels = [
            labeled_tile(rgb_image(original), f"#{view_index:02d} original", Path(camera["img_path"]).stem, 180),
            labeled_tile(
                overlay_mask(original, raw_pixel_mask),
                f"SAM canonical {mask_index}",
                f"pixel {raw_pixel_mask.mean() * 100:.1f}%",
                180,
            ),
            labeled_tile(
                grid_image(ownership_masks_np[view_index].astype(float), "viridis", 0, 1),
                "actual DINO gate",
                f"{ownership_masks_np[view_index].sum()}/1024 patches",
                180,
            ),
        ]
        panel_height = panels[0].height
        card = Image.new("RGB", (540, panel_height + 24), (5, 5, 5))
        for panel_index, panel in enumerate(panels):
            card.paste(panel, (panel_index * 180, 0))
        draw = ImageDraw.Draw(card)
        border = (68, 212, 118) if is_active else (225, 72, 72)
        draw.rectangle((0, panel_height, card.width, card.height), fill=(10, 10, 10))
        draw.text((7, panel_height + 4), reason, fill=border)
        for inset in range(3):
            draw.rectangle((inset, inset, card.width - 1 - inset, card.height - 1 - inset), outline=border)
        view_gate_cards.append(card)
        view_gate_rows.append(
            {
                "view_index": view_index,
                "image": Path(camera["img_path"]).name,
                "crop": infer_crop_side(camera["intrinsics"]),
                "mask_view_index": mask_index,
                "feature_enabled": is_active,
                "training_points": training_points,
                "raw_pixel_mask_fraction": float(raw_pixel_mask.mean()),
                "actual_dino_mask_patches": int(ownership_masks_np[view_index].sum()),
                "gate_reason": reason,
            }
        )
    contact_sheet(view_gate_cards, output / "all_scene_view_gate.jpg", columns=4)

    for view_index in active_indices:
        position = active_position[view_index]
        original = originals_512[view_index]
        pixel_mask = strict_masks_512[view_index]
        patch_mask = ownership_masks_np[view_index]
        card_parts = [
            labeled_tile(rgb_image(original), f"#{view_index:02d} original", Path(cameras[view_index]["img_path"]).stem, 220),
            labeled_tile(overlay_mask(original, pixel_mask), "SAM union", f"pixel {pixel_mask.mean() * 100:.1f}%", 220),
            labeled_tile(grid_image(patch_mask.astype(float), "viridis", 0, 1), "DINO ownership", f"{patch_mask.sum()}/1024 patches", 220),
            labeled_tile(
                grid_image(cosine_maps[view_index], "turbo", cosine_low, cosine_high, patch_mask),
                "DINO cosine",
                f"median {scene_dino_rows[position]['cosine_median']:.3f}",
                220,
            ),
        ]
        card = Image.new("RGB", (440, 2 * card_parts[0].height), (5, 5, 5))
        for card_index, tile in enumerate(card_parts):
            card.paste(tile, ((card_index % 2) * 220, (card_index // 2) * tile.height))
        active_tiles.append(card)
    contact_sheet(active_tiles, output / "active_scene_views.jpg", columns=4)

    chunk_summaries = []
    source_rows = []
    for chunk_position, chunk_index in enumerate(sorted(chunks_json)):
        chunk_output = output / f"chunk_{chunk_index:03d}"
        chunk_output.mkdir(parents=True, exist_ok=True)
        cond_view_index = cond_indices[chunk_position]
        mask_view_index = mask_mapping[cond_view_index]
        position = active_position[cond_view_index]
        original_512 = originals_512[cond_view_index]
        original_1024 = originals_1024[cond_view_index]
        mask_512 = strict_masks_512[cond_view_index]
        mask_1024 = strict_masks_1024[cond_view_index]
        patch_mask = ownership_masks_np[cond_view_index]
        known_mask = depth_known_np[cond_view_index]
        context_input = original_512 * (0.15 + 0.85 * mask_512[..., None])
        strict_input = original_512 if context_encoded else original_512 * mask_512[..., None]
        original_features = original_patch_sets[position]
        strict_features = strict_patch_sets[position]
        context_features = context_patch_sets[chunk_position]
        cosine_original_strict, original_strict_metrics = feature_similarity(
            original_features, strict_features, patch_mask
        )
        cosine_context_strict, context_strict_metrics = feature_similarity(
            context_features, strict_features, patch_mask
        )

        part_masks = []
        if mask_view_index is not None:
            for part_index in range(2):
                part_path = mask_dir / f"view_{mask_view_index:03d}_part_{part_index}.png"
                raw_part = np.asarray(Image.open(part_path).convert("L"), dtype=np.float32) / 255.0
                part_masks.append(resize_mask(raw_part, 512))

        flow_tiles = [
            ("Original cond2D crop", f"local view {cond_view_index}", rgb_image(original_512)),
            ("SAM3.1 union", f"pixel coverage {mask_512.mean() * 100:.1f}%", overlay_mask(original_512, mask_512)),
            ("SAM parts", "cyan tabletop / orange cabinet", overlay_parts(original_512, part_masks)),
            ("Contextual input", "15% mask-out RGB retained", rgb_image(context_input)),
            (
                "Object-token encoder input",
                "full RGB before token gate" if context_encoded else "mask-out RGB = 0",
                rgb_image(strict_input),
            ),
            ("Actual DINO mask", f"{patch_mask.sum()}/1024 patches", grid_image(patch_mask.astype(float), "viridis", 0, 1)),
            ("Track-depth grid", "green known / yellow unknown", ownership_grid(patch_mask, known_mask)),
            ("DINO PCA original", "same PCA basis", pca_image(original_features, pca, pca_low, pca_high)),
            ("DINO PCA context15", "same PCA basis", pca_image(context_features, pca, pca_low, pca_high)),
            ("DINO PCA encoder output", "before token zeroing", pca_image(strict_features, pca, pca_low, pca_high)),
            ("DINO PCA strict final", "non-object tokens zeroed", pca_image(strict_features, pca, pca_low, pca_high, patch_mask)),
            (
                "Context/strict cosine",
                f"range {cosine_low:.2f}-{cosine_high:.2f}",
                grid_image(cosine_context_strict, "turbo", cosine_low, cosine_high, patch_mask),
            ),
        ]
        detailed_panel(flow_tiles, chunk_output / "condition_flow.jpg")

        coords_path = strict / f"coords_{chunk_index:03d}.ply"
        point_cloud = trimesh.load(coords_path, process=False)
        points_world = np.asarray(point_cloud.vertices, dtype=np.float64)
        points_chunk0 = transform_points(points_world, world_to_chunk0)
        source = classify_sources(points_chunk0, cameras, ownership, raw_patch_masks_np)
        view_labels = [f"{index:02d}" for index in range(len(cameras))]
        source_summary = source_plot(
            points_world,
            source,
            view_labels,
            chunk_output / "source_eligibility.png",
            chunk_index,
        )
        roi_masks = []
        for box in run_args["projection_ownership_roi_box"]:
            x0, x1, y0, y1, z0, z1 = box
            roi_masks.append(
                (points_world[:, 0] >= x0)
                & (points_world[:, 0] <= x1)
                & (points_world[:, 1] >= y0)
                & (points_world[:, 1] <= y1)
                & (points_world[:, 2] >= z0)
                & (points_world[:, 2] <= z1)
            )
        source_regions = {
            "tabletop": summarize_sources(source, roi_masks[0]),
            "cabinet": summarize_sources(source, roi_masks[1]),
            "support_shell": summarize_sources(source, ~(roi_masks[0] | roi_masks[1])),
        }

        source_tiles = []
        for view_index in active_indices:
            view_original = originals_512[view_index]
            view_mask = strict_masks_512[view_index]
            final_count = int(source["final"][view_index].sum())
            unknown_count = int(source["unknown"][view_index].sum())
            unknown_fraction = unknown_count / final_count if final_count else 0.0
            diagnostic = view_diag[view_index]
            source_tiles.append(
                projection_thumbnail(
                    view_original,
                    view_mask,
                    source["uv"][view_index],
                    source["final"][view_index],
                    f"#{view_index:02d} {Path(cameras[view_index]['img_path']).stem} {infer_crop_side(cameras[view_index]['intrinsics'])[0].upper()}",
                    f"eligible {final_count} | unknown {unknown_fraction * 100:.0f}% | tracks {diagnostic['projected_training_track_observations']}",
                    active=True,
                )
            )
            source_rows.append(
                {
                    "chunk_index": chunk_index,
                    "view_index": view_index,
                    "image": Path(cameras[view_index]["img_path"]).name,
                    "crop": infer_crop_side(cameras[view_index]["intrinsics"]),
                    "is_cond2d": view_index == cond_view_index,
                    "in_frustum_points": int(source["in_frustum"][view_index].sum()),
                    "raw_sam_points": int(source["raw_sam"][view_index].sum()),
                    "strict_eligible_points": final_count,
                    "surface_points": int(source["surface"][view_index].sum()),
                    "unknown_points": unknown_count,
                }
            )
        contact_sheet(source_tiles, chunk_output / "source_views.jpg", columns=4)

        pixel_patch_average = np.asarray(
            Image.fromarray(np.rint(mask_1024 * 255).astype(np.uint8)).resize(
                (32, 32), Image.Resampling.BOX
            ),
            dtype=np.float32,
        ) / 255.0
        chunk_summary = {
            "chunk_index": chunk_index,
            "cond2d_view_index": cond_view_index,
            "mask_view_index": mask_view_index,
            "image": Path(cameras[cond_view_index]["img_path"]).name,
            "crop": infer_crop_side(cameras[cond_view_index]["intrinsics"]),
            "pixel_mask_fraction": float(mask_1024.mean()),
            "dino_mask_patches": int(patch_mask.sum()),
            "dino_mask_fraction": float(patch_mask.mean()),
            "mixed_dino_patches": int(((pixel_patch_average > 0) & (pixel_patch_average < 1)).sum()),
            "depth_known_patches": int((patch_mask & known_mask).sum()),
            "depth_unknown_patches": int((patch_mask & ~known_mask).sum()),
            "cond2d_dino_original_vs_strict": original_strict_metrics,
            "cond2d_dino_context_vs_strict": context_strict_metrics,
            "source": source_summary,
            "source_regions": source_regions,
        }
        (chunk_output / "metrics.json").write_text(json.dumps(chunk_summary, indent=2) + "\n")
        chunk_summaries.append(chunk_summary)

    with (output / "source_views.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)
    with (output / "scene_dino_similarity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scene_dino_rows[0]))
        writer.writeheader()
        writer.writerows(scene_dino_rows)
    with (output / "view_gate.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(view_gate_rows[0]))
        writer.writeheader()
        writer.writerows(view_gate_rows)

    reproduction_values = [row["mean_abs_u8"] for row in reproduction_diffs]
    summary = {
        "schema": "genrecon.strict-object-feature-flow-audit",
        "strict_reconstruction": str(strict),
        "contextual_reconstruction": str(contextual),
        "dino_model": model_path,
        "rgb_encoding_mode": rgb_encoding_mode,
        "scene_views": len(cameras),
        "canonical_mask_matches": sum(index is not None for index in mask_mapping),
        "active_feature_views": active_indices,
        "active_feature_view_count": len(active_indices),
        "view_gate": view_gate_rows,
        "mask_patch_threshold": run_args["projection_ownership_mask_patch_threshold"],
        "surface_band_m": run_args["projection_ownership_surface_band_m"],
        "allow_unknown": run_args["projection_ownership_allow_unknown"],
        "mask_patches": rebuilt_summary["mask_patches"],
        "depth_known_patches": rebuilt_summary["depth_known_patches"],
        "dino_scene_original_vs_strict": {
            "cosine_mean": float(cosine_values.mean()),
            "cosine_median": float(np.median(cosine_values)),
            "cosine_p10": float(np.percentile(cosine_values, 10)),
            "cosine_visual_scale": [cosine_low, cosine_high],
        },
        "input_reproduction": {
            "max_mean_abs_u8": float(max(reproduction_values)),
            "max_abs_u8": int(max(row["max_abs_u8"] for row in reproduction_diffs)),
            "total_different_values": int(sum(row["different_values"] for row in reproduction_diffs)),
            "views": reproduction_diffs,
        },
        "chunks": chunk_summaries,
        "warning": "Projection eligibility is exact but is not a learned aggregator attribution weight.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    overview_plot(chunk_summaries, scene_dino_rows, cond_indices, output / "overview.png")
    region_source_plot(chunk_summaries, output / "region_sources.png")
    write_html(output, chunk_summaries)

    readme = f"""# Strict object-only feature-flow audit

This audit separates four different concepts that must not be conflated. The audited RGB encoding mode is `{'full_context_before_token_gate' if context_encoded else 'sam_masked_before_dino'}`:

1. Pixel-level SAM3.1 union and part masks.
2. The actual 32x32 patch mask consumed by DINO projection (`threshold={run_args['projection_ownership_mask_patch_threshold']}`).
3. DINO features recomputed with the same local model, crop, resize, and strict black-mask input.
4. Exact projection candidates after in-frustum, active-view, SAM-patch, and track-depth rules.

`condition_flow.jpg` shows the primary cond2D path for each chunk. `source_eligibility.png` and `source_views.jpg` show all scene-view cond3D candidates on the saved strict output occupancy coordinates. Green projected pixels in source thumbnails are exact eligible candidates.

The DINO tensor dump stores FP16 copies of original scene tokens, strict-input scene tokens, contextual-15% cond2D tokens, and actual patch masks. Strict final features are obtained by zeroing non-mask patch tokens from the strict-input tokens.

## Limits

- Candidate eligibility is not learned contribution. Per-view aggregator softmax weights were not persisted by inference.
- DINO PCA colors are a diagnostic projection, not semantic class probabilities.
- Source plots use the generated strict occupancy points, so they diagnose this output rather than an independent ground-truth surface.
- `unknown accepted` means the SAM patch has no nearby registered training-track depth. It is retained because this run used `allow_unknown=true`; it is not evidence of a visible surface.

Open `index.html` for the complete audit. `all_scene_view_gate.jpg` shows all 32 local crops, their raw canonical SAM masks, and the actual post-gate DINO ownership. Machine-readable values are in `summary.json`, `view_gate.csv`, `source_views.csv`, and `scene_dino_similarity.csv`.
"""
    (output / "README.md").write_text(readme)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", type=Path, required=True)
    parser.add_argument("--contextual", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    for folder in (args.strict, args.contextual):
        if not (folder / "cameras.json").is_file():
            raise FileNotFoundError(folder / "cameras.json")
    summary = visualize(args)
    print(
        f"Wrote {len(summary['chunks'])} chunks, {summary['active_feature_view_count']} active views "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
