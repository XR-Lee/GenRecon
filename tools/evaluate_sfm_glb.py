#!/usr/bin/env python3
"""Measure exact COLMAP/SfM point-to-GLB triangle-surface consistency.

The primary direction is SfM -> GLB. The reverse direction is intentionally not
reported as precision because sparse SfM points are not an area-uniform sample
of the scene surface.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree

try:
    from tools.evaluate_view_fidelity import genrecon_glb_to_world_matrix, read_colmap_points
except ModuleNotFoundError:  # Direct execution puts tools/ rather than the repo root on sys.path.
    from evaluate_view_fidelity import genrecon_glb_to_world_matrix, read_colmap_points

DEFAULT_THRESHOLDS_M = (0.02, 0.05, 0.10, 0.20, 0.50, 1.00)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n")


def load_point_cloud(path: str | Path) -> np.ndarray:
    loaded = trimesh.load(str(path), process=False)
    if not isinstance(loaded, (trimesh.PointCloud, trimesh.Trimesh)):
        raise ValueError(f"Input is not point-like geometry: {path}")
    points = np.asarray(loaded.vertices, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError(f"Point cloud has invalid shape {points.shape}: {path}")
    if not np.isfinite(points).all():
        raise ValueError(f"Point cloud contains non-finite coordinates: {path}")
    return points


def distance_summary(
    distances: np.ndarray,
    thresholds_m: tuple[float, ...] = DEFAULT_THRESHOLDS_M,
) -> dict[str, Any]:
    values = np.asarray(distances, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("No finite point-to-surface distances")
    return {
        "count": int(len(values)),
        "max_m": float(np.max(values)),
        "mean_m": float(np.mean(values)),
        "median_m": float(np.median(values)),
        "p75_m": float(np.quantile(values, 0.75)),
        "p90_m": float(np.quantile(values, 0.90)),
        "p95_m": float(np.quantile(values, 0.95)),
        "p99_m": float(np.quantile(values, 0.99)),
        "rmse_m": float(np.sqrt(np.mean(np.square(values)))),
        "threshold_recall": {
            f"{threshold:.3f}": float(np.mean(values <= threshold)) for threshold in thresholds_m
        },
    }


def points_inside_bounds(points: np.ndarray, bounds: np.ndarray, margin_m: float) -> np.ndarray:
    if margin_m < 0 or not math.isfinite(margin_m):
        raise ValueError("ROI margin must be finite and non-negative")
    return np.all((points >= bounds[0] - margin_m) & (points <= bounds[1] + margin_m), axis=1)


def exact_nearest_glb_surface(
    query_points: np.ndarray,
    chunk_paths: list[Path],
    *,
    query_batch_size: int = 100_000,
) -> dict[str, np.ndarray]:
    """Query exact closest triangle points across independently baked GLB chunks."""
    import open3d as o3d

    if not chunk_paths:
        raise ValueError("No GLB chunks were provided")
    query = np.asarray(query_points, dtype=np.float32)
    minimum_distance = np.full(len(query), np.inf, dtype=np.float32)
    closest_point = np.full_like(query, np.nan)
    closest_chunk = np.full(len(query), -1, dtype=np.int32)
    bounds = np.asarray([[np.inf, np.inf, np.inf], [-np.inf, -np.inf, -np.inf]], dtype=np.float64)
    asset_to_world = genrecon_glb_to_world_matrix()

    for chunk_index, path in enumerate(chunk_paths):
        started = time.perf_counter()
        model = o3d.io.read_triangle_model(str(path))
        if not model.meshes:
            raise ValueError(f"GLB chunk contains no triangle meshes: {path}")
        scene = o3d.t.geometry.RaycastingScene()
        chunk_vertices = 0
        chunk_triangles = 0
        for item in model.meshes:
            mesh = item.mesh
            mesh.transform(asset_to_world)
            if not mesh.has_triangles():
                continue
            box = mesh.get_axis_aligned_bounding_box()
            bounds[0] = np.minimum(bounds[0], box.get_min_bound())
            bounds[1] = np.maximum(bounds[1], box.get_max_bound())
            chunk_vertices += len(mesh.vertices)
            chunk_triangles += len(mesh.triangles)
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        if chunk_triangles == 0:
            raise ValueError(f"GLB chunk contains no usable triangles: {path}")

        for start in range(0, len(query), query_batch_size):
            stop = min(start + query_batch_size, len(query))
            tensor = o3d.core.Tensor(query[start:stop], dtype=o3d.core.Dtype.Float32)
            answer = scene.compute_closest_points(tensor)
            candidate_point = answer["points"].numpy()
            candidate_distance = np.linalg.norm(candidate_point - query[start:stop], axis=1)
            current = minimum_distance[start:stop]
            update = candidate_distance < current
            current[update] = candidate_distance[update]
            closest_point[start:stop][update] = candidate_point[update]
            closest_chunk[start:stop][update] = chunk_index

        print(
            f"[sfm-glb] {chunk_index + 1:02d}/{len(chunk_paths)} {path.name}: "
            f"{chunk_vertices:,} vertices, {chunk_triangles:,} faces, "
            f"current median={np.median(minimum_distance):.4f}m, "
            f"{time.perf_counter() - started:.2f}s"
        )
        del model, scene
        gc.collect()

    if not np.isfinite(minimum_distance).all() or not np.isfinite(closest_point).all():
        raise RuntimeError("Closest-surface query produced non-finite output")
    return {
        "bounds": bounds,
        "chunk_index": closest_chunk,
        "closest_point": closest_point,
        "distance": minimum_distance,
    }


def match_clean_points_to_colmap_ids(
    clean_points: np.ndarray,
    quality_points: np.ndarray,
    quality_ids: np.ndarray,
    *,
    tolerance_m: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(quality_points)
    distances, indices = tree.query(clean_points, k=1)
    matched_ids = np.full(len(clean_points), -1, dtype=np.int64)
    matched_ids[distances <= tolerance_m] = quality_ids[indices[distances <= tolerance_m]]
    return matched_ids, distances


def _source_result(
    points: np.ndarray,
    nearest: dict[str, np.ndarray],
    subset: slice,
    bounds: np.ndarray,
    roi_margin_m: float,
) -> dict[str, Any]:
    distances = nearest["distance"][subset]
    closest = nearest["closest_point"][subset]
    roi_mask = points_inside_bounds(points, bounds, roi_margin_m)
    displacement = closest - points
    inlier = distances <= 0.20
    result = {
        "all": distance_summary(distances),
        "bounds_m": np.stack((points.min(axis=0), points.max(axis=0))),
        "inside_glb_aabb_count": int(roi_mask.sum()),
        "outside_glb_aabb_count": int((~roi_mask).sum()),
        "roi": distance_summary(distances[roi_mask]) if np.any(roi_mask) else None,
        "roi_margin_m": roi_margin_m,
    }
    if np.any(inlier):
        result["inlier_displacement_closest_minus_sfm_m"] = {
            "count": int(inlier.sum()),
            "mean_xyz": displacement[inlier].mean(axis=0),
            "median_xyz": np.median(displacement[inlier], axis=0),
        }
    return result


def _write_point_csv(
    path: Path,
    *,
    point_ids: np.ndarray,
    points: np.ndarray,
    nearest_points: np.ndarray,
    distances: np.ndarray,
    chunks: np.ndarray,
    roi_mask: np.ndarray,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "point_id",
                "x_m",
                "y_m",
                "z_m",
                "closest_x_m",
                "closest_y_m",
                "closest_z_m",
                "delta_x_m",
                "delta_y_m",
                "delta_z_m",
                "distance_m",
                "closest_chunk",
                "inside_glb_aabb",
            ]
        )
        for point_id, point, closest, distance, chunk, inside in zip(
            point_ids, points, nearest_points, distances, chunks, roi_mask
        ):
            delta = closest - point
            writer.writerow(
                [
                    int(point_id),
                    *map(float, point),
                    *map(float, closest),
                    *map(float, delta),
                    float(distance),
                    int(chunk),
                    int(inside),
                ]
            )


def _distance_colors(distances: np.ndarray, maximum_m: float = 0.30) -> np.ndarray:
    from matplotlib import colormaps

    normalized = np.clip(np.asarray(distances) / maximum_m, 0, 1)
    return np.rint(colormaps["turbo"](normalized) * 255).astype(np.uint8)


def _write_colored_cloud(path: Path, points: np.ndarray, distances: np.ndarray) -> None:
    trimesh.PointCloud(points, colors=_distance_colors(distances)).export(path)


def _write_plots(
    output: Path,
    clean_points: np.ndarray,
    clean_distances: np.ndarray,
    quality_roi_points: np.ndarray,
    quality_roi_distances: np.ndarray,
    per_chunk: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    scatter = axes[0, 0].scatter(
        clean_points[:, 0],
        clean_points[:, 1],
        c=np.clip(clean_distances, 0, 0.30),
        s=4,
        cmap="turbo",
        vmin=0,
        vmax=0.30,
    )
    axes[0, 0].set_title("Chunker-cleaned SfM points: distance to GLB")
    axes[0, 0].set_xlabel("world x (m)")
    axes[0, 0].set_ylabel("world y (m)")
    axes[0, 0].set_aspect("equal", adjustable="box")
    figure.colorbar(scatter, ax=axes[0, 0], label="distance (m), clipped at 0.30")

    axes[0, 1].scatter(
        quality_roi_points[:, 0],
        quality_roi_points[:, 1],
        c=np.clip(quality_roi_distances, 0, 0.30),
        s=3,
        cmap="turbo",
        vmin=0,
        vmax=0.30,
    )
    axes[0, 1].set_title("Quality-filtered SfM points inside GLB AABB")
    axes[0, 1].set_xlabel("world x (m)")
    axes[0, 1].set_ylabel("world y (m)")
    axes[0, 1].set_aspect("equal", adjustable="box")

    for label, values, color in (
        ("cleaned", clean_distances, "#1677b8"),
        ("quality + ROI", quality_roi_distances, "#d45527"),
    ):
        sorted_values = np.sort(values)
        cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        axes[1, 0].plot(sorted_values, cdf, label=label, color=color, linewidth=2)
    for threshold in (0.02, 0.05, 0.10, 0.20):
        axes[1, 0].axvline(threshold, color="#777777", linewidth=0.7, linestyle="--")
    axes[1, 0].set_xlim(0, 0.60)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_xlabel("exact point-to-triangle distance (m)")
    axes[1, 0].set_ylabel("fraction of SfM points")
    axes[1, 0].set_title("Empirical CDF")
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].legend()

    chunk_ids = [item["chunk"] for item in per_chunk]
    medians = [item["metrics"]["median_m"] * 100 for item in per_chunk]
    p90s = [item["metrics"]["p90_m"] * 100 for item in per_chunk]
    axes[1, 1].bar(chunk_ids, p90s, color="#b6b6b6", label="P90")
    axes[1, 1].bar(chunk_ids, medians, color="#2b8c6b", label="median")
    axes[1, 1].set_xlabel("nearest GLB chunk")
    axes[1, 1].set_ylabel("distance (cm, log scale)")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Cleaned SfM consistency by nearest chunk (GLB AABB only)")
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", alpha=0.2)
    figure.savefig(output / "sfm_glb_consistency.png", dpi=180)
    plt.close(figure)


def _write_chunk_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["chunk", "points", "mean_m", "median_m", "p90_m", "within_0.05m", "within_0.10m"])
        for row in rows:
            metrics = row["metrics"]
            writer.writerow(
                [
                    row["chunk"],
                    metrics["count"],
                    metrics["mean_m"],
                    metrics["median_m"],
                    metrics["p90_m"],
                    metrics["threshold_recall"]["0.050"],
                    metrics["threshold_recall"]["0.100"],
                ]
            )


def _percent(value: float) -> str:
    return f"{value:.1%}"


def _cm(value: float) -> str:
    return f"{value * 100:.2f}cm"


def _write_report(
    output: Path,
    result: dict[str, Any],
    per_chunk: list[dict[str, Any]],
    scene_label: str,
) -> None:
    sources = result["sources"]
    chunk_count = int(result["glb"]["chunk_count"])
    clean_count = int(sources["cleaned"]["all"]["count"])
    lines = [
        f"# {scene_label} COLMAP/SfM 与 GLB 几何一致性",
        "",
        f"本评测在 COLMAP 世界坐标中，对每个 SfM 点计算其到 {chunk_count} 个 GLB chunks 中所有三角面的精确最近距离。",
        "GLB 导出的 `(x,z,-y)` 轴已转回 COLMAP `(x,y,z)`；查询使用 Open3D RaycastingScene BVH，不是表面采样近似。",
        "当前 COLMAP 模型只提供稀疏 `points3D.txt`，所以这是稀疏点到表面的覆盖评测，不是 dense MVS 的双向表面评测。",
        "",
        "## 结果",
        "",
        "| SfM 点集 | 点数 | 均值 | 中位数 | P90 | ≤2cm | ≤5cm | ≤10cm | ≤20cm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label, subset in (
        ("cleaned", "chunker 清理后（全部）", "all"),
        ("cleaned", "chunker 清理后（GLB AABB）", "roi"),
        ("quality_filtered", "error≤2px、track≥3（全部）", "all"),
        ("quality_filtered", "error≤2px、track≥3（GLB AABB）", "roi"),
    ):
        metrics = sources[key][subset]
        thresholds = metrics["threshold_recall"]
        lines.append(
            f"| {label} | {metrics['count']:,} | {_cm(metrics['mean_m'])} | {_cm(metrics['median_m'])} | "
            f"{_cm(metrics['p90_m'])} | {_percent(thresholds['0.020'])} | "
            f"{_percent(thresholds['0.050'])} | {_percent(thresholds['0.100'])} | "
            f"{_percent(thresholds['0.200'])} |"
        )

    clean = sources["cleaned"]["all"]
    quality_roi = sources["quality_filtered"]["roi"]
    worst = sorted(per_chunk, key=lambda item: item["metrics"]["median_m"], reverse=True)[:5]
    lines.extend(
        [
            "",
            f"主结果建议看 `chunker 清理后（全部）`：这是 GenRecon 实际用于分块和相机选择的 {clean_count:,} 个 SfM 点。",
            f"其中 {_percent(clean['threshold_recall']['0.100'])} 距 GLB 表面不超过 10cm，"
            f"中位距离为 {_cm(clean['median_m'])}。",
            "独立的质量过滤 + GLB ROI 点集更严格，保留了未通过空间密度清理的点；",
            f"其中 {_percent(quality_roi['threshold_recall']['0.100'])} 在 10cm 内。",
            "",
            "## 按最近 GLB chunk 分组（GLB AABB 内）",
            "",
            "这里按最近表面的 chunk 分组，不代表 SfM 点原始的 chunk 所有权。",
            "",
            "| chunk | 支撑点 | 中位距离 | P90 | ≤10cm |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for item in worst:
        metrics = item["metrics"]
        lines.append(
            f"| {item['chunk']:03d} | {metrics['count']:,} | {_cm(metrics['median_m'])} | "
            f"{_cm(metrics['p90_m'])} | {_percent(metrics['threshold_recall']['0.100'])} |"
        )

    lines.extend(
        [
            "",
            "## 与相机深度评测的区别",
            "",
            "- 本页的直接 3D 最近面距离回答：SfM 特征点附近是否存在 GLB 表面。它不检查该表面从原相机看是否可见。",
            "- 上级目录的相机深度评测回答：沿原相机射线首先看到的 GLB 表面是否与 SfM 点深度一致。它能发现错误前表面遮挡。",
            "- 因此直接 3D 的 10cm 比例高于可见性深度指标是合理的；两者不能互相替代。",
            "- 不报告 GLB→SfM precision，因为稀疏 SfM 点不是均匀表面真值；额外/幻觉表面应使用独立 dense MVS 或激光真值判断。",
            "",
            "## 文件",
            "",
            "- `sfm_glb_consistency.png`：顶视误差图、CDF 和逐 chunk 统计。",
            "- `cleaned_points_distance.ply`、`quality_roi_points_distance.ply`：按距离着色的 SfM 点云，0.30m 截断色标。",
            "- `cleaned_points.csv`、`quality_points.csv`：每点最近面坐标、误差、最近 chunk 和 ROI 标志。",
            "- `per_chunk.csv`、`summary.json`：逐 chunk 和完整机器可读结果。",
            "",
        ]
    )
    if (output / "rgb_overlays" / "README.md").is_file():
        lines.insert(
            -1,
            "- `rgb_overlays/README.md`：将直接3D误差和可见深度误差投影回原始 RGB 视角。",
        )
    (output / "README.md").write_text("\n".join(lines))


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    clean_points = load_point_cloud(args.clean_points)
    colmap_points = read_colmap_points(args.points3d)
    quality_items = [
        (point_id, point)
        for point_id, point in colmap_points.items()
        if point.reprojection_error <= args.max_reprojection_error
        and point.track_length >= args.min_track_length
    ]
    if not quality_items:
        raise ValueError("COLMAP quality filters removed every point")
    quality_ids = np.asarray([item[0] for item in quality_items], dtype=np.int64)
    quality_points = np.asarray([item[1].xyz for item in quality_items], dtype=np.float32)
    clean_ids, clean_id_distances = match_clean_points_to_colmap_ids(
        clean_points, quality_points, quality_ids
    )

    query_points = np.concatenate((clean_points, quality_points), axis=0)
    nearest = exact_nearest_glb_surface(
        query_points,
        sorted(args.chunks_dir.glob(args.chunk_glob)),
        query_batch_size=args.query_batch_size,
    )
    clean_slice = slice(0, len(clean_points))
    quality_slice = slice(len(clean_points), len(query_points))
    bounds = nearest["bounds"]
    clean_roi = points_inside_bounds(clean_points, bounds, args.roi_margin)
    quality_roi = points_inside_bounds(quality_points, bounds, args.roi_margin)

    per_chunk = []
    clean_chunks = nearest["chunk_index"][clean_slice]
    clean_distances = nearest["distance"][clean_slice]
    for chunk_index in sorted(set(clean_chunks[clean_roi].tolist())):
        mask = (clean_chunks == chunk_index) & clean_roi
        per_chunk.append({"chunk": int(chunk_index), "metrics": distance_summary(clean_distances[mask])})

    result = {
        "elapsed_s": time.perf_counter() - started,
        "glb": {
            "asset_to_colmap_world": genrecon_glb_to_world_matrix(),
            "bounds_colmap_world_m": bounds,
            "chunk_count": len(list(args.chunks_dir.glob(args.chunk_glob))),
            "chunks_dir": args.chunks_dir,
        },
        "inputs": {
            "clean_points": args.clean_points,
            "colmap_points3D": args.points3d,
            "max_reprojection_error_px": args.max_reprojection_error,
            "min_track_length": args.min_track_length,
        },
        "method": "exact-sfm-point-to-glb-triangle-surface",
        "scene_label": args.scene_label,
        "notes": {
            "primary_direction": "SfM -> GLB",
            "reverse_direction_omitted": "Sparse SfM is not an area-uniform surface reference.",
            "visibility": "Nearest surface is not visibility-aware; use the sibling camera-view report for z-buffer consistency.",
        },
        "per_chunk_cleaned_inside_glb_aabb": per_chunk,
        "schema": "genrecon.sfm-to-glb-consistency",
        "schema_version": 1,
        "sources": {
            "cleaned": _source_result(
                clean_points, nearest, clean_slice, bounds, args.roi_margin
            ),
            "quality_filtered": _source_result(
                quality_points, nearest, quality_slice, bounds, args.roi_margin
            ),
        },
    }
    result["sources"]["cleaned"]["colmap_id_match"] = {
        "matched": int(np.sum(clean_ids >= 0)),
        "max_nearest_quality_point_distance_m": float(np.max(clean_id_distances)),
        "tolerance_m": 1e-4,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    clean_nearest = nearest["closest_point"][clean_slice]
    quality_nearest = nearest["closest_point"][quality_slice]
    quality_distances = nearest["distance"][quality_slice]
    _write_point_csv(
        args.output / "cleaned_points.csv",
        point_ids=clean_ids,
        points=clean_points,
        nearest_points=clean_nearest,
        distances=clean_distances,
        chunks=clean_chunks,
        roi_mask=clean_roi,
    )
    _write_point_csv(
        args.output / "quality_points.csv",
        point_ids=quality_ids,
        points=quality_points,
        nearest_points=quality_nearest,
        distances=quality_distances,
        chunks=nearest["chunk_index"][quality_slice],
        roi_mask=quality_roi,
    )
    _write_colored_cloud(args.output / "cleaned_points_distance.ply", clean_points, clean_distances)
    _write_colored_cloud(
        args.output / "quality_roi_points_distance.ply",
        quality_points[quality_roi],
        quality_distances[quality_roi],
    )
    _write_chunk_csv(args.output / "per_chunk.csv", per_chunk)
    _write_plots(
        args.output,
        clean_points,
        clean_distances,
        quality_points[quality_roi],
        quality_distances[quality_roi],
        per_chunk,
    )
    _write_json(args.output / "summary.json", result)
    _write_report(args.output, result, per_chunk, args.scene_label)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-points", type=Path, required=True)
    parser.add_argument("--points3d", type=Path, required=True)
    parser.add_argument("--chunks-dir", type=Path, required=True)
    parser.add_argument("--chunk-glob", default="chunk_*.glb")
    parser.add_argument("--max-reprojection-error", type=float, default=2.0)
    parser.add_argument("--min-track-length", type=int, default=3)
    parser.add_argument("--roi-margin", type=float, default=0.05)
    parser.add_argument("--query-batch-size", type=int, default=100_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scene-label",
        help="Display label for reports; defaults to the output directory's grandparent name",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.scene_label = args.scene_label or args.output.parent.parent.name
    for path in (args.clean_points, args.points3d, args.chunks_dir):
        if not path.exists():
            raise SystemExit(f"Required input does not exist: {path}")
    if args.max_reprojection_error < 0:
        raise SystemExit("--max-reprojection-error must be non-negative")
    if args.min_track_length < 1:
        raise SystemExit("--min-track-length must be at least one")
    if args.query_batch_size < 1:
        raise SystemExit("--query-batch-size must be positive")
    result = evaluate(args)
    clean = result["sources"]["cleaned"]["all"]
    print(
        json.dumps(
            {
                "cleaned_points": clean["count"],
                "median_m": clean["median_m"],
                "p90_m": clean["p90_m"],
                "within_0.10m": clean["threshold_recall"]["0.100"],
                "elapsed_s": result["elapsed_s"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
