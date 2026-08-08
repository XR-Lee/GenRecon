#!/usr/bin/env python3
"""Evaluate object-aware chunk placement without running the generative model.

The test enumerates lattice-aligned grid phases that cover a labelled instance
point cluster, compares the current layout with the best robust-margin layout,
and bootstraps the instance points to measure placement stability.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

try:
    from tools.evaluate_view_fidelity import read_colmap_points
except ModuleNotFoundError:  # Direct execution puts tools/ on sys.path.
    from evaluate_view_fidelity import read_colmap_points


@dataclass(frozen=True)
class LayoutMetrics:
    coverage: float
    margin_10cm: float
    margin_20cm: float
    margin_30cm: float
    margin_q10_m: float
    margin_median_m: float
    margin_mean_m: float


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def best_face_margins(points: np.ndarray, centers: np.ndarray, chunk_size: float) -> np.ndarray:
    """Return each point's largest distance to the nearest face of any covering chunk.

    Negative values mean the point is outside every chunk. A point near one
    chunk boundary can still have a large margin when it is interior to another
    overlapping chunk.
    """
    points = np.asarray(points, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3], got {points.shape}")
    if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) == 0:
        raise ValueError(f"centers must have non-empty shape [K, 3], got {centers.shape}")
    half = 0.5 * float(chunk_size)
    per_axis = half - np.abs(points[:, None, :] - centers[None, :, :])
    per_chunk = per_axis.min(axis=2)
    return per_chunk.max(axis=1)


def layout_metrics(points: np.ndarray, centers: np.ndarray, chunk_size: float) -> LayoutMetrics:
    margins = best_face_margins(points, centers, chunk_size)
    return LayoutMetrics(
        coverage=float(np.mean(margins >= 0.0)),
        margin_10cm=float(np.mean(margins >= 0.10)),
        margin_20cm=float(np.mean(margins >= 0.20)),
        margin_30cm=float(np.mean(margins >= 0.30)),
        margin_q10_m=float(np.quantile(margins, 0.10)),
        margin_median_m=float(np.median(margins)),
        margin_mean_m=float(np.mean(margins)),
    )


def aligned_axis_values(lower: float, upper: float, origin: float, quantum: float) -> np.ndarray:
    """Enumerate values in [lower, upper] aligned to origin + integer * quantum."""
    if lower > upper:
        raise ValueError(f"empty feasible interval [{lower}, {upper}]")
    first = math.ceil((lower - origin) / quantum - 1e-9)
    last = math.floor((upper - origin) / quantum + 1e-9)
    if first > last:
        raise ValueError("feasible interval contains no lattice-aligned coordinate")
    return origin + np.arange(first, last + 1, dtype=np.float64) * quantum


def make_chain(first_x: float, center_y: float, center_z: float, count: int, stride: float) -> np.ndarray:
    return np.asarray(
        [[first_x + index * stride, center_y, center_z] for index in range(count)],
        dtype=np.float64,
    )


def objective_key(metrics: LayoutMetrics) -> tuple[float, float, float]:
    """Prioritize lower-tail robustness, then 30 cm interior coverage and mean margin."""
    return metrics.margin_q10_m, metrics.margin_30cm, metrics.margin_mean_m


def enumerate_layouts(
    points: np.ndarray,
    *,
    chunk_size: float,
    count: int,
    stride: float,
    center_z: float,
    x_origin: float,
    y_origin: float,
    quantum: float,
) -> list[dict[str, Any]]:
    half = 0.5 * chunk_size
    x_lower = float(points[:, 0].max() - (count - 1) * stride - half)
    x_upper = float(points[:, 0].min() + half)
    y_lower = float(points[:, 1].max() - half)
    y_upper = float(points[:, 1].min() + half)
    x_values = aligned_axis_values(x_lower, x_upper, x_origin, quantum)
    y_values = aligned_axis_values(y_lower, y_upper, y_origin, quantum)

    candidates: list[dict[str, Any]] = []
    for first_x in x_values:
        for center_y in y_values:
            centers = make_chain(first_x, center_y, center_z, count, stride)
            metrics = layout_metrics(points, centers, chunk_size)
            candidates.append(
                {
                    "first_x": float(first_x),
                    "center_y": float(center_y),
                    "centers": centers,
                    "metrics": metrics,
                }
            )
    return candidates


def select_adaptive(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("at least one candidate layout is required")
    return max(candidates, key=lambda candidate: objective_key(candidate["metrics"]))


def _select_points(
    points_path: Path,
    roi_boxes: list[list[float]],
    max_reprojection_error: float,
    min_track_length: int,
) -> tuple[np.ndarray, dict[str, int]]:
    points = read_colmap_points(points_path)
    quality = [
        point.xyz
        for point in points.values()
        if point.reprojection_error <= max_reprojection_error and point.track_length >= min_track_length
    ]
    quality_xyz = np.asarray(quality, dtype=np.float64)
    selected = np.zeros(len(quality_xyz), dtype=bool)
    per_box: dict[str, int] = {}
    for index, box in enumerate(roi_boxes):
        x0, x1, y0, y1, z0, z1 = box
        inside = (
            (quality_xyz[:, 0] >= x0)
            & (quality_xyz[:, 0] <= x1)
            & (quality_xyz[:, 1] >= y0)
            & (quality_xyz[:, 1] <= y1)
            & (quality_xyz[:, 2] >= z0)
            & (quality_xyz[:, 2] <= z1)
        )
        per_box[f"roi_{index}"] = int(inside.sum())
        selected |= inside
    return quality_xyz[selected], {"quality_points": len(quality_xyz), **per_box, "union": int(selected.sum())}


def _load_current_layout(path: Path, chunk_ids: list[int]) -> tuple[np.ndarray, float]:
    document = json.loads(path.read_text())
    by_id = {int(chunk["index"]): chunk for chunk in document["chunks"]}
    missing = sorted(set(chunk_ids) - by_id.keys())
    if missing:
        raise ValueError(f"chunk ids missing from transforms: {missing}")
    centers = np.asarray([by_id[index]["crop_center"] for index in chunk_ids], dtype=np.float64)
    centers = centers[np.argsort(centers[:, 0])]
    sizes = [float(by_id[index]["M_chunk_to_original"][0][0]) for index in chunk_ids]
    if not np.allclose(sizes, sizes[0]):
        raise ValueError("selected chunks do not have a common chunk size")
    if not np.allclose(centers[:, 1], centers[0, 1]) or not np.allclose(centers[:, 2], centers[0, 2]):
        raise ValueError("selected chunks must form one x-axis chain")
    return centers, sizes[0]


def _metric_percentiles(candidates: Sequence[dict[str, Any]], current: LayoutMetrics) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in asdict(current):
        values = np.asarray([getattr(candidate["metrics"], name) for candidate in candidates])
        result[name] = float(np.mean(values <= getattr(current, name)))
    return result


def _bootstrap(
    points: np.ndarray,
    candidates: Sequence[dict[str, Any]],
    chunk_size: float,
    samples: int,
    fraction: float,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    subset_size = max(2, int(round(len(points) * fraction)))
    records: list[dict[str, Any]] = []
    for sample_index in range(samples):
        indices = rng.choice(len(points), subset_size, replace=False)
        best: dict[str, Any] | None = None
        best_metrics: LayoutMetrics | None = None
        for candidate in candidates:
            metrics = layout_metrics(points[indices], candidate["centers"], chunk_size)
            if best is None or objective_key(metrics) > objective_key(best_metrics):  # type: ignore[arg-type]
                best = candidate
                best_metrics = metrics
        assert best is not None and best_metrics is not None
        full_metrics = layout_metrics(points, best["centers"], chunk_size)
        records.append(
            {
                "sample": sample_index,
                "first_x": best["first_x"],
                "center_y": best["center_y"],
                "subset_metrics": asdict(best_metrics),
                "full_metrics": asdict(full_metrics),
            }
        )
    return records


def _candidate_rows(label: str, candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        rows.append(
            {
                "family": label,
                "candidate": index,
                "first_x": candidate["first_x"],
                "center_y": candidate["center_y"],
                **asdict(candidate["metrics"]),
            }
        )
    return rows


def _plot_layout(ax, points: np.ndarray, centers: np.ndarray, chunk_size: float, title: str) -> None:
    ax.scatter(points[:, 0], points[:, 1], s=4, c=points[:, 2], cmap="viridis", alpha=0.65)
    half = 0.5 * chunk_size
    for index, center in enumerate(centers):
        ax.add_patch(
            patches.Rectangle(
                (center[0] - half, center[1] - half),
                chunk_size,
                chunk_size,
                facecolor="none",
                edgecolor=f"C{index}",
                linewidth=2,
            )
        )
        ax.text(center[0], center[1], str(index), ha="center", va="center", fontsize=9)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")


def _write_plot(
    output: Path,
    points: np.ndarray,
    chunk_size: float,
    current_centers: np.ndarray,
    adaptive3: dict[str, Any],
    adaptive4: dict[str, Any],
    candidates3: Sequence[dict[str, Any]],
    bootstrap: Sequence[dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    _plot_layout(axes[0, 0], points, current_centers, chunk_size, "Current deterministic layout")
    _plot_layout(axes[0, 1], points, adaptive3["centers"], chunk_size, "Adaptive 3 chunks, same compute")
    _plot_layout(axes[0, 2], points, adaptive4["centers"], chunk_size, "Adaptive 4 chunks, 50% overlap")

    q10 = np.asarray([candidate["metrics"].margin_q10_m for candidate in candidates3])
    mean = np.asarray([candidate["metrics"].margin_mean_m for candidate in candidates3])
    current = layout_metrics(points, current_centers, chunk_size)
    axes[1, 0].hist(q10 * 100, bins=min(24, max(8, len(np.unique(q10)))), color="#4c78a8", alpha=0.8)
    axes[1, 0].axvline(current.margin_q10_m * 100, color="#d62728", label="current")
    axes[1, 0].axvline(adaptive3["metrics"].margin_q10_m * 100, color="#2ca02c", label="adaptive")
    axes[1, 0].set_title("All feasible lattice phases")
    axes[1, 0].set_xlabel("10th-percentile best face margin (cm)")
    axes[1, 0].legend()

    axes[1, 1].hist(mean * 100, bins=min(24, max(8, len(np.unique(mean)))), color="#f58518", alpha=0.8)
    axes[1, 1].axvline(current.margin_mean_m * 100, color="#d62728", label="current")
    axes[1, 1].axvline(adaptive3["metrics"].margin_mean_m * 100, color="#2ca02c", label="adaptive")
    axes[1, 1].set_title("All feasible lattice phases")
    axes[1, 1].set_xlabel("Mean best face margin (cm)")
    axes[1, 1].legend()

    boot_x = np.asarray([record["first_x"] for record in bootstrap])
    boot_y = np.asarray([record["center_y"] for record in bootstrap])
    axes[1, 2].scatter(boot_x, boot_y, s=18, alpha=0.55, color="#54a24b")
    axes[1, 2].scatter(
        [adaptive3["first_x"]], [adaptive3["center_y"]], marker="*", s=180, color="black", label="full-point optimum"
    )
    axes[1, 2].set_title("Adaptive layout under point subsampling")
    axes[1, 2].set_xlabel("first center x (m)")
    axes[1, 2].set_ylabel("chain center y (m)")
    axes[1, 2].legend()
    fig.suptitle("Instance-aware chunk coordinate pretest", fontsize=17)
    fig.savefig(output / "layout_stability.png", dpi=170)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, required=True, help="COLMAP points3D.txt")
    parser.add_argument("--chunk-transforms", type=Path, required=True)
    parser.add_argument("--chunk-ids", type=int, nargs="+", required=True)
    parser.add_argument(
        "--roi-box",
        type=float,
        nargs=6,
        action="append",
        required=True,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-reprojection-error", type=float, default=2.0)
    parser.add_argument("--min-track-length", type=int, default=3)
    parser.add_argument("--atomic-divisions", type=int, default=16)
    parser.add_argument("--adaptive-overlap", type=float, default=0.5)
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    parser.add_argument("--bootstrap-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.adaptive_overlap < 1.0:
        raise ValueError("--adaptive-overlap must be between 0 and 1")
    if not 0.0 < args.bootstrap_fraction <= 1.0:
        raise ValueError("--bootstrap-fraction must be in (0, 1]")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    points, point_counts = _select_points(
        args.points,
        args.roi_box,
        args.max_reprojection_error,
        args.min_track_length,
    )
    if len(points) < 10:
        raise ValueError(f"instance ROI contains too few quality points: {len(points)}")
    current_centers, chunk_size = _load_current_layout(args.chunk_transforms, args.chunk_ids)
    current_metrics = layout_metrics(points, current_centers, chunk_size)
    current_stride = float(np.median(np.diff(current_centers[:, 0])))
    quantum = chunk_size / args.atomic_divisions

    candidates3 = enumerate_layouts(
        points,
        chunk_size=chunk_size,
        count=len(current_centers),
        stride=current_stride,
        center_z=float(current_centers[0, 2]),
        x_origin=float(current_centers[0, 0]),
        y_origin=float(current_centers[0, 1]),
        quantum=quantum,
    )
    adaptive3 = select_adaptive(candidates3)

    adaptive_count = len(current_centers) + 1
    adaptive_stride = chunk_size * (1.0 - args.adaptive_overlap)
    # Quantize the requested stride so overlap coordinates align at every model lattice.
    adaptive_stride = round(adaptive_stride / quantum) * quantum
    candidates4 = enumerate_layouts(
        points,
        chunk_size=chunk_size,
        count=adaptive_count,
        stride=adaptive_stride,
        center_z=float(current_centers[0, 2]),
        x_origin=float(current_centers[0, 0]),
        y_origin=float(current_centers[0, 1]),
        quantum=quantum,
    )
    adaptive4 = select_adaptive(candidates4)

    bootstrap = _bootstrap(
        points,
        candidates3,
        chunk_size,
        args.bootstrap_samples,
        args.bootstrap_fraction,
        args.seed,
    )
    candidate_rows = _candidate_rows("same_compute_3", candidates3) + _candidate_rows("higher_overlap_4", candidates4)
    with (output / "candidate_layouts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)
    flat_bootstrap = []
    for record in bootstrap:
        flat_bootstrap.append(
            {
                "sample": record["sample"],
                "first_x": record["first_x"],
                "center_y": record["center_y"],
                **{f"subset_{key}": value for key, value in record["subset_metrics"].items()},
                **{f"full_{key}": value for key, value in record["full_metrics"].items()},
            }
        )
    with (output / "bootstrap_adaptive.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_bootstrap[0]))
        writer.writeheader()
        writer.writerows(flat_bootstrap)

    boot_x = np.asarray([record["first_x"] for record in bootstrap])
    boot_y = np.asarray([record["center_y"] for record in bootstrap])
    summary = {
        "schema": "genrecon.instance-chunk-layout-pretest",
        "scope": "geometric layout only; no denoiser inference",
        "points": point_counts,
        "roi_boxes": args.roi_box,
        "chunk_size_m": chunk_size,
        "lattice_quantum_m": quantum,
        "current": {"centers": current_centers, "metrics": asdict(current_metrics)},
        "same_compute": {
            "candidate_count": len(candidates3),
            "stride_m": current_stride,
            "adaptive": {"centers": adaptive3["centers"], "metrics": asdict(adaptive3["metrics"])},
            "current_percentile_among_feasible": _metric_percentiles(candidates3, current_metrics),
        },
        "higher_overlap": {
            "candidate_count": len(candidates4),
            "chunk_count": adaptive_count,
            "stride_m": adaptive_stride,
            "overlap_fraction": 1.0 - adaptive_stride / chunk_size,
            "adaptive": {"centers": adaptive4["centers"], "metrics": asdict(adaptive4["metrics"])},
        },
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "point_fraction": args.bootstrap_fraction,
            "first_x_mean_m": float(boot_x.mean()),
            "first_x_std_m": float(boot_x.std()),
            "center_y_mean_m": float(boot_y.mean()),
            "center_y_std_m": float(boot_y.std()),
            "unique_layouts": int(len(set(zip(boot_x.tolist(), boot_y.tolist())))),
        },
    }
    (output / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2) + "\n")
    _write_plot(output, points, chunk_size, current_centers, adaptive3, adaptive4, candidates3, bootstrap)

    current = current_metrics
    improved = adaptive3["metrics"]
    report = f"""# Instance-aware chunk坐标预检

该测试只比较 chunk 几何覆盖和边界裕量，不运行 GenRecon denoiser。当前 chunker 使用确定性规则网格，不是随机坐标初始化；“随机布局”在这里指同尺寸、同数量、同stride且保持体素格对齐的所有可行网格相位。

## 数据

- 高置信 COLMAP 点：{point_counts['quality_points']:,}
- 工作台 ROI 唯一点：{point_counts['union']:,}
- 当前链：{len(current_centers)} chunks，尺寸 {chunk_size:.3f}m，stride {current_stride:.3f}m
- 可行格点相位：{len(candidates3)}

## 同计算量结果

| 指标 | 当前规则布局 | 实例自适应3 chunks |
|---|---:|---:|
| 覆盖率 | {current.coverage:.1%} | {improved.coverage:.1%} |
| 至少10cm interior | {current.margin_10cm:.1%} | {improved.margin_10cm:.1%} |
| 至少20cm interior | {current.margin_20cm:.1%} | {improved.margin_20cm:.1%} |
| 至少30cm interior | {current.margin_30cm:.1%} | {improved.margin_30cm:.1%} |
| 最佳边界裕量 P10 | {current.margin_q10_m*100:.1f}cm | {improved.margin_q10_m*100:.1f}cm |
| 最佳边界裕量中位 | {current.margin_median_m*100:.1f}cm | {improved.margin_median_m*100:.1f}cm |
| 最佳边界裕量均值 | {current.margin_mean_m*100:.1f}cm | {improved.margin_mean_m*100:.1f}cm |

自适应中心：`{np.round(adaptive3['centers'], 4).tolist()}`。

## 高重叠上限

4 chunks、约 {100*(1-adaptive_stride/chunk_size):.0f}% overlap的布局达到 P10 {adaptive4['metrics'].margin_q10_m*100:.1f}cm、平均 {adaptive4['metrics'].margin_mean_m*100:.1f}cm，但增加一个 chunk，不能与3-chunk结果视为同计算量比较。

## Bootstrap

每次随机保留 {args.bootstrap_fraction:.0%} 实例点并重新选布局，共 {args.bootstrap_samples} 次。首个x中心标准差为 {boot_x.std()*100:.2f}cm，y中心标准差为 {boot_y.std()*100:.2f}cm，共选择 {summary['bootstrap']['unique_layouts']} 个量化布局。

## 结论边界

结果说明对象点驱动的坐标选择比当前scene-wide规则相位提供更大的interior margin，并且对稀疏点子采样稳定。它尚不能证明 denoiser disagreement 或最终几何一定改善；下一步必须固定seed、图像条件和模型参数，对当前布局、若干随机相位及自适应布局运行相同的overlap diagnostics。
"""
    (output / "README.md").write_text(report)
    print(f"[layout-pretest] points={len(points):,} candidates={len(candidates3)}")
    print(f"[layout-pretest] current q10={current.margin_q10_m:.4f}m mean={current.margin_mean_m:.4f}m")
    print(f"[layout-pretest] adaptive q10={improved.margin_q10_m:.4f}m mean={improved.margin_mean_m:.4f}m")
    print(f"[layout-pretest] wrote {output}")


if __name__ == "__main__":
    main()
