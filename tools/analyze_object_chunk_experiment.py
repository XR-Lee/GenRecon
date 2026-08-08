#!/usr/bin/env python3
"""Compare object-chunk variants using fixed COLMAP observations and PLY depth renders."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    from tools.evaluate_view_fidelity import read_colmap_cameras, read_colmap_images, read_colmap_points
except ModuleNotFoundError:
    from evaluate_view_fidelity import read_colmap_cameras, read_colmap_images, read_colmap_points


DEFAULT_REGIONS = {
    "worktop": [-2.0, 2.25, -2.15, -1.0, -0.91, -0.84],
    "worktop_core": [-0.70, 0.70, -2.15, -1.0, -0.91, -0.84],
    "cabinet_front": [-2.0, 2.25, -1.45, -1.05, -1.75, -0.92],
}


def parse_variants(values: list[str]) -> list[tuple[str, Path]]:
    result = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"variant must be LABEL=PATH, got {value!r}")
        label, path = value.split("=", 1)
        if not label or not path:
            raise ValueError(f"variant must be LABEL=PATH, got {value!r}")
        result.append((label, Path(path)))
    return result


def inside_box(point: np.ndarray, box: list[float]) -> bool:
    x0, x1, y0, y1, z0, z1 = box
    return x0 <= point[0] <= x1 and y0 <= point[1] <= y1 and z0 <= point[2] <= z1


def summarize_errors(errors: list[float], eligible: int, hits: int) -> dict[str, float | int | None]:
    values = np.asarray(errors, dtype=np.float64)
    return {
        "eligible": eligible,
        "hits": hits,
        "coverage": hits / eligible if eligible else 0.0,
        "median_m": float(np.median(values)) if len(values) else None,
        "p90_m": float(np.percentile(values, 90)) if len(values) else None,
        "within_2cm": float(np.mean(values <= 0.02)) if len(values) else None,
        "within_10cm": float(np.mean(values <= 0.10)) if len(values) else None,
    }


def ply_vertex_count(path: Path) -> int:
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"PLY header is truncated: {path}")
            text = line.decode("ascii").strip()
            if text.startswith("element vertex "):
                count = int(text.split()[2])
            if text == "end_header":
                return count


def latent_summary(path: Path) -> dict[str, dict[str, float]]:
    if not path.is_file():
        return {}
    records = json.loads(path.read_text())["records"]
    result = {}
    for stage in ("sparse_structure", "shape_slat", "texture_slat"):
        subset = [
            record
            for record in records
            if record["stage"] == stage and "roi_velocity_disagreement_rmse" in record
        ]
        if not subset:
            continue
        steps = sorted({int(record["step"]) for record in subset})
        curve = [
            float(np.mean([record["roi_velocity_disagreement_rmse"] for record in subset if record["step"] == step]))
            for step in steps
        ]
        result[stage] = {
            "auc_mean": float(np.mean(curve)),
            "final": curve[-1],
            "peak": float(np.max(curve)),
            "curve": curve,
        }
    return result


def plane_metrics(view_dir: Path, worktop_box: list[float]) -> dict[str, float | int | None]:
    depth = cv2.imread(str(view_dir / "ply_depth_mm.png"), cv2.IMREAD_UNCHANGED)
    if depth is None:
        return {"samples": 0, "fit_p90_m": None, "horizontal_p90_m": None, "tilt_deg": None}
    camera = json.loads((view_dir / "camera.json").read_text())
    intrinsic = np.asarray(camera["intrinsic_render"], dtype=np.float64)
    world_to_camera = np.asarray(camera["world_to_camera"], dtype=np.float64)
    camera_to_world = np.linalg.inv(world_to_camera)
    z = depth.astype(np.float64) / 1000.0
    ys, xs = np.nonzero(z > 0)
    if not len(xs):
        return {"samples": 0, "fit_p90_m": None, "horizontal_p90_m": None, "tilt_deg": None}
    sampled_z = z[ys, xs]
    pixels = np.stack([xs, ys, np.ones_like(xs)], axis=1).astype(np.float64)
    rays = pixels @ np.linalg.inv(intrinsic).T
    camera_points = rays * sampled_z[:, None]
    world = np.concatenate([camera_points, np.ones((len(camera_points), 1))], axis=1) @ camera_to_world.T
    xyz = world[:, :3]
    x0, x1, y0, y1, z0, z1 = worktop_box
    keep = (
        (xyz[:, 0] >= x0)
        & (xyz[:, 0] <= x1)
        & (xyz[:, 1] >= y0)
        & (xyz[:, 1] <= y1)
        & (xyz[:, 2] >= z0 - 0.03)
        & (xyz[:, 2] <= z1 + 0.03)
    )
    xyz = xyz[keep]
    if len(xyz) < 3:
        return {"samples": len(xyz), "fit_p90_m": None, "horizontal_p90_m": None, "tilt_deg": None}
    active = np.ones(len(xyz), dtype=bool)
    for _ in range(3):
        selected = xyz[active]
        center = np.median(selected, axis=0)
        _, _, vh = np.linalg.svd(selected - center, full_matrices=False)
        normal = vh[-1]
        if normal[2] < 0:
            normal = -normal
        residual = np.abs((xyz - center) @ normal)
        threshold = max(0.002, float(np.percentile(residual[active], 90)))
        active = residual <= threshold
    selected = xyz[active]
    center = np.median(selected, axis=0)
    _, _, vh = np.linalg.svd(selected - center, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    fit = np.abs((selected - center) @ normal)
    horizontal = np.abs(selected[:, 2] - np.median(selected[:, 2]))
    tilt = float(np.degrees(np.arccos(np.clip(normal[2] / np.linalg.norm(normal), -1, 1))))
    return {
        "samples": int(len(selected)),
        "fit_p90_m": float(np.percentile(fit, 90)),
        "horizontal_p90_m": float(np.percentile(horizontal, 90)),
        "tilt_deg": tilt,
        "median_z_m": float(np.median(selected[:, 2])),
    }


def analyze_variant(
    root: Path,
    cameras,
    images,
    points,
    anchor_holdout: set[int],
    regions: dict[str, list[float]],
    plane_image: str,
) -> dict[str, Any]:
    fidelity = root / "fidelity"
    input_document = json.loads((root / "cameras.json").read_text())
    input_names = {Path(view["img_path"]).name for view in input_document["scene"]}
    quality_ids = {
        point_id
        for point_id, point in points.items()
        if point.reprojection_error <= 2.0 and point.track_length >= 3
    }
    region_ids = {
        name: {point_id for point_id in quality_ids if inside_box(points[point_id].xyz, box)}
        for name, box in regions.items()
    }
    region_ids["anchor_bank_holdout"] = quality_ids & anchor_holdout
    accum = {
        name: {group: {"eligible": 0, "hits": 0, "errors": []} for group in ("all", "input", "heldout")}
        for name in region_ids
    }
    view_global = {group: [] for group in ("all", "input", "heldout")}

    for image_name, image in images.items():
        group = "input" if image_name in input_names else "heldout"
        view_dir = fidelity / "views" / group / Path(image_name).stem
        depth = cv2.imread(str(view_dir / "ply_depth_mm.png"), cv2.IMREAD_UNCHANGED)
        camera_json = json.loads((view_dir / "camera.json").read_text())
        if depth is None:
            raise FileNotFoundError(view_dir / "ply_depth_mm.png")
        render_h, render_w = depth.shape
        camera = cameras[image.camera_id]
        sx, sy = render_w / camera.width, render_h / camera.height
        per_view_errors = []
        per_view_eligible = 0
        per_view_hits = 0
        for x, y, point_id_float in image.observations:
            point_id = int(point_id_float)
            if point_id not in quality_ids:
                continue
            point_camera = image.world_to_camera @ np.append(points[point_id].xyz, 1.0)
            if point_camera[2] <= 0:
                continue
            px = int(np.clip(round(x * sx), 0, render_w - 1))
            py = int(np.clip(round(y * sy), 0, render_h - 1))
            rendered = float(depth[py, px]) / 1000.0
            error = abs(rendered - float(point_camera[2])) if rendered > 0 else None
            per_view_eligible += 1
            if error is not None:
                per_view_hits += 1
                per_view_errors.append(error)
            for region_name, ids in region_ids.items():
                if point_id not in ids:
                    continue
                for target_group in ("all", group):
                    accum[region_name][target_group]["eligible"] += 1
                    if error is not None:
                        accum[region_name][target_group]["hits"] += 1
                        accum[region_name][target_group]["errors"].append(error)
        if per_view_eligible:
            record = {
                "image": image_name,
                "coverage": per_view_hits / per_view_eligible,
                "median_m": float(np.median(per_view_errors)) if per_view_errors else None,
                "within_10cm": float(np.mean(np.asarray(per_view_errors) <= 0.1)) if per_view_errors else None,
            }
            view_global["all"].append(record)
            view_global[group].append(record)

    region_summary = {
        name: {
            group: summarize_errors(values["errors"], values["eligible"], values["hits"])
            for group, values in groups.items()
        }
        for name, groups in accum.items()
    }
    global_summary = {}
    for group, records in view_global.items():
        medians = [record["median_m"] for record in records if record["median_m"] is not None]
        within = [record["within_10cm"] for record in records if record["within_10cm"] is not None]
        global_summary[group] = {
            "views": len(records),
            "cross_view_median_m": float(np.median(medians)) if medians else None,
            "within_10cm_view_mean": float(np.mean(within)) if within else None,
            "sfm_depth_coverage_view_mean": float(np.mean([record["coverage"] for record in records])) if records else None,
        }
    plane_group = "input" if plane_image in input_names else "heldout"
    return {
        "folder": str(root),
        "mesh_vertices": ply_vertex_count(root / "mesh.ply"),
        "latent": latent_summary(root / "overlap_diagnostics.json"),
        "global_sfm": global_summary,
        "regions": region_summary,
        "plane": plane_metrics(fidelity / "views" / plane_group / Path(plane_image).stem, regions["worktop"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", required=True, help="LABEL=RUN_DIRECTORY")
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--anchors-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plane-image", default="DSC_0903.JPG")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    variants = parse_variants(args.variant)
    cameras = read_colmap_cameras(args.cameras)
    images = read_colmap_images(args.images)
    points = read_colmap_points(args.points)
    anchors = json.loads(args.anchors_json.read_text())
    holdout = set(int(value) for value in anchors["holdout_point_ids"])
    results = {
        label: analyze_variant(
            path, cameras, images, points, holdout, DEFAULT_REGIONS, args.plane_image
        )
        for label, path in variants
    }
    document = {
        "schema": "genrecon.independent-object-chunk-analysis",
        "regions_world": DEFAULT_REGIONS,
        "plane_image": args.plane_image,
        "variants": results,
    }
    (args.output / "summary.json").write_text(json.dumps(document, indent=2) + "\n")

    rows = []
    for label, result in results.items():
        heldout = result["regions"]["worktop_core"]["heldout"]
        rows.append(
            {
                "variant": label,
                "sparse_auc": result["latent"].get("sparse_structure", {}).get("auc_mean"),
                "shape_auc": result["latent"].get("shape_slat", {}).get("auc_mean"),
                "texture_auc": result["latent"].get("texture_slat", {}).get("auc_mean"),
                "heldout_core_median_cm": heldout["median_m"] * 100 if heldout["median_m"] is not None else None,
                "heldout_core_p90_cm": heldout["p90_m"] * 100 if heldout["p90_m"] is not None else None,
                "heldout_core_within10": heldout["within_10cm"],
                "plane_fit_p90_mm": result["plane"]["fit_p90_m"] * 1000 if result["plane"]["fit_p90_m"] is not None else None,
                "global_heldout_median_cm": result["global_sfm"]["heldout"]["cross_view_median_m"] * 100,
                "global_heldout_depth_coverage": result["global_sfm"]["heldout"]["sfm_depth_coverage_view_mean"],
                "mesh_vertices": result["mesh_vertices"],
            }
        )
    with (args.output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = [row["variant"] for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for stage, color in zip(("sparse_auc", "shape_auc", "texture_auc"), ("#4c78a8", "#e45756", "#72b7b2")):
        axes[0, 0].plot(x, [row[stage] for row in rows], marker="o", label=stage.replace("_auc", ""), color=color)
    axes[0, 0].set_title("Pre-aggregation ROI disagreement AUC")
    axes[0, 0].legend()
    axes[0, 1].bar(x - 0.18, [row["heldout_core_median_cm"] for row in rows], 0.36, label="median")
    axes[0, 1].bar(x + 0.18, [row["heldout_core_p90_cm"] for row in rows], 0.36, label="P90")
    axes[0, 1].set_ylabel("cm")
    axes[0, 1].set_title("Heldout worktop-core depth error")
    axes[0, 1].legend()
    axes[1, 0].bar(x, [row["heldout_core_within10"] * 100 for row in rows], color="#54a24b")
    axes[1, 0].set_ylabel("%")
    axes[1, 0].set_title("Heldout worktop-core within 10cm")
    axes[1, 1].bar(x, [row["plane_fit_p90_mm"] for row in rows], color="#f58518")
    axes[1, 1].set_ylabel("mm")
    axes[1, 1].set_title("DSC_0903 tabletop plane fit P90")
    for axis in axes.flat:
        axis.set_xticks(x, labels, rotation=15)
        axis.grid(axis="y", alpha=0.2)
    fig.savefig(args.output / "object_chunk_ablation.png", dpi=180)
    plt.close(fig)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
