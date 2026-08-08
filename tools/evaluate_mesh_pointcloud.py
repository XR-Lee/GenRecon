#!/usr/bin/env python3
"""Evaluate a triangle mesh against one or more aligned laser point clouds.

This is an ROI proxy for datasets such as ETH3D whose public scan-evaluation
asset contains points rather than a watertight triangle mesh. It is not the
official ETH3D depth-map evaluation protocol and does not report normal scores.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

DEFAULT_THRESHOLDS_M = (0.02, 0.05, 0.10)


class PointCloudEvaluationError(RuntimeError):
    pass


def load_meshlab_transforms(path: str | Path) -> dict[str, np.ndarray]:
    """Return filename-to-4x4 transforms from a MeshLab project."""
    root = ET.parse(path).getroot()
    transforms: dict[str, np.ndarray] = {}
    for mesh in root.findall(".//MLMesh"):
        filename = Path(mesh.attrib["filename"]).name
        matrix_node = mesh.find("MLMatrix44")
        if matrix_node is None or matrix_node.text is None:
            raise PointCloudEvaluationError(f"Missing MLMatrix44 for {filename} in {path}")
        values = np.fromstring(matrix_node.text, sep=" ", dtype=np.float64)
        if values.size != 16:
            raise PointCloudEvaluationError(
                f"Expected 16 transform values for {filename} in {path}, got {values.size}"
            )
        transforms[filename] = values.reshape(4, 4)
    if not transforms:
        raise PointCloudEvaluationError(f"No mesh transforms found in {path}")
    return transforms


def _load_triangle_mesh(path: str | Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise PointCloudEvaluationError(f"Predicted input is not a triangle mesh: {path}")
    vertices = np.asarray(loaded.vertices)
    faces = np.asarray(loaded.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise PointCloudEvaluationError(f"Predicted mesh has no vertices: {path}")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise PointCloudEvaluationError(f"Predicted mesh has no triangular faces: {path}")
    if not np.isfinite(vertices).all():
        raise PointCloudEvaluationError(f"Predicted mesh contains non-finite vertices: {path}")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise PointCloudEvaluationError(f"Predicted mesh contains invalid face indices: {path}")
    return loaded


def sample_mesh_surface_batched(
    mesh: trimesh.Trimesh,
    num_samples: int,
    rng: np.random.Generator,
    *,
    face_batch_size: int = 500_000,
) -> np.ndarray:
    """Area-sample a large mesh without materializing every triangle at once."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    areas = np.empty(len(faces), dtype=np.float64)
    for start in range(0, len(faces), face_batch_size):
        stop = min(start + face_batch_size, len(faces))
        triangles = vertices[faces[start:stop]]
        cross = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        areas[start:stop] = 0.5 * np.linalg.norm(cross, axis=1)

    positive = np.isfinite(areas) & (areas > 0.0)
    if not np.any(positive):
        raise PointCloudEvaluationError("Predicted mesh has no non-degenerate faces")
    areas[~positive] = 0.0
    cumulative = np.cumsum(areas)
    total_area = float(cumulative[-1])
    if not math.isfinite(total_area) or total_area <= 0.0:
        raise PointCloudEvaluationError(f"Predicted mesh has invalid area {total_area}")

    face_indices = np.searchsorted(cumulative, rng.random(num_samples) * total_area, side="right")
    np.minimum(face_indices, len(faces) - 1, out=face_indices)
    triangles = vertices[faces[face_indices]]
    root_u = np.sqrt(rng.random(num_samples))
    v = rng.random(num_samples)
    return (
        (1.0 - root_u)[:, None] * triangles[:, 0]
        + (root_u * (1.0 - v))[:, None] * triangles[:, 1]
        + (root_u * v)[:, None] * triangles[:, 2]
    )


def load_cropped_clouds(
    paths: list[str | Path],
    transforms: dict[str, np.ndarray],
    bounds: np.ndarray,
    *,
    batch_size: int = 1_000_000,
) -> tuple[np.ndarray, int]:
    """Load, align, and crop point clouds to bounds; return points and source count."""
    cropped_batches: list[np.ndarray] = []
    source_points = 0
    for path_like in paths:
        path = Path(path_like)
        loaded = trimesh.load(str(path), process=False)
        if not isinstance(loaded, (trimesh.PointCloud, trimesh.Trimesh)):
            raise PointCloudEvaluationError(f"Ground truth is not point-like geometry: {path}")
        points = np.asarray(loaded.vertices)
        if points.ndim != 2 or points.shape[1] != 3:
            raise PointCloudEvaluationError(f"Ground-truth cloud has invalid vertices: {path}")
        source_points += len(points)
        transform = transforms.get(path.name, np.eye(4, dtype=np.float64))
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        for start in range(0, len(points), batch_size):
            aligned = points[start : start + batch_size].astype(np.float64) @ rotation.T + translation
            mask = np.all((aligned >= bounds[0]) & (aligned <= bounds[1]), axis=1)
            if np.any(mask):
                cropped_batches.append(aligned[mask].astype(np.float32))
        del loaded

    if not cropped_batches:
        raise PointCloudEvaluationError("No aligned ground-truth points fall inside the evaluation ROI")
    return np.concatenate(cropped_batches, axis=0), source_points


def distance_metrics(
    predicted_points: np.ndarray,
    ground_truth_points: np.ndarray,
    thresholds_m: tuple[float, ...] = DEFAULT_THRESHOLDS_M,
    *,
    workers: int = -1,
) -> dict:
    gt_tree = cKDTree(ground_truth_points)
    pred_to_gt = gt_tree.query(predicted_points, k=1, workers=workers)[0]
    pred_tree = cKDTree(predicted_points)
    gt_to_pred = pred_tree.query(ground_truth_points, k=1, workers=workers)[0]

    scores = {}
    for threshold in thresholds_m:
        precision = float(np.mean(pred_to_gt <= threshold))
        recall = float(np.mean(gt_to_pred <= threshold))
        harmonic = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores[f"{threshold:.3f}"] = {
            "precision": precision,
            "recall": recall,
            "fscore_harmonic": harmonic,
        }

    return {
        "pred_to_gt_m": {
            "mean": float(np.mean(pred_to_gt)),
            "median": float(np.median(pred_to_gt)),
            "p95": float(np.quantile(pred_to_gt, 0.95)),
        },
        "gt_to_pred_m": {
            "mean": float(np.mean(gt_to_pred)),
            "median": float(np.median(gt_to_pred)),
            "p95": float(np.quantile(gt_to_pred, 0.95)),
        },
        "chamfer_symmetric_mean_m": float(0.5 * (np.mean(pred_to_gt) + np.mean(gt_to_pred))),
        "threshold_scores": scores,
    }


def evaluate(
    predicted_mesh: str | Path,
    ground_truth_clouds: list[str | Path],
    *,
    mlp_path: str | Path | None = None,
    num_pred_samples: int = 200_000,
    max_gt_samples: int = 1_000_000,
    crop_margin_m: float = 0.05,
    seed: int = 42,
    workers: int = -1,
) -> dict:
    if num_pred_samples <= 0 or max_gt_samples <= 0:
        raise ValueError("Sample counts must be positive")
    if crop_margin_m < 0 or not math.isfinite(crop_margin_m):
        raise ValueError("crop_margin_m must be finite and non-negative")

    mesh = _load_triangle_mesh(predicted_mesh)
    vertices = np.asarray(mesh.vertices)
    mesh_bounds = np.stack((vertices.min(axis=0), vertices.max(axis=0))).astype(np.float64)
    crop_bounds = mesh_bounds.copy()
    crop_bounds[0] -= crop_margin_m
    crop_bounds[1] += crop_margin_m

    transforms = load_meshlab_transforms(mlp_path) if mlp_path is not None else {}
    gt_points, source_gt_count = load_cropped_clouds(ground_truth_clouds, transforms, crop_bounds)

    pred_seed, gt_seed = np.random.SeedSequence(seed).spawn(2)
    pred_points = sample_mesh_surface_batched(
        mesh,
        num_pred_samples,
        np.random.default_rng(pred_seed),
    )
    cropped_gt_count = len(gt_points)
    if cropped_gt_count > max_gt_samples:
        rng = np.random.default_rng(gt_seed)
        gt_points = gt_points[rng.choice(cropped_gt_count, max_gt_samples, replace=False)]

    metrics = distance_metrics(pred_points, gt_points, workers=workers)
    return {
        "schema": "genrecon.mesh-to-laser-pointcloud-evaluation",
        "schema_version": 1,
        "protocol": "roi-pointcloud-proxy",
        "protocol_note": (
            "Not the official ETH3D depth-map protocol. GT is cropped to the predicted mesh AABB "
            "plus margin; laser sampling density is not area-uniform; normals are unavailable."
        ),
        "inputs": {
            "predicted_mesh": str(predicted_mesh),
            "ground_truth_clouds": [str(path) for path in ground_truth_clouds],
            "meshlab_alignment": str(mlp_path) if mlp_path is not None else None,
        },
        "sampling": {
            "seed": seed,
            "predicted_surface_samples": len(pred_points),
            "ground_truth_source_points": source_gt_count,
            "ground_truth_points_in_roi": cropped_gt_count,
            "ground_truth_samples_used": len(gt_points),
        },
        "roi": {
            "mesh_bounds_m": mesh_bounds.tolist(),
            "crop_margin_m": crop_margin_m,
            "crop_bounds_m": crop_bounds.tolist(),
        },
        "mesh": {
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
        },
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predicted_mesh")
    parser.add_argument("ground_truth_clouds", nargs="+")
    parser.add_argument("--mlp", default=None, help="MeshLab project containing cloud alignment matrices")
    parser.add_argument("--num-pred-samples", type=int, default=200_000)
    parser.add_argument("--max-gt-samples", type=int, default=1_000_000)
    parser.add_argument("--crop-margin-m", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = evaluate(
        args.predicted_mesh,
        args.ground_truth_clouds,
        mlp_path=args.mlp,
        num_pred_samples=args.num_pred_samples,
        max_gt_samples=args.max_gt_samples,
        crop_margin_m=args.crop_margin_m,
        seed=args.seed,
        workers=args.workers,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
