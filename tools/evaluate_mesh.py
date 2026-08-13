#!/usr/bin/env python3
"""Evaluate a reconstructed mesh with the public GenRecon 3D metrics.

This implements the parts of the GenRecon ScanNet++ 3D protocol that are
described in the paper: deterministic area-weighted surface sampling,
bidirectional nearest-neighbour distances, the 10 cm precision/recall scores,
and bidirectional normal consistency with a 20 cm correspondence cutoff.

The paper's scanner-observation-envelope construction has not been released.
Consequently the default protocol is explicitly named
``paper-like-unclipped``.  An optional GT-axis-aligned-bounding-box crop is
provided as a clearly labelled alternative, not as a replacement for that
unpublished envelope.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree


DEFAULT_NUM_SAMPLES = 200_000
DEFAULT_SEED = 42
DEFAULT_THRESHOLDS_M = (0.02, 0.05, 0.10)
DEFAULT_NORMALIZED_THRESHOLDS = (0.005, 0.01, 0.02)
FSCORE_THRESHOLD_M = 0.1
NORMAL_MAX_DISTANCE_M = 0.2
PROTOCOL_UNCLIPPED = "paper-like-unclipped"
PROTOCOL_GT_AABB = "paper-like-gt-aabb-cropped"


class MeshEvaluationError(RuntimeError):
    """Raised when a mesh cannot be evaluated safely."""


def _as_single_mesh(mesh_or_path: trimesh.Trimesh | str | Path, *, label: str) -> trimesh.Trimesh:
    """Load a mesh and bake scene transforms into one triangle mesh."""

    if isinstance(mesh_or_path, trimesh.Trimesh):
        return mesh_or_path

    path = Path(mesh_or_path)
    if not path.is_file():
        raise MeshEvaluationError(f"{label} mesh does not exist or is not a file: {path}")

    try:
        # Do not force a direct PLY/OBJ mesh through a Scene: doing so creates
        # avoidable multi-gigabyte copies for ScanNet++ scanner meshes.
        loaded = trimesh.load(str(path), process=False)
    except Exception as exc:  # trimesh dispatches many format-specific errors
        raise MeshEvaluationError(f"Failed to load {label} mesh {path}: {exc}") from exc

    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    if not isinstance(loaded, trimesh.Scene):
        raise MeshEvaluationError(
            f"{label} input {path} loaded as {type(loaded).__name__}, not a triangle mesh"
        )

    # Scene.dump applies each node's transform before returning its geometry.
    geometries = [
        geometry
        for geometry in loaded.dump(concatenate=False)
        if isinstance(geometry, trimesh.Trimesh)
    ]
    if not geometries:
        raise MeshEvaluationError(f"{label} scene contains no triangle mesh geometry: {path}")
    return trimesh.util.concatenate(geometries)


def _validated_surface(mesh: trimesh.Trimesh, *, label: str) -> dict[str, Any]:
    """Validate a triangle mesh and return its sampling-ready surface arrays."""

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)

    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) == 0:
        raise MeshEvaluationError(f"{label} mesh has no vertices")
    if faces.ndim != 2 or faces.shape[1:] != (3,) or len(faces) == 0:
        raise MeshEvaluationError(f"{label} mesh has no triangular faces")
    if not np.isfinite(vertices).all():
        raise MeshEvaluationError(f"{label} mesh contains non-finite vertex coordinates")
    if not np.issubdtype(faces.dtype, np.integer):
        raise MeshEvaluationError(f"{label} mesh faces are not integer vertex indices")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise MeshEvaluationError(f"{label} mesh contains out-of-range face indices")

    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_areas = np.linalg.norm(cross, axis=1)
    if not np.isfinite(double_areas).all():
        raise MeshEvaluationError(f"{label} mesh contains non-finite triangle areas")

    positive = double_areas > 0.0
    if not np.any(positive):
        raise MeshEvaluationError(f"{label} mesh has no non-degenerate surface area")

    triangles = triangles[positive]
    double_areas = double_areas[positive]
    normals = cross[positive] / double_areas[:, None]
    areas = double_areas * 0.5
    total_area = float(np.sum(areas, dtype=np.float64))
    if not math.isfinite(total_area) or total_area <= 0.0:
        raise MeshEvaluationError(f"{label} mesh has invalid total surface area {total_area}")

    return {
        "triangles": triangles,
        "normals": normals,
        "areas": areas,
        "total_area": total_area,
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "nondegenerate_face_count": int(np.count_nonzero(positive)),
        "bounds": np.stack((np.min(vertices, axis=0), np.max(vertices, axis=0))),
    }


def sample_mesh_surface(
    mesh: trimesh.Trimesh,
    num_samples: int,
    rng: np.random.Generator,
    *,
    label: str = "mesh",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Sample points uniformly by triangle area and return geometric normals."""

    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    surface = _validated_surface(mesh, label=label)
    cumulative_area = np.cumsum(surface["areas"], dtype=np.float64)

    draws = rng.random(num_samples) * surface["total_area"]
    face_indices = np.searchsorted(cumulative_area, draws, side="right")
    np.minimum(face_indices, len(cumulative_area) - 1, out=face_indices)

    triangles = surface["triangles"][face_indices]
    # sqrt(u) gives a uniform distribution over triangle area.
    root_u = np.sqrt(rng.random(num_samples))
    v = rng.random(num_samples)
    points = (
        (1.0 - root_u)[:, None] * triangles[:, 0]
        + (root_u * (1.0 - v))[:, None] * triangles[:, 1]
        + (root_u * v)[:, None] * triangles[:, 2]
    )
    normals = surface["normals"][face_indices]
    return points, normals, surface


def _clip_to_gt_aabb(
    predicted: trimesh.Trimesh,
    ground_truth_bounds: np.ndarray,
    margin_m: float,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    if not math.isfinite(margin_m) or margin_m < 0.0:
        raise ValueError(f"GT AABB margin must be finite and non-negative, got {margin_m}")

    bounds = np.asarray(ground_truth_bounds, dtype=np.float64).copy()
    bounds[0] -= margin_m
    bounds[1] += margin_m
    clipped = predicted.copy()
    axes = np.eye(3, dtype=np.float64)
    for axis in range(3):
        clipped = clipped.slice_plane(bounds[0], axes[axis], cap=False)
        if len(clipped.faces) == 0:
            raise MeshEvaluationError("Predicted mesh is empty after GT AABB cropping")
        clipped = clipped.slice_plane(bounds[1], -axes[axis], cap=False)
        if len(clipped.faces) == 0:
            raise MeshEvaluationError("Predicted mesh is empty after GT AABB cropping")
    return clipped, bounds


def _mesh_summary(surface: dict[str, Any]) -> dict[str, Any]:
    return {
        "vertices": surface["vertex_count"],
        "faces": surface["face_count"],
        "nondegenerate_faces": surface["nondegenerate_face_count"],
        "surface_area_m2": surface["total_area"],
        "bounds_m": np.asarray(surface["bounds"]).tolist(),
    }


def _normal_consistency(
    source_normals: np.ndarray,
    target_normals: np.ndarray,
    target_indices: np.ndarray,
    distances: np.ndarray,
) -> tuple[float, float]:
    values = np.abs(np.einsum("ij,ij->i", source_normals, target_normals[target_indices]))
    values = np.clip(values, 0.0, 1.0)
    within_cutoff = distances <= NORMAL_MAX_DISTANCE_M
    values[~within_cutoff] = 0.0
    return float(np.mean(values)), float(np.mean(within_cutoff))


def _distance_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean_m": float(np.mean(values)),
        "median_m": float(np.median(values)),
        "p90_m": float(np.quantile(values, 0.90)),
        "p95_m": float(np.quantile(values, 0.95)),
    }


def _threshold_scores(
    pred_to_gt_dist: np.ndarray,
    gt_to_pred_dist: np.ndarray,
    thresholds: tuple[float, ...],
) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    for threshold in thresholds:
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError(f"Evaluation thresholds must be positive and finite: {threshold}")
        precision = float(np.mean(pred_to_gt_dist <= threshold))
        recall = float(np.mean(gt_to_pred_dist <= threshold))
        harmonic = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0.0
            else 0.0
        )
        scores[f"{threshold:.6f}"] = {
            "threshold_m": float(threshold),
            "precision": precision,
            "recall": recall,
            "fscore_harmonic": harmonic,
        }
    return scores


def evaluate_meshes(
    predicted_mesh: trimesh.Trimesh | str | Path,
    ground_truth_mesh: trimesh.Trimesh | str | Path,
    *,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    seed: int = DEFAULT_SEED,
    gt_aabb_margin_m: float | None = None,
    thresholds_m: tuple[float, ...] = DEFAULT_THRESHOLDS_M,
    normalized_thresholds: tuple[float, ...] = DEFAULT_NORMALIZED_THRESHOLDS,
    workers: int = -1,
) -> dict[str, Any]:
    """Evaluate two meshes and return a JSON-serialisable result dictionary."""

    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if workers == 0 or workers < -1:
        raise ValueError("workers must be -1 or a positive integer")
    thresholds_m = tuple(float(value) for value in thresholds_m)
    normalized_thresholds = tuple(float(value) for value in normalized_thresholds)

    predicted = _as_single_mesh(predicted_mesh, label="predicted")
    ground_truth = _as_single_mesh(ground_truth_mesh, label="ground-truth")
    predicted_before_crop = _validated_surface(predicted, label="predicted")
    gt_surface = _validated_surface(ground_truth, label="ground-truth")
    predicted_before_crop_summary = _mesh_summary(predicted_before_crop)
    ground_truth_summary = _mesh_summary(gt_surface)
    ground_truth_bounds = np.asarray(gt_surface["bounds"]).copy()
    # Sampling-ready dictionaries contain one 9-vector per face.  Keep only
    # compact metadata between phases so a multi-million-face GT mesh does not
    # needlessly retain duplicate triangle arrays.
    del predicted_before_crop, gt_surface

    crop: dict[str, Any] = {"mode": "none"}
    protocol = PROTOCOL_UNCLIPPED
    if gt_aabb_margin_m is not None:
        predicted, crop_bounds = _clip_to_gt_aabb(
            predicted, ground_truth_bounds, gt_aabb_margin_m
        )
        crop = {
            "mode": "gt-aabb",
            "margin_m": float(gt_aabb_margin_m),
            "bounds_m": crop_bounds.tolist(),
            "note": "This is not the paper's unpublished scanner observation envelope.",
        }
        protocol = PROTOCOL_GT_AABB

    # Spawn independent, reproducible streams instead of coupling either
    # mesh's samples to the number of random draws used by the other mesh.
    pred_seed, gt_seed = np.random.SeedSequence(seed).spawn(2)
    pred_points, pred_normals, pred_surface = sample_mesh_surface(
        predicted,
        num_samples,
        np.random.default_rng(pred_seed),
        label="predicted",
    )
    predicted_evaluated_summary = _mesh_summary(pred_surface)
    del pred_surface
    gt_points, gt_normals, gt_surface = sample_mesh_surface(
        ground_truth,
        num_samples,
        np.random.default_rng(gt_seed),
        label="ground-truth",
    )
    # This should equal the earlier summary; deriving it from the exact arrays
    # used for sampling also guards against future loader-side changes.
    ground_truth_summary = _mesh_summary(gt_surface)
    del gt_surface

    gt_tree = cKDTree(gt_points)
    pred_to_gt_dist, pred_to_gt_index = gt_tree.query(pred_points, k=1, workers=workers)
    pred_tree = cKDTree(pred_points)
    gt_to_pred_dist, gt_to_pred_index = pred_tree.query(gt_points, k=1, workers=workers)

    pred_to_gt_summary = _distance_summary(pred_to_gt_dist)
    gt_to_pred_summary = _distance_summary(gt_to_pred_dist)
    pred_to_gt_mean = pred_to_gt_summary["mean_m"]
    gt_to_pred_mean = gt_to_pred_summary["mean_m"]
    threshold_scores = _threshold_scores(pred_to_gt_dist, gt_to_pred_dist, thresholds_m)
    gt_diagonal_m = float(np.linalg.norm(ground_truth_bounds[1] - ground_truth_bounds[0]))
    if not math.isfinite(gt_diagonal_m) or gt_diagonal_m <= 0.0:
        raise MeshEvaluationError(f"Ground-truth mesh has invalid AABB diagonal {gt_diagonal_m}")
    normalized_threshold_scores = _threshold_scores(
        pred_to_gt_dist,
        gt_to_pred_dist,
        tuple(value * gt_diagonal_m for value in normalized_thresholds),
    )
    for ratio, score in zip(normalized_thresholds, normalized_threshold_scores.values()):
        score["bbox_diagonal_fraction"] = float(ratio)
    precision = float(np.mean(pred_to_gt_dist <= FSCORE_THRESHOLD_M))
    recall = float(np.mean(gt_to_pred_dist <= FSCORE_THRESHOLD_M))
    arithmetic_fscore = 0.5 * (precision + recall)
    harmonic_fscore = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )

    normal_pred_to_gt, normal_pred_to_gt_valid = _normal_consistency(
        pred_normals, gt_normals, pred_to_gt_index, pred_to_gt_dist
    )
    normal_gt_to_pred, normal_gt_to_pred_valid = _normal_consistency(
        gt_normals, pred_normals, gt_to_pred_index, gt_to_pred_dist
    )

    result = {
        "schema": "genrecon.mesh-evaluation",
        "schema_version": 1,
        "protocol": protocol,
        "protocol_note": (
            "Paper-like public 3D metrics only; the unpublished 1.1 cm scanner "
            "observation envelope and its 15 cm dilation are not implemented."
        ),
        "inputs": {
            "predicted_mesh": str(predicted_mesh) if not isinstance(predicted_mesh, trimesh.Trimesh) else "<in-memory>",
            "ground_truth_mesh": str(ground_truth_mesh) if not isinstance(ground_truth_mesh, trimesh.Trimesh) else "<in-memory>",
        },
        "sampling": {
            "method": "triangle-area-weighted-uniform",
            "samples_per_mesh": int(num_samples),
            "seed": int(seed),
            "independent_rng_streams": True,
        },
        "thresholds_m": {
            "precision_recall": list(thresholds_m),
            "normal_correspondence_max_distance": NORMAL_MAX_DISTANCE_M,
        },
        "crop": crop,
        "meshes": {
            "predicted_before_crop": predicted_before_crop_summary,
            "predicted_evaluated": predicted_evaluated_summary,
            "ground_truth": ground_truth_summary,
        },
        "metrics": {
            "pred_to_gt_m": pred_to_gt_summary,
            "gt_to_pred_m": gt_to_pred_summary,
            "pred_to_gt_mean_m": pred_to_gt_mean,
            "gt_to_pred_mean_m": gt_to_pred_mean,
            "chamfer_symmetric_mean_m": 0.5 * (pred_to_gt_mean + gt_to_pred_mean),
            "threshold_scores": threshold_scores,
            "normalized_threshold_scores": normalized_threshold_scores,
            "ground_truth_bbox_diagonal_m": gt_diagonal_m,
            "precision_at_0_1m": precision,
            "recall_at_0_1m": recall,
            "fscore_arithmetic_at_0_1m": arithmetic_fscore,
            "fscore_harmonic_at_0_1m": harmonic_fscore,
            "normal_consistency_pred_to_gt": normal_pred_to_gt,
            "normal_consistency_gt_to_pred": normal_gt_to_pred,
            "normal_consistency_symmetric_mean": 0.5
            * (normal_pred_to_gt + normal_gt_to_pred),
            "normal_correspondence_fraction_pred_to_gt": normal_pred_to_gt_valid,
            "normal_correspondence_fraction_gt_to_pred": normal_gt_to_pred_valid,
        },
    }

    # Reject accidental NaN/Inf before a result is reported or written.
    json.dumps(result, allow_nan=False)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate predicted and GT meshes with protocol=paper-like-unclipped. "
            "The unpublished GenRecon scanner observation envelope is not implemented."
        )
    )
    parser.add_argument("predicted_mesh", type=Path)
    parser.add_argument("ground_truth_mesh", type=Path)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"Area-weighted samples from each mesh (default: {DEFAULT_NUM_SAMPLES}).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--gt-aabb-margin",
        type=float,
        default=None,
        metavar="METERS",
        help=(
            "Optionally crop only the predicted mesh to the GT AABB expanded by this "
            "margin. This changes protocol to paper-like-gt-aabb-cropped."
        ),
    )
    parser.add_argument(
        "--thresholds-m",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS_M),
        help="Absolute precision/recall thresholds in meters (default: 0.02 0.05 0.10).",
    )
    parser.add_argument(
        "--normalized-thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_NORMALIZED_THRESHOLDS),
        help="Thresholds as fractions of the GT AABB diagonal (default: 0.005 0.01 0.02).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="scipy cKDTree query workers; -1 uses all CPU threads (default: -1).",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = evaluate_meshes(
            args.predicted_mesh,
            args.ground_truth_mesh,
            num_samples=args.num_samples,
            seed=args.seed,
            gt_aabb_margin_m=args.gt_aabb_margin,
            thresholds_m=tuple(args.thresholds_m),
            normalized_thresholds=tuple(args.normalized_thresholds),
            workers=args.workers,
        )
        output = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8")
        sys.stdout.write(output)
        return 0
    except (MeshEvaluationError, OSError, ValueError) as exc:
        print(f"mesh evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
