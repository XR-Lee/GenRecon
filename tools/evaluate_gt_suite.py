#!/usr/bin/env python3
"""Run one canonical geometry evaluation across mesh and point-cloud GT sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

try:
    from tools.evaluate_mesh import (
        DEFAULT_NORMALIZED_THRESHOLDS,
        DEFAULT_NUM_SAMPLES,
        DEFAULT_SEED,
        DEFAULT_THRESHOLDS_M,
        evaluate_meshes,
    )
    from tools.evaluate_mesh_pointcloud import (
        PointCloudEvaluationError,
        distance_metrics,
        load_meshlab_transforms,
        sample_mesh_surface_batched,
    )
except ModuleNotFoundError:
    from evaluate_mesh import (
        DEFAULT_NORMALIZED_THRESHOLDS,
        DEFAULT_NUM_SAMPLES,
        DEFAULT_SEED,
        DEFAULT_THRESHOLDS_M,
        evaluate_meshes,
    )
    from evaluate_mesh_pointcloud import (
        PointCloudEvaluationError,
        distance_metrics,
        load_meshlab_transforms,
        sample_mesh_surface_batched,
    )

SCHEMA = "genrecon.gt-suite-evaluation"
REGISTRY_SCHEMA = "genrecon.gt-calibration-registry"
VALID_GT_TIERS = {"G0-independent-scan", "G1-fusion-reference", "G2-synthetic-exact", "O0-instance-scan"}
VALID_TRACKS = {"scene", "instance"}


class GroundTruthSuiteError(RuntimeError):
    pass


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            GroundTruthSuiteError(f"Non-finite JSON constant {value} in {path}")
        ),
    )


def write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _validate_manifest(manifest: dict[str, Any]) -> None:
    required = {"unit_id", "dataset", "track", "gt_tier", "status", "reference"}
    missing = sorted(required - set(manifest))
    if missing:
        raise GroundTruthSuiteError(f"Manifest is missing fields: {missing}")
    if manifest["track"] not in VALID_TRACKS:
        raise GroundTruthSuiteError(f"Unsupported track {manifest['track']!r}")
    if manifest["gt_tier"] not in VALID_GT_TIERS:
        raise GroundTruthSuiteError(f"Unsupported GT tier {manifest['gt_tier']!r}")
    if manifest["status"] != "prepared":
        raise GroundTruthSuiteError(
            f"Unit {manifest['unit_id']} is not prepared: {manifest['status']}"
        )
    kind = manifest["reference"].get("kind")
    if kind not in {"mesh", "pointcloud"}:
        raise GroundTruthSuiteError(f"Unsupported reference kind {kind!r}")


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [item for item in loaded.dump(concatenate=False) if isinstance(item, trimesh.Trimesh)]
        if not meshes:
            raise GroundTruthSuiteError(f"No triangle geometry in {path}")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh) or not len(loaded.faces):
        raise GroundTruthSuiteError(f"Prediction is not a nonempty triangle mesh: {path}")
    vertices = np.asarray(loaded.vertices)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise GroundTruthSuiteError(f"Prediction has invalid vertices: {path}")
    return loaded


def _load_aligned_points(
    paths: list[Path],
    *,
    transforms: dict[str, np.ndarray],
    crop_bounds: np.ndarray | None,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    batches: list[np.ndarray] = []
    source_count = 0
    cropped_count = 0
    bounds_min = np.full(3, np.inf, dtype=np.float64)
    bounds_max = np.full(3, -np.inf, dtype=np.float64)
    for path in paths:
        loaded = trimesh.load(str(path), process=False)
        if not isinstance(loaded, (trimesh.PointCloud, trimesh.Trimesh)):
            raise GroundTruthSuiteError(f"Reference is not point-like geometry: {path}")
        points = np.asarray(loaded.vertices, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise GroundTruthSuiteError(f"Reference contains invalid points: {path}")
        source_count += len(points)
        matrix = transforms.get(path.name, np.eye(4, dtype=np.float64))
        points = points @ matrix[:3, :3].T + matrix[:3, 3]
        if crop_bounds is not None:
            points = points[np.all((points >= crop_bounds[0]) & (points <= crop_bounds[1]), axis=1)]
        cropped_count += len(points)
        if len(points):
            bounds_min = np.minimum(bounds_min, np.min(points, axis=0))
            bounds_max = np.maximum(bounds_max, np.max(points, axis=0))
            batches.append(points.astype(np.float32))
    if not batches:
        raise GroundTruthSuiteError("No reference points remain after alignment/cropping")
    points = np.concatenate(batches, axis=0)
    if len(points) > max_samples:
        rng = np.random.default_rng(seed)
        points = points[rng.choice(len(points), max_samples, replace=False)]
    return points, {
        "source_points": source_count,
        "points_after_crop": cropped_count,
        "samples_used": len(points),
        "bounds_m": np.stack((bounds_min, bounds_max)).tolist(),
    }


def _distance_values_without_meter_suffix(values: dict[str, Any]) -> dict[str, Any]:
    return {key.removesuffix("_m"): value for key, value in values.items()}


def _normalized_scores_for_coordinate_units(
    scores: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    output = {}
    for item in scores.values():
        converted = dict(item)
        threshold = converted.pop("threshold_m", None)
        if threshold is not None:
            converted["threshold_coordinate_units"] = threshold
        key = f"{float(converted['bbox_diagonal_fraction']):.6f}"
        output[key] = converted
    return output


def _normalized_object_backend(
    raw: dict[str, Any], *, reference_kind: str
) -> dict[str, Any]:
    metrics = raw["metrics"]
    reference_counts = raw.get("reference_counts")
    if isinstance(reference_counts, dict):
        reference_counts = dict(reference_counts)
        bounds = reference_counts.pop("bounds_m", None)
        if bounds is not None:
            reference_counts["bounds_coordinate_units"] = bounds
    return {
        "schema": "genrecon.normalized-object-distance-backend",
        "schema_version": 1,
        "source_backend_schema": raw.get("schema"),
        "reference_kind": reference_kind,
        "coordinate_units": "normalized-object",
        "absolute_threshold_scores": {},
        "normal_consistency": None,
        "reference_counts": reference_counts,
        "metrics": {
            "prediction_to_reference": _distance_values_without_meter_suffix(
                metrics["pred_to_gt_m"]
            ),
            "reference_to_prediction": _distance_values_without_meter_suffix(
                metrics["gt_to_pred_m"]
            ),
            "chamfer_symmetric_mean": metrics["chamfer_symmetric_mean_m"],
            "ground_truth_bbox_diagonal": metrics[
                "ground_truth_bbox_diagonal_m"
            ],
            "normalized_threshold_scores": _normalized_scores_for_coordinate_units(
                metrics["normalized_threshold_scores"]
            ),
        },
    }


def _canonical_mesh_metrics(
    raw: dict[str, Any], *, coordinate_units: str = "meters"
) -> dict[str, Any]:
    metrics = raw["metrics"]
    threshold_scores = {
        f"{float(item['threshold_m']):.3f}": item
        for item in metrics["threshold_scores"].values()
    }
    normalized_scores = {
        f"{float(item['bbox_diagonal_fraction']):.6f}": item
        for item in metrics["normalized_threshold_scores"].values()
    }
    if coordinate_units == "meters":
        return {
            "accuracy_m": metrics["pred_to_gt_m"],
            "completeness_m": metrics["gt_to_pred_m"],
            "chamfer_symmetric_mean_m": metrics["chamfer_symmetric_mean_m"],
            "threshold_scores": threshold_scores,
            "normalized_threshold_scores": normalized_scores,
            "ground_truth_bbox_diagonal_m": metrics[
                "ground_truth_bbox_diagonal_m"
            ],
            "normal_consistency": {
                "prediction_to_gt": metrics["normal_consistency_pred_to_gt"],
                "gt_to_prediction": metrics["normal_consistency_gt_to_pred"],
                "symmetric_mean": metrics["normal_consistency_symmetric_mean"],
                "prediction_correspondence_fraction": metrics[
                    "normal_correspondence_fraction_pred_to_gt"
                ],
                "gt_correspondence_fraction": metrics[
                    "normal_correspondence_fraction_gt_to_pred"
                ],
            },
        }
    return {
        "coordinate_units": coordinate_units,
        "accuracy": _distance_values_without_meter_suffix(metrics["pred_to_gt_m"]),
        "completeness": _distance_values_without_meter_suffix(metrics["gt_to_pred_m"]),
        "chamfer_symmetric_mean": metrics["chamfer_symmetric_mean_m"],
        "normalized_threshold_scores": _normalized_scores_for_coordinate_units(
            normalized_scores
        ),
        "ground_truth_bbox_diagonal": metrics["ground_truth_bbox_diagonal_m"],
        "normal_consistency": None,
        "threshold_scores": {},
    }


def _canonical_point_metrics(
    metrics: dict[str, Any], *, coordinate_units: str = "meters"
) -> dict[str, Any]:
    def direction(values: dict[str, float]) -> dict[str, float]:
        return {
            "mean_m": values["mean"],
            "median_m": values["median"],
            "p90_m": values["p90"],
            "p95_m": values["p95"],
        }

    accuracy = direction(metrics["pred_to_gt_m"])
    completeness = direction(metrics["gt_to_pred_m"])
    if coordinate_units == "meters":
        return {
            "accuracy_m": accuracy,
            "completeness_m": completeness,
            "chamfer_symmetric_mean_m": metrics["chamfer_symmetric_mean_m"],
            "threshold_scores": metrics["threshold_scores"],
            "normalized_threshold_scores": metrics["normalized_threshold_scores"],
            "ground_truth_bbox_diagonal_m": metrics["ground_truth_bbox_diagonal_m"],
            "normal_consistency": None,
        }
    return {
        "coordinate_units": coordinate_units,
        "accuracy": _distance_values_without_meter_suffix(accuracy),
        "completeness": _distance_values_without_meter_suffix(completeness),
        "chamfer_symmetric_mean": metrics["chamfer_symmetric_mean_m"],
        "normalized_threshold_scores": _normalized_scores_for_coordinate_units(
            metrics["normalized_threshold_scores"]
        ),
        "ground_truth_bbox_diagonal": metrics["ground_truth_bbox_diagonal_m"],
        "normal_consistency": None,
        "threshold_scores": {},
    }


def prediction_track_from_manifest(manifest: dict[str, Any]) -> str:
    provenance = manifest.get("prediction_provenance")
    if isinstance(provenance, dict) and isinstance(provenance.get("track"), str):
        return provenance["track"]
    return "prediction-provenance-not-recorded"


def evaluate_unit(
    manifest_path: Path,
    predicted_mesh: Path,
    *,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    max_gt_samples: int = 1_000_000,
    thresholds_m: tuple[float, ...] = DEFAULT_THRESHOLDS_M,
    normalized_thresholds: tuple[float, ...] = DEFAULT_NORMALIZED_THRESHOLDS,
    seed: int = DEFAULT_SEED,
    workers: int = -1,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    predicted_mesh = predicted_mesh.resolve()
    if num_samples <= 0 or max_gt_samples <= 0:
        raise ValueError("Prediction/reference sample counts must be positive")
    manifest = load_json(manifest_path)
    _validate_manifest(manifest)
    if not predicted_mesh.is_file():
        raise GroundTruthSuiteError(f"Predicted mesh does not exist: {predicted_mesh}")
    base = manifest_path.parent
    reference = manifest["reference"]
    coordinate_units = reference.get("coordinate_units", "meters")
    if coordinate_units not in {"meters", "normalized-object"}:
        raise GroundTruthSuiteError(
            f"Unsupported reference coordinate units {coordinate_units!r}"
        )
    reference_paths = [resolve_path(value, base) for value in reference["paths"]]
    missing = [str(path) for path in reference_paths if not path.is_file()]
    if missing:
        raise GroundTruthSuiteError(f"Missing reference files: {missing}")

    backend: dict[str, Any]
    reference_counts: dict[str, Any] | None = None
    absolute_thresholds = thresholds_m if coordinate_units == "meters" else ()
    if reference["kind"] == "mesh":
        if len(reference_paths) != 1:
            raise GroundTruthSuiteError("Mesh reference requires exactly one path")
        backend = evaluate_meshes(
            predicted_mesh,
            reference_paths[0],
            num_samples=num_samples,
            seed=seed,
            gt_aabb_margin_m=reference.get("prediction_aabb_crop_margin_m"),
            thresholds_m=absolute_thresholds,
            normalized_thresholds=normalized_thresholds,
            workers=workers,
        )
        metrics = _canonical_mesh_metrics(
            backend, coordinate_units=coordinate_units
        )
        if coordinate_units != "meters":
            backend = _normalized_object_backend(
                backend, reference_kind=reference["kind"]
            )
    else:
        alignment_path = (
            resolve_path(reference["alignment_mlp"], base)
            if reference.get("alignment_mlp")
            else None
        )
        transforms = load_meshlab_transforms(alignment_path) if alignment_path else {}
        crop_bounds = (
            np.asarray(reference["crop_bounds_m"], dtype=np.float64)
            if reference.get("crop_bounds_m") is not None
            else None
        )
        if crop_bounds is not None and crop_bounds.shape != (2, 3):
            raise GroundTruthSuiteError("crop_bounds_m must have shape [2,3]")
        gt_points, reference_counts = _load_aligned_points(
            reference_paths,
            transforms=transforms,
            crop_bounds=crop_bounds,
            max_samples=max_gt_samples,
            seed=seed,
        )
        mesh = _load_mesh(predicted_mesh)
        pred_points = sample_mesh_surface_batched(
            mesh,
            num_samples,
            np.random.default_rng(seed),
        )
        point_metrics = distance_metrics(
            pred_points,
            gt_points,
            thresholds_m=absolute_thresholds,
            normalized_thresholds=normalized_thresholds,
            ground_truth_bounds=np.asarray(
                reference_counts["bounds_m"], dtype=np.float64
            ),
            workers=workers,
        )
        backend = {
            "schema": "genrecon.aligned-pointcloud-backend",
            "reference_counts": reference_counts,
            "metrics": point_metrics,
        }
        if coordinate_units != "meters":
            backend = _normalized_object_backend(
                backend, reference_kind=reference["kind"]
            )
        metrics = _canonical_point_metrics(
            point_metrics, coordinate_units=coordinate_units
        )

    result = {
        "schema": SCHEMA,
        "schema_version": 1,
        "unit": {
            "unit_id": manifest["unit_id"],
            "physical_scene_group": manifest.get("physical_scene_group", manifest["unit_id"]),
            "dataset": manifest["dataset"],
            "track": manifest["track"],
            "prediction_track": prediction_track_from_manifest(manifest),
            "gt_tier": manifest["gt_tier"],
        },
        "protocol": {
            "alignment": manifest.get("evaluation", {}).get("alignment", "declared-world-frame"),
            "roi": reference.get("roi", "full-reference"),
            "scope": reference.get("scope", "raw-global"),
            "samples_per_prediction": int(num_samples),
            "max_reference_samples": int(max_gt_samples),
            "seed": int(seed),
            "thresholds_m": (
                [float(value) for value in thresholds_m]
                if coordinate_units == "meters"
                else []
            ),
            "normalized_thresholds": [float(value) for value in normalized_thresholds],
        },
        "inputs": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "predicted_mesh": str(predicted_mesh),
            "predicted_mesh_sha256": sha256_file(predicted_mesh),
            "reference_kind": reference["kind"],
            "reference_paths": [str(path) for path in reference_paths],
            "reference_sha256": [sha256_file(path) for path in reference_paths],
        },
        "metrics": metrics,
        "reference_counts": reference_counts,
        "backend": backend,
        "limitations": list(manifest.get("evaluation", {}).get("limitations", [])),
    }
    if coordinate_units != "meters":
        result["protocol"].update(
            {
                "coordinate_units": coordinate_units,
                "primary_threshold_policy": manifest.get("evaluation", {}).get(
                    "primary_threshold_policy", "bbox-diagonal-normalized-only"
                ),
            }
        )
    json.dumps(result, allow_nan=False)
    return result


def _finite_metric(results: list[dict[str, Any]], path: tuple[str, ...]) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for index, result in enumerate(results):
        value: Any = result
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            unit = result.get("unit", {})
            group = (
                unit.get("physical_scene_group")
                or unit.get("unit_id")
                or f"result-{index}"
            )
            grouped[str(group)].append(float(value))
    return [float(np.mean(grouped[key])) for key in sorted(grouped)]


def _aggregate_group(
    results: list[dict[str, Any]], paths: dict[str, tuple[str, ...]]
) -> dict[str, Any]:
    aggregates = {}
    rng = np.random.default_rng(42)
    for label, path in paths.items():
        values = _finite_metric(results, path)
        if values:
            array = np.asarray(values, dtype=np.float64)
            if len(array) == 1:
                ci = [float(array[0]), float(array[0])]
            else:
                draws = rng.choice(array, size=(2000, len(array)), replace=True).mean(axis=1)
                ci = [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
            aggregates[label] = {
                "n": len(values),
                "mean": float(np.mean(array)),
                "median": float(np.median(array)),
                "bootstrap_mean_ci95": ci,
            }
        else:
            aggregates[label] = {
                "n": 0,
                "mean": None,
                "median": None,
                "bootstrap_mean_ci95": None,
            }
    physical_groups = {
        item.get("unit", {}).get("physical_scene_group")
        or item.get("unit", {}).get("unit_id")
        or f"result-{index}"
        for index, item in enumerate(results)
    }
    return {
        "unit_count": len(results),
        "physical_unit_count": len(physical_groups),
        "metrics": aggregates,
    }


def _aggregate_by_label(
    results: list[dict[str, Any]],
    paths: dict[str, tuple[str, ...]],
    key: str,
    missing_label: str,
) -> dict[str, Any]:
    labels = sorted(
        {result["unit"].get(key, missing_label) for result in results}
    )
    return {
        label: _aggregate_group(
            [
                result
                for result in results
                if result["unit"].get(key, missing_label) == label
            ],
            paths,
        )
        for label in labels
    }


def _aggregate_by_tier_and_label(
    results: list[dict[str, Any]],
    paths: dict[str, tuple[str, ...]],
    key: str,
    missing_label: str,
) -> dict[str, Any]:
    tiers = sorted(
        {result["unit"].get("gt_tier", "unknown-gt-tier") for result in results}
    )
    return {
        tier: _aggregate_by_label(
            [
                result
                for result in results
                if result["unit"].get("gt_tier", "unknown-gt-tier") == tier
            ],
            paths,
            key,
            missing_label,
        )
        for tier in tiers
    }


def _summarize_homogeneous_results(
    results: list[dict[str, Any]], protocol_signature: dict[str, Any] | None
) -> dict[str, Any]:
    paths = {
        "chamfer_symmetric_mean_m": ("metrics", "chamfer_symmetric_mean_m"),
        "fscore_at_0.05m": ("metrics", "threshold_scores", "0.050", "fscore_harmonic"),
        "fscore_at_0.10m": ("metrics", "threshold_scores", "0.100", "fscore_harmonic"),
        "normal_consistency": ("metrics", "normal_consistency", "symmetric_mean"),
    }
    normalized_only = any(
        result.get("protocol", {}).get("coordinate_units") == "normalized-object"
        for result in results
    )
    if normalized_only:
        paths = {
            "chamfer_symmetric_mean_coordinate_units": (
                "metrics",
                "chamfer_symmetric_mean",
            ),
            "fscore_at_bbox_0.5pct": (
                "metrics",
                "normalized_threshold_scores",
                "0.005000",
                "fscore_harmonic",
            ),
            "fscore_at_bbox_1pct": (
                "metrics",
                "normalized_threshold_scores",
                "0.010000",
                "fscore_harmonic",
            ),
            "fscore_at_bbox_2pct": (
                "metrics",
                "normalized_threshold_scores",
                "0.020000",
                "fscore_harmonic",
            ),
            "normal_consistency": (
                "metrics",
                "normal_consistency",
                "symmetric_mean",
            ),
        }
    return {
        "result_count": len(results),
        "protocol_signature": protocol_signature,
        "tiers": _aggregate_by_label(
            results, paths, "gt_tier", "unknown-gt-tier"
        ),
        "datasets": _aggregate_by_label(
            results, paths, "dataset", "unknown-dataset"
        ),
        "tracks_by_tier": _aggregate_by_tier_and_label(
            results, paths, "track", "unknown-unit-track"
        ),
        "prediction_tracks_by_tier": _aggregate_by_tier_and_label(
            results,
            paths,
            "prediction_track",
            "prediction-provenance-not-recorded",
        ),
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    protocol_fields = (
        "samples_per_prediction",
        "max_reference_samples",
        "seed",
        "thresholds_m",
        "normalized_thresholds",
        "scope",
        "coordinate_units",
        "primary_threshold_policy",
    )
    signatures: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    missing_protocol = []
    for result in results:
        if "protocol" not in result:
            missing_protocol.append(result)
            continue
        signature = {
            key: result["protocol"][key]
            for key in protocol_fields
            if key in result["protocol"]
        }
        serialized = json.dumps(signature, sort_keys=True)
        signatures.setdefault(serialized, (signature, []))[1].append(result)

    if missing_protocol and signatures:
        raise GroundTruthSuiteError(
            "Cannot aggregate results with mixed present and missing evaluation protocols"
        )
    if not signatures:
        return _summarize_homogeneous_results(results, None)
    if len(signatures) == 1:
        signature, grouped_results = next(iter(signatures.values()))
        return _summarize_homogeneous_results(grouped_results, signature)

    protocol_groups = []
    for serialized in sorted(signatures):
        signature, grouped_results = signatures[serialized]
        group = _summarize_homogeneous_results(grouped_results, signature)
        group["unit_ids"] = sorted(
            result["unit"].get("unit_id", "") for result in grouped_results
        )
        protocol_groups.append(group)
    return {
        "result_count": len(results),
        "protocol_signature": None,
        "protocol_group_count": len(protocol_groups),
        "aggregate_policy": "no-cross-protocol-aggregation",
        "protocol_groups": protocol_groups,
    }


def evaluate_registry(
    registry_path: Path,
    output_root: Path,
    *,
    num_samples: int,
    max_gt_samples: int,
    seed: int,
    workers: int,
) -> dict[str, Any]:
    registry_path = registry_path.resolve()
    registry = load_json(registry_path)
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise GroundTruthSuiteError(f"Unexpected registry schema in {registry_path}")
    results = []
    skipped = []
    for unit in registry["units"]:
        prediction = unit.get("prediction_mesh")
        if unit.get("status") != "prepared":
            skipped.append({"unit_id": unit["unit_id"], "reason": unit.get("status", "unknown-status")})
            continue
        if not prediction:
            skipped.append({"unit_id": unit["unit_id"], "reason": "missing-prediction"})
            continue
        manifest_path = resolve_path(unit["manifest"], registry_path.parent)
        prediction_path = resolve_path(prediction, registry_path.parent)
        if not prediction_path.is_file():
            skipped.append({"unit_id": unit["unit_id"], "reason": "missing-prediction"})
            continue
        result = evaluate_unit(
            manifest_path,
            prediction_path,
            num_samples=num_samples,
            max_gt_samples=max_gt_samples,
            seed=seed,
            workers=workers,
        )
        write_json(output_root / "units" / unit["unit_id"] / "evaluation.json", result)
        results.append(result)
    index = {
        "schema": "genrecon.gt-suite-evaluation-index",
        "schema_version": 1,
        "registry": str(registry_path),
        "summary": summarize_results(results),
        "results": [
            {
                "unit_id": item["unit"]["unit_id"],
                "path": str(Path("units") / item["unit"]["unit_id"] / "evaluation.json"),
                "metrics": item["metrics"],
            }
            for item in results
        ],
        "skipped": skipped,
    }
    write_json(output_root / "index.json", index)
    return index


def validate_result_registry_membership(
    results: list[dict[str, Any]], registry_path: Path
) -> dict[str, Any]:
    registry_path = registry_path.resolve()
    registry = load_json(registry_path)
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise GroundTruthSuiteError(f"Unexpected registry schema in {registry_path}")
    by_id = {unit["unit_id"]: unit for unit in registry.get("units", [])}
    result_ids = {result["unit"]["unit_id"] for result in results}
    unexpected = sorted(result_ids - set(by_id))
    if unexpected:
        raise GroundTruthSuiteError(
            f"Evaluation results are absent from the registry: {unexpected}"
        )
    for result in results:
        unit_id = result["unit"]["unit_id"]
        row = by_id[unit_id]
        prediction_value = row.get("prediction_mesh")
        if row.get("status") != "prepared" or not prediction_value:
            raise GroundTruthSuiteError(
                f"Evaluation result is stale because registry prediction is unavailable for {unit_id}"
            )
        registered_manifest = resolve_path(row["manifest"], registry_path.parent)
        registered_prediction = resolve_path(prediction_value, registry_path.parent)
        result_manifest = Path(result["inputs"]["manifest"]).resolve()
        result_prediction = Path(result["inputs"]["predicted_mesh"]).resolve()
        if registered_manifest != result_manifest:
            raise GroundTruthSuiteError(
                f"Evaluation manifest differs from registry for {unit_id}"
            )
        if registered_prediction != result_prediction:
            raise GroundTruthSuiteError(
                f"Evaluation prediction differs from registry for {unit_id}"
            )
    return registry


def summarize_output(output_root: Path) -> dict[str, Any]:
    output_root = output_root.resolve()
    previous = load_json(output_root / "index.json")
    if previous.get("schema") != "genrecon.gt-suite-evaluation-index":
        raise GroundTruthSuiteError(f"Unexpected evaluation index schema under {output_root}")
    results = []
    for path in sorted((output_root / "units").glob("*/evaluation.json")):
        result = load_json(path)
        if result.get("schema") != SCHEMA:
            raise GroundTruthSuiteError(f"Unexpected evaluation schema in {path}")
        manifest_path = Path(result["inputs"]["manifest"])
        manifest = load_json(manifest_path)
        _validate_manifest(manifest)
        current_unit = {
            "unit_id": manifest["unit_id"],
            "physical_scene_group": manifest.get("physical_scene_group", manifest["unit_id"]),
            "dataset": manifest["dataset"],
            "track": manifest["track"],
            "gt_tier": manifest["gt_tier"],
        }
        stored_unit = {key: result["unit"].get(key) for key in current_unit}
        if current_unit != stored_unit:
            raise GroundTruthSuiteError(
                f"Unit metadata changed since evaluation for {result['unit']['unit_id']}"
            )
        current_prediction_track = prediction_track_from_manifest(manifest)
        stored_prediction_track = result["unit"].get("prediction_track")
        if stored_prediction_track is None:
            if current_prediction_track != "prediction-provenance-not-recorded":
                raise GroundTruthSuiteError(
                    f"Prediction track is missing from a non-legacy evaluation for "
                    f"{result['unit']['unit_id']}"
                )
        elif stored_prediction_track != current_prediction_track:
            raise GroundTruthSuiteError(
                f"Prediction track changed since evaluation for {result['unit']['unit_id']}"
            )
        result["unit"]["prediction_track"] = current_prediction_track
        if manifest["reference"]["kind"] != result["inputs"]["reference_kind"]:
            raise GroundTruthSuiteError(
                f"Reference kind changed since evaluation for {result['unit']['unit_id']}"
            )
        current_alignment = manifest.get("evaluation", {}).get(
            "alignment", "declared-world-frame"
        )
        current_roi = manifest["reference"].get("roi", "full-reference")
        if (
            current_alignment != result["protocol"]["alignment"]
            or current_roi != result["protocol"]["roi"]
        ):
            raise GroundTruthSuiteError(
                f"Alignment/ROI changed since evaluation for {result['unit']['unit_id']}"
            )
        reference_paths = [
            resolve_path(value, manifest_path.parent)
            for value in manifest["reference"]["paths"]
        ]
        if [str(value) for value in reference_paths] != result["inputs"]["reference_paths"]:
            raise GroundTruthSuiteError(
                f"Reference paths changed since evaluation for {result['unit']['unit_id']}"
            )
        current_reference_hashes = [sha256_file(value) for value in reference_paths]
        if current_reference_hashes != result["inputs"]["reference_sha256"]:
            raise GroundTruthSuiteError(
                f"Reference hashes changed since evaluation for {result['unit']['unit_id']}"
            )
        prediction_path = Path(result["inputs"]["predicted_mesh"])
        if sha256_file(prediction_path) != result["inputs"]["predicted_mesh_sha256"]:
            raise GroundTruthSuiteError(
                f"Prediction hash changed since evaluation for {result['unit']['unit_id']}"
            )
        current_scope = manifest["reference"].get("scope", "raw-global")
        stored_scope = result["protocol"].get("scope")
        if stored_scope is not None and stored_scope != current_scope:
            raise GroundTruthSuiteError(
                f"Reference scope changed since evaluation for {result['unit']['unit_id']}"
            )
        current_coordinate_units = manifest["reference"].get(
            "coordinate_units", "meters"
        )
        stored_coordinate_units = result["protocol"].get("coordinate_units")
        if stored_coordinate_units is None:
            if current_coordinate_units != "meters":
                raise GroundTruthSuiteError(
                    f"Coordinate units are missing from a non-meter evaluation for "
                    f"{result['unit']['unit_id']}"
                )
        elif stored_coordinate_units != current_coordinate_units:
            raise GroundTruthSuiteError(
                f"Reference coordinate units changed since evaluation for "
                f"{result['unit']['unit_id']}"
            )
        current_threshold_policy = manifest.get("evaluation", {}).get(
            "primary_threshold_policy", "absolute-and-bbox-normalized"
        )
        stored_threshold_policy = result["protocol"].get(
            "primary_threshold_policy"
        )
        if stored_threshold_policy is None:
            if current_threshold_policy != "absolute-and-bbox-normalized":
                raise GroundTruthSuiteError(
                    f"Threshold policy is missing from a non-legacy evaluation for "
                    f"{result['unit']['unit_id']}"
                )
        elif stored_threshold_policy != current_threshold_policy:
            raise GroundTruthSuiteError(
                f"Threshold policy changed since evaluation for "
                f"{result['unit']['unit_id']}"
            )
        result["inputs"]["manifest_sha256"] = sha256_file(manifest_path)
        result["protocol"]["scope"] = current_scope
        if current_coordinate_units != "meters":
            result["protocol"]["coordinate_units"] = current_coordinate_units
            result["protocol"]["primary_threshold_policy"] = current_threshold_policy
        write_json(path, result)
        results.append(result)
    if not results:
        raise GroundTruthSuiteError(f"No unit evaluations found under {output_root}")
    result_ids = {item["unit"]["unit_id"] for item in results}
    if len(result_ids) != len(results):
        raise GroundTruthSuiteError("Evaluation output contains duplicate unit IDs")
    skipped = []
    registry_value = previous.get("registry")
    if registry_value:
        registry_path = Path(registry_value)
        registry = validate_result_registry_membership(results, registry_path)
        for unit in registry.get("units", []):
            if unit["unit_id"] in result_ids:
                continue
            if unit.get("status") != "prepared":
                reason = unit.get("status", "unknown-status")
            elif not unit.get("prediction_mesh"):
                reason = "missing-prediction"
            else:
                prediction = resolve_path(unit["prediction_mesh"], registry_path.parent)
                reason = "missing-prediction" if not prediction.is_file() else "not-evaluated"
            skipped.append({"unit_id": unit["unit_id"], "reason": reason})
    index = {
        "schema": "genrecon.gt-suite-evaluation-index",
        "schema_version": 1,
        "registry": registry_value,
        "summary": summarize_results(results),
        "results": [
            {
                "unit_id": item["unit"]["unit_id"],
                "path": str(Path("units") / item["unit"]["unit_id"] / "evaluation.json"),
                "metrics": item["metrics"],
            }
            for item in results
        ],
        "skipped": skipped,
    }
    write_json(output_root / "index.json", index)
    return index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    one = subparsers.add_parser("unit")
    one.add_argument("--manifest", type=Path, required=True)
    one.add_argument("--predicted-mesh", type=Path, required=True)
    one.add_argument("--output", type=Path, required=True)
    batch = subparsers.add_parser("batch")
    batch.add_argument("--registry", type=Path, required=True)
    batch.add_argument("--output-root", type=Path, required=True)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--output-root", type=Path, required=True)
    for target in (one, batch):
        target.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
        target.add_argument("--max-gt-samples", type=int, default=1_000_000)
        target.add_argument("--seed", type=int, default=DEFAULT_SEED)
        target.add_argument("--workers", type=int, default=-1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "unit":
            result = evaluate_unit(
                args.manifest,
                args.predicted_mesh,
                num_samples=args.num_samples,
                max_gt_samples=args.max_gt_samples,
                seed=args.seed,
                workers=args.workers,
            )
            write_json(args.output, result)
            print(json.dumps(result["metrics"], indent=2, sort_keys=True))
        elif args.command == "batch":
            result = evaluate_registry(
                args.registry,
                args.output_root,
                num_samples=args.num_samples,
                max_gt_samples=args.max_gt_samples,
                seed=args.seed,
                workers=args.workers,
            )
            print(json.dumps(result["summary"], indent=2, sort_keys=True))
        else:
            result = summarize_output(args.output_root)
            print(json.dumps(result["summary"], indent=2, sort_keys=True))
        return 0
    except (GroundTruthSuiteError, PointCloudEvaluationError, OSError, ValueError) as exc:
        print(f"GT evaluation failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
