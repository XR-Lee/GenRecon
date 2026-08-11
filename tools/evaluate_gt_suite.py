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


def _canonical_mesh_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    metrics = raw["metrics"]
    threshold_scores = {
        f"{float(item['threshold_m']):.3f}": item
        for item in metrics["threshold_scores"].values()
    }
    normalized_scores = {
        f"{float(item['bbox_diagonal_fraction']):.6f}": item
        for item in metrics["normalized_threshold_scores"].values()
    }
    return {
        "accuracy_m": metrics["pred_to_gt_m"],
        "completeness_m": metrics["gt_to_pred_m"],
        "chamfer_symmetric_mean_m": metrics["chamfer_symmetric_mean_m"],
        "threshold_scores": threshold_scores,
        "normalized_threshold_scores": normalized_scores,
        "ground_truth_bbox_diagonal_m": metrics["ground_truth_bbox_diagonal_m"],
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


def _canonical_point_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    def direction(values: dict[str, float]) -> dict[str, float]:
        return {
            "mean_m": values["mean"],
            "median_m": values["median"],
            "p90_m": values["p90"],
            "p95_m": values["p95"],
        }

    return {
        "accuracy_m": direction(metrics["pred_to_gt_m"]),
        "completeness_m": direction(metrics["gt_to_pred_m"]),
        "chamfer_symmetric_mean_m": metrics["chamfer_symmetric_mean_m"],
        "threshold_scores": metrics["threshold_scores"],
        "normalized_threshold_scores": metrics["normalized_threshold_scores"],
        "ground_truth_bbox_diagonal_m": metrics["ground_truth_bbox_diagonal_m"],
        "normal_consistency": None,
    }


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
    reference_paths = [resolve_path(value, base) for value in reference["paths"]]
    missing = [str(path) for path in reference_paths if not path.is_file()]
    if missing:
        raise GroundTruthSuiteError(f"Missing reference files: {missing}")

    backend: dict[str, Any]
    reference_counts: dict[str, Any] | None = None
    if reference["kind"] == "mesh":
        if len(reference_paths) != 1:
            raise GroundTruthSuiteError("Mesh reference requires exactly one path")
        backend = evaluate_meshes(
            predicted_mesh,
            reference_paths[0],
            num_samples=num_samples,
            seed=seed,
            gt_aabb_margin_m=reference.get("prediction_aabb_crop_margin_m"),
            thresholds_m=thresholds_m,
            normalized_thresholds=normalized_thresholds,
            workers=workers,
        )
        metrics = _canonical_mesh_metrics(backend)
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
            thresholds_m=thresholds_m,
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
        metrics = _canonical_point_metrics(point_metrics)

    result = {
        "schema": SCHEMA,
        "schema_version": 1,
        "unit": {
            "unit_id": manifest["unit_id"],
            "physical_scene_group": manifest.get("physical_scene_group", manifest["unit_id"]),
            "dataset": manifest["dataset"],
            "track": manifest["track"],
            "gt_tier": manifest["gt_tier"],
        },
        "protocol": {
            "alignment": manifest.get("evaluation", {}).get("alignment", "declared-world-frame"),
            "roi": reference.get("roi", "full-reference"),
            "scope": reference.get("scope", "raw-global"),
            "samples_per_prediction": int(num_samples),
            "max_reference_samples": int(max_gt_samples),
            "seed": int(seed),
            "thresholds_m": [float(value) for value in thresholds_m],
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


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    protocol_fields = (
        "samples_per_prediction",
        "max_reference_samples",
        "seed",
        "thresholds_m",
        "normalized_thresholds",
        "scope",
    )
    protocol_signatures = {
        json.dumps(
            {key: result["protocol"].get(key) for key in protocol_fields},
            sort_keys=True,
        )
        for result in results
        if "protocol" in result
    }
    if protocol_signatures and (
        len(protocol_signatures) != 1
        or any("protocol" not in result for result in results)
    ):
        raise GroundTruthSuiteError(
            "Cannot aggregate results with mixed or missing evaluation protocols"
        )
    paths = {
        "chamfer_symmetric_mean_m": ("metrics", "chamfer_symmetric_mean_m"),
        "fscore_at_0.05m": ("metrics", "threshold_scores", "0.050", "fscore_harmonic"),
        "fscore_at_0.10m": ("metrics", "threshold_scores", "0.100", "fscore_harmonic"),
        "normal_consistency": ("metrics", "normal_consistency", "symmetric_mean"),
    }
    groupings = {
        "tiers": "gt_tier",
        "datasets": "dataset",
        "tracks": "track",
    }
    summary: dict[str, Any] = {"result_count": len(results)}
    for output_key, unit_key in groupings.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in results:
            grouped[result["unit"][unit_key]].append(result)
        summary[output_key] = {
            label: _aggregate_group(items, paths)
            for label, items in sorted(grouped.items())
        }
    return {
        "result_count": len(results),
        "protocol_signature": (
            json.loads(next(iter(protocol_signatures))) if protocol_signatures else None
        ),
        **{key: value for key, value in summary.items() if key != "result_count"},
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
        if current_unit != result["unit"]:
            raise GroundTruthSuiteError(
                f"Unit metadata changed since evaluation for {result['unit']['unit_id']}"
            )
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
        result["inputs"]["manifest_sha256"] = sha256_file(manifest_path)
        result["protocol"]["scope"] = manifest["reference"].get("scope", "raw-global")
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
        registry = load_json(registry_path)
        if registry.get("schema") != REGISTRY_SCHEMA:
            raise GroundTruthSuiteError(f"Unexpected registry schema in {registry_path}")
        registry_ids = {unit["unit_id"] for unit in registry.get("units", [])}
        unexpected = sorted(result_ids - registry_ids)
        if unexpected:
            raise GroundTruthSuiteError(
                f"Evaluation results are absent from the registry: {unexpected}"
            )
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
