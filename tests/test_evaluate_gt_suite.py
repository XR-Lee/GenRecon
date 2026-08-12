from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from tools.evaluate_gt_suite import (
    GroundTruthSuiteError,
    evaluate_registry,
    evaluate_unit,
    summarize_output,
    summarize_results,
    validate_result_registry_membership,
)


def _square(z: float = 0.0) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=np.asarray(
            [[0, 0, z], [1, 0, z], [1, 1, z], [0, 1, z]],
            dtype=np.float64,
        ),
        faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        process=False,
    )


def _manifest(reference: dict, *, tier: str = "G0-independent-scan", status: str = "prepared") -> dict:
    return {
        "schema": "genrecon.gt-calibration-unit",
        "schema_version": 1,
        "unit_id": "fixture-unit",
        "physical_scene_group": "fixture-scene",
        "dataset": "fixture",
        "track": "scene",
        "gt_tier": tier,
        "status": status,
        "reference": reference,
        "evaluation": {"alignment": "declared-world-frame", "limitations": []},
    }


class EvaluateGroundTruthSuiteTests(unittest.TestCase):
    def test_mesh_reference_is_canonicalized_with_multiscale_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction = root / "prediction.ply"
            reference = root / "reference.ply"
            manifest_path = root / "manifest.json"
            _square(z=0.03).export(prediction)
            _square().export(reference)
            manifest_path.write_text(
                json.dumps(_manifest({"kind": "mesh", "paths": [reference.name]})),
                encoding="utf-8",
            )
            result = evaluate_unit(
                manifest_path,
                prediction,
                num_samples=8_192,
                workers=1,
            )

        self.assertEqual(result["schema"], "genrecon.gt-suite-evaluation")
        self.assertLess(result["metrics"]["threshold_scores"]["0.020"]["fscore_harmonic"], 0.01)
        self.assertGreater(result["metrics"]["threshold_scores"]["0.050"]["fscore_harmonic"], 0.99)
        self.assertAlmostEqual(result["metrics"]["normal_consistency"]["symmetric_mean"], 1.0)
        json.dumps(result, allow_nan=False)

    def test_pointcloud_reference_has_null_normal_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction = root / "prediction.ply"
            reference = root / "reference.ply"
            manifest_path = root / "manifest.json"
            _square().export(prediction)
            grid = np.stack(
                np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32), [0.0]),
                axis=-1,
            ).reshape(-1, 3)
            trimesh.PointCloud(grid).export(reference)
            manifest_path.write_text(
                json.dumps(_manifest({"kind": "pointcloud", "paths": [reference.name]})),
                encoding="utf-8",
            )
            result = evaluate_unit(
                manifest_path,
                prediction,
                num_samples=2_048,
                max_gt_samples=2_048,
                workers=1,
            )

        self.assertIsNone(result["metrics"]["normal_consistency"])
        self.assertEqual(result["reference_counts"]["samples_used"], len(grid))
        self.assertGreater(result["metrics"]["threshold_scores"]["0.050"]["fscore_harmonic"], 0.9)

    def test_nonprepared_unit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction = root / "prediction.ply"
            manifest_path = root / "manifest.json"
            _square().export(prediction)
            manifest_path.write_text(
                json.dumps(
                    _manifest(
                        {"kind": "mesh", "paths": ["missing.ply"]},
                        status="blocked-auth",
                    )
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(GroundTruthSuiteError, "not prepared"):
                evaluate_unit(manifest_path, prediction, num_samples=32, workers=1)

    def test_registry_skip_reasons_distinguish_missing_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = root / "registry.json"
            registry.write_text(
                json.dumps(
                    {
                        "schema": "genrecon.gt-calibration-registry",
                        "units": [
                            {
                                "unit_id": "ready",
                                "status": "prepared",
                                "prediction_mesh": None,
                                "manifest": "ready.json",
                            },
                            {
                                "unit_id": "blocked",
                                "status": "blocked-auth",
                                "prediction_mesh": None,
                                "manifest": "blocked.json",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            index = evaluate_registry(
                registry,
                root / "output",
                num_samples=32,
                max_gt_samples=32,
                seed=42,
                workers=1,
            )

        self.assertEqual(
            index["skipped"],
            [
                {"unit_id": "ready", "reason": "missing-prediction"},
                {"unit_id": "blocked", "reason": "blocked-auth"},
            ],
        )

    def test_result_registry_membership_rejects_stale_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            prediction = root / "prediction.ply"
            manifest.write_text("{}")
            prediction.write_bytes(b"mesh")
            registry = root / "registry.json"
            registry.write_text(
                json.dumps(
                    {
                        "schema": "genrecon.gt-calibration-registry",
                        "units": [
                            {
                                "unit_id": "fixture-unit",
                                "status": "prepared",
                                "manifest": manifest.name,
                                "prediction_mesh": None,
                            }
                        ],
                    }
                )
            )
            result = {
                "unit": {"unit_id": "fixture-unit"},
                "inputs": {
                    "manifest": str(manifest),
                    "predicted_mesh": str(prediction),
                },
            }
            with self.assertRaisesRegex(GroundTruthSuiteError, "stale"):
                validate_result_registry_membership([result], registry)

            other_prediction = root / "other.ply"
            other_prediction.write_bytes(b"other")
            document = json.loads(registry.read_text())
            document["units"][0]["prediction_mesh"] = other_prediction.name
            registry.write_text(json.dumps(document))
            with self.assertRaisesRegex(GroundTruthSuiteError, "differs from registry"):
                validate_result_registry_membership([result], registry)

    def test_summary_rejects_manifest_scope_change_without_reevaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction = root / "prediction.ply"
            reference = root / "reference.ply"
            manifest_path = root / "manifest.json"
            output = root / "evaluations"
            evaluation_path = output / "units" / "fixture-unit" / "evaluation.json"
            registry = root / "registry.json"
            _square().export(prediction)
            _square().export(reference)
            manifest = _manifest(
                {
                    "kind": "mesh",
                    "paths": [reference.name],
                    "scope": "raw-global",
                }
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = evaluate_unit(
                manifest_path, prediction, num_samples=128, workers=1
            )
            evaluation_path.parent.mkdir(parents=True)
            evaluation_path.write_text(json.dumps(result), encoding="utf-8")
            registry.write_text(
                json.dumps(
                    {
                        "schema": "genrecon.gt-calibration-registry",
                        "units": [
                            {
                                "unit_id": "fixture-unit",
                                "status": "prepared",
                                "manifest": manifest_path.name,
                                "prediction_mesh": prediction.name,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (output / "index.json").write_text(
                json.dumps(
                    {
                        "schema": "genrecon.gt-suite-evaluation-index",
                        "registry": str(registry),
                    }
                ),
                encoding="utf-8",
            )
            manifest["reference"]["scope"] = "official-crop-global-reference"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(GroundTruthSuiteError, "scope changed"):
                summarize_output(output)

            manifest["reference"]["scope"] = "raw-global"
            manifest["prediction_provenance"] = {
                "track": "GT-pose-foundation-pseudo-geometry"
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            stored_result = json.loads(evaluation_path.read_text())
            stored_result["unit"].pop("prediction_track")
            evaluation_path.write_text(json.dumps(stored_result), encoding="utf-8")
            with self.assertRaisesRegex(GroundTruthSuiteError, "track is missing"):
                summarize_output(output)

    def test_summary_separates_mixed_protocols_without_cross_scope_mean(self) -> None:
        base = {
            "unit": {
                "unit_id": "a",
                "physical_scene_group": "a",
                "dataset": "fixture",
                "track": "scene",
                "gt_tier": "G0-independent-scan",
            },
            "protocol": {
                "samples_per_prediction": 100,
                "max_reference_samples": 100,
                "seed": 42,
                "thresholds_m": [0.02, 0.05, 0.1],
                "normalized_thresholds": [0.005, 0.01, 0.02],
                "scope": "raw-global",
            },
            "metrics": {
                "chamfer_symmetric_mean_m": 0.1,
                "threshold_scores": {
                    "0.050": {"fscore_harmonic": 0.5},
                    "0.100": {"fscore_harmonic": 0.8},
                },
                "normal_consistency": None,
            },
        }
        changed = json.loads(json.dumps(base))
        changed["unit"]["unit_id"] = "b"
        changed["unit"]["physical_scene_group"] = "b"
        changed["protocol"]["scope"] = "official-crop-global-reference"
        summary = summarize_results([base, changed])

        self.assertEqual(summary["result_count"], 2)
        self.assertIsNone(summary["protocol_signature"])
        self.assertEqual(summary["protocol_group_count"], 2)
        self.assertEqual(summary["aggregate_policy"], "no-cross-protocol-aggregation")
        self.assertNotIn("tiers", summary)
        groups = {group["protocol_signature"]["scope"]: group for group in summary["protocol_groups"]}
        self.assertEqual(groups["raw-global"]["unit_ids"], ["a"])
        self.assertEqual(groups["official-crop-global-reference"]["unit_ids"], ["b"])
        self.assertEqual(
            groups["raw-global"]["tiers"]["G0-independent-scan"]["metrics"]
            ["chamfer_symmetric_mean_m"]["mean"],
            0.1,
        )

    def test_summary_rejects_mixed_present_and_missing_protocols(self) -> None:
        with_protocol = {
            "unit": {
                "unit_id": "a",
                "physical_scene_group": "a",
                "dataset": "fixture",
                "track": "scene",
                "gt_tier": "G0-independent-scan",
            },
            "protocol": {
                "samples_per_prediction": 100,
                "max_reference_samples": 100,
                "seed": 42,
                "thresholds_m": [0.02, 0.05, 0.1],
                "normalized_thresholds": [0.005, 0.01, 0.02],
                "scope": "raw-global",
            },
            "metrics": {},
        }
        without_protocol = json.loads(json.dumps(with_protocol))
        without_protocol.pop("protocol")
        with self.assertRaisesRegex(GroundTruthSuiteError, "mixed present and missing"):
            summarize_results([with_protocol, without_protocol])

    def test_summary_keeps_gt_tiers_separate(self) -> None:
        def result(
            tier: str,
            value: float,
            normal: float | None,
            *,
            group: str | None = None,
        ) -> dict:
            return {
                "unit": {
                    "gt_tier": tier,
                    "dataset": "fixture",
                    "track": "scene",
                    "prediction_track": "prediction-provenance-not-recorded",
                    "physical_scene_group": group or f"fixture-{value}",
                },
                "metrics": {
                    "chamfer_symmetric_mean_m": value,
                    "threshold_scores": {
                        "0.050": {"fscore_harmonic": 1.0 - value},
                        "0.100": {"fscore_harmonic": 1.0},
                    },
                    "normal_consistency": (
                        {"symmetric_mean": normal} if normal is not None else None
                    ),
                },
            }

        summary = summarize_results(
            [
                result("G0-independent-scan", 0.1, 0.8),
                result("G0-independent-scan", 0.2, None),
                result("G2-synthetic-exact", 0.01, 1.0),
            ]
        )
        self.assertEqual(summary["tiers"]["G0-independent-scan"]["unit_count"], 2)
        self.assertAlmostEqual(
            summary["tiers"]["G0-independent-scan"]["metrics"][
                "chamfer_symmetric_mean_m"
            ]["mean"],
            0.15,
        )
        self.assertEqual(
            summary["tiers"]["G0-independent-scan"]["metrics"][
                "normal_consistency"
            ]["n"],
            1,
        )
        self.assertEqual(summary["datasets"]["fixture"]["unit_count"], 3)
        ci = summary["tiers"]["G0-independent-scan"]["metrics"][
            "chamfer_symmetric_mean_m"
        ]["bootstrap_mean_ci95"]
        self.assertEqual(len(ci), 2)
        self.assertLessEqual(ci[0], 0.15)
        self.assertGreaterEqual(ci[1], 0.15)

        grouped = summarize_results(
            [
                result("G0-independent-scan", 0.0, 1.0, group="same"),
                result("G0-independent-scan", 0.0, 1.0, group="same"),
                result("G0-independent-scan", 1.0, 1.0, group="other"),
            ]
        )
        grouped_metric = grouped["tiers"]["G0-independent-scan"]
        self.assertEqual(grouped_metric["physical_unit_count"], 2)
        self.assertEqual(
            grouped_metric["metrics"]["chamfer_symmetric_mean_m"]["mean"],
            0.5,
        )

        foundation = result("G0-independent-scan", 0.25, None)
        foundation["unit"]["prediction_track"] = (
            "GT-pose-foundation-pseudo-geometry"
        )
        split = summarize_results(
            [result("G0-independent-scan", 0.1, None), foundation]
        )
        self.assertEqual(
            set(split["prediction_tracks_by_tier"]["G0-independent-scan"]),
            {
                "prediction-provenance-not-recorded",
                "GT-pose-foundation-pseudo-geometry",
            },
        )
        self.assertEqual(
            split["prediction_tracks_by_tier"]["G0-independent-scan"]
            ["GT-pose-foundation-pseudo-geometry"]["unit_count"],
            1,
        )
        self.assertEqual(
            set(summary["tracks_by_tier"]),
            {"G0-independent-scan", "G2-synthetic-exact"},
        )
        self.assertEqual(
            summary["tracks_by_tier"]["G0-independent-scan"]["scene"][
                "unit_count"
            ],
            2,
        )
        self.assertEqual(
            summary["tracks_by_tier"]["G2-synthetic-exact"]["scene"][
                "unit_count"
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()
