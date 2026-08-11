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
    summarize_results,
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

    def test_summary_rejects_mixed_protocols(self) -> None:
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
        changed["protocol"]["seed"] = 7
        with self.assertRaisesRegex(GroundTruthSuiteError, "mixed"):
            summarize_results([base, changed])

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


if __name__ == "__main__":
    unittest.main()
