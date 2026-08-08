from __future__ import annotations

import unittest

import numpy as np
import trimesh

from tools.evaluate_mesh import MeshEvaluationError, evaluate_meshes


def _unit_square(*, z: float = 0.0) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=np.array(
            [
                [0.0, 0.0, z],
                [1.0, 0.0, z],
                [1.0, 1.0, z],
                [0.0, 1.0, z],
            ],
            dtype=np.float64,
        ),
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        process=False,
    )


def _partial_square(*, xmax: float) -> trimesh.Trimesh:
    mesh = _unit_square()
    mesh.vertices[[1, 2], 0] = xmax
    return mesh


class EvaluateMeshTests(unittest.TestCase):
    def test_identical_mesh_has_full_threshold_scores_and_matching_normals(self) -> None:
        mesh = _unit_square()
        result = evaluate_meshes(mesh, mesh, num_samples=4_096, seed=42, workers=1)
        metrics = result["metrics"]

        self.assertEqual(result["protocol"], "paper-like-unclipped")
        self.assertLess(metrics["chamfer_symmetric_mean_m"], 0.02)
        self.assertEqual(metrics["precision_at_0_1m"], 1.0)
        self.assertEqual(metrics["recall_at_0_1m"], 1.0)
        self.assertEqual(metrics["fscore_arithmetic_at_0_1m"], 1.0)
        self.assertEqual(metrics["fscore_harmonic_at_0_1m"], 1.0)
        self.assertAlmostEqual(metrics["normal_consistency_symmetric_mean"], 1.0)

    def test_translation_beyond_normal_cutoff_scores_zero(self) -> None:
        predicted = _unit_square(z=0.3)
        ground_truth = _unit_square(z=0.0)
        result = evaluate_meshes(
            predicted, ground_truth, num_samples=4_096, seed=7, workers=1
        )
        metrics = result["metrics"]

        self.assertGreaterEqual(metrics["pred_to_gt_mean_m"], 0.3)
        self.assertGreaterEqual(metrics["gt_to_pred_mean_m"], 0.3)
        self.assertLess(metrics["chamfer_symmetric_mean_m"], 0.31)
        self.assertEqual(metrics["precision_at_0_1m"], 0.0)
        self.assertEqual(metrics["recall_at_0_1m"], 0.0)
        self.assertEqual(metrics["fscore_arithmetic_at_0_1m"], 0.0)
        self.assertEqual(metrics["fscore_harmonic_at_0_1m"], 0.0)
        self.assertEqual(metrics["normal_consistency_symmetric_mean"], 0.0)

    def test_sampling_and_metrics_are_deterministic_for_a_seed(self) -> None:
        predicted = _unit_square(z=0.03)
        ground_truth = _unit_square()
        first = evaluate_meshes(
            predicted, ground_truth, num_samples=1_024, seed=1234, workers=1
        )
        second = evaluate_meshes(
            predicted, ground_truth, num_samples=1_024, seed=1234, workers=1
        )

        self.assertEqual(first, second)

    def test_arithmetic_fscore_is_distinct_from_harmonic_for_unequal_pr(self) -> None:
        result = evaluate_meshes(
            _partial_square(xmax=0.4),
            _unit_square(),
            num_samples=8_192,
            seed=42,
            workers=1,
        )
        metrics = result["metrics"]
        precision = metrics["precision_at_0_1m"]
        recall = metrics["recall_at_0_1m"]

        self.assertGreater(precision, recall)
        self.assertAlmostEqual(
            metrics["fscore_arithmetic_at_0_1m"],
            0.5 * (precision + recall),
        )
        self.assertAlmostEqual(
            metrics["fscore_harmonic_at_0_1m"],
            2.0 * precision * recall / (precision + recall),
        )
        self.assertGreater(
            metrics["fscore_arithmetic_at_0_1m"],
            metrics["fscore_harmonic_at_0_1m"],
        )

    def test_empty_mesh_is_rejected(self) -> None:
        empty = trimesh.Trimesh(
            vertices=np.empty((0, 3), dtype=np.float64),
            faces=np.empty((0, 3), dtype=np.int64),
            process=False,
        )
        with self.assertRaisesRegex(MeshEvaluationError, "no vertices"):
            evaluate_meshes(empty, _unit_square(), num_samples=32, workers=1)


if __name__ == "__main__":
    unittest.main()
