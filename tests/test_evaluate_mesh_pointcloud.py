import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from tools.evaluate_mesh_pointcloud import distance_metrics, evaluate, load_meshlab_transforms


class MeshPointCloudEvaluationTests(unittest.TestCase):
    def test_identical_points_have_zero_distance_and_perfect_scores(self) -> None:
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        result = distance_metrics(points, points, workers=1)
        self.assertEqual(result["chamfer_symmetric_mean_m"], 0.0)
        self.assertEqual(result["threshold_scores"]["0.020"]["fscore_harmonic"], 1.0)

    def test_invalid_points_and_thresholds_are_rejected(self) -> None:
        points = np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            distance_metrics(points, points, thresholds_m=(0.0,), workers=1)
        invalid = points.copy()
        invalid[0, 0] = np.nan
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            distance_metrics(invalid, points, workers=1)

    def test_full_reference_penalizes_gt_outside_prediction_aabb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mesh_path = root / "mesh.ply"
            cloud_path = root / "cloud.ply"
            mesh = trimesh.Trimesh(
                vertices=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
                faces=np.asarray([[0, 1, 2]]),
                process=False,
            )
            mesh.export(mesh_path)
            grid = np.asarray(
                [[x, y, 0.0] for x in np.linspace(0.05, 0.8, 8) for y in np.linspace(0.05, 0.8 - x, 5) if x + y < 0.95]
            )
            far = grid + np.asarray([10.0, 0.0, 0.0])
            trimesh.PointCloud(np.concatenate([grid, far])).export(cloud_path)
            cropped = evaluate(
                mesh_path,
                [cloud_path],
                crop_mode="prediction-aabb",
                num_pred_samples=4_096,
                max_gt_samples=4_096,
                workers=1,
            )
            full = evaluate(
                mesh_path,
                [cloud_path],
                crop_mode="full-reference",
                num_pred_samples=4_096,
                max_gt_samples=4_096,
                workers=1,
            )

        self.assertEqual(full["protocol"], "raw-global-pointcloud")
        self.assertIsNone(full["roi"]["crop_bounds_m"])
        self.assertLess(
            full["metrics"]["threshold_scores"]["0.100"]["recall"],
            cropped["metrics"]["threshold_scores"]["0.100"]["recall"],
        )

    def test_meshlab_transform_is_parsed_by_basename(self) -> None:
        xml = """<MeshLabProject><MeshGroup><MLMesh filename="scan1.ply">
<MLMatrix44>1 0 0 1 0 1 0 2 0 0 1 3 0 0 0 1</MLMatrix44>
</MLMesh></MeshGroup></MeshLabProject>"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "alignment.mlp"
            path.write_text(xml, encoding="utf-8")
            transforms = load_meshlab_transforms(path)
        np.testing.assert_array_equal(transforms["scan1.ply"][:3, 3], [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
