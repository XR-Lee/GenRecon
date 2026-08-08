import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.evaluate_mesh_pointcloud import distance_metrics, load_meshlab_transforms


class MeshPointCloudEvaluationTests(unittest.TestCase):
    def test_identical_points_have_zero_distance_and_perfect_scores(self) -> None:
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        result = distance_metrics(points, points, workers=1)
        self.assertEqual(result["chamfer_symmetric_mean_m"], 0.0)
        self.assertEqual(result["threshold_scores"]["0.020"]["fscore_harmonic"], 1.0)

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
