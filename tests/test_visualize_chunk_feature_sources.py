import unittest

import numpy as np

from tools.visualize_chunk_feature_sources import (
    _primary_view_index,
    infer_crop_side,
    project_points,
    transform_points,
)


class ChunkFeatureSourceVisualizationTests(unittest.TestCase):
    def test_transform_points_applies_homogeneous_translation(self):
        points = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 2.0]])
        transform = np.eye(4)
        transform[:3, 3] = [4.0, -2.0, 0.5]
        actual = transform_points(points, transform)
        np.testing.assert_allclose(actual, [[5.0, 0.0, 3.5], [3.0, -2.0, 2.5]])

    def test_project_points_matches_positive_depth_and_image_bounds_rule(self):
        points = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.6, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )
        intrinsic = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
        uv, depth, valid = project_points(points, np.eye(4), intrinsic)
        np.testing.assert_allclose(uv[0], [0.5, 0.5])
        np.testing.assert_allclose(depth, [1.0, 1.0, -1.0])
        self.assertEqual(valid.tolist(), [True, False, False])

    def test_crop_side_uses_shifted_principal_point(self):
        left = np.array([[1.0, 0.0, 0.75], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
        right = left.copy()
        right[0, 2] = 0.25
        center = left.copy()
        center[0, 2] = 0.5
        self.assertEqual(infer_crop_side(left), "left")
        self.assertEqual(infer_crop_side(right), "right")
        self.assertEqual(infer_crop_side(center), "center")

    def test_primary_source_matches_path_intrinsics_and_extrinsics(self):
        identity = np.eye(4).tolist()
        left = [[1.0, 0.0, 0.75], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]
        right = [[1.0, 0.0, 0.25], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]
        scene = [
            {"img_path": "/tmp/frame.jpg", "intrinsics": left, "extrinsics_c0": identity},
            {"img_path": "/tmp/frame.jpg", "intrinsics": right, "extrinsics_c0": identity},
        ]
        cond = {"img_path": "frame.jpg", "intrinsics": right, "extrinsics_c0": identity}
        self.assertEqual(_primary_view_index(scene, cond), 1)


if __name__ == "__main__":
    unittest.main()
