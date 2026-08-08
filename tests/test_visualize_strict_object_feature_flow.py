import unittest

import numpy as np
import torch

from tools.visualize_strict_object_feature_flow import (
    classify_sources,
    feature_similarity,
    transform_boxes,
)


class StrictObjectFeatureFlowTests(unittest.TestCase):
    def test_feature_similarity_reports_identical_and_changed_object_patches(self):
        first = np.zeros((2, 2, 2), dtype=np.float32)
        first[..., 0] = 1.0
        second = first.copy()
        second[0, 0] = [0.0, 1.0]
        mask = np.asarray([[True, True], [False, False]])

        cosine, metrics = feature_similarity(first, second, mask)

        self.assertAlmostEqual(float(cosine[0, 0]), 0.0)
        self.assertAlmostEqual(float(cosine[0, 1]), 1.0)
        self.assertEqual(metrics["patches"], 2)
        self.assertAlmostEqual(metrics["cosine_mean"], 0.5)
        self.assertAlmostEqual(metrics["cosine_median"], 0.5)

    def test_transform_boxes_uses_all_corners(self):
        transform = np.asarray(
            [[2.0, 0.0, 0.0, 1.0], [0.0, 3.0, 0.0, -1.0], [0.0, 0.0, 4.0, 2.0], [0, 0, 0, 1]],
            dtype=np.float64,
        )
        bounds = transform_boxes([[0, 1, -1, 1, 2, 3]], transform)
        self.assertEqual(bounds, [((1.0, -4.0, 10.0), (3.0, 2.0, 14.0))])

    def test_source_classification_separates_surface_unknown_and_occluded(self):
        points = np.asarray([[0, 0, 1], [0, 0, 2], [0, 0, 3]], dtype=np.float64)
        camera = {
            "extrinsics_c0": np.eye(4).tolist(),
            "intrinsics": [[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]],
        }
        mask = torch.zeros((1, 1024), dtype=torch.bool)
        mask[0, 16 * 32 + 16] = True
        known = torch.zeros_like(mask)
        known[0, 16 * 32 + 16] = True
        depth = torch.zeros((1, 1024), dtype=torch.float32)
        depth[0, 16 * 32 + 16] = 2.0
        ownership = {
            "mask": mask,
            "depth_valid": known,
            "surface_depth": depth,
            "surface_band": 0.25,
        }
        raw = mask.numpy().reshape(1, 32, 32)

        result = classify_sources(points, [camera], ownership, raw)

        self.assertEqual(result["free"].tolist(), [[True, False, False]])
        self.assertEqual(result["surface"].tolist(), [[False, True, False]])
        self.assertEqual(result["occluded"].tolist(), [[False, False, True]])
        self.assertEqual(result["final"].tolist(), [[False, True, False]])


if __name__ == "__main__":
    unittest.main()
