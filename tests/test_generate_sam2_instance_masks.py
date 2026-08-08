import unittest

import numpy as np
import torch

from tools.generate_sam2_instance_masks import (
    choose_mask,
    farthest_points_2d,
    project_world_points,
)


class Sam2InstanceMaskTests(unittest.TestCase):
    def test_projection_uses_chunk0_camera_frame_and_normalized_intrinsics(self):
        xyz = np.asarray([[0.0, 0.0, 2.0], [3.0, 0.0, 1.0]])
        uv, valid = project_world_points(xyz, np.eye(4), np.eye(4), np.eye(3))

        np.testing.assert_allclose(uv, [[0.0, 0.0], [3.0, 0.0]])
        self.assertEqual(valid.tolist(), [True, False])

    def test_farthest_points_keep_spatially_separated_prompts(self):
        points = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        sampled = farthest_points_2d(points, 3)

        np.testing.assert_allclose(sampled, [[0.0, 0.0], [3.0, 0.0], [1.0, 0.0]])

    def test_mask_selection_prioritizes_prompt_recall_before_iou_score(self):
        masks = torch.zeros(2, 4, 4, dtype=torch.bool)
        masks[0, 1, 1] = True
        masks[0, 2, 2] = True
        masks[1, 1, 1] = True
        scores = torch.tensor([0.2, 0.9])

        selected, index = choose_mask(masks, scores, np.asarray([[1, 1], [2, 2]]))

        self.assertEqual(index, 0)
        self.assertEqual(int(selected.sum()), 2)


if __name__ == "__main__":
    unittest.main()
