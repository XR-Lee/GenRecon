from __future__ import annotations

import unittest

import numpy as np
import torch

from genrecon.modules.cond_3D.projection import (
    apply_projection_ownership,
    classify_projection_ownership,
)
from inference.projection_ownership import fill_sparse_patch_depth


class ProjectionOwnershipTests(unittest.TestCase):
    def test_sparse_depth_fill_respects_radius_and_allowed_mask(self) -> None:
        depth = np.zeros((3, 3), dtype=np.float32)
        observed = np.zeros((3, 3), dtype=bool)
        allowed = np.ones((3, 3), dtype=bool)
        depth[1, 1] = 2.5
        observed[1, 1] = True
        allowed[0, 0] = False

        filled, valid, distance = fill_sparse_patch_depth(
            depth, observed, allowed, max_patch_distance=1.1
        )

        self.assertEqual(int(valid.sum()), 5)
        self.assertFalse(bool(valid[0, 0]))
        self.assertAlmostEqual(float(filled[1, 2]), 2.5)
        self.assertAlmostEqual(float(distance[2, 2]), np.sqrt(2.0), places=5)

    def test_mask_ownership_changes_only_points_inside_roi(self) -> None:
        points = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        coords_cam = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])
        patch_ids = torch.zeros(1, 2, dtype=torch.long)
        valid = torch.ones(1, 2, dtype=torch.bool)
        ownership = {
            "mode": "mask",
            "mask": torch.tensor([[False]]),
            "roi_bounds": torch.tensor([[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]]),
        }

        filtered = apply_projection_ownership(
            points, coords_cam, patch_ids, valid, ownership, global_img_tokens=0
        )

        self.assertEqual(filtered.tolist(), [[False, True]])

    def test_global_feature_filter_rejects_non_object_patches_outside_roi(self) -> None:
        points = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        coords_cam = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])
        patch_ids = torch.zeros(1, 2, dtype=torch.long)
        valid = torch.ones(1, 2, dtype=torch.bool)
        ownership = {
            "mode": "mask",
            "mask": torch.tensor([[False]]),
            "roi_bounds": torch.tensor([[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]]),
            "global_feature_filter": True,
        }

        filtered = apply_projection_ownership(
            points, coords_cam, patch_ids, valid, ownership, global_img_tokens=0
        )

        self.assertEqual(filtered.tolist(), [[False, False]])

    def test_hard_support_removes_points_outside_object_envelope(self) -> None:
        points = torch.tensor([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]])
        ownership = {
            "mode": "depth",
            "mask": torch.tensor([[True]]),
            "surface_depth": torch.tensor([[1.0]]),
            "depth_valid": torch.tensor([[True]]),
            "roi_bounds": torch.tensor([[[-1.0, -1.0, 0.0], [3.0, 1.0, 2.0]]]),
            "support_bounds": torch.tensor([[[-0.5, -0.5, 0.0], [0.5, 0.5, 2.0]]]),
            "hard_support": True,
            "surface_band": 0.1,
            "min_free_views": 1,
            "allow_unknown": True,
            "patch_resolution": 1,
        }
        extrinsics = torch.eye(4).reshape(1, 1, 4, 4)
        intrinsics = torch.eye(3).reshape(1, 1, 3, 3)

        states = classify_projection_ownership(
            points, extrinsics, intrinsics, ownership, img_patch_res=1
        )

        self.assertEqual(states["inside_support"].tolist(), [True, False])
        self.assertEqual(states["outside_support"].tolist(), [False, True])
        self.assertEqual(states["hard_free"].tolist(), [False, True])

    def test_depth_ownership_separates_free_surface_and_occluded(self) -> None:
        points = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 1.0], [0.0, 0.0, 1.5]])
        ownership = {
            "mode": "depth",
            "mask": torch.tensor([[True]]),
            "surface_depth": torch.tensor([[1.0]]),
            "depth_valid": torch.tensor([[True]]),
            "roi_bounds": torch.tensor([[[-1.0, -1.0, 0.0], [1.0, 1.0, 2.0]]]),
            "surface_band": 0.1,
            "min_free_views": 1,
            "allow_unknown": True,
            "patch_resolution": 1,
        }
        extrinsics = torch.eye(4).reshape(1, 1, 4, 4)
        intrinsics = torch.eye(3).reshape(1, 1, 3, 3)

        states = classify_projection_ownership(
            points,
            extrinsics,
            intrinsics,
            ownership,
            img_patch_res=1,
        )

        self.assertEqual(states["free_votes"].tolist(), [1, 0, 0])
        self.assertEqual(states["surface_votes"].tolist(), [0, 1, 0])
        self.assertEqual(states["occluded_votes"].tolist(), [0, 0, 1])
        self.assertEqual(states["hard_free"].tolist(), [True, False, False])


if __name__ == "__main__":
    unittest.main()
