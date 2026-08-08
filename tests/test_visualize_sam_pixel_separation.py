from __future__ import annotations

import unittest

import numpy as np

from tools.visualize_sam_pixel_separation import (
    binary_iou,
    fragment_metrics,
    mask_metrics,
    part_metrics,
)


class SamPixelSeparationTests(unittest.TestCase):
    def test_mask_metrics_separate_directional_differences(self) -> None:
        sam2 = np.asarray([[1, 1], [0, 0]], dtype=bool)
        sam31 = np.asarray([[0, 1], [1, 0]], dtype=bool)
        metrics = mask_metrics(sam2, sam31, sam2, sam31)

        self.assertAlmostEqual(metrics["iou"], 1 / 3)
        self.assertEqual(metrics["sam2_only_pixels"], 1)
        self.assertEqual(metrics["sam31_only_pixels"], 1)
        self.assertAlmostEqual(metrics["patch_iou"], 1 / 3)

    def test_part_metrics_detect_overlap_and_unexplained_union(self) -> None:
        union = np.asarray([[1, 1], [1, 0]], dtype=bool)
        part0 = np.asarray([[1, 1], [0, 0]], dtype=bool)
        part1 = np.asarray([[1, 0], [0, 0]], dtype=bool)
        metrics = part_metrics(union, [part0, part1])

        self.assertEqual(metrics["overlap_pixels"], 1)
        self.assertEqual(metrics["unexplained_union_pixels"], 1)
        self.assertEqual(metrics["part_pixels_outside_union"], 0)
        self.assertAlmostEqual(metrics["part_union_iou"], 2 / 3)

    def test_fragment_metrics_report_small_disconnected_regions(self) -> None:
        mask = np.zeros((20, 20), dtype=bool)
        mask[1:11, 1:11] = True
        mask[15, 15] = True
        metrics = fragment_metrics(mask, small_component_pixels=4)

        self.assertEqual(metrics["components"], 2)
        self.assertEqual(metrics["significant_components"], 1)
        self.assertAlmostEqual(metrics["largest_component_fraction"], 100 / 101)
        self.assertAlmostEqual(metrics["small_component_pixel_fraction"], 1 / 101)

    def test_empty_masks_have_unit_iou(self) -> None:
        empty = np.zeros((2, 2), dtype=bool)
        self.assertEqual(binary_iou(empty, empty), 1.0)


if __name__ == "__main__":
    unittest.main()
