import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.evaluate_view_fidelity import ColmapCamera, ColmapImage, ColmapPoint
from tools.render_sfm_error_overlays import (
    blend_rgb_glb,
    collect_view_errors,
    draw_point_overlay,
    load_point_surface_metrics,
)


class SfmErrorOverlayTests(unittest.TestCase):
    def test_load_point_surface_metrics_reads_distance_and_roi(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "points.csv"
            path.write_text(
                "point_id,distance_m,inside_glb_aabb\n"
                "7,0.04,1\n"
                "8,0.50,0\n"
            )
            metrics = load_point_surface_metrics(path)
            self.assertEqual(metrics[7]["inside_glb_aabb"], 1)
            self.assertEqual(metrics[7]["distance_m"], 0.04)

    def test_collect_view_errors_combines_direct_and_visibility_depth(self) -> None:
        camera = ColmapCamera(0, "PINHOLE", 100, 50, np.eye(3))
        image = ColmapImage(
            image_id=1,
            camera_id=0,
            name="frame.jpg",
            world_to_camera=np.eye(4),
            observations=np.array([[10, 20, 7], [30, 10, 8]], dtype=float),
        )
        points = {
            7: ColmapPoint(np.array([0, 0, 2.0]), 0.1, 4),
            8: ColmapPoint(np.array([0, 0, 3.0]), 0.1, 4),
        }
        direct = {
            7: {"distance_m": 0.04, "inside_glb_aabb": 1},
            8: {"distance_m": 0.08, "inside_glb_aabb": 1},
        }
        depth = np.zeros((50, 100), dtype=np.float32)
        depth[20, 10] = 2.1

        result = collect_view_errors(image, camera, points, direct, depth)
        np.testing.assert_array_equal(result["depth_valid"], [True, False])
        np.testing.assert_allclose(result["direct_distance_m"], [0.04, 0.08])
        self.assertAlmostEqual(result["visibility_signed_m"][0], 0.1, places=5)
        self.assertTrue(np.isnan(result["visibility_signed_m"][1]))

    def test_rgb_glb_blend_only_changes_valid_render_mask(self) -> None:
        original = np.full((3, 4, 3), 100, dtype=np.uint8)
        rendered = np.full((3, 4, 3), 200, dtype=np.uint8)
        mask = np.zeros((3, 4), dtype=bool)
        mask[1, 2] = True
        blend = blend_rgb_glb(original, rendered, mask, glb_weight=0.5)
        np.testing.assert_array_equal(blend[1, 2], [150, 150, 150])
        np.testing.assert_array_equal(blend[0, 0], original[0, 0])

    def test_draw_overlay_changes_only_neighborhood_of_points(self) -> None:
        original = np.full((32, 48, 3), 100, dtype=np.uint8)
        pixels = np.array([[12, 16]], dtype=np.int32)
        overlay = draw_point_overlay(
            original,
            pixels,
            np.array([0.1], dtype=np.float32),
            maximum=0.3,
            cmap_name="turbo",
            radius=3,
        )
        self.assertEqual(overlay.shape, original.shape)
        self.assertFalse(np.array_equal(overlay[16, 12], original[16, 12]))
        np.testing.assert_array_equal(overlay[0, 0], original[0, 0])


if __name__ == "__main__":
    unittest.main()
