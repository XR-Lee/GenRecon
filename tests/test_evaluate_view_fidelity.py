import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.evaluate_view_fidelity import (
    _write_html,
    _write_json,
    genrecon_glb_to_world_matrix,
    masked_psnr,
    masked_ssim,
    read_colmap_cameras,
    read_colmap_images,
    read_colmap_points,
    rendered_geometry_metrics,
    sparse_depth_metrics,
)


class ViewFidelityTests(unittest.TestCase):
    def test_json_writer_replaces_non_finite_numbers_with_null(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "metrics.json"
            _write_json(
                output,
                {
                    "finite": np.float32(1.25),
                    "missing": float("nan"),
                    "array": np.asarray([np.inf, -np.inf, 2.0]),
                },
            )
            raw = output.read_text()
            self.assertNotIn("NaN", raw)
            self.assertNotIn("Infinity", raw)
            self.assertEqual(
                json.loads(raw),
                {"array": [None, None, 2.0], "finite": 1.25, "missing": None},
            )

    def test_colmap_text_parsers_preserve_camera_and_observations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cameras.txt").write_text("0 PINHOLE 100 50 80 81 49 24\n")
            (root / "images.txt").write_text(
                "1 1 0 0 0 0 0 0 0 nested/frame.jpg\n"
                "10 20 7\n"
            )
            (root / "points3D.txt").write_text(
                "7 0 0 2 255 255 255 0.25 1 0 2 0 3 0\n"
            )

            cameras = read_colmap_cameras(root / "cameras.txt")
            images = read_colmap_images(root / "images.txt")
            points = read_colmap_points(root / "points3D.txt")

            np.testing.assert_allclose(cameras[0].intrinsic, [[80, 0, 49], [0, 81, 24], [0, 0, 1]])
            self.assertEqual(images["frame.jpg"].observations.tolist(), [[10.0, 20.0, 7.0]])
            np.testing.assert_allclose(images["frame.jpg"].world_to_camera, np.eye(4))
            self.assertEqual(points[7].track_length, 3)
            self.assertEqual(points[7].reprojection_error, 0.25)

    def test_genrecon_glb_axis_map_inverts_export_coordinates(self) -> None:
        glb_point = np.array([1.0, 3.0, -2.0, 1.0])
        world_point = genrecon_glb_to_world_matrix() @ glb_point
        np.testing.assert_allclose(world_point, [1.0, 2.0, 3.0, 1.0])

    def test_identical_masked_images_have_perfect_metrics(self) -> None:
        image = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
        mask = np.zeros((12, 16), dtype=bool)
        mask[2:10, 3:14] = True
        self.assertTrue(np.isinf(masked_psnr(image, image, mask)))
        self.assertAlmostEqual(masked_ssim(image, image, mask), 1.0, places=5)

    def test_rendered_geometry_metrics_use_intersection_and_union(self) -> None:
        ply = np.array([[1.0, 2.0], [0.0, 3.0]], dtype=np.float32)
        glb = np.array([[1.01, 2.10], [4.0, 0.0]], dtype=np.float32)
        metrics = rendered_geometry_metrics(ply, glb)
        self.assertAlmostEqual(metrics["mask_iou"], 0.5)
        self.assertAlmostEqual(metrics["intersection_coverage"], 0.5)
        self.assertAlmostEqual(metrics["depth_absolute_difference"]["median_m"], 0.055, places=5)

    def test_report_html_uses_scene_label_and_omits_missing_optional_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            _write_html(
                output,
                [{"group": "input", "image": "frame.jpg"}],
                "scene <label>",
            )
            html = (output / "index.html").read_text()
            self.assertIn("scene &lt;label&gt; fidelity", html)
            self.assertNotIn("sfm_glb_geometry", html)
            self.assertIn("views/input/frame/comparison.jpg", html)

    def test_sparse_depth_matches_colmap_point_camera_z(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cameras.txt").write_text("0 PINHOLE 100 50 80 81 49 24\n")
            (root / "images.txt").write_text("1 1 0 0 0 0 0 0 0 frame.jpg\n10 20 7\n")
            (root / "points3D.txt").write_text(
                "7 0 0 2 255 255 255 0.25 1 0 2 0 3 0\n"
            )
            camera = read_colmap_cameras(root / "cameras.txt")[0]
            image = read_colmap_images(root / "images.txt")["frame.jpg"]
            points = read_colmap_points(root / "points3D.txt")
            depth = np.zeros((50, 100), dtype=np.float32)
            depth[20, 10] = 2.0

            metrics = sparse_depth_metrics(image, camera, points, depth)
            self.assertEqual(metrics["count"], 1)
            self.assertEqual(metrics["mesh_depth_coverage"], 1.0)
            self.assertEqual(metrics["median_m"], 0.0)
            self.assertEqual(metrics["within_0.02m"], 1.0)


if __name__ == "__main__":
    unittest.main()
