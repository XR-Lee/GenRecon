import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

from inference.get_images import IphoneImageSelecter
from tools.prepare_foundation_sfm import (
    align_z_up_and_proxy_scale,
    build_model_inputs,
    evenly_select_with_dynamic,
    foundation_quality_gate,
    model_intrinsics_to_original,
    model_xy_to_original,
    padded_shape,
    stage_visual_review,
    write_binary_ply,
    write_colmap_text,
)


class FoundationSfmTests(unittest.TestCase):
    def test_pad_mapping_and_intrinsics_round_trip_for_landscape(self) -> None:
        record = padded_shape(1600, 900)
        self.assertEqual(
            record,
            {
                "original_width": 1600,
                "original_height": 900,
                "resized_width": 518,
                "resized_height": 294,
                "pad_left": 0,
                "pad_top": 112,
                "target": 518,
            },
        )
        x, y = model_xy_to_original(np.array([259.0]), np.array([259.0]), record)
        self.assertAlmostEqual(float(x[0]), 800.0)
        self.assertAlmostEqual(float(y[0]), 450.0)
        model_k = np.array([[400.0, 0.0, 259.0], [0.0, 410.0, 259.0], [0.0, 0.0, 1.0]])
        original_k = model_intrinsics_to_original(model_k, record)
        self.assertAlmostEqual(original_k[0, 2], 800.0)
        self.assertAlmostEqual(original_k[1, 2], 450.0)
        self.assertAlmostEqual(original_k[0, 0], 400.0 * 1600 / 518)
        self.assertAlmostEqual(original_k[1, 1], 410.0 * 900 / 294)

    def test_even_selection_preserves_coverage_and_prefers_static_frames(self) -> None:
        names = [f"frame_{index:06d}.jpg" for index in range(12)]
        dynamic = {name: 0.2 for name in names}
        dynamic[names[1]] = 0.01
        dynamic[names[4]] = 0.01
        dynamic[names[7]] = 0.01
        dynamic[names[10]] = 0.01
        selected = evenly_select_with_dynamic(names, dynamic, 4)
        self.assertEqual(selected, [names[1], names[4], names[7], names[10]])

    def test_model_inputs_write_rgba_and_keep_dynamic_mask(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate"
            (candidate / "rgb").mkdir(parents=True)
            (candidate / "masks_dynamic").mkdir()
            rgb = np.full((8, 10, 3), 80, dtype=np.uint8)
            dynamic = np.zeros((8, 10), dtype=np.uint8)
            dynamic[2:5, 3:7] = 255
            Image.fromarray(rgb).save(candidate / "rgb" / "frame_000001.jpg")
            Image.fromarray(dynamic).save(candidate / "masks_dynamic" / "frame_000001.png")
            images, valid, model_dynamic, records = build_model_inputs(
                candidate,
                ["frame_000001.jpg"],
                root / "output_rgb",
                "white",
            )
            self.assertEqual(tuple(images.shape), (1, 3, 518, 518))
            self.assertEqual(valid.shape, (1, 518, 518))
            self.assertGreater(int(model_dynamic.sum()), 0)
            self.assertEqual(records[0]["output_name"], "frame_000001.png")
            with Image.open(root / "output_rgb" / "frame_000001.png") as opened:
                self.assertEqual(opened.mode, "RGBA")
                alpha = np.asarray(opened.getchannel("A"))
            self.assertTrue(np.array_equal(alpha, 255 - dynamic))
            self.assertTrue(np.all(images.numpy().transpose(0, 2, 3, 1)[model_dynamic] == 1.0))

    def test_alignment_uses_camera_up_and_proxy_height(self) -> None:
        points = np.array(
            [
                [-1.0, 1.6, 2.0],
                [0.0, 1.6, 2.0],
                [1.0, 1.6, 2.0],
                [-1.0, -1.1, 2.0],
                [1.0, -1.1, 2.0],
            ]
        )
        extrinsic = np.zeros((3, 3, 4), dtype=np.float64)
        extrinsic[:, :3, :3] = np.eye(3)
        centers = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]])
        extrinsic[:, :3, 3] = -centers
        aligned_points, aligned_extrinsic, metadata = align_z_up_and_proxy_scale(
            points, extrinsic, 1.6
        )
        aligned_centers = -np.einsum(
            "sji,sj->si", aligned_extrinsic[:, :3, :3], aligned_extrinsic[:, :3, 3]
        )
        self.assertAlmostEqual(float(np.median(aligned_centers[:, 2])), 1.6, places=5)
        self.assertAlmostEqual(float(np.percentile(aligned_points[:, 2], 2)), 0.0, places=5)
        for rotation in aligned_extrinsic[:, :3, :3]:
            np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-7)
            self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=7)
        self.assertEqual(metadata["gravity_method"], "negative_mean_predicted_camera_y_axis")

    def test_colmap_export_is_consumable_by_iphone_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scene = Path(temporary)
            (scene / "rgb").mkdir()
            records = []
            for index in range(2):
                name = f"frame_{index + 1:06d}.png"
                Image.new("RGBA", (100, 80), (60, 80, 100, 255)).save(scene / "rgb" / name)
                records.append(
                    {
                        "source_name": name.replace(".png", ".jpg"),
                        "output_name": name,
                        "original_width": 100,
                        "original_height": 80,
                    }
                )
            extrinsic = np.zeros((2, 3, 4), dtype=np.float64)
            extrinsic[:, :3, :3] = np.eye(3)
            extrinsic[1, 0, 3] = -0.2
            intrinsics = np.repeat(
                np.array([[[80.0, 0.0, 50.0], [0.0, 80.0, 40.0], [0.0, 0.0, 1.0]]]),
                2,
                axis=0,
            )
            points = np.array([[0.0, 0.0, 2.0], [0.3, 0.1, 2.5]])
            source = np.array([0, 1], dtype=np.int32)
            x = np.empty(2, dtype=np.float32)
            y = np.empty(2, dtype=np.float32)
            for point_index in range(2):
                frame = source[point_index]
                camera = extrinsic[frame, :3, :3] @ points[point_index] + extrinsic[frame, :3, 3]
                x[point_index] = 80 * camera[0] / camera[2] + 50
                y[point_index] = 80 * camera[1] / camera[2] + 40
            metrics = write_colmap_text(
                scene / "colmap_vggt",
                points,
                {
                    "source_frame": source,
                    "x_original": x,
                    "y_original": y,
                    "colors": np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8),
                },
                extrinsic,
                intrinsics,
                records,
            )
            self.assertEqual(metrics["point_count"], 2)
            self.assertLess(metrics["source_observation_reprojection_error_px"]["max"], 1e-5)
            cameras = IphoneImageSelecter(center_crop=True)._get_cameras(
                scene / "colmap_vggt" / "cameras.txt"
            )
            self.assertEqual(len(cameras), 2)
            self.assertEqual(Path(cameras[0]["img_path"]).name, "frame_000001.png")
            self.assertAlmostEqual(float(cameras[0]["intrinsics"][0, 0]), 0.8)

    def test_binary_ply_records_declared_vertex_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "points.ply"
            write_binary_ply(
                path,
                np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8),
                np.array([1.1, 1.2]),
                np.array([2, 3]),
                np.array([0, 1]),
            )
            with path.open("rb") as handle:
                header = b""
                while b"end_header\n" not in header:
                    header += handle.readline()
                payload = handle.read()
            self.assertIn(b"element vertex 2\n", header)
            self.assertEqual(len(payload), 2 * 23)

    def test_visual_review_is_staged_for_requested_subset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_text(
                '{"schema":"review","candidates":['
                '{"candidate_id":"a","disposition":"proceed"},'
                '{"candidate_id":"b","disposition":"reject"}]}'
            )
            output = root / "output"
            output.mkdir()
            stage_visual_review([{"candidate_id": "b"}], output, source)
            import json

            staged = json.loads((output / "visual_review.json").read_text())
            self.assertEqual([item["candidate_id"] for item in staged["candidates"]], ["b"])
            self.assertEqual(staged["summary"], {"reject": 1})

    def test_quality_gate_keeps_pseudo_grade_separate(self) -> None:
        point_metrics = {
            "exported_points": 50_000,
            "cross_view_verified_fraction": 0.9,
            "fallback_includes_unverified_points": False,
        }
        geometry = {
            "camera_centers_proxy_m": [[0, 0, 1.6], [1, 0, 1.6]],
            "point_percentiles_proxy_m": {
                "p01": [-2, -2, 0],
                "p50": [0, 2, 1],
                "p99": [2, 4, 2.7],
            },
        }
        intrinsics = np.repeat(
            np.array([[[800.0, 0.0, 800.0], [0.0, 800.0, 450.0], [0.0, 0.0, 1.0]]]),
            2,
            axis=0,
        )
        records = [
            {"original_width": 1600, "original_height": 900},
            {"original_width": 1600, "original_height": 900},
        ]
        unanchored = foundation_quality_gate(point_metrics, geometry, None, intrinsics, records)
        self.assertEqual(unanchored["grade"], "P-B")
        anchored = foundation_quality_gate(
            point_metrics,
            geometry,
            {
                "matched_camera_count": 8,
                "camera_center_error_normalized_by_baseline": {"p90": 0.1},
                "camera_rotation_error_deg": {"p90": 5.0},
            },
            intrinsics,
            records,
        )
        self.assertEqual(anchored["grade"], "P-A")
        self.assertIn("not_ground_truth", anchored["scope"])

        bad_rotation = foundation_quality_gate(
            point_metrics,
            geometry,
            {
                "matched_camera_count": 8,
                "camera_center_error_normalized_by_baseline": {"p90": 0.1},
                "camera_rotation_error_deg": {"p90": 90.0},
            },
            intrinsics,
            records,
        )
        self.assertEqual(bad_rotation["grade"], "P-C")
        self.assertIn(
            "foundation_camera_rotations_disagree_with_colmap_fragment",
            bad_rotation["warnings"],
        )


if __name__ == "__main__":
    unittest.main()
