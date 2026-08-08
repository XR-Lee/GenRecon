import json
import tempfile
import unittest
from pathlib import Path

import torch

from genrecon.pipelines.types import SelectedImages
from reconstruct_scene import (
    _colmap_image_ids_by_name,
    _farthest_point_indices,
    _load_custom_chunk_layout,
    _mask_cond2d_scene_views,
    _mask_scene_object_features,
    _match_mask_views,
    _prepare_instance_anchors,
    _set_cond2d_scene_views,
    _share_cond2d_scene_view,
    _world_boxes_to_chunk0,
)


class ReconstructSceneDiagnosticTests(unittest.TestCase):
    def test_instance_masks_match_by_image_and_crop_intrinsics_not_array_index(self):
        left = torch.eye(3).tolist()
        right_tensor = torch.eye(3)
        right_tensor[0, 2] = 0.25
        right = right_tensor.tolist()
        canonical = {
            "scene": [
                {"img_path": "a.jpg", "intrinsics": left},
                {"img_path": "a.jpg", "intrinsics": right},
            ]
        }
        current = {
            "scene": [
                {"img_path": "a.jpg", "intrinsics": right},
                {"img_path": "missing.jpg", "intrinsics": left},
                {"img_path": "a.jpg", "intrinsics": left},
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical_path = root / "canonical.json"
            current_path = root / "current.json"
            canonical_path.write_text(json.dumps(canonical))
            current_path.write_text(json.dumps(current))
            mapping = _match_mask_views(current_path, canonical_path)

        self.assertEqual(mapping, [1, None, 0])

    def test_share_cond2d_replaces_selected_chunks_and_updates_metadata(self):
        scene_512 = torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3, 1, 1)
        scene_1024 = scene_512 + 10
        scene_intrinsics = torch.stack([torch.eye(3), torch.eye(3) * 2])
        scene_extrinsics = torch.stack([torch.eye(4), torch.eye(4) * 2])
        selected = SelectedImages(
            scene_images_1024=scene_1024,
            scene_images_512=scene_512,
            scene_intrinsics=scene_intrinsics,
            scene_extrinsics_c0=scene_extrinsics,
            cond2d_images_1024=[torch.zeros(3, 1, 1) for _ in range(3)],
            cond2d_images_512=[torch.zeros(3, 1, 1) for _ in range(3)],
            cond2d_intrinsics=[torch.zeros(3, 3) for _ in range(3)],
            cond2d_extrinsics_c0=[torch.zeros(4, 4) for _ in range(3)],
            chunk_indices=[3, 4, 5],
        )
        document = {
            "scene": [
                {"img_path": "a.jpg", "intrinsics": torch.eye(3).tolist(), "extrinsics_c0": torch.eye(4).tolist()},
                {"img_path": "b.jpg", "intrinsics": (torch.eye(3) * 2).tolist(), "extrinsics_c0": (torch.eye(4) * 2).tolist()},
            ],
            "chunks": [
                {"chunk_index": index, "cond2d_view": {"img_path": "old.jpg"}}
                for index in selected.chunk_indices
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            cameras_json = Path(temporary) / "cameras.json"
            cameras_json.write_text(json.dumps(document))
            _share_cond2d_scene_view(selected, [3, 5], 1, cameras_json)
            updated = json.loads(cameras_json.read_text())

        torch.testing.assert_close(selected.cond2d_images_512[0], scene_512[1])
        torch.testing.assert_close(selected.cond2d_images_1024[2], scene_1024[1])
        torch.testing.assert_close(selected.cond2d_intrinsics[0], scene_intrinsics[1])
        torch.testing.assert_close(selected.cond2d_extrinsics_c0[2], scene_extrinsics[1])
        self.assertEqual(updated["chunks"][0]["cond2d_view"]["img_path"], "b.jpg")
        self.assertEqual(updated["chunks"][1]["cond2d_view"]["img_path"], "old.jpg")
        self.assertEqual(updated["chunks"][2]["cond2d_view"]["img_path"], "b.jpg")

    def test_fixed_cond2d_mapping_can_use_distinct_scene_views(self):
        scene = torch.arange(2, dtype=torch.float32).reshape(2, 1, 1, 1)
        selected = SelectedImages(
            scene_images_1024=scene,
            scene_images_512=scene,
            scene_intrinsics=torch.stack([torch.eye(3), torch.eye(3) * 2]),
            scene_extrinsics_c0=torch.stack([torch.eye(4), torch.eye(4) * 2]),
            cond2d_images_1024=[torch.zeros(1, 1, 1) for _ in range(2)],
            cond2d_images_512=[torch.zeros(1, 1, 1) for _ in range(2)],
            cond2d_intrinsics=[torch.zeros(3, 3) for _ in range(2)],
            cond2d_extrinsics_c0=[torch.zeros(4, 4) for _ in range(2)],
            chunk_indices=[3, 4],
        )
        document = {
            "scene": [
                {"img_path": "a.jpg", "intrinsics": torch.eye(3).tolist(), "extrinsics_c0": torch.eye(4).tolist()},
                {"img_path": "b.jpg", "intrinsics": (torch.eye(3) * 2).tolist(), "extrinsics_c0": (torch.eye(4) * 2).tolist()},
            ],
            "chunks": [{"chunk_index": 3}, {"chunk_index": 4}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            cameras_json = Path(temporary) / "cameras.json"
            cameras_json.write_text(json.dumps(document))
            _set_cond2d_scene_views(selected, {3: 1, 4: 0}, cameras_json)
            updated = json.loads(cameras_json.read_text())

        torch.testing.assert_close(selected.cond2d_images_512[0], scene[1])
        torch.testing.assert_close(selected.cond2d_images_512[1], scene[0])
        self.assertEqual(updated["chunks"][0]["cond2d_view"]["scene_view_index"], 1)
        self.assertEqual(updated["chunks"][1]["cond2d_view"]["scene_view_index"], 0)

    def test_instance_mask_only_changes_selected_cond2d_images(self):
        selected = SelectedImages(
            scene_images_1024=torch.ones(1, 1, 2, 2),
            scene_images_512=torch.ones(1, 1, 1, 1),
            scene_intrinsics=torch.eye(3)[None],
            scene_extrinsics_c0=torch.eye(4)[None],
            cond2d_images_1024=[torch.ones(1, 2, 2), torch.ones(1, 2, 2)],
            cond2d_images_512=[torch.ones(1, 1, 1), torch.ones(1, 1, 1)],
            cond2d_intrinsics=[torch.eye(3), torch.eye(3)],
            cond2d_extrinsics_c0=[torch.eye(4), torch.eye(4)],
            chunk_indices=[3, 4],
        )
        document = {
            "scene": [{"img_path": "a.jpg"}],
            "chunks": [
                {"chunk_index": 3, "cond2d_view": {"scene_view_index": 0}},
                {"chunk_index": 4, "cond2d_view": {"scene_view_index": 0}},
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cameras = root / "cameras.json"
            cameras.write_text(json.dumps(document))
            from PIL import Image
            import numpy as np

            Image.fromarray(np.asarray([[255, 0], [0, 0]], dtype=np.uint8)).save(root / "view_000.png")
            _mask_cond2d_scene_views(selected, [3], root, cameras, background_keep=0.25)

        self.assertAlmostEqual(float(selected.cond2d_images_1024[0][0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(selected.cond2d_images_1024[0][0, 1, 1]), 0.25)
        torch.testing.assert_close(selected.cond2d_images_1024[1], torch.ones(1, 2, 2))

    def test_context_encoded_object_features_leave_rgb_unchanged_before_token_gate(self):
        selected = SelectedImages(
            scene_images_1024=torch.ones(2, 1, 2, 2),
            scene_images_512=torch.ones(2, 1, 1, 1),
            scene_intrinsics=torch.stack([torch.eye(3), torch.eye(3)]),
            scene_extrinsics_c0=torch.stack([torch.eye(4), torch.eye(4)]),
            cond2d_images_1024=[],
            cond2d_images_512=[],
            cond2d_intrinsics=[],
            cond2d_extrinsics_c0=[],
            chunk_indices=[],
        )
        document = {
            "scene": [
                {"img_path": "a.jpg", "intrinsics": torch.eye(3).tolist()},
                {"img_path": "b.jpg", "intrinsics": torch.eye(3).tolist()},
            ],
            "chunks": [],
        }
        summary = {
            "views": [
                {"view_index": 0, "active_parts": [0], "training_points": 3},
                {"view_index": 1, "active_parts": [0], "training_points": 1},
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cameras = root / "cameras.json"
            summary_path = root / "summary.json"
            cameras.write_text(json.dumps(document))
            summary_path.write_text(json.dumps(summary))
            enabled = _mask_scene_object_features(
                selected,
                root,
                summary_path,
                cameras,
                min_training_points=3,
                mask_view_indices=[0, 1],
                mask_rgb_before_encoding=False,
            )
            updated = json.loads(cameras.read_text())

        self.assertEqual(enabled, {0})
        torch.testing.assert_close(selected.scene_images_512, torch.ones_like(selected.scene_images_512))
        torch.testing.assert_close(selected.scene_images_1024, torch.ones_like(selected.scene_images_1024))
        self.assertEqual(updated["scene"][0]["object_feature_rgb_mode"], "context_encoded_before_token_gate")
        self.assertTrue(updated["scene"][0]["object_feature_enabled"])
        self.assertFalse(updated["scene"][1]["object_feature_enabled"])

    def test_custom_layout_requires_model_lattice_alignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = root / "valid.json"
            valid.write_text(json.dumps({"chunk_size_m": 2.0, "centers": [[0, 0, 0], [1.0, 0, 0]]}))
            centers, _, _, rel_t = _load_custom_chunk_layout(valid, root)
            self.assertEqual(centers[1], [1.0, 0.0, 0.0])
            torch.testing.assert_close(rel_t[1], torch.tensor([0.5, 0.0, 0.0]))

            invalid = root / "invalid.json"
            invalid.write_text(json.dumps({"chunk_size_m": 2.0, "centers": [[0, 0, 0], [0.1, 0, 0]]}))
            with self.assertRaisesRegex(ValueError, "1/16"):
                _load_custom_chunk_layout(invalid, root)

    def test_world_boxes_are_transformed_to_chunk0_bounds(self):
        transform = torch.eye(4)
        transform[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
        boxes = _world_boxes_to_chunk0([[0, 2, -1, 1, 4, 5]], transform)

        self.assertEqual(boxes, [((1.0, 1.0, 7.0), (3.0, 3.0, 8.0))])

    def test_farthest_point_sampling_is_deterministic_and_spread(self):
        points = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]]).numpy()
        indices = _farthest_point_indices(points, 3)

        self.assertEqual(indices.tolist(), [0, 3, 1])

    def test_instance_anchors_split_holdout_ids_before_token_sampling(self):
        lines = ["# points"]
        for index in range(10):
            # ID X Y Z R G B ERROR, then three image/point observation pairs.
            lines.append(f"{index + 1} {index / 10} 0 0 0 0 0 0.5 1 0 2 0 3 0")
        with tempfile.TemporaryDirectory() as temporary:
            points_path = Path(temporary) / "points3D.txt"
            points_path.write_text("\n".join(lines))
            anchors, diagnostics = _prepare_instance_anchors(
                points_path,
                [[-1, 2, -1, 1, -1, 1]],
                max_reprojection_error=2.0,
                min_track_length=3,
                token_count=4,
                holdout_fraction=0.3,
                seed=7,
            )

        self.assertEqual(anchors.shape, (4, 3))
        self.assertEqual(diagnostics["training_points"], 7)
        self.assertEqual(diagnostics["holdout_points"], 3)
        self.assertTrue(set(diagnostics["token_point_ids"]).isdisjoint(diagnostics["holdout_point_ids"]))
        self.assertTrue(all(len(image_ids) == 3 for image_ids in diagnostics["token_observation_image_ids"]))

    def test_colmap_image_name_to_id_parser_uses_header_lines(self):
        text = "\n".join(
            [
                "# Image list",
                "7 1 0 0 0 0 0 0 2 folder/a.jpg",
                "10 20 1 30 40 -1",
                "9 1 0 0 0 0 0 0 2 b.jpg",
                "11 21 2",
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "images.txt"
            path.write_text(text)
            result = _colmap_image_ids_by_name(path)

        self.assertEqual(result, {"a.jpg": 7, "b.jpg": 9})


if __name__ == "__main__":
    unittest.main()
