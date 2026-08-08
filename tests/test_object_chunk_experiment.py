from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

from inference.projection_ownership import build_projection_ownership
from tools.analyze_object_chunk_experiment import summarize_errors
from tools.fuse_object_branch_mesh import (
    classify_faces,
    restrict_removal_to_supported_faces,
    write_binary_ply,
)


class ObjectChunkExperimentTests(unittest.TestCase):
    def test_projection_ownership_uses_only_registered_training_track(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            masks = root / "masks"
            masks.mkdir()
            Image.fromarray(np.full((4, 4), 255, dtype=np.uint8)).save(masks / "view_000.png")
            Image.fromarray(np.full((4, 4), 255, dtype=np.uint8)).save(
                masks / "view_000_part_0.png"
            )
            (root / "images.txt").write_text(
                "1 1 0 0 0 0 0 0 1 image.jpg\n0 0 1\n"
            )
            (root / "points3D.txt").write_text(
                "1 0 0 1 0 0 0 0.5 1 0\n"
                "2 0 0 2 0 0 0 0.5 1 1\n"
            )
            camera_document = {
                "scene": [
                    {
                        "img_path": "image.jpg",
                        "extrinsics_c0": torch.eye(4).tolist(),
                        "intrinsics": torch.eye(3).tolist(),
                    }
                ]
            }
            ownership, summary = build_projection_ownership(
                camera_document=camera_document,
                mask_dir=masks,
                anchors_document={"training_point_ids": [1], "holdout_point_ids": [2]},
                colmap_images_path=root / "images.txt",
                colmap_points_path=root / "points3D.txt",
                roi_boxes_world=[[-1, 1, -1, 1, 0, 3]],
                roi_bounds_chunk0=[((-1, -1, 0), (1, 1, 3))],
                world_to_chunk0=torch.eye(4),
                chunk_size_m=1.0,
                mode="depth",
                patch_resolution=2,
                depth_fill_radius_patches=0,
                surface_band_m=0.1,
                min_free_views=1,
                part_count=1,
            )

        self.assertEqual(summary["training_tracks_in_roi"], 1)
        self.assertEqual(summary["views"][0]["projected_training_track_observations"], 1)
        self.assertEqual(int(ownership["depth_valid"].sum()), 1)
        self.assertAlmostEqual(float(ownership["surface_depth"][0, 0]), 1.0)
        self.assertEqual(tuple(ownership["part_mask"].shape), (1, 1, 2, 2))

    def test_support_guard_retains_unsupported_baseline_face(self) -> None:
        vertices = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 0, 0], [11, 0, 0], [10, 1, 0]],
            dtype=np.float32,
        )
        faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        candidate, _ = classify_faces(
            vertices, faces, [[-1, 12, -1, 2, -1, 1]], margin=0, batch_size=10
        )
        supported, retained = restrict_removal_to_supported_faces(
            vertices,
            faces,
            candidate,
            object_surface_vertices=np.asarray([[0.3, 0.3, 0]], dtype=np.float32),
            max_distance=1.0,
            batch_size=10,
        )

        self.assertEqual(supported.tolist(), [True, False])
        self.assertEqual(retained, 1)

    def test_binary_ply_writer_round_trips_triangles_and_colors(self) -> None:
        vertices = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
        )
        colors = np.asarray(
            [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 255, 255, 255]],
            dtype=np.uint8,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mesh.ply"
            write_binary_ply(
                path,
                vertices,
                colors,
                np.asarray([[0, 1, 2]], dtype=np.int32),
                np.asarray([[0, 1, 2]], dtype=np.int32),
                object_vertex_offset=1,
            )
            mesh = trimesh.load(path, force="mesh", process=False)

        self.assertEqual(len(mesh.vertices), 4)
        self.assertEqual(len(mesh.faces), 2)
        self.assertEqual(mesh.faces[1].tolist(), [1, 2, 3])

    def test_error_summary_keeps_coverage_separate_from_accuracy(self) -> None:
        summary = summarize_errors([0.01, 0.20], eligible=4, hits=2)

        self.assertEqual(summary["coverage"], 0.5)
        self.assertAlmostEqual(summary["median_m"], 0.105)
        self.assertEqual(summary["within_10cm"], 0.5)


if __name__ == "__main__":
    unittest.main()
