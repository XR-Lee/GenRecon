from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from inference.get_images import ScannetIphoneImageSelecter

from tools.prepare_da3_scannetpp import (
    AdapterError,
    discover_source_scenes,
    evenly_spaced_indices,
    prepare_dataset,
    prepare_scene,
)


_CAMERA_HEADER = struct.Struct("<iiQQ")
_IMAGE_HEADER = struct.Struct("<idddddddi")
_POINT_HEADER = struct.Struct("<QdddBBBd")


def _write_cameras_binary(path: Path) -> None:
    cameras = [
        # id, model(OPENCV), width, height, fx fy cx cy k1 k2 p1 p2
        (1, 4, 1920, 1440, (1200.0, 1201.0, 960.0, 720.0, 0.01, -0.02, 0.001, -0.001)),
        (2, 4, 1024, 768, (700.0, 701.0, 512.0, 384.0, 0.0, 0.0, 0.0, 0.0)),
    ]
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(cameras)))
        for camera_id, model_id, width, height, params in cameras:
            handle.write(_CAMERA_HEADER.pack(camera_id, model_id, width, height))
            handle.write(struct.pack("<" + "d" * len(params), *params))


def _write_images_binary(
    path: Path,
    images: list[dict[str, object]],
) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(images)))
        for image in images:
            qvec = image.get("qvec", (1.0, 0.0, 0.0, 0.0))
            tvec = image.get("tvec", (0.0, 0.0, 0.0))
            handle.write(
                _IMAGE_HEADER.pack(
                    image["id"],
                    *qvec,
                    *tvec,
                    image["camera_id"],
                )
            )
            handle.write(str(image["name"]).encode("utf-8") + b"\x00")
            observations = image["observations"]
            handle.write(struct.pack("<Q", len(observations)))
            for x, y, point3d_id in observations:
                handle.write(struct.pack("<ddq", x, y, point3d_id))


def _write_points_binary(
    path: Path,
    points: list[dict[str, object]],
) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(points)))
        for point in points:
            handle.write(
                _POINT_HEADER.pack(
                    point["id"],
                    *point["xyz"],
                    *point["rgb"],
                    point["error"],
                )
            )
            track = point["track"]
            handle.write(struct.pack("<Q", len(track)))
            for image_id, point2d_idx in track:
                handle.write(struct.pack("<ii", image_id, point2d_idx))


def _noncomment_lines(path: Path) -> list[str]:
    return [
        line.rstrip("\n")
        for line in path.read_text(encoding="utf-8").splitlines(keepends=True)
        if not line.startswith("#")
    ]


class SyntheticDa3Scene:
    def __init__(
        self,
        root: Path,
        scene_id: str = "09c1414f1b",
        *,
        corrupt_selected_track: bool = False,
        omit_selected_depth: bool = False,
    ) -> None:
        self.scene_root = root / scene_id
        merge = self.scene_root / "merge_dslr_iphone"
        model = merge / "colmap" / "sparse_render_rgb"
        image_root = merge / "images"
        depth_root = merge / "render_depth"
        scans = self.scene_root / "scans"
        model.mkdir(parents=True)
        (image_root / "iphone").mkdir(parents=True)
        (image_root / "render_rgb").mkdir(parents=True)
        depth_root.mkdir(parents=True)
        scans.mkdir(parents=True)

        _write_cameras_binary(model / "cameras.bin")

        images: list[dict[str, object]] = []
        point_tracks: dict[int, list[tuple[int, int]]] = {100: [], 101: [], 102: []}
        for index in range(10):
            image_id = index + 1
            point3d_id = 100 + index % 2
            name = f"iphone/frame_{index:04d}.jpg"
            images.append(
                {
                    "id": image_id,
                    "camera_id": 1,
                    "name": name,
                    "qvec": (1.0, 0.0, 0.0, 0.0),
                    "tvec": (float(index), 0.25, -0.5),
                    "observations": [
                        (100.0 + index, 200.0, point3d_id),
                        (300.0, 400.0, -1),
                    ],
                }
            )
            point_tracks[point3d_id].append((image_id, 0))
            (image_root / name).write_bytes(b"synthetic-jpeg")
            depth_path = depth_root / f"frame_{index:04d}.png"
            # For 10 inputs -> 8 selected indices are 0,1,3,4,5,6,8,9.
            if not (omit_selected_depth and index == 0):
                depth_path.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic-depth")

        images.append(
            {
                "id": 99,
                "camera_id": 2,
                "name": "render_rgb/000000.jpg",
                "observations": [(1.0, 2.0, 102)],
            }
        )
        point_tracks[102].append((99, 0))
        (image_root / "render_rgb" / "000000.jpg").write_bytes(b"render-jpeg")
        _write_images_binary(model / "images.bin", images)

        if corrupt_selected_track:
            # image 1 POINTS2D[1] is untriangulated (-1), so this violates the
            # reciprocal point/observation invariant for selected point 100.
            point_tracks[100][0] = (1, 1)
        points = [
            {
                "id": 100,
                "xyz": (0.0, 0.0, 0.0),
                "rgb": (255, 0, 0),
                "error": 0.25,
                "track": point_tracks[100],
            },
            {
                "id": 101,
                "xyz": (1.0, 2.0, 3.0),
                "rgb": (0, 255, 0),
                "error": 0.5,
                "track": point_tracks[101],
            },
            {
                "id": 102,
                "xyz": (-1.0, -2.0, -3.0),
                "rgb": (0, 0, 255),
                "error": 0.75,
                "track": point_tracks[102],
            },
        ]
        _write_points_binary(model / "points3D.bin", points)
        scans.joinpath("mesh_aligned_0.05.ply").write_bytes(
            b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n"
        )


class PrepareDa3ScannetppTests(unittest.TestCase):
    def test_evenly_spaced_indices_match_expected_ties_to_even(self) -> None:
        self.assertEqual(evenly_spaced_indices(10, 8), [0, 1, 3, 4, 5, 6, 8, 9])
        self.assertEqual(evenly_spaced_indices(9, 8), [0, 1, 2, 3, 5, 6, 7, 8])
        self.assertEqual(evenly_spaced_indices(20, 1), [0])

    def test_prepares_consistent_genrecon_scene_and_eval_links(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = SyntheticDa3Scene(root / "source")
            output = root / "adapted"

            manifest = prepare_scene(source.scene_root, output)
            scene = output / source.scene_root.name

            self.assertEqual(manifest["counts"]["registered_images_all"], 11)
            self.assertEqual(manifest["counts"]["registered_iphone_images"], 10)
            self.assertEqual(manifest["counts"]["selected_images"], 8)
            self.assertEqual(manifest["counts"]["sparse_points_full"], 3)
            self.assertEqual(manifest["counts"]["retained_sparse_tracks"], 8)
            self.assertEqual(manifest["selection"]["indices"], [0, 1, 3, 4, 5, 6, 8, 9])

            colmap = scene / "iphone" / "colmap"
            camera_rows = _noncomment_lines(colmap / "cameras.txt")
            self.assertEqual(len(camera_rows), 1)
            self.assertTrue(camera_rows[0].startswith("1 OPENCV 1920 1440 "))

            image_rows = _noncomment_lines(colmap / "images.txt")
            self.assertEqual(len(image_rows), 16)  # header + POINTS2D for 8 images
            output_image_ids = [int(image_rows[index].split()[0]) for index in range(0, 16, 2)]
            self.assertEqual(output_image_ids, [1, 2, 4, 5, 6, 7, 9, 10])
            self.assertNotIn("iphone/", "\n".join(image_rows))

            point_rows = _noncomment_lines(colmap / "points3D.txt")
            self.assertEqual(len(point_rows), 3)  # full sparse point cloud retained
            point_100 = next(row.split() for row in point_rows if row.startswith("100 "))
            track_100 = list(zip(point_100[8::2], point_100[9::2]))
            self.assertEqual(track_100, [("1", "0"), ("5", "0"), ("7", "0"), ("9", "0")])
            point_102 = next(row.split() for row in point_rows if row.startswith("102 "))
            self.assertEqual(point_102[8:], [])  # non-iPhone track was pruned

            rgb_links = sorted((scene / "iphone" / "rgb").iterdir())
            depth_links = sorted((scene / "render_depth").iterdir())
            self.assertEqual(len(rgb_links), 8)
            self.assertEqual(len(depth_links), 8)
            self.assertTrue(all(path.is_symlink() and path.resolve().is_file() for path in rgb_links))
            self.assertTrue(all(path.is_symlink() and path.resolve().is_file() for path in depth_links))
            mesh = scene / "scans" / "mesh_aligned_0.05.ply"
            self.assertTrue(mesh.is_symlink())
            self.assertTrue(mesh.resolve().is_file())

            on_disk = json.loads((scene / "selection.json").read_text(encoding="utf-8"))
            self.assertEqual(on_disk, manifest)
            self.assertEqual(on_disk["genrecon"]["mode"], "Scannet_iphone")
            self.assertTrue(on_disk["genrecon"]["center_crop_required_for_exact_view_count"])

    def test_rejects_nonreciprocal_selected_track_without_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = SyntheticDa3Scene(root / "source", corrupt_selected_track=True)
            output = root / "adapted"
            with self.assertRaisesRegex(AdapterError, "Inconsistent COLMAP track"):
                prepare_scene(source.scene_root, output)
            self.assertFalse((output / source.scene_root.name).exists())

    def test_requires_selected_depth_and_mesh_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = SyntheticDa3Scene(root / "source", omit_selected_depth=True)
            output = root / "adapted"
            with self.assertRaisesRegex(AdapterError, "render depth paired"):
                prepare_scene(source.scene_root, output)
            self.assertFalse((output / source.scene_root.name).exists())

            manifest = prepare_scene(
                source.scene_root,
                output,
                require_eval_assets=False,
            )
            self.assertFalse(manifest["evaluation_assets_required"])
            self.assertIsNone(manifest["views"][0]["output_depth"])

    def test_discovers_one_or_all_scenes_and_writes_dataset_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_root = root / "extracted"
            first = SyntheticDa3Scene(source_root, "scene_a")
            second = SyntheticDa3Scene(source_root, "scene_b")

            discovered = discover_source_scenes(source_root)
            self.assertEqual(list(discovered), ["scene_a", "scene_b"])

            output = root / "adapted"
            dataset_manifest = prepare_dataset(
                source_root,
                output,
                scene_ids=["scene_a"],
            )
            self.assertEqual(dataset_manifest["scene_ids"], ["scene_a"])
            self.assertTrue((output / first.scene_root.name / "selection.json").is_file())
            self.assertFalse((output / "scene_b").exists())
            self.assertEqual(
                json.loads(
                    (output / "da3_scannetpp_manifest.json").read_text(encoding="utf-8")
                ),
                dataset_manifest,
            )

            updated_manifest = prepare_dataset(
                source_root,
                output,
                scene_ids=["scene_b"],
            )
            self.assertTrue((output / second.scene_root.name / "selection.json").is_file())
            self.assertEqual(updated_manifest["scene_ids"], ["scene_a", "scene_b"])
            self.assertEqual(
                [scene["scene_id"] for scene in updated_manifest["scenes"]],
                ["scene_a", "scene_b"],
            )

            with self.assertRaisesRegex(AdapterError, "different protocol"):
                prepare_dataset(
                    source_root,
                    output,
                    scene_ids=["scene_b"],
                    num_views=4,
                    overwrite=True,
                )
            self.assertEqual(
                json.loads(
                    (output / "da3_scannetpp_manifest.json").read_text(
                        encoding="utf-8"
                    )
                ),
                updated_manifest,
            )

    def test_truncated_images_binary_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = SyntheticDa3Scene(root / "source")
            images_bin = (
                source.scene_root
                / "merge_dslr_iphone"
                / "colmap/sparse_render_rgb/images.bin"
            )
            images_bin.write_bytes(images_bin.read_bytes()[:-5])
            with self.assertRaisesRegex(AdapterError, "Truncated COLMAP binary"):
                prepare_scene(source.scene_root, root / "adapted")

    def test_iphone_undistort_cache_creates_missing_parents(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scene = root / "scene"
            colmap = scene / "iphone" / "colmap"
            rgb = scene / "iphone" / "rgb"
            colmap.mkdir(parents=True)
            rgb.mkdir(parents=True)
            (colmap / "cameras.txt").write_text(
                "1 OPENCV 16 12 10 10 8 6 0.01 -0.01 0 0\n",
                encoding="utf-8",
            )
            (colmap / "images.txt").write_text(
                "1 1 0 0 0 0 0 0 1 frame.jpg\n\n",
                encoding="utf-8",
            )
            cv2.imwrite(str(rgb / "frame.jpg"), np.zeros((12, 16, 3), dtype=np.uint8))
            cache = root / "missing" / "nested" / "cache"

            with patch("inference.get_images._DEFAULT_IPHONE_UNDISTORT_CACHE", cache):
                selector = ScannetIphoneImageSelecter(undistort_cache_dir=cache)
                cameras = selector._get_cameras(colmap / "cameras.txt")

            self.assertEqual(len(cameras), 1)
            self.assertTrue((cache / scene.name / "frame.jpg").is_file())


if __name__ == "__main__":
    unittest.main()
