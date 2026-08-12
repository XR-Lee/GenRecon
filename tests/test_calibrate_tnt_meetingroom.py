from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from tools.calibrate_tnt_meetingroom import (
    TntCalibrationError,
    archive_image_names,
    canonicalize_source_references,
    read_poses,
)


class CalibrateTntMeetingroomTests(unittest.TestCase):
    def test_camera_log_parser_reads_camera_to_world_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "poses.log"
            path.write_text(
                "0 0 0\n"
                "1 0 0 1\n"
                "0 1 0 2\n"
                "0 0 1 3\n"
                "0 0 0 1\n",
                encoding="utf-8",
            )
            poses = read_poses(path)
        self.assertEqual(len(poses), 1)
        np.testing.assert_allclose(poses[0][:3, 3], [1.0, 2.0, 3.0])

    def test_camera_log_parser_rejects_non_homogeneous_pose(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "poses.log"
            path.write_text(
                "0 0 0\n"
                "1 0 0 0\n"
                "0 1 0 0\n"
                "0 0 1 0\n"
                "0 0 0 2\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(TntCalibrationError, "homogeneous"):
                read_poses(path)

    def test_camera_log_parser_rejects_unexpected_frame_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "poses.log"
            path.write_text(
                "7 7 0\n"
                "1 0 0 0\n"
                "0 1 0 0\n"
                "0 0 1 0\n"
                "0 0 0 1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(TntCalibrationError, "mapping"):
                read_poses(path)

    def test_source_references_are_normalized_without_touching_other_fields(self) -> None:
        document = {
            "source": {
                "image_archive": "/old/clone/Meetingroom.zip",
                "camera_log": "/old/clone/Meetingroom_COLMAP.log",
                "image_archive_sha256": "abc",
            },
            "calibration": {"params": [1.0, 2.0, 3.0, 4.0]},
        }
        changed = canonicalize_source_references(
            document,
            Path("/new/clone/Meetingroom.zip"),
            Path("/new/clone/Meetingroom_COLMAP.log"),
        )
        self.assertTrue(changed)
        self.assertEqual(document["source"]["image_archive"], "Meetingroom.zip")
        self.assertEqual(document["source"]["camera_log"], "Meetingroom_COLMAP.log")
        self.assertEqual(document["source"]["image_archive_sha256"], "abc")
        self.assertEqual(document["calibration"]["params"], [1.0, 2.0, 3.0, 4.0])

    def test_archive_images_are_sorted_by_official_frame_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "images.zip"
            with ZipFile(path, "w") as archive:
                archive.writestr("Meetingroom/000002.jpg", b"two")
                archive.writestr("Meetingroom/000001.jpg", b"one")
                archive.writestr("Meetingroom/readme.txt", b"ignored")
            names = archive_image_names(path)
        self.assertEqual(
            names,
            ["Meetingroom/000001.jpg", "Meetingroom/000002.jpg"],
        )


if __name__ == "__main__":
    unittest.main()
