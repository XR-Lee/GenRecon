from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from tools.prepare_gt_calibration_datasets import (
    CalibrationBuildError,
    _backproject_depth,
    _parse_dtu_camera,
    _parse_pose,
    _read_redwood_poses,
    safe_zip_names,
    sha256_file,
    uniform_indices,
    validate,
    write_json,
)


class PrepareGroundTruthCalibrationTests(unittest.TestCase):
    def test_uniform_indices_include_endpoints_without_duplicates(self) -> None:
        self.assertEqual(uniform_indices(10, 4), [0, 3, 6, 9])
        selected = uniform_indices(10, 5, excluded={0, 9})
        self.assertEqual(len(selected), 5)
        self.assertNotIn(0, selected)
        self.assertNotIn(9, selected)

    def test_zip_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.zip"
            with ZipFile(path, "w") as archive:
                archive.writestr("../escape.txt", "bad")
            with ZipFile(path) as archive:
                with self.assertRaisesRegex(CalibrationBuildError, "Unsafe ZIP"):
                    safe_zip_names(archive)

    def test_pose_and_depth_backprojection_use_camera_to_world(self) -> None:
        pose = _parse_pose(
            b"1 0 0 1\n0 1 0 2\n0 0 1 3\n0 0 0 1\n"
        )
        depth = np.asarray([[1000]], dtype=np.uint16)
        points = _backproject_depth(
            depth,
            pose,
            {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
            pixel_stride=1,
        )
        np.testing.assert_allclose(points, [[1.0, 2.0, 4.0]])

    def test_dtu_camera_translation_is_converted_from_mm_to_m(self) -> None:
        payload = b"""extrinsic
1 0 0 1000
0 1 0 -2000
0 0 1 3000
0 0 0 1

intrinsic
100 0 10
0 100 20
0 0 1

425 2.5
"""
        camera = _parse_dtu_camera(payload)
        matrix = np.asarray(camera["world_to_camera"])
        np.testing.assert_allclose(matrix[:3, 3], [1.0, -2.0, 3.0])

    def test_redwood_log_is_parsed_in_five_line_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.log"
            path.write_text(
                "0 0 1\n1 0 0 1\n0 1 0 2\n0 0 1 3\n0 0 0 1\n",
                encoding="utf-8",
            )
            poses = _read_redwood_poses(path)
        self.assertEqual(len(poses), 1)
        np.testing.assert_array_equal(poses[0][:3, 3], [1, 2, 3])

    def test_validator_accepts_declared_auth_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = {
                "schema": "genrecon.gt-calibration-plan",
                "datasets": {
                    "omniobject3d": {
                        "unit_ids": ["object_001"],
                    }
                }
            }
            plan_path = root / "plan.json"
            write_json(plan_path, plan)
            manifest_path = root / "units" / "omniobject3d-object_001" / "manifest.json"
            write_json(
                manifest_path,
                {
                    "schema": "genrecon.gt-calibration-unit",
                    "unit_id": "omniobject3d-object_001",
                    "dataset": "omniobject3d",
                    "track": "instance",
                    "gt_tier": "O0-instance-scan",
                    "physical_scene_group": "omniobject3d:object_001",
                    "status": "blocked-auth",
                    "input": {"conditioning_views": [], "heldout_views": []},
                    "reference": {"paths": []},
                },
            )
            write_json(
                root / "registry.json",
                {
                    "schema": "genrecon.gt-calibration-registry",
                    "plan": "plan.json",
                    "plan_sha256": sha256_file(plan_path),
                    "units": [
                        {
                            "unit_id": "omniobject3d-object_001",
                            "dataset": "omniobject3d",
                            "track": "instance",
                            "gt_tier": "O0-instance-scan",
                            "physical_scene_group": "omniobject3d:object_001",
                            "status": "blocked-auth",
                            "prediction_mesh": None,
                            "manifest": str(manifest_path.relative_to(root)),
                        }
                    ],
                    "summary": {
                        "unit_count": 1,
                        "statuses": {"blocked-auth": 1},
                        "datasets": {"omniobject3d": 1},
                        "tracks": {"instance": 1},
                        "gt_tiers": {"O0-instance-scan": 1},
                        "predictions_available": 0,
                    },
                },
            )
            result = validate(root, plan)
            loaded = json.loads((root / "validation.json").read_text())
        self.assertEqual(result["result"], "pass")
        self.assertEqual(loaded["declared_blockers"][0]["status"], "blocked-auth")


if __name__ == "__main__":
    unittest.main()
