from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import trimesh

import tools.prepare_gt_calibration_datasets as gt_builder
from tools.prepare_gt_calibration_datasets import (
    CalibrationBuildError,
    _backproject_depth,
    _build_tnt_reference,
    _parse_dtu_camera,
    _parse_pose,
    _parse_tnt_alignment,
    _read_redwood_poses,
    _transform_tnt_camera_pose,
    _validate_existing_frozen_hashes,
    safe_zip_names,
    sha256_file,
    uniform_indices,
    validate,
    validate_prediction_override,
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

    def test_tnt_alignment_is_parsed_and_applied_to_camera_pose(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "alignment.txt"
            matrix = np.eye(4)
            matrix[:3, :3] *= 2.0
            matrix[:3, 3] = [1.0, 2.0, 3.0]
            np.savetxt(path, matrix)
            alignment = _parse_tnt_alignment(path)
        pose = np.eye(4)
        pose[:3, 3] = [4.0, 5.0, 6.0]
        transformed = _transform_tnt_camera_pose(pose, alignment)
        self.assertAlmostEqual(alignment["scale"], 2.0)
        np.testing.assert_allclose(transformed[:3, :3], np.eye(3))
        np.testing.assert_allclose(transformed[:3, 3], [9.0, 12.0, 15.0])

    def test_tnt_frozen_hash_rejects_existing_modified_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            artifact = source / "artifact.bin"
            artifact.write_bytes(b"modified")
            with self.assertRaisesRegex(CalibrationBuildError, "frozen SHA256 mismatch"):
                _validate_existing_frozen_hashes(
                    source,
                    [{"artifact.bin": "0" * 64}, {"missing.bin": "1" * 64}],
                )

    def test_tnt_reference_applies_polygon_crop_and_global_voxel_dedup(self) -> None:
        dtype = np.dtype(
            [
                ("x", "<f8"),
                ("y", "<f8"),
                ("z", "<f8"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        )

        def payload(points: list[tuple[float, float, float]]) -> bytes:
            values = np.zeros(len(points), dtype=dtype)
            values["x"], values["y"], values["z"] = np.asarray(points).T
            values["red"] = 10
            values["green"] = 20
            values["blue"] = 30
            header = (
                "ply\nformat binary_little_endian 1.0\n"
                f"element vertex {len(points)}\n"
                "property double x\nproperty double y\nproperty double z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n"
            ).encode("ascii")
            return header + values.tobytes()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / "scans.zip"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "Meetingroom/scan1.ply",
                    payload([(0.0, 0.0, 0.0), (0.001, 0.0, 0.0), (0.02, 0.0, 0.0), (2.0, 2.0, 0.0)]),
                )
                archive.writestr(
                    "Meetingroom/scan2.ply",
                    payload([(0.002, 0.0, 0.0), (0.03, 0.0, 0.0)]),
                )
            crop_path = root / "crop.json"
            write_json(
                crop_path,
                {
                    "axis_min": -1.0,
                    "axis_max": 1.0,
                    "bounding_polygon": [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]],
                    "orthogonal_axis": "Z",
                },
            )
            output = root / "reference.ply"
            metadata = root / "build.json"
            first = _build_tnt_reference(
                archive_path, crop_path, output, metadata, force=True, max_points=None
            )
            first_hash = sha256_file(output)
            second = _build_tnt_reference(
                archive_path, crop_path, output, metadata, force=True, max_points=None
            )
            loaded = trimesh.load(output, process=False)
        self.assertEqual(first["stats"]["source_points"], 6)
        self.assertEqual(first["stats"]["points_after_official_crop_before_deduplication"], 5)
        self.assertEqual(first["stats"]["global_voxel_points"], 3)
        self.assertEqual(first["stats"]["exported_points"], 3)
        self.assertEqual(first_hash, second["output"]["sha256"])
        self.assertEqual(len(loaded.vertices), 3)

    def test_prediction_override_requires_complete_audited_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "outputs" / "unit-a"
            package_dir = candidate / "prediction_official"
            mesh = package_dir / "mesh.ply"
            glb = package_dir / "scene.glb"
            package = package_dir / "manifest.json"
            reconstruction = candidate / "reconstruction"
            source_mesh = reconstruction / "mesh.ply"
            source_glb = reconstruction / "scene.glb"
            input_manifest = root / "inputs" / "unit-a" / "manifest.json"
            plan = {
                "prediction_overrides": {
                    "unit-a": {
                        "mesh": "outputs/unit-a/prediction_official/mesh.ply",
                        "package_manifest": "outputs/unit-a/prediction_official/manifest.json",
                        "track": "GT-pose-foundation-pseudo-geometry",
                    }
                }
            }
            with patch.object(gt_builder, "ROOT", root):
                self.assertIsNone(validate_prediction_override(plan, "unit-a"))
                mesh.parent.mkdir(parents=True)
                mesh.write_bytes(b"mesh")
                with self.assertRaisesRegex(CalibrationBuildError, "Incomplete prediction"):
                    validate_prediction_override(plan, "unit-a")
                glb.write_bytes(b"glTF")
                reconstruction.mkdir(parents=True)
                source_mesh.write_bytes(b"source-mesh")
                source_glb.write_bytes(b"source-glb")
                input_manifest.parent.mkdir(parents=True)
                unit_dir = root / "units" / "unit-a"
                camera = unit_dir / "cameras.json"
                rgb_paths = [
                    unit_dir / "rgb" / "conditioning" / f"{index:03d}.png"
                    for index in range(8)
                ]
                camera.parent.mkdir(parents=True)
                camera.write_bytes(b"camera-metadata")
                for index, path in enumerate(rgb_paths):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(f"conditioning-rgb-{index}".encode())
                unit_manifest = unit_dir / "manifest.json"
                unit_document = {
                    "unit_id": "unit-a",
                    "status": "prepared",
                    "input": {
                        "cameras": "cameras.json",
                        "conditioning_views": [
                            str(path.relative_to(unit_dir)) for path in rgb_paths
                        ],
                    },
                }
                write_json(unit_manifest, unit_document)
                conditioning_contract = gt_builder._conditioning_manifest_contract_sha256(
                    unit_document
                )
                source_records = [
                    {
                        "path": str(unit_manifest.relative_to(root)),
                        "role": "unit-manifest",
                        "size_bytes_at_read": unit_manifest.stat().st_size,
                        "sha256_at_read": sha256_file(unit_manifest),
                        "conditioning_contract_sha256": conditioning_contract,
                    },
                    {
                        "path": str(camera.relative_to(root)),
                        "role": "conditioning-camera-metadata",
                        "size_bytes_at_read": camera.stat().st_size,
                        "sha256_at_read": sha256_file(camera),
                    },
                    *[
                        {
                            "path": str(path.relative_to(root)),
                            "role": "conditioning-rgb",
                            "size_bytes_at_read": path.stat().st_size,
                            "sha256_at_read": sha256_file(path),
                        }
                        for path in rgb_paths
                    ],
                ]
                asset_names = [
                    "colmap_vggt/cameras.txt",
                    "colmap_vggt/images.txt",
                    "colmap_vggt/points3D.txt",
                    *(f"rgb/{index:03d}.png" for index in range(8)),
                ]
                asset_records = []
                for name in asset_names:
                    path = input_manifest.parent / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(name.encode())
                    asset_records.append(
                        {
                            "path": name,
                            "size_bytes": path.stat().st_size,
                            "sha256": sha256_file(path),
                        }
                    )
                input_document = {
                    "schema": "genrecon.gt-representative-foundation-input",
                    "schema_version": 1,
                    "unit_id": "unit-a",
                    "track": "GT-pose-foundation-pseudo-geometry",
                    "source_manifest": str(unit_manifest.relative_to(root)),
                    "source_manifest_conditioning_contract_sha256": conditioning_contract,
                    "source_audit": {
                        "source_files_read": source_records,
                        "heldout_rgb_paths_read": [],
                        "depth_paths_read": [],
                        "reference_paths_read": [],
                        "conditioning_camera_records_used": 8,
                        "heldout_camera_records_used": 0,
                    },
                    "genrecon_input_assets": asset_records,
                    "inference": {"input_views": 8, "elapsed_seconds": 1.0},
                    "work_frame": {"work_to_official": np.eye(4).tolist()},
                }
                write_json(input_manifest, input_document)
                input_contract = gt_builder._representative_input_contract_sha256(
                    input_document
                )
                write_json(
                    package,
                    {
                        "schema": "genrecon.gt-representative-prediction-package",
                        "schema_version": 1,
                        "status": "pass",
                        "unit_id": "unit-a",
                        "track": "GT-pose-foundation-pseudo-geometry",
                        "coordinate_frame": "declared calibration/reference frame",
                        "alignment": {
                            "method": "conditioning-camera-only Sim(3), followed by exact work-to-official inverse",
                            "work_to_official": np.eye(4).tolist(),
                            "gt_geometry_icp_used": False,
                            "reference_geometry_used": False,
                            "heldout_views_used": False,
                        },
                        "source": {
                            "reconstruction": str(reconstruction.relative_to(root)),
                            "sha256": {
                                "mesh": sha256_file(source_mesh),
                                "glb": sha256_file(source_glb),
                            },
                            "input_manifest": str(input_manifest.relative_to(root)),
                            "input_manifest_contract_sha256": input_contract,
                        },
                        "outputs": {
                            "mesh": {
                                "path": "mesh.ply",
                                "size_bytes": mesh.stat().st_size,
                                "sha256": sha256_file(mesh),
                            },
                            "glb": {
                                "path": "scene.glb",
                                "size_bytes": glb.stat().st_size,
                                "sha256": sha256_file(glb),
                            },
                        },
                    },
                )
                result = validate_prediction_override(plan, "unit-a")
                self.assertEqual(result["mesh"], mesh)
                self.assertEqual(result["glb"], glb)
                self.assertEqual(result["track"], "GT-pose-foundation-pseudo-geometry")
                glb.write_bytes(b"changed")
                with self.assertRaisesRegex(CalibrationBuildError, "Invalid prediction"):
                    validate_prediction_override(plan, "unit-a")
                glb.write_bytes(b"glTF")
                source_glb.write_bytes(b"changed-source")
                with self.assertRaisesRegex(CalibrationBuildError, "Invalid prediction"):
                    validate_prediction_override(plan, "unit-a")
                source_glb.write_bytes(b"source-glb")
                points = input_manifest.parent / "colmap_vggt" / "points3D.txt"
                original_points = points.read_bytes()
                points.write_bytes(b"changed-points3D")
                gt_builder._HASH_CACHE.clear()
                with self.assertRaisesRegex(CalibrationBuildError, "GenRecon input asset changed"):
                    validate_prediction_override(plan, "unit-a")
                points.write_bytes(original_points)
                camera.write_bytes(b"changed-camera-metadata")
                gt_builder._HASH_CACHE.clear()
                with self.assertRaisesRegex(
                    CalibrationBuildError, "audited conditioning source changed"
                ):
                    validate_prediction_override(plan, "unit-a")
                camera.write_bytes(b"camera-metadata")
                gt_builder._HASH_CACHE.clear()
                document = json.loads(package.read_text())
                document["alignment"]["work_to_official"][0][3] = 1.0
                write_json(package, document)
                with self.assertRaisesRegex(CalibrationBuildError, "Invalid prediction"):
                    validate_prediction_override(plan, "unit-a")
                document["alignment"]["work_to_official"] = np.eye(4).tolist()
                document["alignment"]["reference_geometry_used"] = True
                write_json(package, document)
                with self.assertRaisesRegex(CalibrationBuildError, "Invalid prediction"):
                    validate_prediction_override(plan, "unit-a")

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
