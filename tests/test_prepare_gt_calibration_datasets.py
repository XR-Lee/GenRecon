from __future__ import annotations

import json
import tarfile
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import trimesh
from PIL import Image

try:
    import OpenEXR
except ModuleNotFoundError:
    OpenEXR = None

import tools.prepare_gt_calibration_datasets as gt_builder
from tools.prepare_gt_calibration_datasets import (
    CalibrationBuildError,
    _backproject_depth,
    _build_tnt_reference,
    _omni_blender_c2w_to_opencv,
    _omni_blender_c2w_to_opencv_with_audit,
    _omni_intrinsic,
    _omni_raycast_mask,
    _omni_raycast_scene,
    _omni_scan_to_render_mesh,
    _omni_select_render_scale,
    _parse_dtu_camera,
    _parse_pose,
    _parse_tnt_alignment,
    _read_redwood_poses,
    _transform_tnt_camera_pose,
    _validate_existing_frozen_hashes,
    _validate_omni_prepared_package,
    safe_tar_members,
    safe_zip_names,
    sha256_file,
    farthest_point_indices,
    prepare_omniobject3d,
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

    def test_farthest_point_indices_are_deterministic_and_disjoint(self) -> None:
        points = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )
        selected = farthest_point_indices(points, 3)
        heldout = farthest_point_indices(points, 3, excluded=set(selected))
        self.assertEqual(selected, farthest_point_indices(points, 3))
        self.assertEqual(len(selected), 3)
        self.assertEqual(len(heldout), 3)
        self.assertFalse(set(selected) & set(heldout))

    def test_tar_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.tar.gz"
            with tarfile.open(path, "w:gz") as archive:
                info = tarfile.TarInfo("../escape.txt")
                payload = b"bad"
                info.size = len(payload)
                archive.addfile(info, BytesIO(payload))
            with tarfile.open(path, "r:gz") as archive:
                with self.assertRaisesRegex(CalibrationBuildError, "Unsafe TAR"):
                    safe_tar_members(archive)

    def test_omni_pose_and_scan_normalization_match_declared_frame(self) -> None:
        blender = np.eye(4)
        blender[:3, 3] = [1.0, 2.0, 3.0]
        opencv = _omni_blender_c2w_to_opencv(blender)
        np.testing.assert_array_equal(opencv[:3, 0], [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(opencv[:3, 1], [0.0, -1.0, 0.0])
        np.testing.assert_array_equal(opencv[:3, 2], [0.0, 0.0, -1.0])
        mesh = trimesh.Trimesh(
            vertices=[[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 1.0]],
            faces=[[0, 1, 2]],
            process=False,
        )
        normalized, record = _omni_scan_to_render_mesh(mesh, 0.1)
        self.assertAlmostEqual(record["selected_uniform_scale"], 0.1)
        self.assertAlmostEqual(np.abs(normalized.vertices).max(), 0.4)
        np.testing.assert_allclose(
            normalized.vertices,
            [[0.2, 0.0, 0.0], [0.0, 0.0, 0.4], [0.0, -0.1, 0.0]],
        )

    def test_omni_scale_selection_only_recovers_impossible_recorded_scale(self) -> None:
        mesh = trimesh.Trimesh(
            vertices=[[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 1.0]],
            faces=[[0, 1, 2]],
            process=False,
        )
        recorded, audit = _omni_select_render_scale(mesh, 0.1)
        self.assertEqual(recorded, 0.1)
        self.assertTrue(audit["recorded_scale_used"])
        recovered, audit = _omni_select_render_scale(mesh, 10.0)
        self.assertAlmostEqual(recovered, 0.99 / 4.0)
        self.assertFalse(audit["recorded_scale_used"])
        self.assertAlmostEqual(audit["selected_output_max_abs_coordinate"], 0.99)

    def test_omni_pose_repair_is_bounded_and_audited(self) -> None:
        blender = np.eye(4)
        blender[:3, :3] = np.asarray(
            [
                [-0.7381741404533386, -0.6746102571487427, 0.00011377158079994842],
                [0.6746102571487427, -0.7381741404533386, 0.00012449148925952613],
                [0.0, 0.0, 0.9999999403953552],
            ]
        )
        repaired, audit = _omni_blender_c2w_to_opencv_with_audit(blender)
        self.assertEqual(audit["method"], "nearest-SO3-SVD")
        self.assertLess(audit["max_abs_rotation_correction"], 1e-4)
        np.testing.assert_allclose(
            repaired[:3, :3].T @ repaired[:3, :3], np.eye(3), atol=1e-12
        )
        malformed = blender.copy()
        malformed[0, 0] += 0.01
        with self.assertRaisesRegex(
            CalibrationBuildError, "exceeds bounded SO\\(3\\) repair limits"
        ):
            _omni_blender_c2w_to_opencv_with_audit(malformed)

    def test_omni_raycast_mask_matches_pinhole_camera(self) -> None:
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        scene = _omni_raycast_scene(mesh)
        pose = np.eye(4)
        pose[:3, 3] = [0.0, 0.0, -3.0]
        intrinsic = _omni_intrinsic(0.8, (64, 64))
        mask = _omni_raycast_mask(scene, pose, intrinsic, (64, 64))
        self.assertEqual(mask.shape, (64, 64))
        self.assertGreater(int(mask.sum()), 0)
        self.assertLess(int(mask.sum()), mask.size)
        self.assertTrue(mask[32, 32])
        self.assertFalse(mask[0, 0])

    def test_omni_fixture_builds_prepared_package_and_isolates_missing_member(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "calibration"
            source = output / "sources" / "omniobject3d"
            repo = source / "downloads" / "OpenXDLab___OmniObject3D-New"
            image_archive = repo / "raw" / "blender_renders" / "fixture.tar.gz"
            scan_archive = repo / "raw" / "raw_scans" / "fixture.tar.gz"
            image_archive.parent.mkdir(parents=True)
            scan_archive.parent.mkdir(parents=True)

            mesh = trimesh.Trimesh(
                vertices=[
                    [-1.0, -1.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [-1.0, 1.0, 0.0],
                ],
                faces=[[0, 1, 2], [0, 2, 3]],
                process=False,
            )
            obj = trimesh.exchange.obj.export_obj(mesh).encode()
            focal = 80.0
            angle = 2.0 * np.arctan(32.0 / focal)
            frames = []
            rgb_payloads = {}
            normal_payloads = {}
            depth_payloads = {}
            for index, theta in enumerate(
                np.linspace(0.0, 2.0 * np.pi, 100, endpoint=False)
            ):
                center = np.asarray(
                    [1.2 * np.cos(theta), 1.2 * np.sin(theta), 0.8]
                )
                forward = -center / np.linalg.norm(center)
                right = np.cross(forward, [0.0, 0.0, 1.0])
                right /= np.linalg.norm(right)
                down = np.cross(forward, right)
                c2w = np.eye(4)
                c2w[:3, :3] = np.column_stack((right, -down, -forward))
                c2w[:3, 3] = center
                name = f"r_{index}"
                frames.append(
                    {
                        "file_path": name,
                        "transform_matrix": c2w.tolist(),
                        "scale": 0.1,
                    }
                )
                rgb = np.zeros((64, 64, 3), dtype=np.uint8)
                rgb[12:52, 12:52] = [0, 120, 140]
                normal = np.zeros((64, 64, 4), dtype=np.uint8)
                normal[12:52, 12:52, :3] = [128, 128, 255]
                normal[..., 3] = 255
                for values, destination in (
                    (rgb, rgb_payloads),
                    (normal, normal_payloads),
                ):
                    payload = BytesIO()
                    Image.fromarray(values).save(payload, format="PNG")
                    destination[name] = payload.getvalue()
                if OpenEXR is not None:
                    depth = np.full((64, 64, 3), 65504.0, dtype=np.float32)
                    depth[12:52, 12:52] = 3.0
                    with tempfile.NamedTemporaryFile(suffix=".exr") as temporary:
                        OpenEXR.File({}, {"RGB": depth}).write(temporary.name)
                        depth_payloads[name] = Path(temporary.name).read_bytes()
                else:
                    depth_payloads[name] = b"OpenEXR-unavailable"
            transforms = {"camera_angle_x": float(angle), "frames": frames}

            def write_image_archive(*, omit_depth: str | None = None) -> None:
                with tarfile.open(image_archive, "w:gz") as archive:
                    entries = {
                        "fixture_001/render/transforms.json": json.dumps(
                            transforms
                        ).encode(),
                    }
                    for name in rgb_payloads:
                        entries[f"fixture_001/render/images/{name}.png"] = (
                            rgb_payloads[name]
                        )
                        entries[
                            f"fixture_001/render/normals/{name}_normal.png"
                        ] = normal_payloads[name]
                        if name != omit_depth:
                            entries[
                                f"fixture_001/render/depths/{name}_depth.exr"
                            ] = depth_payloads[name]
                    for member_name, payload in entries.items():
                        info = tarfile.TarInfo(member_name)
                        info.size = len(payload)
                        archive.addfile(info, BytesIO(payload))

            write_image_archive()
            with tarfile.open(scan_archive, "w:gz") as archive:
                info = tarfile.TarInfo("fixture_001/Scan/Scan.obj")
                info.size = len(obj)
                archive.addfile(info, BytesIO(obj))

            index_path = source / "metadata" / "openxlab_file_index.json"

            def write_file_index() -> None:
                write_json(
                    index_path,
                    {
                        "schema": "genrecon.omniobject3d-openxlab-file-index",
                        "dataset_repo": "OpenXDLab/OmniObject3D-New",
                        "files": [
                            {
                                "path": "/raw/blender_renders/fixture.tar.gz",
                                "size": image_archive.stat().st_size,
                                "sha256": sha256_file(image_archive),
                            },
                            {
                                "path": "/raw/raw_scans/fixture.tar.gz",
                                "size": scan_archive.stat().st_size,
                                "sha256": sha256_file(scan_archive),
                            },
                        ],
                    },
                )

            write_file_index()
            plan = {
                "datasets": {
                    "omniobject3d": {
                        "track": "instance",
                        "gt_tier": "O0-instance-scan",
                        "capture_kind": "scan-derived-rendered-object-multiview-and-real-object-scan",
                        "license_status": "CC-BY-4.0",
                        "source_url": "https://example.invalid",
                        "unit_ids": ["fixture_001"],
                    }
                }
            }
            if OpenEXR is None:
                self.skipTest("OpenEXR is required for the OmniObject3D fixture")
            with patch.object(
                gt_builder, "OMNI_MAX_SILHOUETTE_EDGE_ERROR_PX", 1e6
            ), patch.object(
                gt_builder, "OMNI_MIN_SILHOUETTE_BBOX_IOU", 0.0
            ), patch.object(gt_builder, "OMNI_RENDER_RESOLUTION", 64):
                rows = prepare_omniobject3d(
                    plan, output, output / "sources", force=True
                )
            manifest = json.loads(
                (output / rows[0]["manifest"]).read_text(encoding="utf-8")
            )
            unit = output / "units" / rows[0]["unit_id"]
            camera = json.loads((unit / "cameras.json").read_text())
            alignment = json.loads((unit / "scan_render_alignment.json").read_text())
            with Image.open(unit / "rgb" / "conditioning" / "000.png") as image:
                rendered = np.asarray(image)

            self.assertEqual(rows[0]["status"], "prepared")
            self.assertEqual(manifest["source"]["object_id"], "fixture_001")
            self.assertEqual(manifest["source"]["adapter_version"], 3)
            self.assertEqual(
                manifest["reference"]["coordinate_units"], "normalized-object"
            )
            self.assertEqual(len(manifest["input"]["conditioning_views"]), 8)
            self.assertEqual(len(manifest["input"]["heldout_views"]), 8)
            self.assertEqual(len(camera["conditioning"]), 8)
            self.assertEqual(len(camera["heldout"]), 8)
            selected = {
                item["source_index"]
                for item in camera["conditioning"] + camera["heldout"]
            }
            self.assertEqual(len(selected), 16)
            self.assertEqual(alignment["view_count"], 100)
            self.assertTrue(np.all(rendered[:10, :10] == 255))
            self.assertEqual(rendered[20, 20].tolist(), [0, 120, 140])
            with patch.object(
                gt_builder, "OMNI_MAX_SILHOUETTE_EDGE_ERROR_PX", 1e6
            ), patch.object(
                gt_builder, "OMNI_MIN_SILHOUETTE_BBOX_IOU", 0.0
            ), patch.object(gt_builder, "OMNI_RENDER_RESOLUTION", 64):
                _validate_omni_prepared_package(
                    manifest, unit / "manifest.json", camera
                )
                alignment["fast_projection_pass_count"] -= 1
                write_json(unit / "scan_render_alignment.json", alignment)
                with self.assertRaisesRegex(
                    CalibrationBuildError, "alignment summary is stale"
                ):
                    _validate_omni_prepared_package(
                        manifest, unit / "manifest.json", camera
                    )
                alignment["fast_projection_pass_count"] += 1
                write_json(unit / "scan_render_alignment.json", alignment)

            write_image_archive(omit_depth="r_99")
            gt_builder._HASH_CACHE.clear()
            write_file_index()
            with patch.object(
                gt_builder, "OMNI_MAX_SILHOUETTE_EDGE_ERROR_PX", 1e6
            ), patch.object(
                gt_builder, "OMNI_MIN_SILHOUETTE_BBOX_IOU", 0.0
            ), patch.object(gt_builder, "OMNI_RENDER_RESOLUTION", 64):
                rows = prepare_omniobject3d(
                    plan, output, output / "sources", force=True
                )
            failed = json.loads(
                (output / rows[0]["manifest"]).read_text(encoding="utf-8")
            )
            self.assertEqual(rows[0]["status"], "source-incomplete")
            self.assertIn("found 0", failed["blocker"])
            self.assertEqual(failed["reference"]["paths"], [])
            self.assertFalse((unit / "reference" / "scan_normalized.ply").exists())

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
