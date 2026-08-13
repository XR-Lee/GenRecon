from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.prepare_gt_representative_genrecon import (
    _white_background_foreground_mask,
    assign_visible_observations,
    camera_to_world_from_record,
    conditioning_manifest_contract_sha256,
    genrecon_input_asset_records,
    intrinsics_from_record,
    load_conditioning_contract,
    package_prediction,
    representative_input_contract_sha256,
    representative_quality_gate,
    trajectory_alignment,
    transform_binary_ply,
    transform_world_similarity,
    umeyama_similarity,
    validate_all,
    validate_genrecon_input_assets,
    validate_prediction_package,
    validate_source_audit,
    wrap_glb_with_transform,
)


class GroundTruthRepresentativeGenReconTests(unittest.TestCase):
    def test_rgb_foreground_mask_removes_only_border_connected_exact_white(self) -> None:
        rgb = np.full((9, 9, 3), 255, dtype=np.uint8)
        rgb[2:7, 2:7] = [20, 80, 40]
        rgb[4, 4] = 255
        rgb[0, 4] = [254, 254, 254]

        foreground = _white_background_foreground_mask(rgb)

        self.assertFalse(foreground[0, 0])
        self.assertTrue(foreground[2, 2])
        self.assertTrue(foreground[4, 4])
        self.assertTrue(foreground[0, 4])
        self.assertAlmostEqual(float(foreground.mean()), 26 / 81)

    def test_camera_schema_normalizes_world_to_camera_and_dict_intrinsics(self) -> None:
        world_to_camera = np.eye(4)
        world_to_camera[:3, 3] = [1.0, -2.0, 3.0]
        record = {
            "world_to_camera": world_to_camera.tolist(),
            "intrinsics": {
                "fx": 500.0,
                "fy": 510.0,
                "cx": 320.0,
                "cy": 240.0,
            },
        }
        c2w, diagnostic = camera_to_world_from_record(record, return_diagnostic=True)
        intrinsic = intrinsics_from_record(record, (640, 480))
        np.testing.assert_allclose(c2w[:3, 3], [-1.0, 2.0, -3.0])
        self.assertLess(diagnostic["nearest_so3_correction_frobenius"], 1e-12)
        np.testing.assert_allclose(intrinsic, [[500, 0, 320], [0, 510, 240], [0, 0, 1]])

    def test_pose_normalization_accepts_only_small_finite_precision_drift(self) -> None:
        pose = np.eye(4)
        pose[:3, :3] *= 0.9998
        normalized, diagnostic = camera_to_world_from_record(
            {"camera_to_world": pose.tolist()}, return_diagnostic=True
        )
        np.testing.assert_allclose(normalized[:3, :3], np.eye(3), atol=1e-12)
        self.assertGreater(diagnostic["nearest_so3_correction_frobenius"], 0.0)
        pose[:3, :3] *= 0.99
        with self.assertRaisesRegex(RuntimeError, "SO\\(3\\) normalization gate"):
            camera_to_world_from_record({"camera_to_world": pose.tolist()})

    def test_umeyama_recovers_orientation_preserving_similarity(self) -> None:
        source = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.2, 0.3, 1.0]]
        )
        angle = np.deg2rad(30.0)
        rotation = np.asarray(
            [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
        )
        target = 2.4 * (source @ rotation.T) + [3.0, -4.0, 1.5]
        result = umeyama_similarity(source, target)
        self.assertAlmostEqual(result["scale"], 2.4, places=10)
        np.testing.assert_allclose(result["rotation"], rotation, atol=1e-10)
        np.testing.assert_allclose(result["translation"], [3.0, -4.0, 1.5], atol=1e-10)
        self.assertLess(float(np.max(result["errors"])), 1e-10)

    def test_world_similarity_preserves_pinhole_projections(self) -> None:
        points = np.asarray([[0.1, -0.2, 3.0], [0.6, 0.3, 4.0], [-0.4, 0.2, 2.5]])
        extrinsic = np.zeros((2, 3, 4), dtype=np.float64)
        extrinsic[:, :3, :3] = np.eye(3)
        extrinsic[1, :3, 3] = [-0.4, 0.0, 0.0]
        angle = np.deg2rad(-20.0)
        rotation = np.asarray(
            [[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]]
        )
        transformed_points, transformed_extrinsic = transform_world_similarity(
            points, extrinsic, 1.7, rotation, np.asarray([2.0, -1.0, 0.5])
        )
        for old, new in zip(extrinsic, transformed_extrinsic):
            camera_old = points @ old[:3, :3].T + old[:3, 3]
            camera_new = transformed_points @ new[:3, :3].T + new[:3, 3]
            np.testing.assert_allclose(
                camera_old[:, :2] / camera_old[:, 2:3],
                camera_new[:, :2] / camera_new[:, 2:3],
                atol=1e-10,
            )

    def test_trajectory_alignment_uses_conditioning_centers_and_rotations(self) -> None:
        c2w_predicted = np.repeat(np.eye(4)[None], 4, axis=0)
        c2w_predicted[:, :3, 3] = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        extrinsic = np.repeat(np.eye(4)[None, :3], 4, axis=0)
        extrinsic[:, :3, :3] = np.transpose(c2w_predicted[:, :3, :3], (0, 2, 1))
        extrinsic[:, :3, 3] = -c2w_predicted[:, :3, 3]
        official = c2w_predicted.copy()
        official[:, :3, 3] = 3.0 * official[:, :3, 3] + [4.0, 5.0, 6.0]
        result = trajectory_alignment(extrinsic, official)
        self.assertAlmostEqual(result["scale"], 3.0)
        self.assertLess(result["center_error_normalized_by_baseline"]["max"], 1e-10)
        self.assertLess(result["rotation_error_deg"]["max"], 1e-8)

    def test_trajectory_alignment_uses_normalized_object_labels_without_meter_keys(self) -> None:
        c2w = np.repeat(np.eye(4)[None], 4, axis=0)
        c2w[:, :3, 3] = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        extrinsic = np.repeat(np.eye(4)[None, :3], 4, axis=0)
        extrinsic[:, :3, 3] = -c2w[:, :3, 3]
        result = trajectory_alignment(
            extrinsic, c2w, coordinate_units="normalized-object"
        )

        def keys(value: object) -> list[str]:
            if isinstance(value, dict):
                return [
                    *value.keys(),
                    *(item for child in value.values() for item in keys(child)),
                ]
            if isinstance(value, list):
                return [item for child in value for item in keys(child)]
            return []

        self.assertEqual(result["coordinate_units"], "normalized-object")
        self.assertIn("target_baseline_diagonal", result)
        self.assertIn("center_error", result)
        self.assertFalse(any(name.endswith("_m") for name in keys(result)))

    def test_conditioning_contract_does_not_resolve_forbidden_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unit = root / "unit-a"
            (unit / "rgb").mkdir(parents=True)
            conditioning = []
            heldout = []
            for index in range(8):
                rgb = unit / "rgb" / f"c{index}.png"
                Image.new("RGB", (10, 8), (index, 20, 30)).save(rgb)
                conditioning.append(
                    {
                        "order": index,
                        "rgb": f"rgb/c{index}.png",
                        "camera_to_world": np.eye(4).tolist(),
                        "intrinsics": {"fx": 8, "fy": 8, "cx": 5, "cy": 4},
                    }
                )
                heldout.append({"order": index, "rgb": f"missing/heldout-{index}.png"})
            (unit / "cameras.json").write_text(
                json.dumps({"conditioning": conditioning, "heldout": heldout})
            )
            (unit / "manifest.json").write_text(
                json.dumps(
                    {
                        "unit_id": "unit-a",
                        "status": "prepared",
                        "input": {
                            "cameras": "cameras.json",
                            "conditioning_views": [f"rgb/c{i}.png" for i in range(8)],
                            "heldout_views": [f"missing/heldout-{i}.png" for i in range(8)],
                            "conditioning_depths": ["missing/depth.png"],
                        },
                        "reference": {"paths": ["missing/reference.ply"]},
                    }
                )
            )
            contract = load_conditioning_contract("unit-a", root)
        self.assertEqual(len(contract["views"]), 8)
        self.assertEqual(len(contract["source_files_read"]), 10)
        self.assertEqual(contract["declared_but_forbidden"]["reference_geometry"], ["missing/reference.ply"])

    def test_visible_observation_assignment_uses_only_projecting_cameras(self) -> None:
        points = np.asarray([[0.0, 0.0, 2.0], [5.0, 0.0, 2.0]])
        intrinsic = np.repeat(
            np.asarray([[[8.0, 0.0, 5.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]]]),
            2,
            axis=0,
        )
        extrinsic = np.repeat(np.eye(4)[None, :3], 2, axis=0)
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for index in range(2):
                path = Path(temporary) / f"{index}.png"
                Image.new("RGB", (10, 8), (10 + index, 20, 30)).save(path)
                paths.append(path)
            result = assign_visible_observations(points, intrinsic, extrinsic, paths)
        np.testing.assert_array_equal(result["keep"], [True, False])
        self.assertEqual(result["source_frame"].tolist(), [0])

    def test_representative_gate_marks_sparse_overlap_as_marginal(self) -> None:
        protocol = {
            "max_center_error_p90_baseline": 0.25,
            "max_rotation_error_p90_deg": 35.0,
            "minimum_official_camera_visible_fraction": 0.5,
            "minimum_cross_view_verified_fraction": 0.1,
            "low_overlap_max_opportunity_fraction": 0.05,
            "low_overlap_minimum_verified_points": 1000,
            "low_overlap_minimum_verified_fraction_given_overlap": 0.5,
        }
        trajectory = {
            "center_error_normalized_by_baseline": {"p90": 0.14},
            "rotation_error_deg": {"p90": 8.1},
        }
        sparse = {
            "cross_view_verified_fraction": 0.008,
            "cross_view_overlap_opportunity_fraction": 0.013,
            "cross_view_verified_fraction_given_overlap": 0.68,
            "cross_view_verified_points": 2100,
        }
        gate = representative_quality_gate(sparse, trajectory, 0.97, protocol)
        self.assertEqual(gate["grade"], "P-C")
        self.assertEqual(gate["decision"], "marginal")
        self.assertTrue(gate["low_overlap_conditional_consistency_pass"])
        sparse["cross_view_verified_points"] = 999
        failed = representative_quality_gate(sparse, trajectory, 0.97, protocol)
        self.assertEqual(failed["decision"], "fail")
        self.assertIn("cross_view_verified_fraction_too_low", failed["reasons"])

    def test_representative_input_contract_ignores_runtime_but_not_geometry(self) -> None:
        document = {
            "inference": {
                "input_views": 8,
                "elapsed_seconds": 1.2,
                "peak_memory_mib": 100.0,
            },
            "model": {
                "checkpoint_sha256": "abc",
                "checkpoint_path": "/machine/a/checkpoint.pt",
            },
            "source_manifest_sha256_at_read": "before-registration",
            "source_audit": {
                "source_files_read": [
                    {
                        "role": "unit-manifest",
                        "size_bytes_at_read": 100,
                        "sha256_at_read": "before-registration",
                        "conditioning_contract_sha256": "condition-contract",
                    },
                    {"role": "conditioning-rgb", "sha256_at_read": "rgb-a"},
                ]
            },
            "work_frame": {"work_to_official": np.eye(4).tolist()},
        }
        expected = representative_input_contract_sha256(document)
        replay = json.loads(json.dumps(document))
        replay["inference"]["elapsed_seconds"] = 9.9
        replay["inference"]["peak_memory_mib"] = 200.0
        replay["model"]["checkpoint_path"] = "/machine/b/checkpoint.pt"
        replay["source_manifest_sha256_at_read"] = "after-registration"
        replay["source_audit"]["source_files_read"][0]["size_bytes_at_read"] = 200
        replay["source_audit"]["source_files_read"][0]["sha256_at_read"] = "after-registration"
        self.assertEqual(representative_input_contract_sha256(replay), expected)
        replay["source_audit"]["source_files_read"][1]["sha256_at_read"] = "rgb-b"
        self.assertNotEqual(representative_input_contract_sha256(replay), expected)
        replay = json.loads(json.dumps(document))
        replay["work_frame"]["work_to_official"][0][3] = 1.0
        self.assertNotEqual(representative_input_contract_sha256(replay), expected)

    def test_prediction_registration_is_excluded_from_conditioning_contract_hash(self) -> None:
        manifest = {
            "unit_id": "unit-a",
            "status": "prepared",
            "input": {"cameras": "cameras.json", "conditioning_views": ["000.png"]},
            "reference": {"paths": ["reference.ply"]},
        }
        expected = conditioning_manifest_contract_sha256(manifest)
        registered = json.loads(json.dumps(manifest))
        registered["prediction_mesh"] = "prediction.ply"
        registered["prediction_provenance"] = {"track": "fixture"}
        self.assertEqual(conditioning_manifest_contract_sha256(registered), expected)
        registered["input"]["cameras"] = "other-cameras.json"
        self.assertNotEqual(conditioning_manifest_contract_sha256(registered), expected)

    def test_source_audit_detects_conditioning_file_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            camera_path = root / "cameras.json"
            rgb_paths = [root / f"{index:03d}.png" for index in range(8)]
            manifest = {
                "unit_id": "unit-a",
                "prediction_mesh": None,
                "input": {
                    "cameras": camera_path.name,
                    "conditioning_views": [path.name for path in rgb_paths],
                },
            }
            manifest_path.write_text(json.dumps(manifest))
            camera_path.write_text(json.dumps({"conditioning": []}))
            for index, path in enumerate(rgb_paths):
                path.write_bytes(f"rgb-{index}".encode())
            contract_hash = conditioning_manifest_contract_sha256(manifest)
            audit = {
                "source_files_read": [
                    {
                        "path": str(manifest_path),
                        "role": "unit-manifest",
                        "size_bytes_at_read": manifest_path.stat().st_size,
                        "sha256_at_read": "not-used-for-contract",
                        "conditioning_contract_sha256": contract_hash,
                    },
                    {
                        "path": str(camera_path),
                        "role": "conditioning-camera-metadata",
                        "size_bytes_at_read": camera_path.stat().st_size,
                        "sha256_at_read": __import__("hashlib").sha256(
                            camera_path.read_bytes()
                        ).hexdigest(),
                    },
                    *[
                        {
                            "path": str(path),
                            "role": "conditioning-rgb",
                            "size_bytes_at_read": path.stat().st_size,
                            "sha256_at_read": __import__("hashlib").sha256(
                                path.read_bytes()
                            ).hexdigest(),
                        }
                        for path in rgb_paths
                    ],
                ]
            }
            input_manifest = {
                "source_manifest": str(manifest_path),
                "source_manifest_conditioning_contract_sha256": contract_hash,
            }
            validate_source_audit(input_manifest, audit)
            camera_path.write_text(json.dumps({"conditioning": [1]}))
            with self.assertRaisesRegex(RuntimeError, "conditioning source changed"):
                validate_source_audit(input_manifest, audit)

    def test_source_audit_rejects_substituted_rgb_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unit = root / "unit"
            unit.mkdir()
            camera = unit / "cameras.json"
            camera.write_text("{}")
            declared = [unit / f"declared-{index}.png" for index in range(8)]
            substituted = [unit / f"substituted-{index}.png" for index in range(8)]
            for path in [*declared, *substituted]:
                path.write_bytes(b"rgb")
            manifest = {
                "unit_id": "unit-a",
                "input": {
                    "cameras": camera.name,
                    "conditioning_views": [path.name for path in declared],
                },
            }
            manifest_path = unit / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            contract_hash = conditioning_manifest_contract_sha256(manifest)
            audit = {
                "source_files_read": [
                    {
                        "path": str(manifest_path),
                        "role": "unit-manifest",
                        "conditioning_contract_sha256": contract_hash,
                    },
                    {
                        "path": str(camera),
                        "role": "conditioning-camera-metadata",
                        "size_bytes_at_read": camera.stat().st_size,
                        "sha256_at_read": __import__("hashlib").sha256(
                            camera.read_bytes()
                        ).hexdigest(),
                    },
                    *[
                        {
                            "path": str(path),
                            "role": "conditioning-rgb",
                            "size_bytes_at_read": path.stat().st_size,
                            "sha256_at_read": __import__("hashlib").sha256(
                                path.read_bytes()
                            ).hexdigest(),
                        }
                        for path in substituted
                    ],
                ]
            }
            input_manifest = {
                "source_manifest": str(manifest_path),
                "source_manifest_conditioning_contract_sha256": contract_hash,
            }
            with self.assertRaisesRegex(RuntimeError, "conditioning split"):
                validate_source_audit(input_manifest, audit)

    def test_genrecon_input_asset_contract_detects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scene = Path(temporary)
            for relative in (
                "colmap_vggt/cameras.txt",
                "colmap_vggt/images.txt",
                "colmap_vggt/points3D.txt",
                *(f"rgb/{index:03d}.png" for index in range(8)),
            ):
                path = scene / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(relative.encode())
            manifest = {"genrecon_input_assets": genrecon_input_asset_records(scene)}
            validate_genrecon_input_assets(scene, manifest)
            (scene / "colmap_vggt" / "points3D.txt").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "input asset changed"):
                validate_genrecon_input_assets(scene, manifest)

    def test_inference_validator_does_not_overwrite_asset_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "inputs"
            prediction_root = root / "predictions"
            scene = input_root / "candidates" / "unit-a"
            (scene / "genrecon_preflight").mkdir(parents=True)
            prediction_root.mkdir(parents=True)
            unit_manifest = prediction_root / "unit-manifest.json"
            camera_source = prediction_root / "cameras.json"
            rgb_sources = [prediction_root / f"rgb-{index}.png" for index in range(8)]
            source_document = {
                "unit_id": "unit-a",
                "status": "prepared",
                "input": {
                    "cameras": camera_source.name,
                    "conditioning_views": [path.name for path in rgb_sources],
                },
            }
            unit_manifest.write_text(json.dumps(source_document))
            camera_source.write_text("{}")
            for path in rgb_sources:
                path.write_bytes(b"rgb")
            contract_hash = conditioning_manifest_contract_sha256(source_document)
            source_records = [
                {
                    "path": str(unit_manifest),
                    "role": "unit-manifest",
                    "size_bytes_at_read": unit_manifest.stat().st_size,
                    "sha256_at_read": "not-used-for-contract",
                    "conditioning_contract_sha256": contract_hash,
                },
                {
                    "path": str(camera_source),
                    "role": "conditioning-camera-metadata",
                    "size_bytes_at_read": camera_source.stat().st_size,
                    "sha256_at_read": __import__("hashlib").sha256(
                        camera_source.read_bytes()
                    ).hexdigest(),
                },
                *[
                    {
                        "path": str(path),
                        "role": "conditioning-rgb",
                        "size_bytes_at_read": path.stat().st_size,
                        "sha256_at_read": __import__("hashlib").sha256(
                            path.read_bytes()
                        ).hexdigest(),
                    }
                    for path in rgb_sources
                ],
            ]
            (scene / "status.json").write_text(
                json.dumps({"status": "completed", "point_count": 123})
            )
            for relative in (
                "colmap_vggt/cameras.txt",
                "colmap_vggt/images.txt",
                "colmap_vggt/points3D.txt",
                *(f"rgb/{index:03d}.png" for index in range(8)),
            ):
                path = scene / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(relative.encode())
            (scene / "manifest.json").write_text(
                json.dumps(
                    {
                        "quality_gate": {"decision": "foundation_pass"},
                        "source_manifest": str(unit_manifest),
                        "source_manifest_conditioning_contract_sha256": contract_hash,
                        "source_audit": {
                            "heldout_rgb_paths_read": [],
                            "depth_paths_read": [],
                            "reference_paths_read": [],
                            "heldout_camera_records_used": 0,
                            "conditioning_camera_records_used": 8,
                            "source_files_read": source_records,
                        },
                        "genrecon_input_assets": genrecon_input_asset_records(scene),
                    }
                )
            )
            (scene / "genrecon_preflight" / "preflight.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "closest_camera_fallback_count": 0,
                        "chunk_count": 2,
                    }
                )
            )
            asset_validation = prediction_root / "validation.json"
            asset_validation.write_text(json.dumps({"schema": "asset-validator"}))

            result = validate_all(
                ["unit-a"], input_root, prediction_root, require_packages=False
            )

            self.assertEqual(result["result"], "pass")
            self.assertEqual(
                json.loads(asset_validation.read_text()), {"schema": "asset-validator"}
            )
            self.assertEqual(
                json.loads((prediction_root / "inference_validation.json").read_text())[
                    "schema"
                ],
                "genrecon.gt-representative-inference-validation",
            )

    def test_normalized_package_reuse_and_tamper_guards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "inputs"
            prediction_root = root / "predictions"
            input_scene = input_root / "candidates" / "unit-a"
            reconstruction = prediction_root / "candidates" / "unit-a" / "reconstruction"
            input_scene.mkdir(parents=True)
            reconstruction.mkdir(parents=True)
            (input_scene / "genrecon_preflight").mkdir()
            matrix = np.eye(4)
            camera_selection = {
                "policy": "frustum-and-minimum-projected-chunk-area",
                "minimum_projected_chunk_area": 0.2,
                "required_fallback_count": 0,
                "rationale": "small-object-frustum-visibility",
            }
            input_manifest = {
                "unit_id": "unit-a",
                "coordinate_units": "normalized-object",
                "genrecon_camera_selection": camera_selection,
                "work_frame": {"work_to_official": matrix.tolist()},
                "limitations": [],
            }
            (input_scene / "manifest.json").write_text(json.dumps(input_manifest))
            camera_document = {
                "scene": [],
                "chunks": [
                    {
                        "chunk_index": 0,
                        "cond2d_view": {
                            "selection_mode": "visible-projected-area",
                            "projected_chunk_area": 0.25,
                            "minimum_projected_chunk_area": 0.2,
                        },
                    }
                ],
            }
            preflight = {
                "status": "passed",
                "chunk_count": 1,
                "closest_camera_fallback_count": 0,
                "camera_selection": {
                    "policy": "frustum-and-minimum-projected-chunk-area",
                    "minimum_projected_chunk_area": 0.2,
                    "required_fallback_count": 0,
                    "selected_projected_chunk_area_min": 0.25,
                    "selected_projected_chunk_area_max": 0.25,
                    "selected_chunks_meeting_area_gate": 1,
                },
            }
            (input_scene / "genrecon_preflight" / "preflight.json").write_text(
                json.dumps(preflight)
            )
            (input_scene / "genrecon_preflight" / "cameras.json").write_text(
                json.dumps(camera_document)
            )
            (reconstruction / "args.json").write_text(
                json.dumps({"min_projected_chunk_area": 0.2})
            )
            (reconstruction / "cameras.json").write_text(
                json.dumps(camera_document)
            )

            header = (
                "ply\nformat binary_little_endian 1.0\n"
                "element vertex 1\nproperty float x\nproperty float y\nproperty float z\n"
                "element face 0\nproperty list uchar int vertex_indices\nend_header\n"
            ).encode("ascii")
            vertex = np.asarray([(1.0, 0.0, 0.0)], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
            (reconstruction / "mesh.ply").write_bytes(header + vertex.tobytes())
            document = {
                "asset": {"version": "2.0"},
                "scenes": [{"nodes": [0]}],
                "scene": 0,
                "nodes": [{"mesh": 0}],
                "meshes": [{}],
            }
            payload = json.dumps(document, separators=(",", ":")).encode("utf-8")
            payload += b" " * ((-len(payload)) % 4)
            total = 12 + 8 + len(payload)
            (reconstruction / "scene.glb").write_bytes(
                struct.pack("<4sII", b"glTF", 2, total)
                + struct.pack("<I4s", len(payload), b"JSON")
                + payload
            )

            first = package_prediction(
                "unit-a", input_root, prediction_root, force=True
            )
            input_manifest["work_frame"]["work_to_official"][0][3] = 5.0
            (input_scene / "manifest.json").write_text(json.dumps(input_manifest))
            second = package_prediction(
                "unit-a", input_root, prediction_root, force=False
            )

            self.assertEqual(
                first["coordinate_units"], "normalized-object"
            )
            self.assertNotEqual(
                first["source"]["input_manifest_contract_sha256"],
                second["source"]["input_manifest_contract_sha256"],
            )
            self.assertNotEqual(
                first["outputs"]["mesh"]["sha256"],
                second["outputs"]["mesh"]["sha256"],
            )
            self.assertEqual(second["alignment"]["work_to_official"][0][3], 5.0)

            package = (
                prediction_root
                / "candidates"
                / "unit-a"
                / "prediction_official"
            )
            package_manifest = package / "manifest.json"
            preflight_cameras = (
                input_scene / "genrecon_preflight" / "cameras.json"
            )
            reconstruction_args = reconstruction / "args.json"
            reconstruction_cameras = reconstruction / "cameras.json"
            self.assertEqual(
                validate_prediction_package(package, "unit-a"),
                package / "mesh.ply",
            )

            def assert_rejected(path, mutate, message) -> None:
                original = json.loads(path.read_text())
                changed = json.loads(path.read_text())
                mutate(changed)
                path.write_text(json.dumps(changed))
                try:
                    with self.assertRaisesRegex(RuntimeError, message):
                        validate_prediction_package(package, "unit-a")
                finally:
                    path.write_text(json.dumps(original))
                self.assertEqual(
                    validate_prediction_package(package, "unit-a"),
                    package / "mesh.ply",
                )

            assert_rejected(
                input_scene / "manifest.json",
                lambda value: value["genrecon_camera_selection"].__setitem__(
                    "minimum_projected_chunk_area", 0.3
                ),
                "input manifest changed",
            )
            assert_rejected(
                preflight_cameras,
                lambda value: value["chunks"][0]["cond2d_view"].__setitem__(
                    "selection_mode", "closest-camera-fallback"
                ),
                "Preflight chunk did not pass projected-area camera selection",
            )
            assert_rejected(
                reconstruction_args,
                lambda value: value.__setitem__("min_projected_chunk_area", 0.3),
                "source args changed",
            )
            assert_rejected(
                reconstruction_cameras,
                lambda value: value["chunks"][0]["cond2d_view"].__setitem__(
                    "selection_mode", "closest-camera-fallback"
                ),
                "source cameras changed",
            )
            assert_rejected(
                package_manifest,
                lambda value: value["source"]["camera_selection"].__setitem__(
                    "minimum_projected_chunk_area", 0.3
                ),
                "camera-selection contract mismatch",
            )
            for source_name in ("args", "cameras"):
                assert_rejected(
                    package_manifest,
                    lambda value, key=source_name: value["source"]["sha256"].__setitem__(
                        key, "0" * 64
                    ),
                    f"source {source_name} changed",
                )

    def test_representative_plan_includes_omni_video_unit(self) -> None:
        plan = json.loads(
            (Path(__file__).parents[1] / "configs/eval/gt_representative_inference_v1.json")
            .read_text(encoding="utf-8")
        )
        self.assertIn("omniobject3d-bottle_045", plan["unit_ids"])
        self.assertIn("coordinate-unit-preserving", plan["protocol"]["work_frame"])

    def test_binary_ply_and_glb_apply_same_matrix(self) -> None:
        matrix = np.eye(4)
        matrix[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        matrix[:3, 3] = [2.0, 3.0, 4.0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ply = root / "source.ply"
            header = (
                "ply\nformat binary_little_endian 1.0\n"
                "element vertex 1\nproperty float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                "element face 0\nproperty list uchar int vertex_indices\nend_header\n"
            ).encode("ascii")
            dtype = np.dtype(
                [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")]
            )
            vertex = np.zeros(1, dtype=dtype)
            vertex["x"], vertex["y"], vertex["z"] = 1.0, 0.0, 0.0
            ply.write_bytes(header + vertex.tobytes())
            output_ply = root / "output.ply"
            transform_binary_ply(ply, output_ply, matrix)
            with output_ply.open("rb") as handle:
                while handle.readline() != b"end_header\n":
                    pass
                transformed = np.frombuffer(handle.read(dtype.itemsize), dtype=dtype)
            np.testing.assert_allclose(
                [transformed["x"][0], transformed["y"][0], transformed["z"][0]],
                [2.0, 4.0, 4.0],
            )

            document = {"asset": {"version": "2.0"}, "scenes": [{"nodes": [0]}], "scene": 0, "nodes": [{"mesh": 0}], "meshes": [{}]}
            payload = json.dumps(document, separators=(",", ":")).encode("utf-8")
            payload += b" " * ((-len(payload)) % 4)
            glb = root / "source.glb"
            total = 12 + 8 + len(payload)
            glb.write_bytes(struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(payload), b"JSON") + payload)
            output_glb = root / "output.glb"
            wrap_glb_with_transform(glb, output_glb, matrix)
            with output_glb.open("rb") as handle:
                handle.read(12)
                length, _ = struct.unpack("<I4s", handle.read(8))
                updated = json.loads(handle.read(length).rstrip(b" \0"))
            stored = np.asarray(updated["nodes"][-1]["matrix"]).reshape(4, 4).T
            np.testing.assert_allclose(stored, matrix)
            self.assertEqual(updated["scenes"][0]["nodes"], [1])


if __name__ == "__main__":
    unittest.main()
