import json
import struct
import tempfile
import unittest
from pathlib import Path

from tools.run_foundation_genrecon_batch import (
    glb_command,
    parse_glb,
    parse_mesh_ply,
    profile_is_complete,
    reconstruct_command,
    selected_candidates,
    validate_outputs,
)


class FoundationGenreconBatchTests(unittest.TestCase):
    def test_reconstruction_protocol_matches_foundation_preflight(self) -> None:
        command = reconstruct_command(Path("/input"), Path("/output"))
        self.assertIn("colmap_vggt", command)
        self.assertEqual(command[command.index("--num_imgs_per_scene") + 1], "8")
        self.assertEqual(command[command.index("--chunk_size_factor") + 1], "1.08")
        self.assertEqual(command[command.index("--proj_batch_voxels") + 1], "256")
        self.assertNotIn("--center_crop", command)
        self.assertNotIn("--manual_z_bounds", command)
        self.assertNotIn("--min_track_len", command)
        self.assertNotIn("--min_projected_chunk_area", command)

    def test_object_scale_protocol_records_projected_area_gate(self) -> None:
        command = reconstruct_command(
            Path("/input"),
            Path("/output"),
            min_projected_chunk_area=0.2,
        )
        self.assertEqual(
            command[command.index("--min_projected_chunk_area") + 1], "0.2"
        )

    def test_profile_reuse_rejects_stale_reconstruction_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "mesh.ply"
            artifact.write_bytes(b"mesh")
            profile = root / "profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "success": True,
                        "return_code": 0,
                        "command": ["old"],
                        "artifacts": [
                            {
                                "path": str(artifact),
                                "size_bytes": artifact.stat().st_size,
                            }
                        ],
                    }
                )
            )
            self.assertTrue(
                profile_is_complete(profile, [artifact], expected_command=["old"])
            )
            self.assertFalse(
                profile_is_complete(profile, [artifact], expected_command=["new"])
            )

    def test_native_sfm_protocol_uses_explicit_track_filters(self) -> None:
        command = reconstruct_command(
            Path("/input"),
            Path("/output"),
            colmap_subdir="colmap_sfm",
            max_reproj_error=2.0,
            min_track_len=4,
        )
        self.assertEqual(command[command.index("--colmap_subdir") + 1], "colmap_sfm")
        self.assertEqual(command[command.index("--max_reproj_error") + 1], "2")
        self.assertEqual(command[command.index("--min_track_len") + 1], "4")

    def test_glb_protocol_is_resumable_and_resource_bounded(self) -> None:
        command = glb_command(Path("/output"))
        self.assertEqual(command[command.index("--texture_size") + 1], "4096")
        self.assertEqual(command[command.index("--simplify_threshold") + 1], "300000")
        self.assertIn("--skip_fill_holes", command)
        self.assertIn("--skip_remesh", command)
        self.assertIn("--chunks_dir", command)

    def test_mesh_ply_payload_validation(self) -> None:
        header = (
            b"ply\nformat binary_little_endian 1.0\n"
            b"element vertex 3\n"
            b"property float x\nproperty float y\nproperty float z\n"
            b"property uchar red\nproperty uchar green\nproperty uchar blue\n"
            b"property uchar metallic\nproperty uchar roughness\nproperty uchar alpha\n"
            b"element face 1\nproperty list uchar int vertex_indices\nend_header\n"
        )
        vertices = b"".join(
            struct.pack("<fffBBBBBB", float(index), 0.0, 0.0, 1, 2, 3, 4, 5, 255)
            for index in range(3)
        )
        face = struct.pack("<Biii", 3, 0, 1, 2)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mesh.ply"
            path.write_bytes(header + vertices + face)
            parsed = parse_mesh_ply(path)
        self.assertEqual(parsed["vertices"], 3)
        self.assertEqual(parsed["faces"], 1)

    def test_glb_requires_embedded_base_and_metallic_roughness_textures(self) -> None:
        document = {
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"material": 0}]}],
            "materials": [
                {
                    "pbrMetallicRoughness": {
                        "baseColorTexture": {"index": 0},
                        "metallicRoughnessTexture": {"index": 1},
                    }
                }
            ],
            "textures": [{"source": 0}, {"source": 1}],
            "images": [{"bufferView": 0}, {"bufferView": 1}],
            "accessors": [],
        }
        payload = json.dumps(document, separators=(",", ":")).encode("utf-8")
        payload += b" " * ((-len(payload)) % 4)
        glb = struct.pack("<4sII", b"glTF", 2, 20 + len(payload))
        glb += struct.pack("<I4s", len(payload), b"JSON") + payload
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scene.glb"
            path.write_bytes(glb)
            parsed = parse_glb(path)
        self.assertEqual(parsed["meshes"], 1)
        self.assertTrue(parsed["pbr_textures_embedded"])

    def test_consumable_candidates_are_prioritized_without_exclusion(self) -> None:
        index = {
            "candidates": [
                {
                    "candidate_id": "reject",
                    "status": "completed",
                    "preflight": "passed",
                    "visual_disposition": "reject_current_shot",
                    "foundation_grade": "P-F",
                },
                {
                    "candidate_id": "pilot",
                    "status": "completed",
                    "preflight": "passed",
                    "visual_disposition": "proceed_foundation_pilot",
                    "foundation_grade": "P-B",
                },
                {
                    "candidate_id": "failed",
                    "status": "completed",
                    "preflight": "failed",
                    "visual_disposition": "proceed_foundation_pilot",
                    "foundation_grade": "P-A",
                },
            ]
        }
        selected = selected_candidates(index, [])
        self.assertEqual([item["candidate_id"] for item in selected], ["pilot", "reject"])

    def test_summary_and_validation_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            output_root = root / "output"
            report_root = root / "report"
            output_root.mkdir(parents=True)

            first = validate_outputs(
                [], input_root, output_root, report_root, "deterministic-test"
            )
            first_index = (output_root / "index.json").read_bytes()
            first_validation = (output_root / "validation.json").read_bytes()
            second = validate_outputs(
                [], input_root, output_root, report_root, "deterministic-test"
            )

            self.assertNotIn("created_utc", json.loads(first_index))
            self.assertNotIn("validated_utc", first)
            self.assertEqual(first, second)
            self.assertEqual(first_index, (output_root / "index.json").read_bytes())
            self.assertEqual(
                first_validation, (output_root / "validation.json").read_bytes()
            )


if __name__ == "__main__":
    unittest.main()
