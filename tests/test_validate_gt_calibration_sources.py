from __future__ import annotations

import hashlib
import json
import tarfile
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from tools.validate_gt_calibration_sources import (
    SourceValidationError,
    validate_sources,
    validate_tar_gzip,
    validate_tnt_alignment,
    validate_tnt_scans,
    validate_zip,
)


def _add_tar_payload(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, BytesIO(payload))


def _write_omni_render_archive(
    path: Path, object_id: str, *, omit: str | None = None, unsafe: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    transforms = json.dumps(
        {
            "camera_angle_x": 0.7,
            "frames": [
                {"file_path": f"r_{index}"} for index in range(100)
            ],
        }
    ).encode()
    with tarfile.open(path, "w:gz") as archive:
        _add_tar_payload(
            archive, f"{object_id}/render/transforms.json", transforms
        )
        for index in range(100):
            names = (
                f"{object_id}/render/images/r_{index}.png",
                f"{object_id}/render/normals/r_{index}_normal.png",
                f"{object_id}/render/depths/r_{index}_depth.exr",
            )
            for name in names:
                if name != omit:
                    _add_tar_payload(archive, name, b"payload")
        if unsafe:
            _add_tar_payload(archive, "../escape", b"unsafe")


def _write_omni_scan_archive(path: Path, object_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        _add_tar_payload(archive, f"{object_id}/Scan/Scan.obj", b"v 0 0 0\n")
        _add_tar_payload(archive, f"{object_id}/Scan/Scan.mtl", b"newmtl m\n")
        _add_tar_payload(archive, f"{object_id}/Scan/Scan.jpg", b"jpeg")


class ValidateGroundTruthSourcesTests(unittest.TestCase):
    def test_valid_zip_crc_and_collection_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "sources" / "fixture"
            source.mkdir(parents=True)
            archive = source / "fixture.zip"
            with ZipFile(archive, "w", compression=ZIP_DEFLATED) as handle:
                handle.writestr("data/value.txt", "content")
            quota = source / "blocked.quota-blocked.html"
            quota.write_text("<title>Google Drive - Quota exceeded</title>")
            historical = source / "resolved.previous-quota-response.html"
            historical.write_text("<title>Google Drive - Quota exceeded</title>")
            result = validate_sources(root)

        self.assertEqual(result["result"], "pass")
        self.assertEqual(result["counts"]["zip"], 1)
        self.assertEqual(result["counts"]["declared_quota_pages"], 1)
        self.assertEqual(result["counts"]["historical_download_failures"], 1)
        self.assertTrue(result["declared_quota_pages"][0]["quota_message_present"])
        self.assertEqual(
            result["historical_download_failures"][0]["status"],
            "historical-resolved-download-failure",
        )

    def test_empty_source_tree_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = validate_sources(Path(tmp))
        self.assertEqual(result["result"], "fail")
        self.assertIn("No source archives", result["errors"][0])

    def test_tnt_scan_archive_requires_eleven_binary_plys_and_positions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Meetingroom_individual_scans.zip"
            with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
                for index in range(1, 12):
                    archive.writestr(
                        f"Meetingroom/Meetingroom{index:02d}.ply",
                        b"ply\nformat binary_little_endian 1.0\nend_header\n",
                    )
                archive.writestr(
                    "Meetingroom/scanner_pos.txt",
                    "\n".join(f"scan{i:02d}.ply 0 0 0" for i in range(1, 13)),
                )
            result = validate_tnt_scans(path)
        self.assertEqual(result["ply_scans"], 11)
        self.assertEqual(result["scanner_position_records"], 12)
        self.assertEqual(result["crc"], "pass")

    def test_tnt_alignment_requires_a_valid_sim3(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "alignment.txt"
            matrix = np.eye(4)
            matrix[:3, :3] *= 1.5
            np.savetxt(path, matrix)
            result = validate_tnt_alignment(path)
        self.assertAlmostEqual(result["sim3_scale"], 1.5)
        self.assertLess(result["rotation_orthogonality_error"], 1e-12)

    def test_omni_tar_contract_requires_exact_safe_100_view_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = root / "fixture.tar.gz"
            _write_omni_render_archive(valid, "fixture_001")
            record = validate_tar_gzip(
                valid, render_object_ids=("fixture_001",)
            )
            self.assertEqual(record["stream_integrity"], "pass")
            self.assertEqual(record["selected_object_payloads"][0]["rgb_members"], 100)
            self.assertEqual(record["selected_object_payloads"][0]["depth_members"], 100)

            missing_name = "fixture_001/render/depths/r_99_depth.exr"
            incomplete = root / "incomplete.tar.gz"
            _write_omni_render_archive(
                incomplete, "fixture_001", omit=missing_name
            )
            with self.assertRaisesRegex(
                SourceValidationError, "Incomplete 100-view render payload"
            ):
                validate_tar_gzip(
                    incomplete, render_object_ids=("fixture_001",)
                )

            unsafe = root / "unsafe.tar.gz"
            _write_omni_render_archive(unsafe, "fixture_001", unsafe=True)
            with self.assertRaisesRegex(SourceValidationError, "Unsafe TAR path"):
                validate_tar_gzip(
                    unsafe, render_object_ids=("fixture_001",)
                )

    def test_omni_source_archives_bind_to_plan_and_openxlab_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            omni = root / "sources" / "omniobject3d"
            download = omni / "downloads" / "OpenXDLab___OmniObject3D-New"
            render = download / "raw" / "blender_renders" / "fixture.tar.gz"
            scan = download / "raw" / "raw_scans" / "fixture.tar.gz"
            _write_omni_render_archive(render, "fixture_001")
            _write_omni_scan_archive(scan, "fixture_001")
            plan = root / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "datasets": {
                            "omniobject3d": {
                                "adapter": {"adapter_version": 3},
                                "unit_ids": ["fixture_001"],
                                "rejected_candidates": [],
                            }
                        }
                    }
                )
            )

            def index_record(path: Path, openxlab_path: str) -> dict[str, object]:
                return {
                    "dataset_id": 6639,
                    "path": openxlab_path,
                    "size": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }

            records = [
                index_record(render, "/raw/blender_renders/fixture.tar.gz"),
                index_record(scan, "/raw/raw_scans/fixture.tar.gz"),
            ]
            index_path = omni / "metadata" / "openxlab_file_index.json"
            index_path.parent.mkdir(parents=True)

            def write_index() -> None:
                index_path.write_text(
                    json.dumps(
                        {
                            "schema": "genrecon.omniobject3d-openxlab-file-index",
                            "schema_version": 1,
                            "dataset_repo": "OpenXDLab/OmniObject3D-New",
                            "dataset_id": 6639,
                            "file_count": len(records),
                            "total_bytes": sum(int(item["size"]) for item in records),
                            "all_files_have_sha256": True,
                            "files": records,
                        }
                    )
                )

            write_index()
            nested_archive_directory = download / "raw" / "camera.tar.gz"
            nested_archive_directory.mkdir(parents=True)
            (nested_archive_directory / "not-an-archive.txt").write_text("directory fixture")
            result = validate_sources(root, plan_path=plan)
            self.assertEqual(result["result"], "pass")
            self.assertEqual(result["counts"]["tar_gzip"], 2)
            self.assertEqual(result["counts"]["omniobject3d_selected_render_payloads"], 1)
            self.assertEqual(result["counts"]["omniobject3d_selected_scan_payloads"], 1)

            records[0]["sha256"] = "0" * 64
            write_index()
            tampered = validate_sources(root, plan_path=plan)
            self.assertEqual(tampered["result"], "fail")
            self.assertTrue(
                any("OpenXLab SHA256 mismatch" in error for error in tampered["errors"])
            )

    def test_invalid_zip_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.zip"
            path.write_bytes(b"not a zip")
            with self.assertRaisesRegex(SourceValidationError, "Invalid ZIP"):
                validate_zip(path)


if __name__ == "__main__":
    unittest.main()
