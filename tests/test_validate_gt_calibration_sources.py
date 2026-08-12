from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from tools.validate_gt_calibration_sources import (
    SourceValidationError,
    validate_sources,
    validate_tnt_alignment,
    validate_tnt_scans,
    validate_zip,
)


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

    def test_invalid_zip_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.zip"
            path.write_bytes(b"not a zip")
            with self.assertRaisesRegex(SourceValidationError, "Invalid ZIP"):
                validate_zip(path)


if __name__ == "__main__":
    unittest.main()
