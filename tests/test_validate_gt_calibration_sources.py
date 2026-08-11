from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from tools.validate_gt_calibration_sources import (
    SourceValidationError,
    validate_sources,
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
            result = validate_sources(root)

        self.assertEqual(result["result"], "pass")
        self.assertEqual(result["counts"]["zip"], 1)
        self.assertEqual(result["counts"]["declared_quota_pages"], 1)
        self.assertTrue(result["declared_quota_pages"][0]["quota_message_present"])

    def test_empty_source_tree_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = validate_sources(Path(tmp))
        self.assertEqual(result["result"], "fail")
        self.assertIn("No source archives", result["errors"][0])

    def test_invalid_zip_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.zip"
            path.write_bytes(b"not a zip")
            with self.assertRaisesRegex(SourceValidationError, "Invalid ZIP"):
                validate_zip(path)


if __name__ == "__main__":
    unittest.main()
