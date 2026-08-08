from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from tools.profile_command import profile_command


class ProfileCommandTests(unittest.TestCase):
    def test_records_success_output_and_expected_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = profile_command(
                [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; Path('mesh.ply').write_bytes(b'ply')",
                ],
                cwd=root,
                log_path=root / "run.log",
                result_path=root / "profile.json",
                expected_files=[Path("mesh.ply")],
                sample_interval_s=0.01,
                progress_interval_s=60.0,
                label="test",
            )

            self.assertTrue(result["success"])
            self.assertEqual(result["return_code"], 0)
            self.assertEqual(result["artifacts"][0]["size_bytes"], 3)
            self.assertEqual(
                json.loads((root / "profile.json").read_text(encoding="utf-8")),
                result,
            )

    def test_missing_expected_artifact_fails_even_when_command_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = profile_command(
                [sys.executable, "-c", "print('done')"],
                cwd=root,
                log_path=root / "run.log",
                result_path=root / "profile.json",
                expected_files=[Path("missing.ply")],
                sample_interval_s=0.01,
                progress_interval_s=60.0,
            )

            self.assertFalse(result["success"])
            self.assertEqual(result["return_code"], 0)
            self.assertEqual(len(result["missing_expected_artifacts"]), 1)

    def test_preexisting_unchanged_artifact_is_not_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "stale.ply").write_bytes(b"old mesh")
            result = profile_command(
                [sys.executable, "-c", "print('did not write the mesh')"],
                cwd=root,
                log_path=root / "run.log",
                result_path=root / "profile.json",
                expected_files=[Path("stale.ply")],
                sample_interval_s=0.01,
                progress_interval_s=60.0,
            )

            self.assertFalse(result["success"])
            self.assertEqual(result["missing_expected_artifacts"], [])
            self.assertEqual(len(result["stale_expected_artifacts"]), 1)
            self.assertFalse(result["artifacts"][0]["produced_or_updated_this_run"])


if __name__ == "__main__":
    unittest.main()
