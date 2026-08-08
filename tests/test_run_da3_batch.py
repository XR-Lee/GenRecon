from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.run_da3_batch import inspect_scene


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_profile(path: Path, artifact: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "success": True,
                "return_code": 0,
                "missing_expected_artifacts": [],
                "stale_expected_artifacts": [],
                "artifacts": [
                    {
                        "path": str(artifact.resolve()),
                        "produced_or_updated_this_run": True,
                        "size_bytes": artifact.stat().st_size,
                        "sha256": _sha256(artifact),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


class RunDa3BatchTests(unittest.TestCase):
    def test_missing_scene_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            inspection = inspect_scene(Path(temp), "scene")
            self.assertFalse(inspection["complete"])
            self.assertTrue(inspection["reasons"])

    def test_complete_scene_requires_matching_profiles_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "outputs/scene/reconstruction"
            report = root / "reports/generated/scene"
            output.mkdir(parents=True)
            report.mkdir(parents=True)
            mesh = output / "mesh.ply"
            glb = output / "scene.glb"
            mesh.write_bytes(b"mesh")
            glb.write_bytes(b"glb")
            (output / "to_glb_inputs.pt").write_bytes(b"inputs")
            (output / "chunk_inputs.pt").write_bytes(b"chunks")
            _write_profile(report / "reconstruct_profile.json", mesh)
            _write_profile(report / "glb_profile.json", glb)
            (report / "mesh_metrics_unclipped.json").write_text(
                json.dumps(
                    {
                        "schema": "genrecon.mesh-evaluation",
                        "protocol": "paper-like-unclipped",
                        "sampling": {"samples_per_mesh": 200_000},
                        "metrics": {"chamfer_symmetric_mean_m": 0.1},
                    }
                ),
                encoding="utf-8",
            )

            inspection = inspect_scene(root, "scene")

            self.assertTrue(inspection["complete"])
            self.assertEqual(inspection["metrics"]["chamfer_symmetric_mean_m"], 0.1)

    def test_changed_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "outputs/scene/reconstruction"
            report = root / "reports/generated/scene"
            output.mkdir(parents=True)
            report.mkdir(parents=True)
            mesh = output / "mesh.ply"
            mesh.write_bytes(b"original")
            _write_profile(report / "reconstruct_profile.json", mesh)
            mesh.write_bytes(b"changed")

            inspection = inspect_scene(root, "scene")

            self.assertFalse(inspection["complete"])
            self.assertTrue(any("differs from profile" in item for item in inspection["reasons"]))


if __name__ == "__main__":
    unittest.main()
