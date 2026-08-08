from __future__ import annotations

import fcntl
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from tools.run_da3_postprocess import (
    REQUIRED_METRICS,
    _run_command,
    build_parser,
    inspect_postprocess_scene,
    run_postprocess,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_profile(path: Path, artifacts: list[Path], *, success: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "success": success,
                "return_code": 0 if success else 1,
                "missing_expected_artifacts": [],
                "stale_expected_artifacts": [],
                "artifacts": [
                    {
                        "path": str(artifact.resolve()),
                        "produced_or_updated_this_run": True,
                        "size_bytes": artifact.stat().st_size,
                        "sha256": _sha256(artifact),
                    }
                    for artifact in artifacts
                ],
            }
        ),
        encoding="utf-8",
    )


def _paths(root: Path, scene_id: str) -> dict[str, Path]:
    output = root / "outputs" / scene_id / "reconstruction"
    report = root / "reports" / "generated" / scene_id
    return {
        "output": output,
        "report": report,
        "mesh": output / "mesh.ply",
        "to_glb_inputs": output / "to_glb_inputs.pt",
        "chunk_inputs": output / "chunk_inputs.pt",
        "glb": output / "scene.glb",
        "reconstruct_profile": report / "reconstruct_profile.json",
        "glb_profile": report / "glb_profile.json",
        "metrics": report / "mesh_metrics_unclipped.json",
        "ground_truth": root
        / "data"
        / "da3-adapted"
        / scene_id
        / "scans"
        / "mesh_aligned_0.05.ply",
    }


def _prepare_reconstruction(root: Path, scene_id: str, *, with_gt: bool = True) -> None:
    paths = _paths(root, scene_id)
    paths["output"].mkdir(parents=True, exist_ok=True)
    paths["mesh"].write_bytes(b"mesh")
    paths["to_glb_inputs"].write_bytes(b"saved glb inputs")
    paths["chunk_inputs"].write_bytes(b"saved chunk inputs")
    _write_profile(
        paths["reconstruct_profile"],
        [paths["mesh"], paths["to_glb_inputs"], paths["chunk_inputs"]],
    )
    if with_gt:
        paths["ground_truth"].parent.mkdir(parents=True, exist_ok=True)
        paths["ground_truth"].write_bytes(b"ground truth")


def _write_metrics(path: Path, predicted: Path, ground_truth: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "genrecon.mesh-evaluation",
                "schema_version": 1,
                "protocol": "paper-like-unclipped",
                "inputs": {
                    "predicted_mesh": str(predicted),
                    "ground_truth_mesh": str(ground_truth),
                },
                "sampling": {"samples_per_mesh": 200_000, "seed": 42},
                "crop": {"mode": "none"},
                "metrics": {name: 0.25 for name in REQUIRED_METRICS},
            }
        ),
        encoding="utf-8",
    )


def _complete_glb(paths: dict[str, Path]) -> None:
    paths["glb"].write_bytes(b"glb")
    _write_profile(paths["glb_profile"], [paths["glb"]])


def _write_manifest(root: Path, scenes: list[str]) -> None:
    manifest = root / "data" / "da3-adapted" / "da3_scannetpp_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"scene_ids": scenes}), encoding="utf-8")


def _args(root: Path, scenes: list[str], *extra: str):
    return build_parser().parse_args(
        [
            "--root",
            str(root),
            "--expected-scene-count",
            str(len(scenes)),
            "--timeout-seconds",
            "10",
            *extra,
        ]
    )


class FakeRunner:
    def __init__(self, *, fail_metrics_scene: str | None = None, timeout_scene: str | None = None):
        self.calls: list[tuple[list[str], float]] = []
        self.fail_metrics_scene = fail_metrics_scene
        self.timeout_scene = timeout_scene

    def __call__(self, command, **kwargs):
        command = list(command)
        self.calls.append((command, kwargs["timeout_s"]))
        command_text = " ".join(command)
        is_metrics = any(token.endswith("evaluate_mesh.py") for token in command)
        if is_metrics and self.timeout_scene and self.timeout_scene in command_text:
            return {
                "return_code": -2,
                "timed_out": True,
                "duration_s": 0.01,
                "log_path": str(kwargs["log_path"]),
            }
        if is_metrics and self.fail_metrics_scene and self.fail_metrics_scene in command_text:
            return {
                "return_code": 2,
                "timed_out": False,
                "duration_s": 0.01,
                "log_path": str(kwargs["log_path"]),
            }
        if is_metrics:
            output = Path(command[command.index("--output") + 1])
            _write_metrics(output, Path(command[2]), Path(command[3]))
        else:
            glb = Path(command[command.index("--expect-file") + 1])
            profile = Path(command[command.index("--result") + 1])
            glb.write_bytes(b"recovered glb")
            _write_profile(profile, [glb])
        return {
            "return_code": 0,
            "timed_out": False,
            "duration_s": 0.01,
            "log_path": str(kwargs["log_path"]),
        }


class MalformedMetricsRunner(FakeRunner):
    def __call__(self, command, **kwargs):
        command = list(command)
        if any(token.endswith("evaluate_mesh.py") for token in command):
            self.calls.append((command, kwargs["timeout_s"]))
            output = Path(command[command.index("--output") + 1])
            output.write_text('{"schema": "wrong"}', encoding="utf-8")
            return {
                "return_code": 0,
                "timed_out": False,
                "duration_s": 0.01,
                "log_path": str(kwargs["log_path"]),
            }
        return super().__call__(command, **kwargs)


class FailedGlbRunner(FakeRunner):
    def __call__(self, command, **kwargs):
        command = list(command)
        self.calls.append((command, kwargs["timeout_s"]))
        return {
            "return_code": 1,
            "timed_out": False,
            "duration_s": 0.01,
            "log_path": str(kwargs["log_path"]),
        }


class RunDa3PostprocessTests(unittest.TestCase):
    def test_complete_scene_invokes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scenes = ["complete"]
            _write_manifest(root, scenes)
            _prepare_reconstruction(root, scenes[0])
            paths = _paths(root, scenes[0])
            _write_metrics(paths["metrics"], paths["mesh"], paths["ground_truth"])
            _complete_glb(paths)
            runner = FakeRunner()

            result = run_postprocess(_args(root, scenes), command_runner=runner)

            self.assertEqual(result, 0)
            self.assertEqual(runner.calls, [])
            summary = json.loads(
                (root / "reports/generated/da3_postprocess_summary.json").read_text()
            )
            self.assertEqual(summary["records"][0]["state"], "skipped-complete")
            self.assertFalse(
                (root / "reports/generated/da3_postprocess_summary.json.tmp").exists()
            )

    def test_dry_run_plans_both_actions_without_scene_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scenes = ["pending"]
            _write_manifest(root, scenes)
            _prepare_reconstruction(root, scenes[0])
            paths = _paths(root, scenes[0])
            runner = FakeRunner()

            result = run_postprocess(
                _args(root, scenes, "--dry-run"), command_runner=runner
            )

            self.assertEqual(result, 0)
            self.assertEqual(runner.calls, [])
            self.assertFalse(paths["metrics"].exists())
            self.assertFalse(paths["glb"].exists())
            self.assertFalse((paths["report"] / "mesh_metrics_postprocess.log").exists())
            summary = json.loads(
                (root / "reports/generated/da3_postprocess_summary.json").read_text()
            )
            record = summary["records"][0]
            self.assertEqual(record["state"], "dry-run-pending")
            self.assertEqual(record["planned_actions"], ["metrics", "glb"])

    def test_recovers_metrics_then_glb_using_only_saved_inputs_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scenes = ["recover"]
            _write_manifest(root, scenes)
            _prepare_reconstruction(root, scenes[0])
            paths = _paths(root, scenes[0])
            runner = FakeRunner()

            result = run_postprocess(_args(root, scenes), command_runner=runner)

            self.assertEqual(result, 0)
            self.assertEqual(len(runner.calls), 2)
            metric_command, glb_command = (call[0] for call in runner.calls)
            self.assertTrue(any(token.endswith("evaluate_mesh.py") for token in metric_command))
            self.assertTrue(any(token.endswith("profile_command.py") for token in glb_command))
            self.assertTrue(any(token.endswith("chunked_to_glb.py") for token in glb_command))
            flattened = " ".join(metric_command + glb_command)
            self.assertNotIn("reconstruct_scene.py", flattened)
            self.assertNotIn("run_smoke.sh", flattened)
            self.assertIn(str(paths["to_glb_inputs"]), glb_command)
            self.assertIn(str(paths["chunk_inputs"]), glb_command)
            self.assertIn("--chunks_dir", glb_command)
            self.assertIn(str(paths["output"] / "chunks"), glb_command)
            self.assertNotIn("--no_chunk_cache", glb_command)
            self.assertTrue(inspect_postprocess_scene(root, scenes[0])["complete"])
            self.assertFalse(list(paths["report"].glob(".*.postprocess.*.tmp")))

    def test_one_deadline_passes_reduced_time_to_second_action(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scenes = ["deadline"]
            _write_manifest(root, scenes)
            _prepare_reconstruction(root, scenes[0])
            runner = FakeRunner()
            times = iter([0.0, 1.0, 4.0, 5.0])

            result = run_postprocess(
                _args(root, scenes),
                command_runner=runner,
                clock=lambda: next(times),
            )

            self.assertEqual(result, 0)
            self.assertEqual([timeout for _, timeout in runner.calls], [9.0, 6.0])

    def test_metrics_failure_still_recovers_glb_and_continues_next_scene(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scenes = ["bad", "good"]
            _write_manifest(root, scenes)
            for scene in scenes:
                _prepare_reconstruction(root, scene)
            runner = FakeRunner(fail_metrics_scene="bad")

            result = run_postprocess(_args(root, scenes), command_runner=runner)

            self.assertEqual(result, 1)
            call_text = [" ".join(command) for command, _ in runner.calls]
            self.assertEqual(len(call_text), 4)
            self.assertIn("bad", call_text[0])
            self.assertIn("bad", call_text[1])
            self.assertIn("chunked_to_glb.py", call_text[1])
            self.assertIn("good", call_text[2])
            summary = json.loads(
                (root / "reports/generated/da3_postprocess_summary.json").read_text()
            )
            self.assertEqual([item["state"] for item in summary["records"]], ["failed", "complete"])

    def test_timeout_skips_same_scene_glb_but_continues(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scenes = ["slow", "next"]
            _write_manifest(root, scenes)
            for scene in scenes:
                _prepare_reconstruction(root, scene)
            runner = FakeRunner(timeout_scene="slow")

            result = run_postprocess(_args(root, scenes), command_runner=runner)

            self.assertEqual(result, 1)
            call_text = [" ".join(command) for command, _ in runner.calls]
            self.assertEqual(len(call_text), 3)
            self.assertIn("slow", call_text[0])
            self.assertNotIn("slow", " ".join(call_text[1:]))
            self.assertIn("next", call_text[1])
            summary = json.loads(
                (root / "reports/generated/da3_postprocess_summary.json").read_text()
            )
            self.assertEqual(
                [item["state"] for item in summary["records"]],
                ["timed-out", "complete"],
            )

    def test_changed_intermediate_blocks_all_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scenes = ["changed"]
            _write_manifest(root, scenes)
            _prepare_reconstruction(root, scenes[0])
            _paths(root, scenes[0])["chunk_inputs"].write_bytes(b"changed after profile")
            runner = FakeRunner()

            result = run_postprocess(_args(root, scenes), command_runner=runner)

            self.assertEqual(result, 1)
            self.assertEqual(runner.calls, [])
            summary = json.loads(
                (root / "reports/generated/da3_postprocess_summary.json").read_text()
            )
            record = summary["records"][0]
            self.assertEqual(record["state"], "blocked-invalid-reconstruction")
            self.assertTrue(
                any(
                    "differs from profile" in reason
                    for reason in record["inspection_before"]["reconstruction"]["reasons"]
                )
            )

    def test_invalid_temporary_metrics_preserve_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scenes = ["metrics"]
            _write_manifest(root, scenes)
            _prepare_reconstruction(root, scenes[0])
            paths = _paths(root, scenes[0])
            _complete_glb(paths)
            original = b'{"old": "invalid but preserved on failure"}'
            paths["metrics"].write_bytes(original)

            result = run_postprocess(
                _args(root, scenes), command_runner=MalformedMetricsRunner()
            )

            self.assertEqual(result, 1)
            self.assertEqual(paths["metrics"].read_bytes(), original)
            self.assertFalse(list(paths["report"].glob(".*.postprocess.*.tmp")))

    def test_failed_glb_preserves_existing_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scenes = ["glb"]
            _write_manifest(root, scenes)
            _prepare_reconstruction(root, scenes[0])
            paths = _paths(root, scenes[0])
            _write_metrics(paths["metrics"], paths["mesh"], paths["ground_truth"])
            paths["glb"].write_bytes(b"old glb")
            original = b'{"success": false, "sentinel": "keep me"}'
            paths["glb_profile"].write_bytes(original)

            result = run_postprocess(
                _args(root, scenes), command_runner=FailedGlbRunner()
            )

            self.assertEqual(result, 1)
            self.assertEqual(paths["glb_profile"].read_bytes(), original)
            self.assertFalse(list(paths["report"].glob(".*.postprocess.*.tmp")))

    def test_missing_ground_truth_blocks_metrics_but_still_recovers_glb(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scenes = ["missing-gt"]
            _write_manifest(root, scenes)
            _prepare_reconstruction(root, scenes[0], with_gt=False)
            runner = FakeRunner()

            result = run_postprocess(_args(root, scenes), command_runner=runner)

            self.assertEqual(result, 1)
            self.assertEqual(len(runner.calls), 1)
            self.assertIn("chunked_to_glb.py", " ".join(runner.calls[0][0]))
            summary = json.loads(
                (root / "reports/generated/da3_postprocess_summary.json").read_text()
            )
            actions = summary["records"][0]["actions"]
            self.assertEqual(actions[0]["state"], "blocked-missing-ground-truth")
            self.assertEqual(actions[1]["state"], "complete")

    def test_held_batch_lock_rejects_even_dry_run_before_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scenes = ["scene"]
            _write_manifest(root, scenes)
            lock_path = root / "reports/generated/da3_batch_summary.json.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("a+", encoding="utf-8") as held:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "active DA3 batch"):
                    run_postprocess(
                        _args(root, scenes, "--dry-run"), command_runner=FakeRunner()
                    )
            self.assertFalse(
                (root / "reports/generated/da3_postprocess_summary.json").exists()
            )

    def test_summary_cannot_alias_batch_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scenes = ["scene"]
            _write_manifest(root, scenes)
            batch_lock = root / "reports/generated/da3_batch_summary.json.lock"

            with self.assertRaisesRegex(ValueError, "unsafe path alias"):
                run_postprocess(
                    _args(
                        root,
                        scenes,
                        "--summary",
                        "reports/generated/da3_batch_summary.json.lock",
                        "--dry-run",
                    ),
                    command_runner=FakeRunner(),
                )

            self.assertFalse(batch_lock.exists())

    def test_outer_command_timeout_terminates_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = _run_command(
                [sys.executable, "-c", "import time; time.sleep(5)"],
                cwd=root,
                env={},
                log_path=root / "timeout.log",
                timeout_s=0.05,
                kill_grace_s=0.05,
            )
            self.assertTrue(result["timed_out"])
            self.assertNotEqual(result["return_code"], 0)
            self.assertLess(result["duration_s"], 2.0)


if __name__ == "__main__":
    unittest.main()
