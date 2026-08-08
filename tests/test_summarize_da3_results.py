from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from tools.summarize_da3_results import (
    DISTANCE_METRICS,
    REQUIRED_METRICS,
    build_parser,
    run,
)


def _paths(root: Path, scene_id: str) -> dict[str, Path]:
    output = root / "outputs" / scene_id / "reconstruction"
    report = root / "reports" / "generated" / scene_id
    return {
        "output": output,
        "report": report,
        "mesh": output / "mesh.ply",
        "glb": output / "scene.glb",
        "metrics": report / "mesh_metrics_unclipped.json",
        "reconstruct_profile": report / "reconstruct_profile.json",
        "glb_profile": report / "glb_profile.json",
        "ground_truth": root
        / "data"
        / "da3-adapted"
        / scene_id
        / "scans"
        / "mesh_aligned_0.05.ply",
    }


def _write_ply(path: Path, *, vertices: int = 3, faces: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (
            "ply\n"
            "format ascii 1.0\n"
            f"element vertex {vertices}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            f"element face {faces}\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
            "0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n"
        ).encode("ascii")
    )


def _write_glb(path: Path, *, valid: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if valid:
        json_chunk = b"{}  "
        path.write_bytes(
            struct.pack("<4sII", b"glTF", 2, 20 + len(json_chunk))
            + struct.pack("<I4s", len(json_chunk), b"JSON")
            + json_chunk
        )
    else:
        path.write_bytes(b"not a valid glb file")


def _write_profile(
    path: Path,
    artifact: Path,
    *,
    success: bool = True,
    duration: float = 12.5,
    gpu_mib: float = 4321.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "success": success,
                "return_code": 0 if success else 1,
                "duration_s": duration,
                "gpu": {
                    "peak_global_memory_mib": gpu_mib,
                    "peak_process_memory_mib": gpu_mib - 100,
                },
                "missing_expected_artifacts": [] if success else [str(artifact)],
                "stale_expected_artifacts": [],
                "artifacts": [
                    {
                        "path": str(artifact.resolve()),
                        "produced_or_updated_this_run": success,
                        "size_bytes": artifact.stat().st_size if artifact.exists() else None,
                        # The summarizer must never read the artifact to verify this.
                        "sha256": "deliberately-not-a-real-hash",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_metrics(path: Path, mesh: Path, ground_truth: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ground_truth.parent.mkdir(parents=True, exist_ok=True)
    _write_ply(ground_truth)
    path.write_text(
        json.dumps(
            {
                "schema": "genrecon.mesh-evaluation",
                "schema_version": 1,
                "protocol": "paper-like-unclipped",
                "sampling": {"samples_per_mesh": 200_000, "seed": 42},
                "crop": {"mode": "none"},
                "inputs": {
                    "predicted_mesh": str(mesh.resolve()),
                    "ground_truth_mesh": str(ground_truth.resolve()),
                },
                "meshes": {
                    "predicted_evaluated": {"vertices": 3, "faces": 1}
                },
                "metrics": {
                    name: value if name in DISTANCE_METRICS else 0.5
                    for name in REQUIRED_METRICS
                },
            }
        ),
        encoding="utf-8",
    )


def _prepare_geometry(root: Path, scene_id: str, value: float) -> None:
    paths = _paths(root, scene_id)
    _write_ply(paths["mesh"], vertices=3, faces=1)
    _write_profile(paths["reconstruct_profile"], paths["mesh"])
    _write_metrics(paths["metrics"], paths["mesh"], paths["ground_truth"], value)


def _prepare_full(root: Path, scene_id: str, value: float) -> None:
    _prepare_geometry(root, scene_id, value)
    paths = _paths(root, scene_id)
    _write_glb(paths["glb"])
    _write_profile(paths["glb_profile"], paths["glb"], duration=7.5, gpu_mib=5000)


def _write_reconstruction_args(
    root: Path,
    scene_id: str,
    *,
    max_chunks_per_group: int | None,
    max_inflated_voxels: int | None,
) -> None:
    output = _paths(root, scene_id)["output"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "args.json").write_text(
        json.dumps(
            {
                "joint_decode_max_chunks_per_group": max_chunks_per_group,
                "joint_decode_max_inflated_voxels": max_inflated_voxels,
            }
        ),
        encoding="utf-8",
    )


def _write_inputs(root: Path, scene_ids: list[str]) -> None:
    manifest = root / "data" / "da3-adapted" / "da3_scannetpp_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({"num_views_per_scene": 8, "scene_ids": scene_ids}),
        encoding="utf-8",
    )
    preflight = root / "reports" / "generated" / "preflight" / "summary.json"
    preflight.parent.mkdir(parents=True, exist_ok=True)
    preflight.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "scene_id": scene_id,
                        "status": "ok",
                        "chunk_count": index + 2,
                        "joint_aabb_volume_ratio": index + 1.25,
                    }
                    for index, scene_id in enumerate(scene_ids)
                ]
            }
        ),
        encoding="utf-8",
    )


def _args(root: Path, scene_count: int):
    return build_parser().parse_args(
        [
            "--root",
            str(root),
            "--expected-scene-count",
            str(scene_count),
            "--output-json",
            "summary.json",
            "--output-markdown",
            "summary.md",
        ]
    )


class SummarizeDa3ResultsTests(unittest.TestCase):
    def test_lists_only_explicit_oom_retry_grouped_fallbacks(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_inputs(root, ["fallback", "default"])
            _prepare_full(root, "fallback", 0.25)
            _prepare_full(root, "default", 0.5)
            _write_reconstruction_args(
                root,
                "fallback",
                max_chunks_per_group=4,
                max_inflated_voxels=40_000,
            )
            _write_reconstruction_args(
                root,
                "default",
                max_chunks_per_group=None,
                max_inflated_voxels=None,
            )

            summary = run(_args(root, 2))

            fallback = summary["resource_fallbacks"][
                "oom_retry_grouped_joint_decode"
            ]
            self.assertTrue(fallback["deterministic"])
            self.assertEqual(fallback["scene_count"], 1)
            self.assertEqual(
                fallback["scenes"],
                [
                    {
                        "scene_id": "fallback",
                        "args_path": "outputs/fallback/reconstruction/args.json",
                        "joint_decode_max_chunks_per_group": 4,
                        "joint_decode_max_inflated_voxels": 40_000,
                    }
                ],
            )
            self.assertTrue(summary["scenes"][0]["oom_retry_grouped_fallback"]["used"])
            self.assertFalse(summary["scenes"][1]["oom_retry_grouped_fallback"]["used"])
            self.assertEqual(
                summary["protocol"]["low_vram_decoder_policy"][
                    "large_scene_max_chunks_per_group"
                ],
                8,
            )

            machine = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(machine["resource_fallbacks"], summary["resource_fallbacks"])
            markdown = (root / "summary.md").read_text(encoding="utf-8")
            self.assertIn("仅用于初次重建发生 OOM 的场景", markdown)
            self.assertIn("deterministic resource fallback", markdown)
            self.assertIn("| fallback | 4 | 40000 |", markdown)

    def test_classifies_all_states_and_aggregates_only_valid_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scene_ids = ["full", "geometry", "failed", "pending"]
            _write_inputs(root, scene_ids)
            _prepare_full(root, "full", 1.0)
            _prepare_geometry(root, "geometry", 3.0)

            failed = _paths(root, "failed")
            failed["report"].mkdir(parents=True, exist_ok=True)
            failed["reconstruct_profile"].write_text(
                json.dumps(
                    {
                        "success": False,
                        "return_code": 1,
                        "artifacts": [],
                    }
                ),
                encoding="utf-8",
            )
            (failed["report"] / "batch_driver.log").write_text(
                "Traceback\nRuntimeError: CUDA out of memory\n", encoding="utf-8"
            )
            batch = root / "reports" / "generated" / "da3_batch_summary.json"
            batch.write_text(
                json.dumps(
                    {
                        "finished_utc": "2026-01-01T00:00:00+00:00",
                        "records": [
                            {"scene_id": "full", "state": "complete"},
                            {"scene_id": "geometry", "state": "failed"},
                            {
                                "scene_id": "failed",
                                "state": "failed",
                                "return_code": 1,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            summary = run(_args(root, len(scene_ids)))

            self.assertEqual(
                [scene["status"] for scene in summary["scenes"]],
                [
                    "full_complete",
                    "geometry_complete",
                    "reconstruction_failed",
                    "not_attempted",
                ],
            )
            self.assertEqual(summary["counts"]["metric_eligible"], 2)
            aggregate = summary["metric_aggregate"]["chamfer_symmetric_mean_m"]
            self.assertEqual(aggregate, {"count": 2, "mean": 2.0, "median": 2.0})
            full = summary["scenes"][0]
            self.assertEqual(full["mesh"]["vertices"], 3)
            self.assertEqual(full["mesh"]["faces"], 1)
            self.assertEqual(full["mesh"]["count_source"], "ply_header")
            self.assertEqual(full["glb"]["size_bytes"], 24)
            self.assertEqual(
                full["profiles"]["glb"]["peak_gpu_global_memory_mib"], 5000.0
            )
            self.assertIn("CUDA out of memory", summary["scenes"][2]["failure_reason"])
            self.assertFalse(summary["validation"]["full_artifact_hashing"])
            self.assertEqual(
                summary["protocol"]["low_vram_decoder_policy"]["single_pass_max_chunks"],
                16,
            )

            machine = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(machine["counts"], summary["counts"])
            markdown = (root / "summary.md").read_text(encoding="utf-8")
            self.assertIn("不是严格论文复现", markdown)
            self.assertIn("16 GiB low-VRAM 执行策略", markdown)
            self.assertIn("meshlab outputs/<scene_id>/reconstruction/mesh.ply", markdown)
            self.assertFalse(list(root.glob(".*summary*tmp")))

    def test_corrupt_optional_summaries_are_warnings_not_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_inputs(root, ["complete"])
            _prepare_full(root, "complete", 0.25)
            (root / "reports/generated/preflight/summary.json").write_text(
                "{broken", encoding="utf-8"
            )
            batch = root / "reports/generated/da3_batch_summary.json"
            batch.write_text("[]", encoding="utf-8")
            postprocess = root / "reports/generated/da3_postprocess_summary.json"
            postprocess.write_text(
                json.dumps(
                    {
                        "finished_utc": "2026-01-01T00:00:00+00:00",
                        "records": [{"scene_id": "complete", "actions": None}],
                    }
                ),
                encoding="utf-8",
            )

            summary = run(_args(root, 1))

            self.assertEqual(summary["counts"]["full_complete"], 1)
            self.assertGreaterEqual(len(summary["inputs"]["warnings"]), 3)
            self.assertTrue(
                any("cannot read JSON" in warning for warning in summary["inputs"]["warnings"])
            )

    def test_structurally_invalid_glb_is_geometry_complete_even_with_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_inputs(root, ["scene"])
            _prepare_geometry(root, "scene", 0.5)
            paths = _paths(root, "scene")
            _write_glb(paths["glb"], valid=False)
            _write_profile(paths["glb_profile"], paths["glb"])

            summary = run(_args(root, 1))

            scene = summary["scenes"][0]
            self.assertEqual(scene["status"], "geometry_complete")
            self.assertFalse(scene["glb"]["valid"])
            self.assertTrue(any("GLB" in reason for reason in scene["glb"]["reasons"]))

    def test_nonfinite_or_out_of_range_metrics_never_enter_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_inputs(root, ["scene"])
            _prepare_geometry(root, "scene", 0.5)
            paths = _paths(root, "scene")
            document = json.loads(paths["metrics"].read_text(encoding="utf-8"))
            document["metrics"]["chamfer_symmetric_mean_m"] = float("nan")
            document["metrics"]["precision_at_0_1m"] = 1.5
            paths["metrics"].write_text(json.dumps(document), encoding="utf-8")

            summary = run(_args(root, 1))

            self.assertEqual(summary["scenes"][0]["status"], "reconstruction_failed")
            self.assertEqual(
                summary["metric_aggregate"]["chamfer_symmetric_mean_m"]["count"], 0
            )

    def test_refuses_to_overwrite_a_live_input_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_inputs(root, ["scene"])
            batch = root / "reports/generated/da3_batch_summary.json"
            original = '{"records": [], "sentinel": true}'
            batch.write_text(original, encoding="utf-8")
            args = build_parser().parse_args(
                [
                    "--root",
                    str(root),
                    "--expected-scene-count",
                    "1",
                    "--output-json",
                    "reports/generated/da3_batch_summary.json",
                    "--output-markdown",
                    "summary.md",
                ]
            )

            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                run(args)

            self.assertEqual(batch.read_text(encoding="utf-8"), original)

    def test_refuses_unfinished_summary_unless_partial_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_inputs(root, ["scene"])
            batch = root / "reports/generated/da3_batch_summary.json"
            batch.write_text(
                json.dumps({"finished_utc": None, "records": []}), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "summary is unfinished"):
                run(_args(root, 1))
            self.assertFalse((root / "summary.json").exists())

            args = build_parser().parse_args(
                [
                    "--root",
                    str(root),
                    "--expected-scene-count",
                    "1",
                    "--allow-partial",
                    "--output-json",
                    "summary.json",
                    "--output-markdown",
                    "summary.md",
                ]
            )
            summary = run(args)
            self.assertIn("batch summary is unfinished", summary["inputs"]["warnings"])


if __name__ == "__main__":
    unittest.main()
