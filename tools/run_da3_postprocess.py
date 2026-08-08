#!/usr/bin/env python3
"""Recover DA3 ScanNet++ metrics and GLBs without rerunning reconstruction.

The tool is intentionally narrower than ``run_da3_batch.py``.  It trusts a
scene only when the reconstruction profile validates the mesh and both saved
GLB inputs, then it may run the existing mesh evaluator and/or the resumable
chunked GLB converter.  It never invokes ``reconstruct_scene.py`` or the smoke
pipeline.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


METRICS_SCHEMA = "genrecon.mesh-evaluation"
METRICS_SCHEMA_VERSION = 1
METRICS_PROTOCOL = "paper-like-unclipped"
METRICS_NUM_SAMPLES = 200_000
METRICS_SEED = 42
REQUIRED_METRICS = {
    "pred_to_gt_mean_m",
    "gt_to_pred_mean_m",
    "chamfer_symmetric_mean_m",
    "precision_at_0_1m",
    "recall_at_0_1m",
    "fscore_arithmetic_at_0_1m",
    "fscore_harmonic_at_0_1m",
    "normal_consistency_pred_to_gt",
    "normal_consistency_gt_to_pred",
    "normal_consistency_symmetric_mean",
    "normal_correspondence_fraction_pred_to_gt",
    "normal_correspondence_fraction_gt_to_pred",
}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    return False


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _json_temporary_path(path)
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_temporary_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".tmp")


def _reject_path_aliases(paths: dict[str, Path]) -> None:
    """Reject aliases that could replace an inode while another process locks it."""

    items = list(paths.items())
    for index, (left_name, left_path) in enumerate(items):
        for right_name, right_path in items[index + 1 :]:
            same = left_path.resolve() == right_path.resolve()
            if not same and left_path.exists() and right_path.exists():
                try:
                    same = left_path.samefile(right_path)
                except OSError:
                    same = False
            if same:
                raise ValueError(
                    f"unsafe path alias: {left_name} and {right_name} both resolve to "
                    f"{left_path.resolve()}"
                )


def _profile_artifact_reasons(profile_path: Path, artifacts: Sequence[Path]) -> list[str]:
    reasons: list[str] = []
    if not profile_path.is_file():
        return [f"missing profile {profile_path}"]
    try:
        profile = _load_json(profile_path)
    except ValueError as exc:
        return [str(exc)]
    if not isinstance(profile, dict):
        return [f"profile root is not an object: {profile_path}"]
    if profile.get("success") is not True or profile.get("return_code") != 0:
        reasons.append(f"profile did not succeed: {profile_path}")
    if profile.get("missing_expected_artifacts") or profile.get("stale_expected_artifacts"):
        reasons.append(f"profile reports missing/stale artifacts: {profile_path}")

    records: dict[Path, dict[str, Any]] = {}
    raw_records = profile.get("artifacts", [])
    if not isinstance(raw_records, list):
        reasons.append(f"profile artifacts are not a list: {profile_path}")
        raw_records = []
    for item in raw_records:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            continue
        try:
            records[Path(item["path"]).resolve()] = item
        except (OSError, RuntimeError):
            continue

    for artifact in artifacts:
        try:
            if not artifact.is_file() or artifact.stat().st_size <= 0:
                reasons.append(f"missing or empty artifact {artifact}")
                continue
            record = records.get(artifact.resolve())
            if record is None:
                reasons.append(f"profile has no record for {artifact}")
                continue
            if record.get("produced_or_updated_this_run") is not True:
                reasons.append(f"profile did not mark artifact fresh: {artifact}")
            if record.get("size_bytes") != artifact.stat().st_size:
                reasons.append(f"artifact size differs from profile: {artifact}")
            elif record.get("sha256") != _sha256(artifact):
                reasons.append(f"artifact hash differs from profile: {artifact}")
        except OSError as exc:
            reasons.append(f"cannot validate artifact {artifact}: {exc}")
    return reasons


def _normalise_input_path(value: Any, root: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _metrics_reasons(
    metrics_path: Path,
    *,
    root: Path,
    predicted_mesh: Path,
    ground_truth_mesh: Path,
) -> tuple[list[str], dict[str, Any] | None]:
    if not metrics_path.is_file():
        return [f"missing metrics {metrics_path}"], None
    try:
        metrics = _load_json(metrics_path)
    except ValueError as exc:
        return [str(exc)], None
    if not isinstance(metrics, dict):
        return [f"metrics root is not an object: {metrics_path}"], None

    reasons: list[str] = []
    if metrics.get("schema") != METRICS_SCHEMA:
        reasons.append(f"unexpected metrics schema: {metrics_path}")
    if metrics.get("schema_version") != METRICS_SCHEMA_VERSION:
        reasons.append(f"unexpected metrics schema version: {metrics_path}")
    if metrics.get("protocol") != METRICS_PROTOCOL:
        reasons.append(f"unexpected metrics protocol: {metrics_path}")
    sampling = metrics.get("sampling")
    if not isinstance(sampling, dict):
        reasons.append(f"missing metrics sampling object: {metrics_path}")
    else:
        if sampling.get("samples_per_mesh") != METRICS_NUM_SAMPLES:
            reasons.append(f"unexpected metrics sample count: {metrics_path}")
        if sampling.get("seed") != METRICS_SEED:
            reasons.append(f"unexpected metrics seed: {metrics_path}")

    inputs = metrics.get("inputs")
    if not isinstance(inputs, dict):
        reasons.append(f"missing metrics inputs object: {metrics_path}")
    else:
        predicted = _normalise_input_path(inputs.get("predicted_mesh"), root)
        ground_truth = _normalise_input_path(inputs.get("ground_truth_mesh"), root)
        if predicted != predicted_mesh.resolve():
            reasons.append(f"metrics predicted mesh does not match scene: {metrics_path}")
        if ground_truth != ground_truth_mesh.resolve():
            reasons.append(f"metrics ground-truth mesh does not match scene: {metrics_path}")

    crop = metrics.get("crop")
    if not isinstance(crop, dict) or crop.get("mode") != "none":
        reasons.append(f"metrics are not the unclipped protocol output: {metrics_path}")

    values = metrics.get("metrics")
    if not isinstance(values, dict):
        reasons.append(f"missing metrics values object: {metrics_path}")
    else:
        missing = sorted(REQUIRED_METRICS - set(values))
        if missing:
            reasons.append(f"metrics values are incomplete ({', '.join(missing)}): {metrics_path}")
        for name in REQUIRED_METRICS & set(values):
            value = values[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                reasons.append(f"metric {name} is not finite numeric data: {metrics_path}")
    if not _all_finite(metrics):
        reasons.append(f"metrics contain non-finite values: {metrics_path}")
    return reasons, metrics


def _scene_paths(
    scene_id: str,
    *,
    data_root: Path,
    output_root: Path,
    report_root: Path,
) -> dict[str, Path]:
    output = output_root / scene_id / "reconstruction"
    report = report_root / scene_id
    return {
        "output": output,
        "report": report,
        "mesh": output / "mesh.ply",
        "to_glb_inputs": output / "to_glb_inputs.pt",
        "chunk_inputs": output / "chunk_inputs.pt",
        "chunks": output / "chunks",
        "glb": output / "scene.glb",
        "reconstruct_profile": report / "reconstruct_profile.json",
        "glb_profile": report / "glb_profile.json",
        "metrics": report / "mesh_metrics_unclipped.json",
        "ground_truth": data_root / scene_id / "scans" / "mesh_aligned_0.05.ply",
    }


def inspect_postprocess_scene(
    root: Path,
    scene_id: str,
    *,
    data_root: Path | None = None,
    output_root: Path | None = None,
    report_root: Path | None = None,
) -> dict[str, Any]:
    """Validate the immutable reconstruction and the two recoverable products."""

    root = root.resolve()
    data_root = data_root or root / "data" / "da3-adapted"
    output_root = output_root or root / "outputs"
    report_root = report_root or root / "reports" / "generated"
    paths = _scene_paths(
        scene_id,
        data_root=data_root,
        output_root=output_root,
        report_root=report_root,
    )
    reconstruction_artifacts = [
        paths["mesh"],
        paths["to_glb_inputs"],
        paths["chunk_inputs"],
    ]
    reconstruction_reasons = _profile_artifact_reasons(
        paths["reconstruct_profile"], reconstruction_artifacts
    )
    metrics_reasons, loaded_metrics = _metrics_reasons(
        paths["metrics"],
        root=root,
        predicted_mesh=paths["mesh"],
        ground_truth_mesh=paths["ground_truth"],
    )
    glb_reasons = _profile_artifact_reasons(paths["glb_profile"], [paths["glb"]])
    return {
        "scene_id": scene_id,
        "complete": not reconstruction_reasons and not metrics_reasons and not glb_reasons,
        "reconstruction": {
            "valid": not reconstruction_reasons,
            "reasons": reconstruction_reasons,
            "profile": str(paths["reconstruct_profile"]),
            "mesh": str(paths["mesh"]),
            "to_glb_inputs": str(paths["to_glb_inputs"]),
            "chunk_inputs": str(paths["chunk_inputs"]),
        },
        "metrics": {
            "valid": not metrics_reasons,
            "reasons": metrics_reasons,
            "path": str(paths["metrics"]),
            "values": loaded_metrics.get("metrics") if loaded_metrics else None,
        },
        "glb": {
            "valid": not glb_reasons,
            "reasons": glb_reasons,
            "profile": str(paths["glb_profile"]),
            "path": str(paths["glb"]),
        },
        "ground_truth": str(paths["ground_truth"]),
    }


def plan_scene(inspection: dict[str, Any]) -> list[str]:
    if not inspection["reconstruction"]["valid"]:
        return []
    actions: list[str] = []
    if not inspection["metrics"]["valid"]:
        actions.append("metrics")
    if not inspection["glb"]["valid"]:
        actions.append("glb")
    return actions


def _terminate_process_group(process: subprocess.Popen[Any], grace_s: float) -> None:
    if process.poll() is not None:
        return
    process_group = process.pid
    try:
        os.killpg(process_group, signal.SIGINT)
    except ProcessLookupError:
        process.wait()
        return

    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        process.poll()  # reap the direct child without assuming its group is gone
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        process.wait()


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    timeout_s: float,
    kill_grace_s: float,
) -> dict[str, Any]:
    if not command:
        raise ValueError("command must not be empty")
    if timeout_s <= 0:
        raise ValueError("command timeout must be positive")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        log.write(f"# started_utc: {_utc_now()}\n")
        log.write(f"# timeout_s: {timeout_s}\n")
        log.write(f"# command: {shlex.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        timed_out = False
        try:
            return_code = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_group(process, grace_s=kill_grace_s)
            return_code = process.returncode
            log.write(f"# timed_out_utc: {_utc_now()}\n")
    return {
        "command": list(command),
        "return_code": return_code,
        "timed_out": timed_out,
        "duration_s": time.monotonic() - started,
        "log_path": str(log_path),
    }


CommandRunner = Callable[..., dict[str, Any]]


def _base_action(name: str, command: Sequence[str], timeout_s: float) -> dict[str, Any]:
    return {
        "name": name,
        "state": "running",
        "started_utc": _utc_now(),
        "finished_utc": None,
        "timeout_s": timeout_s,
        "command": list(command),
    }


def _run_metrics_action(
    *,
    root: Path,
    paths: dict[str, Path],
    timeout_s: float,
    kill_grace_s: float,
    env: dict[str, str],
    command_runner: CommandRunner,
) -> dict[str, Any]:
    if not paths["ground_truth"].is_file():
        return {
            "name": "metrics",
            "state": "blocked-missing-ground-truth",
            "started_utc": None,
            "finished_utc": _utc_now(),
            "duration_s": 0.0,
            "reason": f"missing ground-truth mesh {paths['ground_truth']}",
        }

    paths["report"].mkdir(parents=True, exist_ok=True)
    temporary = paths["metrics"].with_name(
        f".{paths['metrics'].name}.postprocess.{os.getpid()}.tmp"
    )
    temporary.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(root / "tools" / "evaluate_mesh.py"),
        str(paths["mesh"]),
        str(paths["ground_truth"]),
        "--num-samples",
        str(METRICS_NUM_SAMPLES),
        "--seed",
        str(METRICS_SEED),
        "--output",
        str(temporary),
    ]
    action = _base_action("metrics", command, timeout_s)
    try:
        result = command_runner(
            command,
            cwd=root,
            env=env,
            log_path=paths["report"] / "mesh_metrics_postprocess.log",
            timeout_s=timeout_s,
            kill_grace_s=kill_grace_s,
        )
        action.update(result)
        if result.get("timed_out"):
            action["state"] = "timed-out"
        elif result.get("return_code") != 0:
            action["state"] = "failed"
        else:
            reasons, _ = _metrics_reasons(
                temporary,
                root=root,
                predicted_mesh=paths["mesh"],
                ground_truth_mesh=paths["ground_truth"],
            )
            if reasons:
                action["state"] = "failed-validation"
                action["validation_reasons"] = reasons
            else:
                temporary.replace(paths["metrics"])
                action["state"] = "complete"
    except Exception as exc:  # keep later actions/scenes recoverable
        action["state"] = "failed-to-run"
        action["error"] = f"{type(exc).__name__}: {exc}"
        action.setdefault("duration_s", 0.0)
    finally:
        temporary.unlink(missing_ok=True)
        action["finished_utc"] = _utc_now()
    return action


def _run_glb_action(
    *,
    root: Path,
    scene_id: str,
    paths: dict[str, Path],
    timeout_s: float,
    kill_grace_s: float,
    env: dict[str, str],
    command_runner: CommandRunner,
) -> dict[str, Any]:
    paths["report"].mkdir(parents=True, exist_ok=True)
    temporary_profile = paths["glb_profile"].with_name(
        f".{paths['glb_profile'].name}.postprocess.{os.getpid()}.tmp"
    )
    temporary_profile.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(root / "tools" / "profile_command.py"),
        "--cwd",
        str(root),
        "--label",
        f"genrecon-glb-{scene_id}-postprocess",
        "--log",
        str(paths["report"] / "glb.log"),
        "--result",
        str(temporary_profile),
        "--expect-file",
        str(paths["glb"]),
        "--",
        sys.executable,
        str(root / "chunked_to_glb.py"),
        "--inputs",
        str(paths["to_glb_inputs"]),
        "--chunk_inputs",
        str(paths["chunk_inputs"]),
        "--output_dir",
        str(paths["output"]),
        "--chunks_dir",
        str(paths["chunks"]),
    ]
    action = _base_action("glb", command, timeout_s)
    try:
        result = command_runner(
            command,
            cwd=root,
            env=env,
            log_path=paths["report"] / "glb_postprocess_driver.log",
            timeout_s=timeout_s,
            kill_grace_s=kill_grace_s,
        )
        action.update(result)
        if result.get("timed_out"):
            action["state"] = "timed-out"
        elif result.get("return_code") != 0:
            action["state"] = "failed"
        else:
            reasons = _profile_artifact_reasons(temporary_profile, [paths["glb"]])
            if reasons:
                action["state"] = "failed-validation"
                action["validation_reasons"] = reasons
            else:
                temporary_profile.replace(paths["glb_profile"])
                action["state"] = "complete"
    except Exception as exc:  # keep later scenes recoverable
        action["state"] = "failed-to-run"
        action["error"] = f"{type(exc).__name__}: {exc}"
        action.setdefault("duration_s", 0.0)
    finally:
        temporary_profile.unlink(missing_ok=True)
        action["finished_utc"] = _utc_now()
    return action


def _process_scene(
    *,
    root: Path,
    scene_id: str,
    data_root: Path,
    output_root: Path,
    report_root: Path,
    timeout_s: float,
    kill_grace_s: float,
    offline: bool,
    dry_run: bool,
    command_runner: CommandRunner,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    started_clock = clock()
    deadline = started_clock + timeout_s
    started_utc = _utc_now()
    before = inspect_postprocess_scene(
        root,
        scene_id,
        data_root=data_root,
        output_root=output_root,
        report_root=report_root,
    )
    planned = plan_scene(before)
    record: dict[str, Any] = {
        "scene_id": scene_id,
        "state": "running",
        "started_utc": started_utc,
        "finished_utc": None,
        "duration_s": None,
        "planned_actions": planned,
        "actions": [],
        "inspection_before": before,
        "inspection_after": before,
    }
    if before["complete"]:
        record["state"] = "skipped-complete"
    elif not before["reconstruction"]["valid"]:
        record["state"] = "blocked-invalid-reconstruction"
    elif dry_run:
        record["state"] = "dry-run-pending"
    else:
        paths = _scene_paths(
            scene_id,
            data_root=data_root,
            output_root=output_root,
            report_root=report_root,
        )
        env = os.environ.copy()
        if offline:
            env["HF_HUB_OFFLINE"] = "1"
        else:
            env.pop("HF_HUB_OFFLINE", None)

        timed_out = False
        for name in planned:
            remaining = deadline - clock()
            if remaining <= 0:
                record["actions"].append(
                    {
                        "name": name,
                        "state": "not-run-time-budget-exhausted",
                        "started_utc": None,
                        "finished_utc": _utc_now(),
                        "duration_s": 0.0,
                    }
                )
                timed_out = True
                break
            if name == "metrics":
                action = _run_metrics_action(
                    root=root,
                    paths=paths,
                    timeout_s=remaining,
                    kill_grace_s=kill_grace_s,
                    env=env,
                    command_runner=command_runner,
                )
            else:
                action = _run_glb_action(
                    root=root,
                    scene_id=scene_id,
                    paths=paths,
                    timeout_s=remaining,
                    kill_grace_s=kill_grace_s,
                    env=env,
                    command_runner=command_runner,
                )
            record["actions"].append(action)
            if action["state"] == "timed-out":
                timed_out = True
                break

        after = inspect_postprocess_scene(
            root,
            scene_id,
            data_root=data_root,
            output_root=output_root,
            report_root=report_root,
        )
        record["inspection_after"] = after
        if after["complete"]:
            record["state"] = "complete"
        elif timed_out:
            record["state"] = "timed-out"
        else:
            record["state"] = "failed"

    record["finished_utc"] = _utc_now()
    record["duration_s"] = max(0.0, clock() - started_clock)
    return record


def _validate_scene_id(scene_id: str) -> None:
    if not scene_id or Path(scene_id).name != scene_id or scene_id in {".", ".."}:
        raise ValueError(f"unsafe scene ID: {scene_id!r}")


def _open_lock(path: Path, owner_message: str) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"{owner_message}: {path}") from exc
    return handle


def _close_lock(handle: Any) -> None:
    fcntl.flock(handle, fcntl.LOCK_UN)
    handle.close()


def _state_counts(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        state = str(record["state"])
        counts[state] = counts.get(state, 0) + 1
    return dict(sorted(counts.items()))


def run_postprocess(
    args: argparse.Namespace,
    *,
    command_runner: CommandRunner | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        raise ValueError("timeout must be finite and positive")
    if not math.isfinite(args.kill_grace_seconds) or args.kill_grace_seconds < 0:
        raise ValueError("kill grace must be finite and non-negative")
    if args.expected_scene_count < 0:
        raise ValueError("expected scene count must be non-negative")

    root = args.root.resolve()
    manifest_path = _resolve(root, args.manifest)
    data_root = _resolve(root, args.data_root)
    output_root = _resolve(root, args.output_root)
    report_root = _resolve(root, args.report_root)
    summary_path = _resolve(root, args.summary)
    batch_summary_path = _resolve(root, args.batch_summary)
    batch_lock_path = batch_summary_path.with_suffix(batch_summary_path.suffix + ".lock")
    postprocess_lock_path = summary_path.with_suffix(summary_path.suffix + ".lock")
    _reject_path_aliases(
        {
            "postprocess summary": summary_path,
            "postprocess summary temporary": _json_temporary_path(summary_path),
            "postprocess lock": postprocess_lock_path,
            "batch summary": batch_summary_path,
            "batch summary temporary": _json_temporary_path(batch_summary_path),
            "batch lock": batch_lock_path,
        }
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dataset manifest not found: {manifest_path}")
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest root is not an object: {manifest_path}")
    manifest_scenes = manifest.get("scene_ids")
    if not isinstance(manifest_scenes, list) or not all(
        isinstance(scene_id, str) for scene_id in manifest_scenes
    ):
        raise ValueError(f"invalid scene_ids in {manifest_path}")
    if len(manifest_scenes) != len(set(manifest_scenes)):
        raise ValueError(f"manifest contains duplicate scene IDs: {manifest_path}")
    for scene_id in manifest_scenes:
        _validate_scene_id(scene_id)
    if args.expected_scene_count and len(manifest_scenes) != args.expected_scene_count:
        raise ValueError(
            f"expected {args.expected_scene_count} manifest scenes, found {len(manifest_scenes)}"
        )
    requested = args.scene_ids or manifest_scenes
    unknown = sorted(set(requested) - set(manifest_scenes))
    if unknown:
        raise ValueError(f"requested scene IDs are absent from manifest: {unknown}")
    scenes = list(dict.fromkeys(requested))

    runner = command_runner or _run_command
    batch_lock = _open_lock(
        batch_lock_path,
        "active DA3 batch owns lock; wait before postprocessing",
    )
    postprocess_lock = None
    try:
        postprocess_lock = _open_lock(
            postprocess_lock_path, "another DA3 postprocessor owns lock"
        )
        summary: dict[str, Any] = {
            "schema": "genrecon.da3-postprocess",
            "schema_version": 1,
            "started_utc": _utc_now(),
            "finished_utc": None,
            "root": str(root),
            "manifest": str(manifest_path),
            "data_root": str(data_root),
            "output_root": str(output_root),
            "report_root": str(report_root),
            "batch_summary": str(batch_summary_path),
            "offline": args.offline,
            "dry_run": args.dry_run,
            "timeout_s_per_scene": args.timeout_seconds,
            "kill_grace_s": args.kill_grace_seconds,
            "timeout_note": (
                "One deadline is shared by recovery subprocesses in a scene; artifact "
                "validation I/O and process-group termination grace are outside the "
                "interruptible subprocess budget."
            ),
            "scene_ids": scenes,
            "counts": {},
            "records": [],
        }
        _atomic_write_json(summary_path, summary)
        try:
            for index, scene_id in enumerate(scenes, start=1):
                print(f"[{index}/{len(scenes)}] {scene_id}: inspecting", flush=True)
                try:
                    record = _process_scene(
                        root=root,
                        scene_id=scene_id,
                        data_root=data_root,
                        output_root=output_root,
                        report_root=report_root,
                        timeout_s=args.timeout_seconds,
                        kill_grace_s=args.kill_grace_seconds,
                        offline=args.offline,
                        dry_run=args.dry_run,
                        command_runner=runner,
                        clock=clock,
                    )
                except Exception as exc:  # corrupt one scene must not stop the other 19
                    record = {
                        "scene_id": scene_id,
                        "state": "internal-error",
                        "started_utc": None,
                        "finished_utc": _utc_now(),
                        "duration_s": 0.0,
                        "planned_actions": [],
                        "actions": [],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                summary["records"].append(record)
                summary["counts"] = _state_counts(summary["records"])
                _atomic_write_json(summary_path, summary)
                print(f"[{index}/{len(scenes)}] {scene_id}: {record['state']}", flush=True)
        finally:
            summary["finished_utc"] = _utc_now()
            summary["counts"] = _state_counts(summary["records"])
            _atomic_write_json(summary_path, summary)
    finally:
        if postprocess_lock is not None:
            _close_lock(postprocess_lock)
        _close_lock(batch_lock)

    if args.dry_run:
        return 0
    complete_states = {"complete", "skipped-complete"}
    return 0 if all(record["state"] in complete_states for record in summary["records"]) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/da3-adapted/da3_scannetpp_manifest.json"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/da3-adapted"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--report-root", type=Path, default=Path("reports/generated")
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/generated/da3_postprocess_summary.json"),
    )
    parser.add_argument(
        "--batch-summary",
        type=Path,
        default=Path("reports/generated/da3_batch_summary.json"),
        help="Its lock is held to prevent races with the reconstruction batch.",
    )
    parser.add_argument("--scene", action="append", dest="scene_ids")
    parser.add_argument(
        "--expected-scene-count",
        type=int,
        default=20,
        help="Expected manifest size; use 0 to disable the check.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=7_200.0,
        help=(
            "Shared recovery-subprocess budget per scene; validation I/O and "
            "termination grace may add wall time."
        ),
    )
    parser.add_argument("--kill-grace-seconds", type=float, default=30.0)
    parser.add_argument(
        "--offline",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run_postprocess(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"postprocess setup failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
