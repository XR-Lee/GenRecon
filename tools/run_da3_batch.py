#!/usr/bin/env python3
"""Run the DA3 ScanNet++ smoke pipeline sequentially with resumable validation."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc


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


def _validated_profile_artifact(profile_path: Path, artifact_path: Path) -> str | None:
    if not profile_path.is_file():
        return f"missing profile {profile_path}"
    try:
        profile = _load_json(profile_path)
    except ValueError as exc:
        return str(exc)
    if profile.get("success") is not True or profile.get("return_code") != 0:
        return f"profile did not succeed: {profile_path}"
    if profile.get("missing_expected_artifacts") or profile.get("stale_expected_artifacts"):
        return f"profile reports missing/stale artifacts: {profile_path}"
    if not artifact_path.is_file() or artifact_path.stat().st_size <= 0:
        return f"missing or empty artifact {artifact_path}"

    resolved = artifact_path.resolve()
    record = next(
        (
            item
            for item in profile.get("artifacts", [])
            if Path(item.get("path", "")).resolve() == resolved
        ),
        None,
    )
    if record is None:
        return f"profile has no record for {artifact_path}"
    if record.get("produced_or_updated_this_run") is not True:
        return f"profile did not mark artifact fresh: {artifact_path}"
    if record.get("size_bytes") != artifact_path.stat().st_size:
        return f"artifact size differs from profile: {artifact_path}"
    if record.get("sha256") != _sha256(artifact_path):
        return f"artifact hash differs from profile: {artifact_path}"
    return None


def inspect_scene(root: Path, scene_id: str) -> dict[str, Any]:
    output = root / "outputs" / scene_id / "reconstruction"
    reports = root / "reports" / "generated" / scene_id
    mesh = output / "mesh.ply"
    glb = output / "scene.glb"
    metrics_path = reports / "mesh_metrics_unclipped.json"

    reasons: list[str] = []
    for profile_path, artifact_path in (
        (reports / "reconstruct_profile.json", mesh),
        (reports / "glb_profile.json", glb),
    ):
        reason = _validated_profile_artifact(profile_path, artifact_path)
        if reason is not None:
            reasons.append(reason)

    for intermediate in (output / "to_glb_inputs.pt", output / "chunk_inputs.pt"):
        if not intermediate.is_file() or intermediate.stat().st_size <= 0:
            reasons.append(f"missing or empty intermediate {intermediate}")

    metrics: dict[str, Any] | None = None
    if not metrics_path.is_file():
        reasons.append(f"missing metrics {metrics_path}")
    else:
        try:
            loaded_metrics = _load_json(metrics_path)
            if not isinstance(loaded_metrics, dict):
                raise ValueError(f"metrics root is not an object: {metrics_path}")
            metrics = loaded_metrics
            if metrics.get("schema") != "genrecon.mesh-evaluation":
                reasons.append(f"unexpected metrics schema: {metrics_path}")
            if metrics.get("protocol") != "paper-like-unclipped":
                reasons.append(f"unexpected metrics protocol: {metrics_path}")
            if metrics.get("sampling", {}).get("samples_per_mesh") != 200_000:
                reasons.append(f"unexpected metrics sample count: {metrics_path}")
            if not _all_finite(metrics):
                reasons.append(f"metrics contain non-finite values: {metrics_path}")
        except ValueError as exc:
            reasons.append(str(exc))

    return {
        "complete": not reasons,
        "reasons": reasons,
        "mesh": str(mesh),
        "mesh_size_bytes": mesh.stat().st_size if mesh.is_file() else None,
        "glb": str(glb),
        "glb_size_bytes": glb.stat().st_size if glb.is_file() else None,
        "metrics": metrics.get("metrics") if metrics is not None else None,
    }


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _terminate_process_group(process: subprocess.Popen[Any], grace_s: float = 300.0) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def _run_scene(
    root: Path,
    scene_id: str,
    *,
    pipeline_config: Path,
    offline: bool,
    timeout_s: float,
) -> tuple[int | None, bool, float, Path]:
    report_dir = root / "reports" / "generated" / scene_id
    report_dir.mkdir(parents=True, exist_ok=True)
    log_path = report_dir / "batch_driver.log"
    env = os.environ.copy()
    env.update(
        {
            "SCENE_ID": scene_id,
            "PIPELINE_CONFIG": str(pipeline_config),
        }
    )
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
    else:
        env.pop("HF_HUB_OFFLINE", None)

    started = time.monotonic()
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        log.write(f"# scene_id: {scene_id}\n")
        log.write(f"# started_utc: {_utc_now()}\n")
        log.write(f"# timeout_s: {timeout_s}\n")
        log.flush()
        process = subprocess.Popen(
            ["bash", "scripts/run_smoke.sh"],
            cwd=root,
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
            _terminate_process_group(process)
            return_code = process.returncode
            log.write(f"# timed_out_utc: {_utc_now()}\n")
    return return_code, timed_out, time.monotonic() - started, log_path


def run_batch(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    manifest_path = (root / args.manifest).resolve()
    pipeline_config = (root / args.pipeline_config).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    if not pipeline_config.is_file():
        raise FileNotFoundError(f"Pipeline config not found: {pipeline_config}")
    manifest = _load_json(manifest_path)
    manifest_scenes = manifest.get("scene_ids")
    if not isinstance(manifest_scenes, list) or not all(
        isinstance(scene_id, str) for scene_id in manifest_scenes
    ):
        raise ValueError(f"Invalid scene_ids in {manifest_path}")

    requested = args.scene_ids or manifest_scenes
    unknown = sorted(set(requested) - set(manifest_scenes))
    if unknown:
        raise ValueError(f"Requested scene IDs are absent from manifest: {unknown}")
    scenes = list(dict.fromkeys(requested))

    summary_path = (root / args.summary).resolve()
    lock_path = summary_path.with_suffix(summary_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"Another DA3 batch owns {lock_path}") from exc

    summary: dict[str, Any] = {
        "schema": "genrecon.da3-batch",
        "schema_version": 1,
        "started_utc": _utc_now(),
        "finished_utc": None,
        "root": str(root),
        "manifest": str(manifest_path),
        "pipeline_config": str(pipeline_config),
        "offline": args.offline,
        "timeout_s_per_scene": args.timeout_seconds,
        "scene_ids": scenes,
        "records": [],
    }
    _atomic_write_json(summary_path, summary)

    try:
        for index, scene_id in enumerate(scenes, start=1):
            before = inspect_scene(root, scene_id)
            print(
                f"[{index}/{len(scenes)}] {scene_id}: "
                f"{'already complete' if before['complete'] else 'running'}",
                flush=True,
            )
            if before["complete"]:
                record = {
                    "scene_id": scene_id,
                    "state": "skipped-complete",
                    "started_utc": None,
                    "finished_utc": _utc_now(),
                    "duration_s": 0.0,
                    "return_code": None,
                    "timed_out": False,
                    "inspection": before,
                }
            elif args.dry_run:
                record = {
                    "scene_id": scene_id,
                    "state": "dry-run-pending",
                    "started_utc": None,
                    "finished_utc": _utc_now(),
                    "duration_s": 0.0,
                    "return_code": None,
                    "timed_out": False,
                    "inspection": before,
                }
            else:
                free_bytes = shutil.disk_usage(root).free
                if free_bytes < args.min_free_gib * 2**30:
                    record = {
                        "scene_id": scene_id,
                        "state": "blocked-low-disk",
                        "started_utc": None,
                        "finished_utc": _utc_now(),
                        "duration_s": 0.0,
                        "return_code": None,
                        "timed_out": False,
                        "free_bytes": free_bytes,
                        "inspection": before,
                    }
                    summary["records"].append(record)
                    _atomic_write_json(summary_path, summary)
                    break

                started_utc = _utc_now()
                return_code, timed_out, duration_s, log_path = _run_scene(
                    root,
                    scene_id,
                    pipeline_config=pipeline_config,
                    offline=args.offline,
                    timeout_s=args.timeout_seconds,
                )
                after = inspect_scene(root, scene_id)
                record = {
                    "scene_id": scene_id,
                    "state": "complete" if after["complete"] else "failed",
                    "started_utc": started_utc,
                    "finished_utc": _utc_now(),
                    "duration_s": duration_s,
                    "return_code": return_code,
                    "timed_out": timed_out,
                    "log_path": str(log_path),
                    "inspection": after,
                }
                print(
                    f"[{index}/{len(scenes)}] {scene_id}: {record['state']} "
                    f"in {duration_s:.1f}s (rc={return_code}, timeout={timed_out})",
                    flush=True,
                )
            summary["records"].append(record)
            _atomic_write_json(summary_path, summary)
    finally:
        summary["finished_utc"] = _utc_now()
        _atomic_write_json(summary_path, summary)
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()

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
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=Path("data/hf-cache/converted/dinov3-vitl16-timm-qkvb/pipeline.json"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/generated/da3_batch_summary.json"),
    )
    parser.add_argument("--scene", action="append", dest="scene_ids")
    parser.add_argument("--timeout-seconds", type=float, default=7_200.0)
    parser.add_argument("--min-free-gib", type=float, default=10.0)
    parser.add_argument(
        "--offline",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.timeout_seconds <= 0 or args.min_free_gib < 0:
        raise SystemExit("timeout and minimum free disk must be positive")
    try:
        return run_batch(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"batch setup failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
