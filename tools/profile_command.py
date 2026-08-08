#!/usr/bin/env python3
"""Run a command while recording reproducible CPU/GPU resource telemetry."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import psutil


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _parse_csv_numbers(line: str) -> list[float]:
    return [float(part.strip()) for part in line.split(",")]


def _gpu_sample(process_ids: set[int]) -> dict[str, float | int] | None:
    try:
        global_query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        rows = [row for row in global_query.stdout.splitlines() if row.strip()]
        if not rows:
            return None
        memory_used, memory_total, utilization, power, temperature = _parse_csv_numbers(rows[0])

        process_memory = 0.0
        compute_query = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        for row in compute_query.stdout.splitlines():
            fields = [field.strip() for field in row.split(",")]
            if len(fields) != 2:
                continue
            try:
                pid = int(fields[0])
                used = float(fields[1])
            except ValueError:
                continue
            if pid in process_ids:
                process_memory += used

        return {
            "global_memory_used_mib": memory_used,
            "global_memory_total_mib": memory_total,
            "process_memory_used_mib": process_memory,
            "utilization_percent": utilization,
            "power_w": power,
            "temperature_c": temperature,
        }
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _process_tree_metrics(process: psutil.Process) -> tuple[set[int], int]:
    processes = [process]
    try:
        processes.extend(process.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    pids: set[int] = set()
    rss_bytes = 0
    for item in processes:
        try:
            pids.add(item.pid)
            rss_bytes += item.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return pids, rss_bytes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if record["exists"]:
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = _sha256(path)
    return record


def _artifact_state(path: Path) -> tuple[int, int] | None:
    """Return a cheap identity for detecting stale outputs without pre-hashing them."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_size, stat.st_mtime_ns


def profile_command(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    result_path: Path,
    expected_files: Sequence[Path] = (),
    sample_interval_s: float = 1.0,
    progress_interval_s: float = 30.0,
    label: str = "command",
) -> dict[str, Any]:
    if not command:
        raise ValueError("command must not be empty")
    if sample_interval_s <= 0.0 or progress_interval_s <= 0.0:
        raise ValueError("sampling intervals must be positive")

    cwd = cwd.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_artifacts = [path if path.is_absolute() else cwd / path for path in expected_files]
    initial_artifact_states = {
        str(path): _artifact_state(path) for path in resolved_artifacts
    }
    start_wall = time.monotonic()
    start_utc = _utc_now()
    peak_rss_bytes = 0
    gpu_samples = 0
    gpu_peak_process_mib = 0.0
    gpu_peak_global_mib = 0.0
    gpu_peak_utilization = 0.0
    gpu_peak_power_w = 0.0
    gpu_peak_temperature_c = 0.0

    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        log.write(f"# label: {label}\n")
        log.write(f"# started_utc: {start_utc}\n")
        log.write(f"# cwd: {cwd}\n")
        log.write(f"# command: {shlex.join(command)}\n")
        log.flush()
        child = subprocess.Popen(
            list(command),
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
        tracked = psutil.Process(child.pid)
        next_progress = start_wall
        while True:
            return_code = child.poll()
            pids, rss_bytes = _process_tree_metrics(tracked)
            peak_rss_bytes = max(peak_rss_bytes, rss_bytes)
            gpu = _gpu_sample(pids)
            if gpu is not None:
                gpu_samples += 1
                gpu_peak_process_mib = max(
                    gpu_peak_process_mib, float(gpu["process_memory_used_mib"])
                )
                gpu_peak_global_mib = max(
                    gpu_peak_global_mib, float(gpu["global_memory_used_mib"])
                )
                gpu_peak_utilization = max(
                    gpu_peak_utilization, float(gpu["utilization_percent"])
                )
                gpu_peak_power_w = max(gpu_peak_power_w, float(gpu["power_w"]))
                gpu_peak_temperature_c = max(
                    gpu_peak_temperature_c, float(gpu["temperature_c"])
                )

            now = time.monotonic()
            if now >= next_progress:
                process_gpu = float(gpu["process_memory_used_mib"]) if gpu else 0.0
                print(
                    f"[{label}] elapsed={now - start_wall:.1f}s "
                    f"rss={rss_bytes / 2**30:.2f}GiB "
                    f"gpu_process={process_gpu:.0f}MiB "
                    f"gpu_process_peak={gpu_peak_process_mib:.0f}MiB",
                    flush=True,
                )
                next_progress = now + progress_interval_s
            if return_code is not None:
                break
            time.sleep(sample_interval_s)

    end_utc = _utc_now()
    duration_s = time.monotonic() - start_wall
    artifacts = []
    stale_artifacts = []
    for path in resolved_artifacts:
        record = _artifact_record(path)
        initial_state = initial_artifact_states[str(path)]
        final_state = _artifact_state(path)
        record["existed_before_run"] = initial_state is not None
        record["produced_or_updated_this_run"] = (
            final_state is not None and final_state != initial_state
        )
        artifacts.append(record)
        if not record["produced_or_updated_this_run"]:
            stale_artifacts.append(record["path"])
    missing_artifacts = [record["path"] for record in artifacts if not record["exists"]]
    result = {
        "schema": "genrecon.profiled-command",
        "schema_version": 1,
        "label": label,
        "command": list(command),
        "command_shell_escaped": shlex.join(command),
        "cwd": str(cwd),
        "started_utc": start_utc,
        "finished_utc": end_utc,
        "duration_s": duration_s,
        "return_code": return_code,
        "success": return_code == 0 and not stale_artifacts,
        "peak_process_tree_rss_bytes": peak_rss_bytes,
        "gpu": {
            "samples": gpu_samples,
            "peak_process_memory_mib": gpu_peak_process_mib,
            "peak_global_memory_mib": gpu_peak_global_mib,
            "peak_utilization_percent": gpu_peak_utilization,
            "peak_power_w": gpu_peak_power_w,
            "peak_temperature_c": gpu_peak_temperature_c,
        },
        "log_path": str(log_path.resolve()),
        "artifacts": artifacts,
        "missing_expected_artifacts": missing_artifacts,
        "stale_expected_artifacts": stale_artifacts,
    }
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--label", default="command")
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--progress-interval", type=float, default=30.0)
    parser.add_argument("--expect-file", action="append", type=Path, default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        result = profile_command(
            command,
            cwd=args.cwd,
            log_path=args.log,
            result_path=args.result,
            expected_files=args.expect_file,
            sample_interval_s=args.sample_interval,
            progress_interval_s=args.progress_interval,
            label=args.label,
        )
    except (OSError, ValueError) as exc:
        print(f"profiled command failed to start: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["success"] else (result["return_code"] or 3)


if __name__ == "__main__":
    raise SystemExit(main())
