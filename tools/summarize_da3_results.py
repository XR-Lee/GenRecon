#!/usr/bin/env python3
"""Build a lightweight, read-only summary of the DA3 ScanNet++ run.

Only output report files are written.  Meshes are validated from their PLY
headers, GLBs from their fixed-size headers, and profiled artifacts by path and
size.  In particular, this tool deliberately does *not* hash or load the large
mesh/GLB artifacts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import statistics
import struct
import sys
from pathlib import Path
from typing import Any


REQUIRED_METRICS = (
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
)
DISTANCE_METRICS = {
    "pred_to_gt_mean_m",
    "gt_to_pred_mean_m",
    "chamfer_symmetric_mean_m",
}
UNIT_INTERVAL_METRICS = set(REQUIRED_METRICS) - DISTANCE_METRICS
PRIMARY_METRICS = (
    "chamfer_symmetric_mean_m",
    "fscore_arithmetic_at_0_1m",
    "precision_at_0_1m",
    "recall_at_0_1m",
    "normal_consistency_symmetric_mean",
)
OOM_RETRY_GROUPED_ARG_NAMES = (
    "joint_decode_max_chunks_per_group",
    "joint_decode_max_inflated_voxels",
)
STATUS_ORDER = (
    "full_complete",
    "geometry_complete",
    "reconstruction_failed",
    "not_attempted",
)
STATUS_ZH = {
    "full_complete": "完整（几何+指标+GLB）",
    "geometry_complete": "几何完成（GLB 缺失/失败）",
    "reconstruction_failed": "运行后未形成有效几何结果",
    "not_attempted": "未尝试",
}
METRIC_ZH = {
    "chamfer_symmetric_mean_m": "Chamfer 对称均值（m）",
    "fscore_arithmetic_at_0_1m": "F-score 算术均值@0.1m",
    "precision_at_0_1m": "Precision@0.1m",
    "recall_at_0_1m": "Recall@0.1m",
    "normal_consistency_symmetric_mean": "法向一致性对称均值",
}
_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _safe_json(path: Path, max_bytes: int = 16 * 1024 * 1024) -> tuple[Any | None, str | None]:
    try:
        if path.stat().st_size > max_bytes:
            return None, f"JSON exceeds lightweight read limit ({max_bytes} bytes): {path}"
        return json.loads(path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, f"missing JSON: {path}"
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"cannot read JSON {path}: {exc}"


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except (OSError, ValueError):
        return str(path.resolve())


def _normalise_record_path(value: Any, root: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_write_text(path, payload)


def _json_safe(value: Any) -> Any:
    """Remove non-finite or exotic values inherited from damaged input JSON."""

    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def _read_ply_header(path: Path, max_bytes: int = 1024 * 1024) -> dict[str, Any]:
    result: dict[str, Any] = {
        "valid": False,
        "format": None,
        "vertices": None,
        "faces": None,
        "header_bytes_read": 0,
        "reasons": [],
    }
    try:
        size = path.stat().st_size
        if size <= 0:
            result["reasons"].append(f"empty mesh: {path}")
            return result
        with path.open("rb") as handle:
            header = handle.read(min(size, max_bytes))
    except OSError as exc:
        result["reasons"].append(f"cannot read mesh {path}: {exc}")
        return result
    result["header_bytes_read"] = len(header)
    marker = b"end_header"
    marker_at = header.find(marker)
    if marker_at < 0:
        result["reasons"].append(f"PLY end_header not found within {max_bytes} bytes: {path}")
        return result
    line_end = header.find(b"\n", marker_at)
    header_end = len(header) if line_end < 0 else line_end + 1
    try:
        text = header[:header_end].decode("ascii")
    except UnicodeDecodeError:
        result["reasons"].append(f"PLY header is not ASCII: {path}")
        return result
    lines = [line.strip() for line in text.splitlines()]
    if not lines or lines[0] != "ply":
        result["reasons"].append(f"invalid PLY signature: {path}")
    format_lines = [line.split() for line in lines if line.startswith("format ")]
    if len(format_lines) == 1 and len(format_lines[0]) >= 3:
        result["format"] = format_lines[0][1]
        if result["format"] not in {
            "ascii",
            "binary_little_endian",
            "binary_big_endian",
        }:
            result["reasons"].append(f"unsupported PLY format in {path}: {result['format']}")
    else:
        result["reasons"].append(f"missing or ambiguous PLY format: {path}")
    elements: dict[str, int] = {}
    for line in lines:
        fields = line.split()
        if len(fields) == 3 and fields[0] == "element":
            try:
                elements[fields[1]] = int(fields[2])
            except ValueError:
                result["reasons"].append(f"invalid PLY element count in {path}: {line}")
    result["vertices"] = elements.get("vertex")
    result["faces"] = elements.get("face")
    if not isinstance(result["vertices"], int) or result["vertices"] <= 0:
        result["reasons"].append(f"PLY has no positive vertex count: {path}")
    if not isinstance(result["faces"], int) or result["faces"] <= 0:
        result["reasons"].append(f"PLY has no positive face count: {path}")
    if size <= header_end:
        result["reasons"].append(f"PLY has no body data: {path}")
    result["valid"] = not result["reasons"]
    return result


def _read_glb_header(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "valid": False,
        "version": None,
        "declared_size_bytes": None,
        "reasons": [],
    }
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            header = handle.read(20)
    except OSError as exc:
        result["reasons"].append(f"cannot read GLB {path}: {exc}")
        return result
    if len(header) < 20:
        result["reasons"].append(f"GLB header is truncated: {path}")
        return result
    magic, version, declared_size = struct.unpack("<4sII", header[:12])
    json_chunk_size, json_chunk_type = struct.unpack("<I4s", header[12:20])
    result["version"] = version
    result["declared_size_bytes"] = declared_size
    if magic != b"glTF":
        result["reasons"].append(f"invalid GLB signature: {path}")
    if version != 2:
        result["reasons"].append(f"unsupported GLB version {version}: {path}")
    if declared_size != size:
        result["reasons"].append(
            f"GLB declared size {declared_size} differs from file size {size}: {path}"
        )
    if size < 20:
        result["reasons"].append(f"GLB is too small: {path}")
    if json_chunk_type != b"JSON":
        result["reasons"].append(f"GLB first chunk is not JSON: {path}")
    if json_chunk_size <= 0 or json_chunk_size % 4 != 0:
        result["reasons"].append(f"invalid GLB JSON chunk size {json_chunk_size}: {path}")
    if 20 + json_chunk_size > size:
        result["reasons"].append(f"GLB JSON chunk exceeds file size: {path}")
    result["valid"] = not result["reasons"]
    return result


def _profile_summary(
    profile_path: Path,
    artifact_path: Path,
    *,
    root: Path,
) -> dict[str, Any]:
    loaded, error = _safe_json(profile_path)
    reasons: list[str] = []
    if error:
        reasons.append(error)
        loaded = None
    if loaded is not None and not isinstance(loaded, dict):
        reasons.append(f"profile root is not an object: {profile_path}")
        loaded = None

    artifact_size = artifact_path.stat().st_size if artifact_path.is_file() else None
    matched_record: dict[str, Any] | None = None
    if isinstance(loaded, dict):
        if loaded.get("success") is not True or loaded.get("return_code") != 0:
            reasons.append(
                f"profile did not succeed (return_code={loaded.get('return_code')}): {profile_path}"
            )
        if loaded.get("missing_expected_artifacts") or loaded.get("stale_expected_artifacts"):
            reasons.append(f"profile reports missing/stale artifacts: {profile_path}")
        raw_artifacts = loaded.get("artifacts")
        if not isinstance(raw_artifacts, list):
            reasons.append(f"profile artifacts are not a list: {profile_path}")
            raw_artifacts = []
        expected = artifact_path.resolve()
        for candidate in raw_artifacts:
            if not isinstance(candidate, dict):
                continue
            candidate_path = _normalise_record_path(candidate.get("path"), root)
            if candidate_path == expected:
                matched_record = candidate
                break
        if matched_record is None:
            reasons.append(f"profile has no artifact record for {artifact_path}")
        else:
            if matched_record.get("produced_or_updated_this_run") is not True:
                reasons.append(f"profile did not mark artifact fresh: {artifact_path}")
            if matched_record.get("exists") is False:
                reasons.append(f"profile reports artifact absent: {artifact_path}")
            if artifact_size is None or artifact_size <= 0:
                reasons.append(f"missing or empty artifact: {artifact_path}")
            elif matched_record.get("size_bytes") != artifact_size:
                reasons.append(f"artifact size differs from profile: {artifact_path}")

    gpu = loaded.get("gpu") if isinstance(loaded, dict) else None
    if not isinstance(gpu, dict):
        gpu = {}
    return {
        "path": _relative(profile_path, root),
        "readable": isinstance(loaded, dict),
        "valid_for_artifact": not reasons,
        "success": loaded.get("success") if isinstance(loaded, dict) else None,
        "return_code": loaded.get("return_code") if isinstance(loaded, dict) else None,
        "duration_s": _finite_number(loaded.get("duration_s")) if isinstance(loaded, dict) else None,
        "peak_gpu_global_memory_mib": _finite_number(gpu.get("peak_global_memory_mib")),
        "peak_gpu_process_memory_mib": _finite_number(gpu.get("peak_process_memory_mib")),
        "peak_process_tree_rss_bytes": (
            int(loaded["peak_process_tree_rss_bytes"])
            if isinstance(loaded, dict)
            and _finite_number(loaded.get("peak_process_tree_rss_bytes")) is not None
            else None
        ),
        "reasons": reasons,
    }


def _metrics_summary(
    metrics_path: Path,
    mesh_path: Path,
    ground_truth_path: Path,
    *,
    root: Path,
) -> dict[str, Any]:
    loaded, error = _safe_json(metrics_path)
    reasons: list[str] = []
    if error:
        reasons.append(error)
        loaded = None
    if loaded is not None and not isinstance(loaded, dict):
        reasons.append(f"metrics root is not an object: {metrics_path}")
        loaded = None
    values: dict[str, float] = {}
    recorded_counts: dict[str, int | None] = {"vertices": None, "faces": None}
    if isinstance(loaded, dict):
        if loaded.get("schema") != "genrecon.mesh-evaluation":
            reasons.append(f"unexpected metrics schema: {metrics_path}")
        if loaded.get("schema_version") != 1:
            reasons.append(f"unexpected metrics schema version: {metrics_path}")
        if loaded.get("protocol") != "paper-like-unclipped":
            reasons.append(f"unexpected metrics protocol: {metrics_path}")
        sampling = loaded.get("sampling")
        if not isinstance(sampling, dict):
            reasons.append(f"missing metrics sampling object: {metrics_path}")
        else:
            if sampling.get("samples_per_mesh") != 200_000:
                reasons.append(f"unexpected metrics sample count: {metrics_path}")
            if sampling.get("seed") != 42:
                reasons.append(f"unexpected metrics seed: {metrics_path}")
        crop = loaded.get("crop")
        if not isinstance(crop, dict) or crop.get("mode") != "none":
            reasons.append(f"metrics are not unclipped: {metrics_path}")
        inputs = loaded.get("inputs")
        if not isinstance(inputs, dict):
            reasons.append(f"missing metrics inputs: {metrics_path}")
        else:
            predicted = _normalise_record_path(inputs.get("predicted_mesh"), root)
            ground_truth = _normalise_record_path(inputs.get("ground_truth_mesh"), root)
            if predicted != mesh_path.resolve():
                reasons.append(f"metrics predicted mesh does not match scene: {metrics_path}")
            if ground_truth != ground_truth_path.resolve():
                reasons.append(f"metrics ground truth does not match scene: {metrics_path}")
        raw_values = loaded.get("metrics")
        if not isinstance(raw_values, dict):
            reasons.append(f"missing metrics values object: {metrics_path}")
        else:
            for name in REQUIRED_METRICS:
                value = _finite_number(raw_values.get(name))
                if value is None:
                    reasons.append(f"metric {name} is missing or non-finite: {metrics_path}")
                else:
                    values[name] = value
                    if name in DISTANCE_METRICS and value < 0:
                        reasons.append(f"metric {name} is negative: {metrics_path}")
                    if name in UNIT_INTERVAL_METRICS and not 0 <= value <= 1:
                        reasons.append(
                            f"metric {name} is outside the [0, 1] interval: {metrics_path}"
                        )
        meshes = loaded.get("meshes")
        if isinstance(meshes, dict):
            evaluated = meshes.get("predicted_evaluated")
            if isinstance(evaluated, dict):
                for name in ("vertices", "faces"):
                    value = evaluated.get(name)
                    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                        recorded_counts[name] = value
    return {
        "path": _relative(metrics_path, root),
        "valid": not reasons,
        "values": values if values else None,
        "recorded_mesh_counts": recorded_counts,
        "reasons": reasons,
    }


def _records_by_scene(document: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(document, dict) or not isinstance(document.get("records"), list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for record in document["records"]:
        if isinstance(record, dict) and isinstance(record.get("scene_id"), str):
            result[record["scene_id"]] = record
    return result


def _record_excerpt(record: dict[str, Any] | None, *, postprocess: bool = False) -> dict[str, Any] | None:
    if record is None:
        return None
    excerpt: dict[str, Any] = {
        "state": record.get("state"),
        "return_code": record.get("return_code"),
        "timed_out": record.get("timed_out"),
        "duration_s": _finite_number(record.get("duration_s")),
    }
    if postprocess:
        actions: list[dict[str, Any]] = []
        raw_actions = record.get("actions")
        if not isinstance(raw_actions, list):
            raw_actions = []
        for action in raw_actions:
            if not isinstance(action, dict):
                continue
            actions.append(
                {
                    "name": action.get("name"),
                    "state": action.get("state"),
                    "return_code": action.get("return_code"),
                    "timed_out": action.get("timed_out"),
                    "duration_s": _finite_number(action.get("duration_s")),
                    "reason": action.get("reason") or action.get("error"),
                    "validation_reasons": action.get("validation_reasons"),
                }
            )
        excerpt["actions"] = actions
    return excerpt


def _record_failure_reasons(
    batch_record: dict[str, Any] | None,
    postprocess_record: dict[str, Any] | None,
) -> list[str]:
    reasons: list[str] = []
    if isinstance(batch_record, dict):
        state = batch_record.get("state")
        if state not in (None, "complete", "skipped-complete"):
            detail = f"batch state={state}"
            if batch_record.get("timed_out") is True:
                detail += ", timed_out=true"
            if batch_record.get("return_code") is not None:
                detail += f", return_code={batch_record.get('return_code')}"
            reasons.append(detail)
    if isinstance(postprocess_record, dict):
        state = postprocess_record.get("state")
        if state not in (None, "complete", "skipped-complete"):
            reasons.append(f"postprocess state={state}")
        actions = postprocess_record.get("actions")
        if isinstance(actions, list):
            for action in actions:
                if not isinstance(action, dict):
                    continue
                action_state = action.get("state")
                if action_state in (None, "complete"):
                    continue
                detail = f"postprocess {action.get('name')} state={action_state}"
                reason = action.get("reason") or action.get("error")
                if reason:
                    detail += f": {reason}"
                reasons.append(detail)
    return reasons


def _tail_failure_hint(paths: list[Path], max_bytes: int = 64 * 1024) -> str | None:
    patterns = (
        "out of memory",
        "cuda error",
        "runtimeerror",
        "memoryerror",
        "traceback",
        "exception",
        "timed_out",
        "timed out",
        "killed",
        "failed",
        "error:",
    )
    for path in paths:
        try:
            size = path.stat().st_size
            with path.open("rb") as handle:
                handle.seek(max(0, size - max_bytes))
                text = handle.read(max_bytes).decode("utf-8", errors="replace")
        except OSError:
            continue
        candidates: list[str] = []
        for raw_line in text.replace("\r", "\n").splitlines():
            line = _ANSI_ESCAPE.sub("", raw_line).strip()
            lowered = line.lower()
            if line and any(pattern in lowered for pattern in patterns):
                candidates.append(" ".join(line.split()))
        if candidates:
            return candidates[-1][:500]
    return None


def _preflight_by_scene(document: Any) -> dict[str, dict[str, Any]]:
    records = _records_by_scene(document)
    keep = (
        "status",
        "chunk_count",
        "chunk_size_world_m",
        "clean_point_count",
        "joint_aabb_factor_xyz",
        "joint_aabb_volume_ratio",
        "relative_span_chunks",
    )
    return {
        scene_id: {name: record.get(name) for name in keep}
        for scene_id, record in records.items()
    }


def _oom_retry_grouped_fallback(args_path: Path, *, root: Path) -> dict[str, Any]:
    """Detect the explicit grouped-decoder overrides used for an OOM retry."""

    loaded: Any = None
    reasons: list[str] = []
    if args_path.exists():
        loaded, error = _safe_json(args_path)
        if error:
            reasons.append(error)
        elif not isinstance(loaded, dict):
            reasons.append(f"reconstruction args root is not an object: {args_path}")
            loaded = None
    values = {
        name: loaded.get(name) if isinstance(loaded, dict) else None
        for name in OOM_RETRY_GROUPED_ARG_NAMES
    }
    return {
        "used": any(value is not None for value in values.values()),
        "args_path": _relative(args_path, root),
        **values,
        "reasons": reasons,
    }


def _scene_summary(
    scene_id: str,
    *,
    root: Path,
    preflight: dict[str, dict[str, Any]],
    batch_record: dict[str, Any] | None,
    postprocess_record: dict[str, Any] | None,
) -> dict[str, Any]:
    output_dir = root / "outputs" / scene_id / "reconstruction"
    report_dir = root / "reports" / "generated" / scene_id
    data_dir = root / "data" / "da3-adapted" / scene_id
    mesh_path = output_dir / "mesh.ply"
    glb_path = output_dir / "scene.glb"
    metrics_path = report_dir / "mesh_metrics_unclipped.json"
    ground_truth_path = data_dir / "scans" / "mesh_aligned_0.05.ply"
    oom_retry_grouped_fallback = _oom_retry_grouped_fallback(
        output_dir / "args.json", root=root
    )

    ply = _read_ply_header(mesh_path)
    reconstruct_profile = _profile_summary(
        report_dir / "reconstruct_profile.json", mesh_path, root=root
    )
    mesh_reasons = list(ply["reasons"]) + list(reconstruct_profile["reasons"])
    mesh_valid = ply["valid"] and reconstruct_profile["valid_for_artifact"]

    metrics = _metrics_summary(
        metrics_path, mesh_path, ground_truth_path, root=root
    )
    recorded_counts = metrics["recorded_mesh_counts"]
    for name in ("vertices", "faces"):
        header_count = ply[name]
        recorded_count = recorded_counts[name]
        if (
            header_count is not None
            and recorded_count is not None
            and header_count != recorded_count
        ):
            metrics["reasons"].append(
                f"metrics {name} count {recorded_count} differs from PLY header "
                f"{header_count}: {metrics_path}"
            )
            metrics["valid"] = False
    glb_header = _read_glb_header(glb_path)
    glb_profile = _profile_summary(report_dir / "glb_profile.json", glb_path, root=root)
    glb_reasons = list(glb_header["reasons"]) + list(glb_profile["reasons"])
    glb_valid = glb_header["valid"] and glb_profile["valid_for_artifact"]

    attempt_evidence: list[str] = []
    if batch_record is not None:
        attempt_evidence.append("batch summary record")
    evidence_paths = (
        report_dir / "reconstruct_profile.json",
        report_dir / "reconstruct.log",
        report_dir / "batch_driver.log",
        metrics_path,
        report_dir / "glb_profile.json",
        report_dir / "glb.log",
        report_dir / "glb_postprocess_driver.log",
        mesh_path,
        glb_path,
        output_dir / "to_glb_inputs.pt",
        output_dir / "chunk_inputs.pt",
    )
    attempt_evidence.extend(
        _relative(path, root) for path in evidence_paths if path.exists()
    )
    if isinstance(postprocess_record, dict) and isinstance(
        postprocess_record.get("actions"), list
    ) and postprocess_record["actions"]:
        attempt_evidence.append("postprocess action record")
    attempted = bool(attempt_evidence)
    geometry_valid = mesh_valid and metrics["valid"]
    if geometry_valid and glb_valid:
        status = "full_complete"
    elif geometry_valid:
        status = "geometry_complete"
    elif attempted:
        status = "reconstruction_failed"
    else:
        status = "not_attempted"

    if status == "geometry_complete":
        failure_reasons = _record_failure_reasons(batch_record, postprocess_record)
        failure_reasons.extend(f"GLB: {reason}" for reason in glb_reasons)
        hint_paths = [
            report_dir / "glb_postprocess_driver.log",
            report_dir / "glb.log",
            report_dir / "batch_driver.log",
        ]
    elif status == "reconstruction_failed":
        failure_reasons = _record_failure_reasons(batch_record, postprocess_record)
        failure_reasons.extend(f"mesh: {reason}" for reason in mesh_reasons)
        failure_reasons.extend(f"metrics: {reason}" for reason in metrics["reasons"])
        hint_paths = [
            report_dir / "batch_driver.log",
            report_dir / "reconstruct.log",
            report_dir / "glb_postprocess_driver.log",
        ]
    elif status == "not_attempted":
        failure_reasons = ["no reconstruction attempt evidence found"]
        hint_paths = []
    else:
        failure_reasons = []
        hint_paths = []
    failure_hint = _tail_failure_hint(hint_paths)
    if failure_hint:
        failure_reasons.insert(0, failure_hint)

    mesh_size = mesh_path.stat().st_size if mesh_path.is_file() else None
    glb_size = glb_path.stat().st_size if glb_path.is_file() else None
    if ply["vertices"] is not None and ply["faces"] is not None:
        count_source = "ply_header"
    elif recorded_counts["vertices"] is not None and recorded_counts["faces"] is not None:
        count_source = "metrics_json"
    else:
        count_source = "unavailable"
    return {
        "scene_id": scene_id,
        "status": status,
        "attempted": attempted,
        "attempt_evidence": attempt_evidence,
        "preflight": preflight.get(scene_id),
        "oom_retry_grouped_fallback": oom_retry_grouped_fallback,
        "mesh": {
            "path": _relative(mesh_path, root),
            "valid": mesh_valid,
            "size_bytes": mesh_size,
            "vertices": ply["vertices"] if ply["vertices"] is not None else recorded_counts["vertices"],
            "faces": ply["faces"] if ply["faces"] is not None else recorded_counts["faces"],
            "count_source": count_source,
            "ply_format": ply["format"],
            "header_bytes_read": ply["header_bytes_read"],
            "reasons": mesh_reasons,
        },
        "metrics": metrics,
        "glb": {
            "path": _relative(glb_path, root),
            "valid": glb_valid,
            "size_bytes": glb_size,
            "version": glb_header["version"],
            "reasons": glb_reasons,
        },
        "profiles": {
            "reconstruction": reconstruct_profile,
            "glb": glb_profile,
        },
        "batch": _record_excerpt(batch_record),
        "postprocess": _record_excerpt(postprocess_record, postprocess=True),
        "failure_reason": failure_reasons[0] if failure_reasons else None,
        "failure_reasons": failure_reasons,
    }


def _aggregate_metrics(scenes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    eligible = {
        "full_complete",
        "geometry_complete",
    }
    for name in REQUIRED_METRICS:
        values: list[float] = []
        for scene in scenes:
            if scene["status"] not in eligible:
                continue
            raw_values = scene["metrics"].get("values")
            value = _finite_number(raw_values.get(name)) if isinstance(raw_values, dict) else None
            if value is not None:
                values.append(value)
        result[name] = {
            "count": len(values),
            "mean": statistics.fmean(values) if values else None,
            "median": statistics.median(values) if values else None,
        }
    return result


def _fmt_number(value: Any, digits: int = 4) -> str:
    converted = _finite_number(value)
    return "—" if converted is None else f"{converted:.{digits}f}"


def _fmt_seconds(value: Any) -> str:
    converted = _finite_number(value)
    return "—" if converted is None else f"{converted:.1f}"


def _fmt_mib(value: Any) -> str:
    converted = _finite_number(value)
    return "—" if converted is None else f"{converted / (1024 * 1024):.1f}"


def _md_cell(value: Any, limit: int = 90) -> str:
    if value is None:
        return "—"
    text = " ".join(str(value).replace("|", "\\|").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_markdown(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    aggregate = summary["metric_aggregate"]
    oom_retry_fallback = summary["resource_fallbacks"][
        "oom_retry_grouped_joint_decode"
    ]
    lines = [
        "# GenRecon × DA3 ScanNet++ 全场景结果",
        "",
        f"> 生成时间：`{summary['generated_utc']}`",
        "",
        "## 协议说明",
        "",
        f"本报告覆盖 Hugging Face DA3-BENCH 中当前 manifest 的 {summary['protocol']['manifest_scene_count']} 个 ScanNet++ 场景。每场景固定选择 8 张确定性 iPhone 视图，随机种子 42，生成配置 512；几何指标采用 `paper-like-unclipped`。这是可复跑的公开子集实验，**不是严格论文复现**：论文使用 25 场景，且其未公开的扫描观测包络与 15 cm 膨胀裁剪没有实现。",
        "",
        "16 GiB low-VRAM 执行策略：512 场景在不超过 16 chunks 时保留单次联合解码；更大场景使用互斥的递归空间分区，并为每组外扩 16 个输入体素作为边界上下文。相机、采样、阈值和生成分辨率不变。",
        "",
        "显式分组解码覆盖仅用于初次重建发生 OOM 的场景，是确定性的资源回退（deterministic resource fallback），不改变相机、采样、阈值或生成分辨率。下表依据各场景 `outputs/<scene_id>/reconstruction/args.json` 中两个 `joint_decode_*` 字段的非空值自动生成。",
        "",
        "## 初次 OOM 场景的分组解码回退",
        "",
        f"共识别出 {oom_retry_fallback['scene_count']} 个使用回退的场景。",
        "",
        "| 场景 | max chunks/group | max inflated voxels | 参数来源 |",
        "|---|---:|---:|---|",
    ]
    if oom_retry_fallback["scenes"]:
        for fallback in oom_retry_fallback["scenes"]:
            lines.append(
                "| "
                + " | ".join(
                    (
                        fallback["scene_id"],
                        _md_cell(fallback.get("joint_decode_max_chunks_per_group")),
                        _md_cell(fallback.get("joint_decode_max_inflated_voxels")),
                        _md_cell(fallback.get("args_path")),
                    )
                )
                + " |"
            )
    else:
        lines.append("| — | — | — | 未识别到非空覆盖参数 |")
    lines.extend(
        [
            "",
            "轻量验证只读取 PLY/GLB 头部，并核对 profile 中的成功状态、产物路径和大小；不会加载完整网格或重新计算大文件哈希。",
            "",
            "## 完成情况",
            "",
            "| 状态 | 数量 |",
            "|---|---:|",
        ]
    )
    for status in STATUS_ORDER:
        lines.append(f"| {STATUS_ZH[status]} | {counts[status]} |")
    lines.extend(
        [
            f"| 合计 | {counts['total']} |",
            "",
            f"指标汇总只包含几何有效的 {counts['metric_eligible']} 个场景（`full_complete` + `geometry_complete`）。",
            "",
            "## 有限指标汇总",
            "",
            "| 指标 | n | 均值 | 中位数 |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in PRIMARY_METRICS:
        item = aggregate[name]
        lines.append(
            f"| {METRIC_ZH[name]} | {item['count']} | {_fmt_number(item['mean'], 6)} | {_fmt_number(item['median'], 6)} |"
        )
    lines.extend(
        [
            "",
            "## 每场景结果",
            "",
            "| 场景 | 状态 | CD↓ | F@0.1m↑ | NC↑ | 顶点 / 面 | PLY / GLB MiB | chunks / AABB | 重建 / GLB 秒 | GPU 峰值 MiB | 说明 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for scene in summary["scenes"]:
        values = scene["metrics"].get("values") or {}
        preflight = scene.get("preflight") or {}
        reconstruction = scene["profiles"]["reconstruction"]
        glb_profile = scene["profiles"]["glb"]
        gpu_values = [
            value
            for value in (
                reconstruction.get("peak_gpu_global_memory_mib"),
                glb_profile.get("peak_gpu_global_memory_mib"),
            )
            if _finite_number(value) is not None
        ]
        gpu_peak = max(gpu_values) if gpu_values else None
        vertices = scene["mesh"].get("vertices")
        faces = scene["mesh"].get("faces")
        counts_text = (
            f"{vertices:,} / {faces:,}"
            if isinstance(vertices, int) and isinstance(faces, int)
            else "—"
        )
        chunks = preflight.get("chunk_count")
        aabb = preflight.get("joint_aabb_volume_ratio")
        chunks_text = (
            f"{chunks} / {_fmt_number(aabb, 3)}"
            if chunks is not None or aabb is not None
            else "—"
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    scene["scene_id"],
                    STATUS_ZH[scene["status"]],
                    _fmt_number(values.get("chamfer_symmetric_mean_m")),
                    _fmt_number(values.get("fscore_arithmetic_at_0_1m")),
                    _fmt_number(values.get("normal_consistency_symmetric_mean")),
                    counts_text,
                    f"{_fmt_mib(scene['mesh'].get('size_bytes'))} / {_fmt_mib(scene['glb'].get('size_bytes'))}",
                    chunks_text,
                    f"{_fmt_seconds(reconstruction.get('duration_s'))} / {_fmt_seconds(glb_profile.get('duration_s'))}",
                    _fmt_number(gpu_peak, 0),
                    _md_cell(scene.get("failure_reason")),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "状态定义：`full_complete` = 有效 PLY、有限且协议匹配的指标、有效 GLB；`geometry_complete` = PLY 与指标有效，但 GLB 缺失或失败；`reconstruction_failed` = 已有运行证据，但没有同时得到有效 PLY 与指标；`not_attempted` = 未发现重建运行证据。",
            "",
            "## 查看三角面",
            "",
            "原始三角面和线框优先看 `mesh.ply`；带纹理、便于整体浏览时看 `scene.glb`。MeshLab 命令：",
            "",
            "```bash",
            "meshlab outputs/<scene_id>/reconstruction/mesh.ply",
            "meshlab outputs/<scene_id>/reconstruction/scene.glb",
            "```",
            "",
            "在 MeshLab 中打开 `Render → Show Wireframe`（或工具栏线框按钮）即可检查三角剖分。",
        ]
    )
    warnings = summary["inputs"].get("warnings", [])
    if warnings:
        lines.extend(["", "## 输入警告", ""])
        lines.extend(f"- {_md_cell(warning, 300)}" for warning in warnings)
    return "\n".join(lines) + "\n"


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    manifest_path = (root / args.manifest).resolve()
    preflight_path = (root / args.preflight_summary).resolve()
    batch_path = (root / args.batch_summary).resolve()
    postprocess_path = (root / args.postprocess_summary).resolve()

    manifest, manifest_error = _safe_json(manifest_path)
    if manifest_error:
        raise ValueError(manifest_error)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("scene_ids"), list):
        raise ValueError(f"manifest scene_ids is not a list: {manifest_path}")
    raw_scene_ids = manifest["scene_ids"]
    if not all(isinstance(scene_id, str) for scene_id in raw_scene_ids):
        raise ValueError(f"manifest has non-string scene IDs: {manifest_path}")
    for scene_id in raw_scene_ids:
        if not scene_id or Path(scene_id).name != scene_id or scene_id in {".", ".."}:
            raise ValueError(f"unsafe scene ID in manifest: {scene_id!r}")
    scene_ids = list(dict.fromkeys(raw_scene_ids))
    if len(scene_ids) != len(raw_scene_ids):
        raise ValueError(f"manifest has duplicate scene IDs: {manifest_path}")
    if args.expected_scene_count is not None and len(scene_ids) != args.expected_scene_count:
        raise ValueError(
            f"expected {args.expected_scene_count} manifest scenes, found {len(scene_ids)}"
        )

    warnings: list[str] = []
    manifest_views = manifest.get("num_views_per_scene")
    if manifest_views is not None and manifest_views != 8:
        warnings.append(
            f"manifest num_views_per_scene is {manifest_views!r}, expected fixed 8-view protocol"
        )
    manifest_scene_records = manifest.get("scenes")
    if isinstance(manifest_scene_records, list):
        mismatched_selections: list[str] = []
        for record in manifest_scene_records:
            if not isinstance(record, dict) or not isinstance(record.get("scene_id"), str):
                continue
            counts = record.get("counts")
            if isinstance(counts, dict) and counts.get("selected_images") != 8:
                mismatched_selections.append(record["scene_id"])
        if mismatched_selections:
            warnings.append(
                "manifest scenes do not have exactly 8 selected images: "
                + ", ".join(mismatched_selections)
            )
    loaded_inputs: dict[str, Any] = {}
    for name, path in (
        ("preflight", preflight_path),
        ("batch", batch_path),
        ("postprocess", postprocess_path),
    ):
        document, error = _safe_json(path)
        if error:
            warnings.append(error)
            document = None
        elif not isinstance(document, dict):
            warnings.append(f"{name} summary root is not an object: {path}")
            document = None
        loaded_inputs[name] = document

    for name in ("batch", "postprocess"):
        document = loaded_inputs[name]
        if isinstance(document, dict) and document.get("finished_utc") is None:
            message = f"{name} summary is unfinished"
            if not getattr(args, "allow_partial", False):
                raise ValueError(
                    f"{message}; wait for it to finish or pass --allow-partial explicitly"
                )
            warnings.append(message)

    preflight = _preflight_by_scene(loaded_inputs["preflight"])
    batch = _records_by_scene(loaded_inputs["batch"])
    postprocess = _records_by_scene(loaded_inputs["postprocess"])
    missing_preflight = [scene_id for scene_id in scene_ids if scene_id not in preflight]
    if missing_preflight:
        warnings.append(
            f"preflight summary lacks {len(missing_preflight)} manifest scene(s): "
            + ", ".join(missing_preflight)
        )
    scenes = [
        _scene_summary(
            scene_id,
            root=root,
            preflight=preflight,
            batch_record=batch.get(scene_id),
            postprocess_record=postprocess.get(scene_id),
        )
        for scene_id in scene_ids
    ]
    oom_retry_grouped_scenes = [
        {
            "scene_id": scene["scene_id"],
            "args_path": scene["oom_retry_grouped_fallback"]["args_path"],
            **{
                name: scene["oom_retry_grouped_fallback"][name]
                for name in OOM_RETRY_GROUPED_ARG_NAMES
            },
        }
        for scene in scenes
        if scene["oom_retry_grouped_fallback"]["used"]
    ]
    status_counts = {status: 0 for status in STATUS_ORDER}
    for scene in scenes:
        status_counts[scene["status"]] += 1
    counts = {
        "total": len(scenes),
        **status_counts,
        "metric_eligible": status_counts["full_complete"] + status_counts["geometry_complete"],
    }
    return _json_safe({
        "schema": "genrecon.da3-final-summary",
        "schema_version": 1,
        "generated_utc": _utc_now(),
        "root": str(root),
        "protocol": {
            "name": "paper-like-unclipped",
            "dataset": "Hugging Face DA3-BENCH ScanNet++ manifest subset",
            "manifest_scene_count": len(scene_ids),
            "views_per_scene": 8,
            "view_source": "deterministic iPhone selection",
            "seed": 42,
            "generation_resolution": 512,
            "low_vram_decoder_policy": {
                "single_pass_max_chunks": 16,
                "large_scene_max_chunks_per_group": 8,
                "large_scene_max_inflated_voxels": 80_000,
                "group_overlap_input_voxels": 16,
                "group_ownership": "disjoint-recursive-spatial-partition",
            },
            "manifest_views_per_scene": manifest_views,
            "strict_paper_reproduction": False,
            "paper_scene_count": 25,
            "note": "Public paper-like metrics without the unpublished scanner observation envelope and 15 cm dilation crop.",
        },
        "validation": {
            "mode": "lightweight-structural-and-profile-size",
            "full_artifact_hashing": False,
            "mesh": "PLY header plus successful profile artifact path/size",
            "glb": "20-byte GLB/container header plus successful profile artifact path/size",
            "metrics": "schema/protocol/input paths, required finite values, and valid ranges",
        },
        "resource_fallbacks": {
            "oom_retry_grouped_joint_decode": {
                "deterministic": True,
                "scope": "only scenes whose initial reconstruction attempt failed with OOM",
                "detection": (
                    "non-null joint_decode_max_chunks_per_group or "
                    "joint_decode_max_inflated_voxels in reconstruction args.json"
                ),
                "scene_count": len(oom_retry_grouped_scenes),
                "scenes": oom_retry_grouped_scenes,
            }
        },
        "inputs": {
            "manifest": _relative(manifest_path, root),
            "preflight_summary": _relative(preflight_path, root),
            "batch_summary": _relative(batch_path, root),
            "batch_finished_utc": (
                loaded_inputs["batch"].get("finished_utc")
                if isinstance(loaded_inputs["batch"], dict)
                else None
            ),
            "postprocess_summary": _relative(postprocess_path, root),
            "postprocess_finished_utc": (
                loaded_inputs["postprocess"].get("finished_utc")
                if isinstance(loaded_inputs["postprocess"], dict)
                else None
            ),
            "warnings": warnings,
        },
        "counts": counts,
        "metric_aggregate": _aggregate_metrics(scenes),
        "scenes": scenes,
    })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: inferred from this script)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/da3-adapted/da3_scannetpp_manifest.json"),
    )
    parser.add_argument(
        "--preflight-summary",
        type=Path,
        default=Path("reports/generated/preflight/summary.json"),
    )
    parser.add_argument(
        "--batch-summary",
        type=Path,
        default=Path("reports/generated/da3_batch_summary.json"),
    )
    parser.add_argument(
        "--postprocess-summary",
        type=Path,
        default=Path("reports/generated/da3_postprocess_summary.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/generated/da3_final_summary.json"),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("reports/DA3_ALL_SCENES_REPORT_zh.md"),
    )
    parser.add_argument(
        "--expected-scene-count",
        type=int,
        default=20,
        help="Fail if the manifest count differs; use 0 to disable",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow an explicitly unfinished batch/postprocess summary (not for final reports)",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.expected_scene_count == 0:
        args.expected_scene_count = None
    root = args.root.resolve()
    output_json = (root / args.output_json).resolve()
    output_markdown = (root / args.output_markdown).resolve()
    if output_json == output_markdown:
        raise ValueError("JSON and Markdown output paths must differ")
    protected_inputs = {
        (root / args.manifest).resolve(),
        (root / args.preflight_summary).resolve(),
        (root / args.batch_summary).resolve(),
        (root / args.postprocess_summary).resolve(),
    }
    if output_json in protected_inputs or output_markdown in protected_inputs:
        raise ValueError("report outputs must not overwrite manifest or live summaries")
    summary = build_summary(args)
    markdown = render_markdown(summary)
    _atomic_write_json(output_json, summary)
    _atomic_write_text(output_markdown, markdown)
    return summary


def main() -> int:
    try:
        args = build_parser().parse_args()
        summary = run(args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    counts = summary["counts"]
    print(
        "DA3 summary: "
        f"full={counts['full_complete']}, geometry={counts['geometry_complete']}, "
        f"failed={counts['reconstruction_failed']}, not_attempted={counts['not_attempted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
