#!/usr/bin/env python3
"""Run current GenRecon and PBR GLB export for foundation-SfM candidates.

The runner is scene-resumable. It preserves the source foundation grade and
visual disposition because successful mesh generation is not geometry proof.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import struct
import subprocess
from pathlib import Path
from typing import Any

try:
    from tools.profile_command import profile_command
except ModuleNotFoundError:
    from profile_command import profile_command

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = ROOT / "data" / "internet-zero-shot" / "foundation-sfm-v1"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "internet-zero-shot" / "foundation-genrecon-v1"
DEFAULT_REPORT_ROOT = ROOT / "reports" / "generated" / "internet-zero-shot" / "foundation-genrecon-v1"
PIPELINE_CONFIG = ROOT / "data" / "hf-cache" / "converted" / "dinov3-vitl16-timm-qkvb" / "pipeline.json"
SS_CHECKPOINT = ROOT / "weights" / "ss" / "checkpoints" / "sparse_structure.pt"
SHAPE_CHECKPOINT = ROOT / "weights" / "shape" / "checkpoints" / "shape_slat.pt"
TEXTURE_CHECKPOINT = ROOT / "weights" / "texture" / "checkpoints" / "texture_slat.pt"
EXPECTED_MODEL_HASHES = {
    SS_CHECKPOINT: "e18c1caddb2357dbf5839f0f7e1569c50d855fcb47e0871483d99c91a14e2bb7",
    SHAPE_CHECKPOINT: "d9e13be151a213bf67565d2a17341fe97328e455814fb2328a59366344e122fb",
    TEXTURE_CHECKPOINT: "28f99217a4fbcd04f36a5f576905975ae8b874ad63548d6ab8afdb96cf03ed47",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant {value} in {path}")
        ),
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=False, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_provenance() -> dict[str, Any]:
    assets = []
    for path, expected_hash in EXPECTED_MODEL_HASHES.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required GenRecon checkpoint is missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Checkpoint SHA256 mismatch: {path}")
        assets.append(
            {
                "path": path,
                "size_bytes": path.stat().st_size,
                "sha256": actual_hash,
            }
        )
    for path in (
        PIPELINE_CONFIG,
        ROOT / "reconstruct_scene.py",
        ROOT / "chunked_to_glb.py",
        Path(__file__).resolve(),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required inference asset is missing: {path}")
        assets.append(
            {
                "path": path,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return {
        "git_commit": commit,
        "worktree_dirty": bool(status),
        "worktree_status": status,
        "assets": assets,
    }


def profile_artifact_record(
    profile: dict[str, Any] | None, path: Path
) -> dict[str, Any] | None:
    if profile is None:
        return None
    resolved = path.resolve()
    return next(
        (
            item
            for item in profile.get("artifacts", [])
            if Path(item.get("path", "")).resolve() == resolved
        ),
        None,
    )


def profile_is_complete(profile_path: Path, expected_files: list[Path]) -> bool:
    if not profile_path.is_file():
        return False
    try:
        profile = load_json(profile_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if profile.get("success") is not True or profile.get("return_code") != 0:
        return False
    records = {
        str(Path(item.get("path", "")).resolve()): item
        for item in profile.get("artifacts", [])
    }
    for path in expected_files:
        if not path.is_file() or path.stat().st_size <= 0:
            return False
        artifact = records.get(str(path.resolve()))
        if artifact is None or artifact.get("size_bytes") != path.stat().st_size:
            return False
    return True


def reconstruct_command(
    scene_root: Path,
    reconstruction_root: Path,
    *,
    colmap_subdir: str = "colmap_vggt",
    max_reproj_error: float | None = None,
    min_track_len: int | None = None,
) -> list[str]:
    command = [
        str(ROOT / ".venv" / "bin" / "python"),
        str(ROOT / "reconstruct_scene.py"),
        "--mode",
        "Iphone",
        "--path",
        str(scene_root),
        "--colmap_subdir",
        colmap_subdir,
        "--output_path",
        str(reconstruction_root),
        "--ss_ckpt",
        str(SS_CHECKPOINT),
        "--shape_ckpt",
        str(SHAPE_CHECKPOINT),
        "--tex_ckpt",
        str(TEXTURE_CHECKPOINT),
        "--ss_config",
        str(ROOT / "configs" / "gen" / "ss_flow_img" / "genrecon.json"),
        "--shape_config",
        str(ROOT / "configs" / "gen" / "slat_flow_img2shape" / "genrecon_512.json"),
        "--tex_config",
        str(ROOT / "configs" / "gen" / "slat_flow_imgshape2tex" / "genrecon_512.json"),
        "--pipeline_config",
        str(PIPELINE_CONFIG),
        "--num_imgs_per_scene",
        "8",
        "--seed",
        "42",
        "--chunk_size_factor",
        "1.08",
        "--min_overlap_factor",
        "4",
        "--proj_batch_voxels",
        "256",
        "--save_imgs",
    ]
    if max_reproj_error is not None:
        command.extend(["--max_reproj_error", f"{max_reproj_error:g}"])
    if min_track_len is not None:
        command.extend(["--min_track_len", str(min_track_len)])
    return command


def glb_command(reconstruction_root: Path) -> list[str]:
    return [
        str(ROOT / ".venv" / "bin" / "python"),
        str(ROOT / "chunked_to_glb.py"),
        "--inputs",
        str(reconstruction_root / "to_glb_inputs.pt"),
        "--chunk_inputs",
        str(reconstruction_root / "chunk_inputs.pt"),
        "--output_dir",
        str(reconstruction_root),
        "--chunks_dir",
        str(reconstruction_root / "chunks_300k"),
        "--texture_size",
        "4096",
        "--simplify_threshold",
        "300000",
        "--skip_fill_holes",
        "--skip_remesh",
    ]


def parse_mesh_ply(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = handle.readline()
            if not line:
                raise ValueError(f"Truncated PLY header: {path}")
            header += line
    text = header.decode("ascii")
    if "format binary_little_endian 1.0" not in text:
        raise ValueError(f"Unexpected PLY format: {path}")
    vertex_match = re.search(r"element vertex (\d+)", text)
    face_match = re.search(r"element face (\d+)", text)
    if vertex_match is None or face_match is None:
        raise ValueError(f"PLY is missing vertex/face counts: {path}")
    vertices = int(vertex_match.group(1))
    faces = int(face_match.group(1))
    expected_size = len(header) + vertices * 18 + faces * 13
    if path.stat().st_size != expected_size:
        raise ValueError(
            f"PLY payload size mismatch for triangle mesh: {path.stat().st_size} != {expected_size}"
        )
    if vertices <= 0 or faces <= 0:
        raise ValueError(f"PLY mesh is empty: {path}")
    return {
        "vertices": vertices,
        "faces": faces,
        "size_bytes": path.stat().st_size,
    }


def parse_glb(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        header = handle.read(12)
        if len(header) != 12:
            raise ValueError(f"Truncated GLB header: {path}")
        magic, version, total_length = struct.unpack("<4sII", header)
        if magic != b"glTF" or version != 2 or total_length != path.stat().st_size:
            raise ValueError(f"Invalid GLB header: {path}")
        json_length, json_type = struct.unpack("<I4s", handle.read(8))
        if json_type != b"JSON":
            raise ValueError(f"GLB first chunk is not JSON: {path}")
        document = json.loads(handle.read(json_length).rstrip(b" \x00"))
    keys = ("scenes", "nodes", "meshes", "materials", "textures", "images", "accessors")
    counts = {key: len(document.get(key, [])) for key in keys}
    if counts["scenes"] != 1 or counts["meshes"] <= 0:
        raise ValueError(f"GLB has no scene meshes: {path}")
    if counts["materials"] != counts["meshes"]:
        raise ValueError(f"GLB mesh/material counts differ: {path}")
    if counts["textures"] != 2 * counts["meshes"] or counts["images"] != 2 * counts["meshes"]:
        raise ValueError(f"GLB does not have two embedded PBR textures per mesh: {path}")
    for material in document["materials"]:
        pbr = material.get("pbrMetallicRoughness", {})
        if "baseColorTexture" not in pbr or "metallicRoughnessTexture" not in pbr:
            raise ValueError(f"GLB material is missing PBR texture bindings: {path}")
    return {**counts, "size_bytes": path.stat().st_size, "pbr_textures_embedded": True}


def candidate_paths(
    candidate_id: str, input_root: Path, output_root: Path, report_root: Path
) -> dict[str, Path]:
    return {
        "input": input_root / "candidates" / candidate_id,
        "output": output_root / "candidates" / candidate_id,
        "reconstruction": output_root / "candidates" / candidate_id / "reconstruction",
        "report": report_root / "candidates" / candidate_id,
    }


def candidate_grade(candidate: dict[str, Any]) -> Any:
    return candidate.get("input_grade", candidate.get("grade", candidate.get("foundation_grade")))


def inspect_candidate(
    candidate: dict[str, Any],
    input_root: Path,
    output_root: Path,
    report_root: Path,
    track_name: str = "foundation-sfm",
) -> dict[str, Any]:
    identifier = candidate["candidate_id"]
    paths = candidate_paths(identifier, input_root, output_root, report_root)
    reconstruction = paths["reconstruction"]
    reconstruct_profile = paths["report"] / "reconstruct_profile.json"
    glb_profile = paths["report"] / "glb_profile.json"
    reconstruct_artifacts = [
        reconstruction / "mesh.ply",
        reconstruction / "to_glb_inputs.pt",
        reconstruction / "chunk_inputs.pt",
    ]
    reconstruct_complete = profile_is_complete(reconstruct_profile, reconstruct_artifacts)
    glb_complete = profile_is_complete(glb_profile, [reconstruction / "scene.glb"])
    record: dict[str, Any] = {
        "candidate_id": identifier,
        "title": candidate.get("title"),
        "track": candidate.get("track", track_name),
        "input_grade": candidate_grade(candidate),
        "input_preflight": candidate.get("preflight"),
        "input_chunk_count": candidate.get("chunks"),
        "input_fallback_count": candidate.get("fallback"),
        "foundation_grade": candidate_grade(candidate),
        "foundation_preflight": candidate.get("preflight"),
        "foundation_chunk_count": candidate.get("chunks"),
        "foundation_fallback_count": candidate.get("fallback"),
        "visual_disposition": candidate.get("visual_disposition"),
        "input_directory": paths["input"],
        "output_directory": paths["output"],
        "reconstruction_directory": reconstruction,
        "report_directory": paths["report"],
        "reconstruct_profile": reconstruct_profile,
        "glb_profile": glb_profile,
        "reconstruct_status": "complete" if reconstruct_complete else "pending_or_failed",
        "glb_status": "complete" if glb_complete else "pending_or_failed",
        "mesh": None,
        "glb": None,
        "reconstruct_duration_s": None,
        "glb_duration_s": None,
        "reconstruct_peak_gpu_mib": None,
        "glb_peak_gpu_mib": None,
        "reconstruct_peak_rss_bytes": None,
        "glb_peak_rss_bytes": None,
        "intermediates": {},
        "closest_camera_fallbacks": None,
        "errors": [],
    }
    loaded_reconstruct_profile = None
    loaded_glb_profile = None
    if reconstruct_profile.is_file():
        loaded_reconstruct_profile = load_json(reconstruct_profile)
        record["reconstruct_duration_s"] = loaded_reconstruct_profile.get("duration_s")
        record["reconstruct_peak_gpu_mib"] = loaded_reconstruct_profile.get("gpu", {}).get(
            "peak_process_memory_mib"
        )
        record["reconstruct_peak_rss_bytes"] = loaded_reconstruct_profile.get(
            "peak_process_tree_rss_bytes"
        )
        if loaded_reconstruct_profile.get("success") is not True:
            record["errors"].append("reconstruct_profile_failed")
    if glb_profile.is_file():
        loaded_glb_profile = load_json(glb_profile)
        record["glb_duration_s"] = loaded_glb_profile.get("duration_s")
        record["glb_peak_gpu_mib"] = loaded_glb_profile.get("gpu", {}).get(
            "peak_process_memory_mib"
        )
        record["glb_peak_rss_bytes"] = loaded_glb_profile.get("peak_process_tree_rss_bytes")
        if loaded_glb_profile.get("success") is not True:
            record["errors"].append("glb_profile_failed")
    reconstruct_log = paths["report"] / "reconstruct.log"
    if reconstruct_log.is_file():
        log_text = reconstruct_log.read_text(encoding="utf-8", errors="replace")
        record["closest_camera_fallbacks"] = log_text.count("falling back to closest camera")
    glb_log = paths["report"] / "glb.log"
    record["empty_glb_chunks"] = []
    if glb_log.is_file():
        glb_log_text = glb_log.read_text(encoding="utf-8", errors="replace")
        record["empty_glb_chunks"] = [
            int(value)
            for value in re.findall(r"\[chunked_to_glb\] chunk (\d+): empty, skipping", glb_log_text)
        ]
    if reconstruct_complete:
        try:
            mesh_path = reconstruction / "mesh.ply"
            record["mesh"] = parse_mesh_ply(mesh_path)
            mesh_artifact = profile_artifact_record(loaded_reconstruct_profile, mesh_path)
            record["mesh"]["sha256"] = mesh_artifact.get("sha256") if mesh_artifact else None
            for name in ("to_glb_inputs.pt", "chunk_inputs.pt"):
                intermediate_path = reconstruction / name
                artifact = profile_artifact_record(loaded_reconstruct_profile, intermediate_path)
                record["intermediates"][name] = {
                    "path": intermediate_path,
                    "size_bytes": intermediate_path.stat().st_size,
                    "sha256": artifact.get("sha256") if artifact else None,
                }
        except (OSError, ValueError) as error:
            record["errors"].append(f"mesh_validation: {error}")
    if glb_complete:
        try:
            glb_path = reconstruction / "scene.glb"
            record["glb"] = parse_glb(glb_path)
            glb_artifact = profile_artifact_record(loaded_glb_profile, glb_path)
            record["glb"]["sha256"] = glb_artifact.get("sha256") if glb_artifact else None
        except (OSError, ValueError, json.JSONDecodeError, struct.error) as error:
            record["errors"].append(f"glb_validation: {error}")
    chunk_cache = reconstruction / "chunks_300k"
    record["cached_chunk_glbs"] = len(list(chunk_cache.glob("chunk_*.glb"))) if chunk_cache.is_dir() else 0
    record["complete"] = (
        reconstruct_complete
        and glb_complete
        and record["mesh"] is not None
        and record["glb"] is not None
        and not record["errors"]
    )
    return record


def write_summary(
    candidates: list[dict[str, Any]],
    input_root: Path,
    output_root: Path,
    report_root: Path,
    track_name: str = "foundation-sfm",
) -> dict[str, Any]:
    records = [
        inspect_candidate(item, input_root, output_root, report_root, track_name)
        for item in candidates
    ]
    summary = {
        "schema": f"genrecon.{track_name}-genrecon-index",
        "schema_version": 1,
        "created_utc": utc_now(),
        "track": track_name,
        "source_index": input_root / "index.json",
        "source_foundation_index": input_root / "index.json",
        "output_root": output_root,
        "report_root": report_root,
        "summary": {
            "candidate_count": len(records),
            "reconstruct_complete": sum(row["reconstruct_status"] == "complete" for row in records),
            "glb_complete": sum(row["glb_status"] == "complete" for row in records),
            "fully_validated": sum(bool(row["complete"]) for row in records),
            "mesh_vertices": sum((row["mesh"] or {}).get("vertices", 0) for row in records),
            "mesh_faces": sum((row["mesh"] or {}).get("faces", 0) for row in records),
            "input_chunks": sum(row["input_chunk_count"] or 0 for row in records),
            "foundation_chunks": sum(row["foundation_chunk_count"] or 0 for row in records),
            "nonempty_glb_primitives": sum(row["cached_chunk_glbs"] for row in records),
            "explicitly_empty_glb_chunks": sum(len(row["empty_glb_chunks"]) for row in records),
            "closest_camera_fallbacks": sum(row["closest_camera_fallbacks"] or 0 for row in records),
            "mesh_bytes": sum((row["mesh"] or {}).get("size_bytes", 0) for row in records),
            "glb_bytes": sum((row["glb"] or {}).get("size_bytes", 0) for row in records),
            "reconstruct_duration_s": sum(row["reconstruct_duration_s"] or 0 for row in records),
            "glb_duration_s": sum(row["glb_duration_s"] or 0 for row in records),
            "peak_reconstruct_gpu_mib": max(
                (row["reconstruct_peak_gpu_mib"] or 0 for row in records), default=0
            ),
            "peak_glb_gpu_mib": max((row["glb_peak_gpu_mib"] or 0 for row in records), default=0),
        },
        "candidates": records,
    }
    write_json(output_root / "index.json", summary)
    output_root.mkdir(parents=True, exist_ok=True)
    fields = [
        "candidate_id",
        "track",
        "input_grade",
        "input_preflight",
        "input_chunk_count",
        "visual_disposition",
        "foundation_grade",
        "foundation_preflight",
        "foundation_chunk_count",
        "reconstruct_status",
        "glb_status",
        "complete",
        "reconstruct_duration_s",
        "glb_duration_s",
        "closest_camera_fallbacks",
        "cached_chunk_glbs",
        "empty_glb_chunks",
        "reconstruct_peak_gpu_mib",
        "glb_peak_gpu_mib",
        "reconstruction_directory",
    ]
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return summary


def validate_outputs(
    candidates: list[dict[str, Any]],
    input_root: Path,
    output_root: Path,
    report_root: Path,
    track_name: str = "foundation-sfm",
) -> dict[str, Any]:
    summary = write_summary(candidates, input_root, output_root, report_root, track_name)
    errors: list[str] = []
    deliverable_hash_bytes = 0
    chunk_glbs_validated = 0
    for record in summary["candidates"]:
        if not record["complete"]:
            errors.append(f"{record['candidate_id']}: {', '.join(record['errors']) or 'incomplete'}")
            continue
        reconstruction = Path(record["reconstruction_directory"])
        for key, filename in (("mesh", "mesh.ply"), ("glb", "scene.glb")):
            deliverable = reconstruction / filename
            actual_hash = sha256_file(deliverable)
            deliverable_hash_bytes += deliverable.stat().st_size
            if actual_hash != record[key].get("sha256"):
                errors.append(f"{record['candidate_id']}: {filename} SHA256 differs from profile")
        cached_glbs = sorted((reconstruction / "chunks_300k").glob("chunk_*.glb"))
        for chunk_path in cached_glbs:
            try:
                chunk = parse_glb(chunk_path)
                if chunk["meshes"] != 1 or chunk["materials"] != 1:
                    raise ValueError("chunk GLB does not contain exactly one mesh/material")
                chunk_glbs_validated += 1
            except (OSError, ValueError, json.JSONDecodeError, struct.error) as error:
                errors.append(f"{record['candidate_id']}: invalid cached {chunk_path.name}: {error}")
        expected_chunks = int(record["foundation_chunk_count"])
        empty_chunks = len(record["empty_glb_chunks"])
        expected_nonempty_chunks = expected_chunks - empty_chunks
        if len(set(record["empty_glb_chunks"])) != empty_chunks:
            errors.append(f"{record['candidate_id']}: duplicate empty chunk records")
        if record["glb"]["meshes"] != expected_nonempty_chunks:
            errors.append(
                f"{record['candidate_id']}: GLB meshes {record['glb']['meshes']} != expected "
                f"nonempty chunks {expected_nonempty_chunks}"
            )
        if record["cached_chunk_glbs"] != expected_nonempty_chunks:
            errors.append(
                f"{record['candidate_id']}: cached GLBs {record['cached_chunk_glbs']} != expected "
                f"nonempty chunks {expected_nonempty_chunks}"
            )
    strict_json_files = 0
    for root in (output_root, report_root):
        for path in root.rglob("*.json"):
            try:
                load_json(path)
                strict_json_files += 1
            except (OSError, ValueError, json.JSONDecodeError) as error:
                errors.append(f"strict JSON: {path}: {error}")
    result = {
        "schema": f"genrecon.{track_name}-genrecon-validation",
        "track": track_name,
        "schema_version": 1,
        "validated_utc": utc_now(),
        "result": "pass" if not errors else "fail",
        "counts": {
            **summary["summary"],
            "strict_json_files": strict_json_files,
            "deliverable_hash_bytes": deliverable_hash_bytes,
            "cached_chunk_glbs_validated": chunk_glbs_validated,
        },
        "checks": {
            "all_reconstructions_profiled": summary["summary"]["reconstruct_complete"] == len(candidates),
            "all_mesh_ply_payloads_valid": all(row["mesh"] is not None for row in summary["candidates"]),
            "all_deliverable_sha256_match_profiles": not any(
                "SHA256 differs" in error for error in errors
            ),
            "all_glbs_v2_with_embedded_pbr": all(row["glb"] is not None for row in summary["candidates"]),
            "all_cached_chunk_glbs_v2_with_embedded_pbr": chunk_glbs_validated
            == summary["summary"]["nonempty_glb_primitives"],
            "nonempty_and_explicitly_empty_chunks_cover_preflight": not any(
                "expected nonempty chunks" in error or "empty chunk" in error for error in errors
            ),
            "strict_json": not any(error.startswith("strict JSON") for error in errors),
        },
        "errors": errors,
    }
    write_json(output_root / "validation.json", result)
    return result


def selected_candidates(index: dict[str, Any], requested: list[str]) -> list[dict[str, Any]]:
    candidates = [
        item
        for item in index["candidates"]
        if item.get("status") == "completed" and item.get("preflight") in {"passed", "marginal"}
    ]
    by_id = {item["candidate_id"]: item for item in candidates}
    if requested:
        missing = sorted(set(requested) - set(by_id))
        if missing:
            raise ValueError(f"Requested candidates are not consumable: {missing}")
        return [by_id[identifier] for identifier in dict.fromkeys(requested)]
    disposition_order = {"proceed_foundation_pilot": 0, "refine_then_retry": 1, "reject_current_shot": 2}
    grade_order = {"P-A": 0, "P-B": 1, "P-C": 2, "P-F": 3}
    return sorted(
        candidates,
        key=lambda item: (
            disposition_order.get(item.get("visual_disposition"), 9),
            grade_order.get(candidate_grade(item), 9),
            item["candidate_id"],
        ),
    )


def configure_environment() -> None:
    defaults = {
        "CUDA_HOME": "/usr/local/cuda-12.6",
        "TORCH_CUDA_ARCH_LIST": "8.6",
        "HF_HOME": str(ROOT / "data" / "hf-cache"),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "OMP_NUM_THREADS": "8",
        "HF_HUB_OFFLINE": "1",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def run_batch(args: argparse.Namespace, candidates: list[dict[str, Any]]) -> int:
    lock_path = args.output_root / ".batch.lock"
    args.output_root.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"Another foundation GenRecon batch owns {lock_path}") from error

    config_path = args.output_root / "config.json"
    if not config_path.is_file() or args.refresh_provenance:
        write_json(
            config_path,
            {
                "schema": f"genrecon.{args.track_name}-genrecon-config",
                "track": args.track_name,
                "schema_version": 1,
                "created_utc": utc_now(),
                "source_root": args.input_root,
                "output_root": args.output_root,
                "report_root": args.report_root,
                "candidate_ids": [item["candidate_id"] for item in candidates],
                "protocol": {
                    "mode": "Iphone",
                    "colmap_subdir": args.colmap_subdir,
                    "max_reproj_error": args.max_reproj_error,
                    "min_track_len": args.min_track_len,
                    "num_imgs_per_scene": 8,
                    "center_crop": False,
                    "seed": 42,
                    "chunk_size_factor": 1.08,
                    "min_overlap_factor": 4,
                    "proj_batch_voxels": 256,
                    "glb_texture_size": 4096,
                    "glb_simplify_threshold_per_chunk": 300000,
                    "glb_skip_fill_holes": True,
                    "glb_skip_remesh": True,
                },
                "provenance": command_provenance(),
            },
        )

    run_state = {
        "schema": f"genrecon.{args.track_name}-genrecon-batch-run",
        "track": args.track_name,
        "schema_version": 1,
        "started_utc": utc_now(),
        "finished_utc": None,
        "candidate_ids": [item["candidate_id"] for item in candidates],
        "records": [],
    }
    write_json(args.report_root / "batch_run.json", run_state)
    had_failure = False
    try:
        for index, candidate in enumerate(candidates, start=1):
            identifier = candidate["candidate_id"]
            paths = candidate_paths(identifier, args.input_root, args.output_root, args.report_root)
            paths["reconstruction"].mkdir(parents=True, exist_ok=True)
            paths["report"].mkdir(parents=True, exist_ok=True)
            mesh = paths["reconstruction"] / "mesh.ply"
            to_glb = paths["reconstruction"] / "to_glb_inputs.pt"
            chunk_inputs = paths["reconstruction"] / "chunk_inputs.pt"
            scene_glb = paths["reconstruction"] / "scene.glb"
            reconstruct_profile = paths["report"] / "reconstruct_profile.json"
            glb_profile = paths["report"] / "glb_profile.json"
            print(
                f"[{index}/{len(candidates)}] {identifier}: grade={candidate_grade(candidate)} "
                f"visual={candidate['visual_disposition']}",
                flush=True,
            )
            record = {"candidate_id": identifier, "started_utc": utc_now(), "stages": {}}
            reconstruct_ready = profile_is_complete(
                reconstruct_profile, [mesh, to_glb, chunk_inputs]
            )
            if reconstruct_ready and not args.force_reconstruct:
                record["stages"]["reconstruct"] = "skipped_complete"
                print(f"[{index}/{len(candidates)}] {identifier}: reconstruction already complete", flush=True)
            else:
                profile = profile_command(
                    reconstruct_command(
                        paths["input"],
                        paths["reconstruction"],
                        colmap_subdir=args.colmap_subdir,
                        max_reproj_error=args.max_reproj_error,
                        min_track_len=args.min_track_len,
                    ),
                    cwd=ROOT,
                    log_path=paths["report"] / "reconstruct.log",
                    result_path=reconstruct_profile,
                    expected_files=[mesh, to_glb, chunk_inputs],
                    progress_interval_s=30.0,
                    label=f"{args.track_name}-genrecon-{identifier}",
                )
                reconstruct_ready = profile.get("success") is True
                record["stages"]["reconstruct"] = "complete" if reconstruct_ready else "failed"
            if not reconstruct_ready:
                had_failure = True
                record["stages"]["glb"] = "blocked_reconstruction_failed"
                print(f"[{index}/{len(candidates)}] {identifier}: reconstruction failed", flush=True)
            else:
                glb_ready = profile_is_complete(glb_profile, [scene_glb])
                if glb_ready and not args.force_glb:
                    record["stages"]["glb"] = "skipped_complete"
                    print(f"[{index}/{len(candidates)}] {identifier}: GLB already complete", flush=True)
                else:
                    profile = profile_command(
                        glb_command(paths["reconstruction"]),
                        cwd=ROOT,
                        log_path=paths["report"] / "glb.log",
                        result_path=glb_profile,
                        expected_files=[scene_glb],
                        progress_interval_s=30.0,
                        label=f"{args.track_name}-glb-{identifier}",
                    )
                    glb_ready = profile.get("success") is True
                    record["stages"]["glb"] = "complete" if glb_ready else "failed"
                if not glb_ready:
                    had_failure = True
                    print(f"[{index}/{len(candidates)}] {identifier}: GLB conversion failed", flush=True)
            record["finished_utc"] = utc_now()
            run_state["records"].append(record)
            write_summary(
                candidates,
                args.input_root,
                args.output_root,
                args.report_root,
                args.track_name,
            )
            write_json(args.report_root / "batch_run.json", run_state)
            if had_failure and args.fail_fast:
                break
    finally:
        run_state["finished_utc"] = utc_now()
        write_json(args.report_root / "batch_run.json", run_state)
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()
    return 1 if had_failure else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "summarize", "validate", "all"))
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--track-name", default="foundation-sfm")
    parser.add_argument("--colmap-subdir", default="colmap_vggt")
    parser.add_argument("--max-reproj-error", type=float)
    parser.add_argument("--min-track-len", type=int)
    parser.add_argument("--force-reconstruct", action="store_true")
    parser.add_argument("--force-glb", action="store_true")
    parser.add_argument("--refresh-provenance", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    args.report_root = args.report_root.resolve()
    source_index = load_json(args.input_root / "index.json")
    candidates = selected_candidates(source_index, args.candidate)
    configure_environment()
    return_code = 0
    if args.stage in {"run", "all"}:
        return_code = run_batch(args, candidates)
    if args.stage in {"summarize", "all"}:
        summary = write_summary(
            candidates, args.input_root, args.output_root, args.report_root, args.track_name
        )
        print(f"[summary] {summary['summary']}", flush=True)
    if args.stage in {"validate", "all"}:
        validation = validate_outputs(
            candidates, args.input_root, args.output_root, args.report_root, args.track_name
        )
        print(f"[validation] {validation['result']} {validation['counts']}", flush=True)
        if validation["result"] != "pass":
            return_code = 1
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
