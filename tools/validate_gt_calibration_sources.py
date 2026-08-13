#!/usr/bin/env python3
"""Deep source-archive validation for the frozen GT calibration collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile

import cv2
import numpy as np
import py7zr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "data" / "gt-calibration-v1"
DEFAULT_PLAN = ROOT / "configs" / "eval" / "gt_calibration_v1.json"
OMNI_DATASET_REPO = "OpenXDLab/OmniObject3D-New"
OMNI_INDEX_SCHEMA = "genrecon.omniobject3d-openxlab-file-index"
OMNI_DOWNLOAD_RELATIVE = Path(
    "downloads/OpenXDLab___OmniObject3D-New"
)
TNT_EXPECTED_MD5 = {
    "Meetingroom.mp4": "5beeb4e21ca5b8fda31235cf15972393",
    "Meetingroom.zip": "754932b99adcfc602908c5bda917c5a3",
}
TNT_FROZEN_SHA256 = {
    "Meetingroom_individual_scans.zip": "2653a0849c7e8023c4280f041777977409c26f5d1c3f2f3f6238f91e8d7227f6",
    "Meetingroom_alignment.txt": "fed12060a3f347e1eaf4a806c43966cf5972cd1dfc6032a4e412a814eda7392f",
    "Meetingroom_intrinsics.json": "efa8dcdc70f81361c8ff5b17384d63fc157fa7d523429674cf1f067d183c72b0",
}


class SourceValidationError(RuntimeError):
    pass


def digest(path: Path, algorithm: str, block_size: int = 8 << 20) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            value.update(block)
    return value.hexdigest()


def validate_zip(path: Path) -> dict[str, Any]:
    try:
        with ZipFile(path) as archive:
            names = archive.namelist()
            bad = archive.testzip()
    except BadZipFile as exc:
        raise SourceValidationError(f"Invalid ZIP {path}: {exc}") from exc
    if bad is not None:
        raise SourceValidationError(f"ZIP CRC failure in {path}: {bad}")
    if not names:
        raise SourceValidationError(f"Empty ZIP archive: {path}")
    return {"kind": "zip", "entries": len(names), "crc": "pass"}


def validate_7z(path: Path) -> dict[str, Any]:
    try:
        with py7zr.SevenZipFile(path) as archive:
            names = archive.getnames()
            bad = archive.test()
    except Exception as exc:
        raise SourceValidationError(f"Invalid 7z {path}: {exc}") from exc
    if bad is not None:
        raise SourceValidationError(f"7z CRC failure in {path}: {bad}")
    if not names:
        raise SourceValidationError(f"Empty 7z archive: {path}")
    return {"kind": "7z", "entries": len(names), "crc": "pass"}


def _normalized_tar_name(name: str) -> str:
    return name[2:] if name.startswith("./") else name


def _validate_tar_member(member: tarfile.TarInfo, path: Path) -> str:
    member_path = PurePosixPath(member.name)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise SourceValidationError(f"Unsafe TAR path in {path}: {member.name!r}")
    if member.issym() or member.islnk() or member.isdev():
        raise SourceValidationError(
            f"Unsupported TAR member in {path}: {member.name!r}"
        )
    return _normalized_tar_name(member.name)


def _validate_omni_transforms(
    payload: bytes, *, object_id: str, archive_path: Path
) -> dict[str, Any]:
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceValidationError(
            f"Invalid transforms.json for {object_id} in {archive_path}: {exc}"
        ) from exc
    frames = document.get("frames")
    if not isinstance(frames, list) or len(frames) != 100:
        raise SourceValidationError(
            f"Expected 100 transform frames for {object_id} in {archive_path}"
        )
    names = [str(frame.get("file_path", "")) for frame in frames]
    expected_names = {f"r_{index}" for index in range(100)}
    if len(names) != len(set(names)) or set(names) != expected_names:
        raise SourceValidationError(
            f"Invalid 100-view filename set for {object_id} in {archive_path}"
        )
    camera_angle_x = document.get("camera_angle_x")
    if (
        not isinstance(camera_angle_x, (int, float))
        or not math.isfinite(float(camera_angle_x))
        or float(camera_angle_x) <= 0.0
    ):
        raise SourceValidationError(
            f"Invalid camera_angle_x for {object_id} in {archive_path}"
        )
    return {
        "object_id": object_id,
        "transforms_frames": len(frames),
        "source_view_names": len(set(names)),
        "camera_angle_x": float(camera_angle_x),
    }


def validate_tar_gzip(
    path: Path,
    *,
    render_object_ids: tuple[str, ...] = (),
    scan_object_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    if render_object_ids and scan_object_ids:
        raise ValueError("A TAR cannot be both a render and scan contract")
    selected_ids = render_object_ids or scan_object_ids
    selected_files: dict[str, dict[str, tuple[int, int]]] = {
        object_id: {} for object_id in selected_ids
    }
    transforms_payloads: dict[str, bytes] = {}
    entries = 0
    files = 0
    uncompressed_file_bytes = 0
    seen_file_names: set[str] = set()
    try:
        with tarfile.open(path, "r|gz") as archive:
            for member in archive:
                entries += 1
                name = _validate_tar_member(member, path)
                if not member.isfile():
                    continue
                files += 1
                uncompressed_file_bytes += member.size
                if name in seen_file_names:
                    raise SourceValidationError(
                        f"Duplicate TAR file member in {path}: {name!r}"
                    )
                seen_file_names.add(name)
                for object_id in selected_ids:
                    prefix = f"{object_id}/"
                    if not name.startswith(prefix):
                        continue
                    selected_files[object_id][name] = (member.size, member.offset_data)
                    transforms_name = f"{object_id}/render/transforms.json"
                    if render_object_ids and name == transforms_name:
                        handle = archive.extractfile(member)
                        if handle is None:
                            raise SourceValidationError(
                                f"Cannot read {transforms_name} from {path}"
                            )
                        payload = handle.read()
                        if len(payload) != member.size:
                            raise SourceValidationError(
                                f"Short read for {transforms_name} in {path}"
                            )
                        transforms_payloads[object_id] = payload
    except SourceValidationError:
        raise
    except (OSError, tarfile.TarError, EOFError) as exc:
        raise SourceValidationError(f"Invalid TAR/GZip {path}: {exc}") from exc
    if entries == 0 or files == 0:
        raise SourceValidationError(f"Empty TAR/GZip archive: {path}")

    payload_records = []
    expected_views = {f"r_{index}" for index in range(100)}
    for object_id in selected_ids:
        selected = selected_files[object_id]
        if render_object_ids:
            transforms_name = f"{object_id}/render/transforms.json"
            rgb_names = {
                f"{object_id}/render/images/{name}.png" for name in expected_views
            }
            normal_names = {
                f"{object_id}/render/normals/{name}_normal.png"
                for name in expected_views
            }
            depth_names = {
                f"{object_id}/render/depths/{name}_depth.exr"
                for name in expected_views
            }
            expected = {transforms_name, *rgb_names, *normal_names, *depth_names}
            actual = set(selected)
            if actual != expected:
                missing = sorted(expected - actual)
                unexpected = sorted(actual - expected)
                raise SourceValidationError(
                    f"Incomplete 100-view render payload for {object_id} in {path}: "
                    f"missing={missing[:3]} unexpected={unexpected[:3]}"
                )
            if any(selected[name][0] <= 0 for name in expected):
                raise SourceValidationError(
                    f"Empty render member for {object_id} in {path}"
                )
            transform_record = _validate_omni_transforms(
                transforms_payloads[object_id],
                object_id=object_id,
                archive_path=path,
            )
            payload_records.append(
                {
                    **transform_record,
                    "rgb_members": len(rgb_names),
                    "normal_members": len(normal_names),
                    "depth_members": len(depth_names),
                    "transforms_members": 1,
                    "result": "pass",
                }
            )
        else:
            expected_name = f"{object_id}/Scan/Scan.obj"
            if expected_name not in selected or selected[expected_name][0] <= 0:
                raise SourceValidationError(
                    f"Expected one nonempty Scan.obj for {object_id} in {path}"
                )
            payload_records.append(
                {
                    "object_id": object_id,
                    "scan_obj_members": 1,
                    "selected_object_file_members": len(selected),
                    "result": "pass",
                }
            )
    return {
        "kind": "tar-gzip",
        "entries": entries,
        "files": files,
        "uncompressed_file_bytes": uncompressed_file_bytes,
        "stream_integrity": "pass",
        "path_safety": "pass",
        "selected_object_payloads": payload_records,
    }


def validate_video(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SourceValidationError(f"Video cannot be opened: {path}")
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if frame_count <= 0 or fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise SourceValidationError(f"Video metadata is invalid: {path}")
    sample_candidates = {0, frame_count // 2, max(0, frame_count - 10)}
    sampled = []
    for index in sorted(sample_candidates):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok or frame is None or frame.shape[:2] != (height, width):
            capture.release()
            raise SourceValidationError(f"Video sample frame {index} failed: {path}")
        sampled.append(index)
    capture.release()
    return {
        "kind": "video",
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "sampled_frames": sampled,
    }


def validate_tnt_scans(path: Path) -> dict[str, Any]:
    record = validate_zip(path)
    with ZipFile(path) as archive:
        names = archive.namelist()
        ply_names = sorted(name for name in names if Path(name).suffix.lower() == ".ply")
        scanner_positions = [name for name in names if Path(name).name == "scanner_pos.txt"]
        if len(ply_names) != 11 or len(scanner_positions) != 1:
            raise SourceValidationError(
                f"Expected 11 PLY scans and one scanner_pos.txt in {path}, "
                f"found {len(ply_names)} and {len(scanner_positions)}"
            )
        scanner_position_name = scanner_positions[0]
        scanner_position_lines = [
            line
            for line in archive.read(scanner_position_name).decode("ascii").splitlines()
            if line.strip()
        ]
        if len(scanner_position_lines) != 12:
            raise SourceValidationError(
                f"Expected 12 scanner position records in {path}, found "
                f"{len(scanner_position_lines)}"
            )
        headers = {}
        for name in ply_names:
            with archive.open(name) as handle:
                header = handle.read(256)
            if not header.startswith(b"ply\nformat binary_little_endian 1.0\n"):
                raise SourceValidationError(f"Unexpected PLY header in {path}: {name}")
            headers[Path(name).name] = "binary_little_endian"
    record.update(
        {
            "official_payload": "Meetingroom individual prealigned scans",
            "ply_scans": len(ply_names),
            "scanner_positions": scanner_position_name,
            "scanner_position_records": len(scanner_position_lines),
            "ply_headers": headers,
        }
    )
    return record


def validate_tnt_alignment(path: Path) -> dict[str, Any]:
    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise SourceValidationError(f"Invalid T&T alignment matrix: {path}")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-10):
        raise SourceValidationError(f"Invalid homogeneous row in T&T alignment: {path}")
    determinant = float(np.linalg.det(matrix[:3, :3]))
    if determinant <= 0.0:
        raise SourceValidationError(f"T&T alignment has non-positive scale: {path}")
    scale = float(np.cbrt(determinant))
    rotation = matrix[:3, :3] / scale
    orthogonality_error = float(np.linalg.norm(rotation.T @ rotation - np.eye(3)))
    if orthogonality_error > 1e-8:
        raise SourceValidationError(f"T&T alignment is not a valid Sim(3): {path}")
    return {
        "kind": "alignment",
        "shape": [4, 4],
        "sim3_scale": scale,
        "rotation_orthogonality_error": orthogonality_error,
    }


def validate_tnt_intrinsics(path: Path, source_root: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    gates = document.get("quality_gates")
    if (
        document.get("schema") != "genrecon.tnt-fixed-pose-intrinsics"
        or document.get("schema_version") != 2
        or document.get("protocol", {}).get("revision") != "v2-single-thread-geometry"
        or document.get("result") != "pass"
        or not isinstance(gates, dict)
        or not gates
        or not all(gates.values())
    ):
        raise SourceValidationError(f"T&T intrinsics artifact did not pass: {path}")
    expected = {
        "image_archive_sha256": digest(source_root / "Meetingroom.zip", "sha256"),
        "camera_log_sha256": digest(source_root / "Meetingroom_COLMAP.log", "sha256"),
    }
    for key, value in expected.items():
        if document.get("source", {}).get(key) != value:
            raise SourceValidationError(f"T&T intrinsics source hash mismatch for {key}")
    return {
        "kind": "derived-calibration",
        "calibration_result": "pass",
        "camera_model": document["calibration"]["model"],
        "params": document["calibration"]["params"],
        "points3D": document["final_stats"]["points3D"],
        "observations": document["final_stats"]["observations"],
        "mean_reprojection_error_px": document["final_stats"][
            "mean_reprojection_error_px"
        ],
    }


def _omni_category(object_id: str) -> str:
    category, separator, suffix = object_id.rpartition("_")
    if not separator or not category or len(suffix) != 3 or not suffix.isdigit():
        raise SourceValidationError(f"Invalid OmniObject3D object ID: {object_id!r}")
    return category


def _prepare_omni_source_contract(
    source_root: Path, plan_path: Path
) -> tuple[dict[Path, dict[str, Any]], dict[str, Any] | None, list[str]]:
    omni_root = source_root / "omniobject3d"
    if not omni_root.is_dir():
        return {}, None, []
    errors: list[str] = []
    index_path = omni_root / "metadata" / "openxlab_file_index.json"
    if not plan_path.is_file() or not index_path.is_file():
        missing = [
            str(path)
            for path in (plan_path, index_path)
            if not path.is_file()
        ]
        return {}, None, [f"Missing OmniObject3D plan/index: {missing}"]
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {}, None, [f"Invalid OmniObject3D plan/index JSON: {exc}"]

    dataset = plan.get("datasets", {}).get("omniobject3d")
    if (
        not isinstance(dataset, dict)
        or dataset.get("adapter", {}).get("adapter_version") != 3
        or not isinstance(dataset.get("unit_ids"), list)
        or not dataset["unit_ids"]
    ):
        errors.append("Frozen plan does not declare OmniObject3D adapter v3 units")
        return {}, None, errors
    object_ids = tuple(dataset["unit_ids"])
    if len(object_ids) != len(set(object_ids)):
        errors.append("Frozen OmniObject3D unit IDs are not unique")
        return {}, None, errors
    try:
        categories: dict[str, list[str]] = {}
        for object_id in object_ids:
            categories.setdefault(_omni_category(object_id), []).append(object_id)
    except SourceValidationError as exc:
        errors.append(str(exc))
        return {}, None, errors
    rejected = {
        item.get("object_id")
        for item in dataset.get("rejected_candidates", [])
        if isinstance(item, dict)
    }
    selected_rejected = sorted(set(object_ids) & rejected)
    if selected_rejected:
        errors.append(
            f"Rejected OmniObject3D candidates remain selected: {selected_rejected}"
        )

    files = index.get("files")
    if (
        index.get("schema") != OMNI_INDEX_SCHEMA
        or index.get("schema_version") != 1
        or index.get("dataset_repo") != OMNI_DATASET_REPO
        or index.get("all_files_have_sha256") is not True
        or not isinstance(files, list)
    ):
        errors.append("Unexpected OmniObject3D OpenXLab metadata index contract")
        return {}, None, errors
    index_by_path: dict[str, dict[str, Any]] = {}
    indexed_bytes = 0
    for item in files:
        if not isinstance(item, dict):
            errors.append("OpenXLab metadata index contains a non-object record")
            continue
        openxlab_path = item.get("path")
        size = item.get("size")
        sha256 = item.get("sha256")
        if (
            not isinstance(openxlab_path, str)
            or not openxlab_path.startswith("/")
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            errors.append(f"Invalid OpenXLab metadata record: {item!r}")
            continue
        if openxlab_path in index_by_path:
            errors.append(f"Duplicate OpenXLab metadata path: {openxlab_path}")
            continue
        index_by_path[openxlab_path] = item
        indexed_bytes += size
    if index.get("file_count") != len(files) or index.get("total_bytes") != indexed_bytes:
        errors.append("OpenXLab metadata index count/byte totals are inconsistent")

    download_root = omni_root / OMNI_DOWNLOAD_RELATIVE
    contexts: dict[Path, dict[str, Any]] = {}
    for local_path in sorted(
        path for path in download_root.rglob("*.tar.gz") if path.is_file()
    ):
        openxlab_path = "/" + local_path.relative_to(download_root).as_posix()
        expected = index_by_path.get(openxlab_path)
        if expected is None:
            errors.append(
                f"Local OmniObject3D TAR is absent from OpenXLab index: {local_path}"
            )
            continue
        contexts[local_path.resolve()] = {
            "openxlab_path": openxlab_path,
            "expected_size_bytes": expected["size"],
            "expected_sha256": expected["sha256"],
            "dataset_id": expected.get("dataset_id"),
            "render_object_ids": (),
            "scan_object_ids": (),
        }

    for category, selected_ids in sorted(categories.items()):
        specifications = (
            ("blender_renders", "render_object_ids"),
            ("raw_scans", "scan_object_ids"),
        )
        for directory, contract_key in specifications:
            openxlab_path = f"/raw/{directory}/{category}.tar.gz"
            local_path = (download_root / openxlab_path.removeprefix("/")).resolve()
            expected = index_by_path.get(openxlab_path)
            if expected is None:
                errors.append(f"OpenXLab index is missing selected archive {openxlab_path}")
                continue
            if not local_path.is_file():
                errors.append(f"Selected OmniObject3D archive is missing: {local_path}")
                continue
            context = contexts.get(local_path)
            if context is None:
                context = {
                    "openxlab_path": openxlab_path,
                    "expected_size_bytes": expected["size"],
                    "expected_sha256": expected["sha256"],
                    "dataset_id": expected.get("dataset_id"),
                    "render_object_ids": (),
                    "scan_object_ids": (),
                }
                contexts[local_path] = context
            context[contract_key] = tuple(sorted(selected_ids))

    provenance = {
        "calibration_plan": str(plan_path.resolve()),
        "calibration_plan_sha256": digest(plan_path, "sha256"),
        "adapter_version": dataset["adapter"]["adapter_version"],
        "selected_object_ids": list(object_ids),
        "selected_categories": sorted(categories),
        "rejected_object_ids": sorted(value for value in rejected if value),
        "openxlab_index": str(index_path.resolve()),
        "openxlab_index_sha256": digest(index_path, "sha256"),
        "openxlab_dataset_repo": index["dataset_repo"],
        "openxlab_dataset_id": index.get("dataset_id"),
        "openxlab_indexed_files": len(files),
        "openxlab_indexed_bytes": indexed_bytes,
        "locally_downloaded_tar_archives": len(contexts),
    }
    return contexts, provenance, errors


def validate_sources(root: Path, *, plan_path: Path = DEFAULT_PLAN) -> dict[str, Any]:
    source_root = root / "sources"
    paths = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file()
        and (
            path.suffix.lower() in {".zip", ".7z", ".mp4"}
            or path.name.lower().endswith(".tar.gz")
        )
    )
    records = []
    omni_contexts, omni_provenance, omni_errors = _prepare_omni_source_contract(
        source_root, plan_path.resolve()
    )
    errors = list(omni_errors)
    if not paths:
        errors.append(f"No source archives or videos found under {source_root}")
    total_bytes = 0
    for path in paths:
        record: dict[str, Any] = {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
        }
        total_bytes += path.stat().st_size
        try:
            if path.name == "Meetingroom_individual_scans.zip":
                record.update(validate_tnt_scans(path))
            elif path.suffix.lower() == ".zip":
                record.update(validate_zip(path))
            elif path.suffix.lower() == ".7z":
                record.update(validate_7z(path))
            elif path.name.lower().endswith(".tar.gz"):
                context = omni_contexts.get(path.resolve(), {})
                record.update(
                    validate_tar_gzip(
                        path,
                        render_object_ids=context.get("render_object_ids", ()),
                        scan_object_ids=context.get("scan_object_ids", ()),
                    )
                )
                if "omniobject3d" in path.parts:
                    if not context:
                        raise SourceValidationError(
                            f"OmniObject3D TAR lacks metadata binding: {path}"
                        )
                    record["openxlab"] = {
                        "path": context["openxlab_path"],
                        "dataset_id": context["dataset_id"],
                        "size_bytes": context["expected_size_bytes"],
                        "sha256": context["expected_sha256"],
                    }
                    if path.stat().st_size != context["expected_size_bytes"]:
                        raise SourceValidationError(
                            f"OpenXLab size mismatch for {path}: "
                            f"{path.stat().st_size} != {context['expected_size_bytes']}"
                        )
            else:
                record.update(validate_video(path))
            record["sha256"] = digest(path, "sha256")
            context = omni_contexts.get(path.resolve())
            if context is not None and record["sha256"] != context["expected_sha256"]:
                raise SourceValidationError(
                    f"OpenXLab SHA256 mismatch for {path}: "
                    f"{record['sha256']} != {context['expected_sha256']}"
                )
            frozen_sha256 = TNT_FROZEN_SHA256.get(path.name)
            if frozen_sha256 is not None:
                record["frozen_sha256"] = frozen_sha256
                if record["sha256"] != frozen_sha256:
                    raise SourceValidationError(
                        f"Frozen SHA256 mismatch for {path}: "
                        f"{record['sha256']} != {frozen_sha256}"
                    )
            expected_md5 = TNT_EXPECTED_MD5.get(path.name)
            if expected_md5 is not None:
                actual_md5 = digest(path, "md5")
                record["official_md5"] = expected_md5
                record["md5"] = actual_md5
                if actual_md5 != expected_md5:
                    raise SourceValidationError(
                        f"Official MD5 mismatch for {path}: {actual_md5} != {expected_md5}"
                    )
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            errors.append(record["error"])
        records.append(record)
    auxiliary_records = []
    tnt_root = source_root / "tanks-and-temples"
    if tnt_root.is_dir():
        auxiliary_specs = (
            (tnt_root / "Meetingroom_alignment.txt", validate_tnt_alignment),
            (
                tnt_root / "Meetingroom_intrinsics.json",
                lambda value: validate_tnt_intrinsics(value, tnt_root),
            ),
        )
    else:
        auxiliary_specs = ()
    for path, validator in auxiliary_specs:
        if not path.is_file():
            errors.append(f"Missing T&T auxiliary source: {path}")
            continue
        record: dict[str, Any] = {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
        }
        try:
            record.update(validator(path))
            record["sha256"] = digest(path, "sha256")
            frozen_sha256 = TNT_FROZEN_SHA256.get(path.name)
            if frozen_sha256 is not None:
                record["frozen_sha256"] = frozen_sha256
                if record["sha256"] != frozen_sha256:
                    raise SourceValidationError(
                        f"Frozen SHA256 mismatch for {path}: "
                        f"{record['sha256']} != {frozen_sha256}"
                    )
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            errors.append(record["error"])
        auxiliary_records.append(record)

    quota_pages = []
    for path in sorted(source_root.rglob("*.quota-blocked.html")):
        text = path.read_text(encoding="utf-8", errors="replace")
        quota_present = "Quota exceeded" in text
        quota_pages.append(
            {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": digest(path, "sha256"),
                "quota_message_present": quota_present,
            }
        )
        if not quota_present:
            errors.append(f"Declared quota page does not contain quota marker: {path}")
    historical_failures = []
    for path in sorted(source_root.rglob("*.previous-quota-response.html")):
        text = path.read_text(encoding="utf-8", errors="replace")
        historical_failures.append(
            {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": digest(path, "sha256"),
                "quota_message_present": "Quota exceeded" in text,
                "status": "historical-resolved-download-failure",
            }
        )
    result = {
        "schema": "genrecon.gt-calibration-source-validation",
        "schema_version": 2,
        "result": "pass" if not errors else "fail",
        "counts": {
            "archives_and_videos": len(records),
            "validated_bytes": total_bytes,
            "zip": sum(record.get("kind") == "zip" for record in records),
            "seven_zip": sum(record.get("kind") == "7z" for record in records),
            "video": sum(record.get("kind") == "video" for record in records),
            "tar_gzip": sum(record.get("kind") == "tar-gzip" for record in records),
            "omniobject3d_tar_gzip": sum(
                record.get("kind") == "tar-gzip" and "openxlab" in record
                for record in records
            ),
            "omniobject3d_selected_render_archives": sum(
                bool(record.get("selected_object_payloads"))
                and "/raw/blender_renders/" in record.get("openxlab", {}).get("path", "")
                for record in records
            ),
            "omniobject3d_selected_scan_archives": sum(
                bool(record.get("selected_object_payloads"))
                and "/raw/raw_scans/" in record.get("openxlab", {}).get("path", "")
                for record in records
            ),
            "omniobject3d_selected_render_payloads": sum(
                len(record.get("selected_object_payloads", []))
                for record in records
                if "/raw/blender_renders/" in record.get("openxlab", {}).get("path", "")
            ),
            "omniobject3d_selected_scan_payloads": sum(
                len(record.get("selected_object_payloads", []))
                for record in records
                if "/raw/raw_scans/" in record.get("openxlab", {}).get("path", "")
            ),
            "declared_quota_pages": len(quota_pages),
            "historical_download_failures": len(historical_failures),
            "auxiliary_records": len(auxiliary_records),
        },
        "omniobject3d": omni_provenance,
        "records": records,
        "auxiliary_records": auxiliary_records,
        "declared_quota_pages": quota_pages,
        "historical_download_failures": historical_failures,
        "errors": errors,
    }
    json.dumps(result, allow_nan=False)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "source_validation.json"
    result = validate_sources(root, plan_path=args.plan.resolve())
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"[source-validation] {result['result']} {result['counts']}")
    for error in result["errors"][:20]:
        print(f"  - {error}")
    return 0 if result["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
