#!/usr/bin/env python3
"""Deep source-archive validation for the frozen GT calibration collection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

import cv2
import numpy as np
import py7zr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "data" / "gt-calibration-v1"
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


def validate_sources(root: Path) -> dict[str, Any]:
    source_root = root / "sources"
    paths = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".zip", ".7z", ".mp4"}
    )
    records = []
    errors = []
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
            else:
                record.update(validate_video(path))
            record["sha256"] = digest(path, "sha256")
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
        "schema_version": 1,
        "result": "pass" if not errors else "fail",
        "counts": {
            "archives_and_videos": len(records),
            "validated_bytes": total_bytes,
            "zip": sum(record.get("kind") == "zip" for record in records),
            "seven_zip": sum(record.get("kind") == "7z" for record in records),
            "video": sum(record.get("kind") == "video" for record in records),
            "declared_quota_pages": len(quota_pages),
            "historical_download_failures": len(historical_failures),
            "auxiliary_records": len(auxiliary_records),
        },
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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "source_validation.json"
    result = validate_sources(root)
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
