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
import py7zr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "data" / "gt-calibration-v1"
TNT_EXPECTED_MD5 = {
    "Meetingroom.mp4": "5beeb4e21ca5b8fda31235cf15972393",
    "Meetingroom.zip": "754932b99adcfc602908c5bda917c5a3",
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
            if path.suffix.lower() == ".zip":
                record.update(validate_zip(path))
            elif path.suffix.lower() == ".7z":
                record.update(validate_7z(path))
            else:
                record.update(validate_video(path))
            record["sha256"] = digest(path, "sha256")
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
        },
        "records": records,
        "declared_quota_pages": quota_pages,
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
