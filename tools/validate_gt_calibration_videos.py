#!/usr/bin/env python3
"""Release-validate representative GT calibration visualization videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from tools.export_foundation_genrecon_videos import probe_video
    from tools.export_gt_calibration_videos import (
        AVAILABLE_STATUS,
        BLOCKED_STATUS,
        COMPARISON_SIZE,
        MISSING_PREDICTION_STATUS,
        PANEL_SIZE,
        VisualizationError,
        load_json,
        sha256_file,
        write_json,
    )
except ModuleNotFoundError:
    from export_foundation_genrecon_videos import probe_video
    from export_gt_calibration_videos import (
        AVAILABLE_STATUS,
        BLOCKED_STATUS,
        COMPARISON_SIZE,
        MISSING_PREDICTION_STATUS,
        PANEL_SIZE,
        VisualizationError,
        load_json,
        sha256_file,
        write_json,
    )

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "gt-calibration-v1" / "visualizations-v1"
FONT_REGULAR = Path("/usr/share/fonts/truetype/lato/Lato-Medium.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/lato/Lato-Semibold.ttf")


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def expected_video_contract(status: str) -> dict[str, tuple[int, int, int]]:
    """Return video key to (frames, width, height)."""
    if status == AVAILABLE_STATUS:
        return {
            "source": (16, PANEL_SIZE[0], PANEL_SIZE[1]),
            "reference": (16, PANEL_SIZE[0], PANEL_SIZE[1]),
            "prediction": (16, PANEL_SIZE[0], PANEL_SIZE[1]),
            "comparison": (16, COMPARISON_SIZE[0], COMPARISON_SIZE[1]),
        }
    if status == BLOCKED_STATUS:
        return {"status": (8, COMPARISON_SIZE[0], COMPARISON_SIZE[1])}
    raise ValueError(f"Unsupported visualization status: {status}")


def expected_role_order() -> list[tuple[str, int]]:
    return [("conditioning", index) for index in range(8)] + [
        ("heldout", index) for index in range(8)
    ]


def expected_release_frame_counts(available: int, blocked: int) -> dict[str, int]:
    available_contract = expected_video_contract(AVAILABLE_STATUS)
    blocked_contract = expected_video_contract(BLOCKED_STATUS)
    return {
        "camera_frames": available * len(expected_role_order()),
        "decoded_candidate_video_frames": (
            available * sum(spec[0] for spec in available_contract.values())
            + blocked * sum(spec[0] for spec in blocked_contract.values())
        ),
        "overview_video_frames": (
            available * available_contract["comparison"][0]
            + blocked * blocked_contract["status"][0]
        ),
    }


def _check_file_hash(
    path: Path,
    expected_hash: str | None,
    errors: list[str],
    label: str,
    expected_size: int | None = None,
) -> bool:
    if not path.is_file():
        errors.append(f"{label}: missing file {path}")
        return False
    if expected_size is not None and path.stat().st_size != expected_size:
        errors.append(
            f"{label}: size mismatch {path.stat().st_size} != {expected_size}"
        )
        return False
    if expected_hash is not None and sha256_file(path) != expected_hash:
        errors.append(f"{label}: SHA256 mismatch")
        return False
    return True


def _validate_video(
    video: dict[str, Any],
    expected: tuple[int, int, int],
    errors: list[str],
    label: str,
) -> dict[str, Any] | None:
    path = Path(video.get("path", ""))
    if not _check_file_hash(
        path,
        video.get("sha256"),
        errors,
        label,
        expected_size=video.get("size_bytes"),
    ):
        return None
    actual = probe_video(path)
    expected_frames, expected_width, expected_height = expected
    checks = {
        "decoded_frame_count": expected_frames,
        "reported_frame_count": expected_frames,
        "width": expected_width,
        "height": expected_height,
        "fps": 2.0,
    }
    for key, expected_value in checks.items():
        actual_value = actual[key]
        if key == "fps":
            if abs(actual_value - expected_value) > 1e-6:
                errors.append(f"{label}: {key} {actual_value} != {expected_value}")
        elif actual_value != expected_value:
            errors.append(f"{label}: {key} {actual_value} != {expected_value}")
        if video.get(key) != actual_value:
            errors.append(f"{label}: probed {key} differs from manifest")
    if actual["minimum_decoded_rgb_std"] <= 0.5:
        errors.append(f"{label}: decoded video contains a blank frame")
    return actual


def _validate_geometry_sources(
    render: dict[str, Any], errors: list[str], label: str
) -> None:
    for source_index, source in enumerate(render.get("sources", [])):
        path = ROOT / source["path"]
        _check_file_hash(
            path,
            source.get("sha256"),
            errors,
            f"{label} source {source_index}",
            expected_size=source.get("size_bytes"),
        )


def _validate_available_candidate(
    document: dict[str, Any], registry_row: dict[str, Any], errors: list[str]
) -> dict[str, Any]:
    unit_id = document["unit_id"]
    if registry_row["status"] != "prepared":
        errors.append(f"{unit_id}: available visualization is not registry-prepared")
    inputs = document.get("inputs", {})
    source_manifest = ROOT / inputs.get("manifest", "")
    _check_file_hash(
        source_manifest,
        inputs.get("manifest_sha256"),
        errors,
        f"{unit_id} source manifest",
    )
    frames = document.get("frames", [])
    if len(frames) != 16:
        errors.append(f"{unit_id}: expected 16 camera frames, found {len(frames)}")
    roles = [(frame.get("role"), frame.get("order")) for frame in frames]
    if roles != expected_role_order():
        errors.append(f"{unit_id}: camera frame role/order is not frozen 8+8")
    indices = [frame.get("index") for frame in frames]
    if indices != list(range(16)):
        errors.append(f"{unit_id}: frame indices are not contiguous 0-15")

    reference_coverages = []
    prediction_coverages = []
    for frame in frames:
        index = frame.get("index")
        file_specs = (
            ("source_rgb", "source_rgb_sha256"),
            ("frame", "frame_sha256"),
            ("reference_frame", "reference_frame_sha256"),
            ("prediction_frame", "prediction_frame_sha256"),
            ("comparison_frame", "comparison_frame_sha256"),
        )
        for path_key, hash_key in file_specs:
            _check_file_hash(
                ROOT / frame.get(path_key, ""),
                frame.get(hash_key),
                errors,
                f"{unit_id} frame {index} {path_key}",
            )
        if frame.get("rgb_std", 0.0) <= 5.0:
            errors.append(f"{unit_id}: blank source frame {index}")
        coverage = frame.get("reference_coverage")
        if not isinstance(coverage, (int, float)) or coverage <= 0.0001:
            errors.append(f"{unit_id}: empty reference render {index}")
        else:
            reference_coverages.append(float(coverage))
        if frame.get("reference_masked_rgb_std", 0.0) <= 0.5:
            errors.append(f"{unit_id}: flat reference render {index}")
        if document.get("prediction_status") == AVAILABLE_STATUS:
            prediction_coverage = frame.get("prediction_coverage")
            if (
                not isinstance(prediction_coverage, (int, float))
                or prediction_coverage <= 0.0001
            ):
                errors.append(f"{unit_id}: empty prediction render {index}")
            else:
                prediction_coverages.append(float(prediction_coverage))
            if frame.get("prediction_masked_rgb_std", 0.0) <= 0.5:
                errors.append(f"{unit_id}: flat prediction render {index}")
        elif frame.get("prediction_coverage") is not None:
            errors.append(f"{unit_id}: missing prediction has numeric coverage")

    reference_render = document.get("reference_render", {})
    if reference_render.get("provenance", {}).get("role") != "evaluation-reference":
        errors.append(f"{unit_id}: reference render provenance is not evaluation-only")
    if len(reference_render.get("frames", [])) != 16:
        errors.append(f"{unit_id}: reference render manifest does not contain 16 frames")
    _validate_geometry_sources(reference_render, errors, f"{unit_id} reference")

    prediction_render = document.get("prediction_render", {})
    if document.get("prediction_status") == AVAILABLE_STATUS:
        if prediction_render.get("provenance", {}).get("role") != "genrecon-prediction":
            errors.append(f"{unit_id}: prediction render provenance is invalid")
        if prediction_render.get("provenance", {}).get("reference_geometry_used") is not False:
            errors.append(f"{unit_id}: prediction render does not exclude GT geometry")
        if len(prediction_render.get("sources", [])) != 1:
            errors.append(f"{unit_id}: prediction render must have exactly one mesh source")
        _validate_geometry_sources(prediction_render, errors, f"{unit_id} prediction")
    elif document.get("prediction_status") == MISSING_PREDICTION_STATUS:
        if prediction_render.get("availability") != MISSING_PREDICTION_STATUS:
            errors.append(f"{unit_id}: missing prediction status card is not explicit")
        if prediction_render.get("sources"):
            errors.append(f"{unit_id}: missing prediction unexpectedly has geometry sources")
    else:
        errors.append(f"{unit_id}: unsupported prediction status")

    contract = expected_video_contract(AVAILABLE_STATUS)
    if set(document.get("videos", {})) != set(contract):
        errors.append(f"{unit_id}: candidate video set is incomplete")
    decoded = 0
    video_bytes = 0
    for kind, expected in contract.items():
        video = document.get("videos", {}).get(kind)
        if video is None:
            continue
        actual = _validate_video(video, expected, errors, f"{unit_id} {kind} video")
        if actual:
            decoded += actual["decoded_frame_count"]
            video_bytes += actual["size_bytes"]
    return {
        "camera_frames": len(frames),
        "decoded_candidate_video_frames": decoded,
        "video_bytes": video_bytes,
        "reference_coverages": reference_coverages,
        "prediction_coverages": prediction_coverages,
    }


def _validate_blocked_candidate(
    document: dict[str, Any], registry_row: dict[str, Any], errors: list[str]
) -> dict[str, Any]:
    unit_id = document["unit_id"]
    if registry_row["status"] != BLOCKED_STATUS:
        errors.append(f"{unit_id}: blocker visualization does not match registry status")
    forbidden = {"frames", "reference_render", "prediction_render", "inputs"} & set(document)
    if forbidden:
        errors.append(f"{unit_id}: blocker fabricates unavailable fields {sorted(forbidden)}")
    if document.get("prediction_status") != BLOCKED_STATUS:
        errors.append(f"{unit_id}: blocker prediction status is inconsistent")
    contract = expected_video_contract(BLOCKED_STATUS)
    if set(document.get("videos", {})) != set(contract):
        errors.append(f"{unit_id}: blocker must expose only one status video")
    video = document.get("videos", {}).get("status")
    actual = (
        _validate_video(video, contract["status"], errors, f"{unit_id} status video")
        if video
        else None
    )
    return {
        "camera_frames": 0,
        "decoded_candidate_video_frames": (
            actual["decoded_frame_count"] if actual else 0
        ),
        "video_bytes": actual["size_bytes"] if actual else 0,
        "reference_coverages": [],
        "prediction_coverages": [],
    }


def write_dataset_contact(
    candidates: list[dict[str, Any]], destination: Path
) -> dict[str, Any]:
    canvas = Image.new("RGB", (1920, 1440), (19, 23, 25))
    for index, document in enumerate(candidates[:7]):
        poster_path = ROOT / document["poster"]
        with Image.open(poster_path) as opened:
            poster = opened.convert("RGB").resize((960, 360), Image.Resampling.LANCZOS)
        canvas.paste(poster, ((index % 2) * 960, (index // 2) * 360))
    draw = ImageDraw.Draw(canvas)
    x, y = 960, 1080
    draw.rectangle((x, y, 1920, 1440), fill=(25, 30, 33))
    draw.rectangle((x, y, x + 12, 1440), fill=(31, 147, 156))
    draw.text((x + 46, y + 50), "RELEASE CONTACT", font=_font(27, bold=True), fill=(246, 248, 248))
    available = sum(document.get("status") == AVAILABLE_STATUS for document in candidates)
    blocked = sum(document.get("status") == BLOCKED_STATUS for document in candidates)
    prediction_available = sum(
        document.get("prediction_status") == AVAILABLE_STATUS for document in candidates
    )
    prediction_missing = sum(
        document.get("prediction_status") == MISSING_PREDICTION_STATUS
        for document in candidates
    )
    lines = [
        f"{len(candidates)} dataset representatives",
        f"{available} prepared source + reference representatives",
        f"{prediction_available} representatives with GenRecon prediction",
        f"{prediction_missing} missing prediction; {blocked} source blockers",
        "Frozen 8 conditioning + 8 heldout cameras",
    ]
    for line_index, line in enumerate(lines):
        draw.text(
            (x + 46, y + 112 + line_index * 38),
            line,
            font=_font(19),
            fill=(184, 195, 200),
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=92, subsampling=0)
    return {
        "path": str(destination.resolve()),
        "size_bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "width": canvas.width,
        "height": canvas.height,
    }


def validate_release(output: Path) -> dict[str, Any]:
    output = output.resolve()
    index_path = output / "index.json"
    index = load_json(index_path)
    errors: list[str] = []
    if index.get("schema") != "genrecon.gt-calibration-visualization-index":
        errors.append("unexpected visualization index schema")

    plan_path = ROOT / index.get("plan", "")
    registry_path = ROOT / index.get("registry", "")
    exporter_path = ROOT / index.get("tool", "")
    _check_file_hash(plan_path, index.get("plan_sha256"), errors, "plan")
    _check_file_hash(registry_path, index.get("registry_sha256"), errors, "registry")
    _check_file_hash(exporter_path, index.get("tool_sha256"), errors, "exporter")
    plan = load_json(plan_path)
    registry = load_json(registry_path)
    plan_entries = plan.get("datasets", [])
    representatives = index.get("representatives", [])
    if len(plan_entries) != 7 or len(representatives) != 7:
        errors.append("plan/index must contain exactly seven dataset representatives")
    expected_units = [entry["unit_id"] for entry in plan_entries]
    actual_units = [entry["unit_id"] for entry in representatives]
    if actual_units != expected_units or len(set(actual_units)) != len(actual_units):
        errors.append("index representative order/membership differs from frozen plan")
    if len({entry["dataset"] for entry in representatives}) != 7:
        errors.append("index does not contain seven unique datasets")
    registry_units = {row["unit_id"]: row for row in registry["units"]}
    plan_by_unit = {entry["unit_id"]: entry for entry in plan_entries}

    expected_build_contract = {
        "plan_sha256": index.get("plan_sha256"),
        "registry_sha256": index.get("registry_sha256"),
        "tool_sha256": index.get("tool_sha256"),
    }
    candidate_documents = []
    totals = {
        "camera_frames": 0,
        "decoded_candidate_video_frames": 0,
        "video_bytes": 0,
    }
    reference_coverages: list[float] = []
    prediction_coverages: list[float] = []
    available = 0
    blocked = 0
    prediction_available = 0
    prediction_missing = 0
    for representative in representatives:
        unit_id = representative["unit_id"]
        manifest_path = ROOT / representative.get("manifest", "")
        if not manifest_path.is_file():
            errors.append(f"{unit_id}: missing candidate manifest")
            continue
        document = load_json(manifest_path)
        candidate_documents.append(document)
        for key in (
            "dataset",
            "dataset_label",
            "unit_id",
            "scene_label",
            "status",
            "prediction_status",
            "review_status",
        ):
            if document.get(key) != representative.get(key):
                errors.append(f"{unit_id}: index/candidate mismatch for {key}")
        expected_review = plan_by_unit.get(unit_id, {}).get(
            "review_status", "not-reviewed"
        )
        if document.get("review_status") != expected_review:
            errors.append(f"{unit_id}: candidate review status differs from frozen plan")
        if document.get("build_contract") != expected_build_contract:
            errors.append(f"{unit_id}: candidate build contract is stale")
        registry_row = registry_units.get(unit_id)
        if registry_row is None:
            errors.append(f"{unit_id}: representative is absent from registry")
            continue
        if document["dataset"] != registry_row["dataset"]:
            errors.append(f"{unit_id}: candidate/registry dataset mismatch")
        poster_path = ROOT / document.get("poster", "")
        _check_file_hash(
            poster_path,
            document.get("poster_sha256"),
            errors,
            f"{unit_id} poster",
        )
        if document["status"] == AVAILABLE_STATUS:
            available += 1
            if document["prediction_status"] == AVAILABLE_STATUS:
                prediction_available += 1
            elif document["prediction_status"] == MISSING_PREDICTION_STATUS:
                prediction_missing += 1
            detail = _validate_available_candidate(document, registry_row, errors)
        elif document["status"] == BLOCKED_STATUS:
            blocked += 1
            detail = _validate_blocked_candidate(document, registry_row, errors)
        else:
            errors.append(f"{unit_id}: unsupported candidate status")
            continue
        for key in totals:
            totals[key] += detail[key]
        reference_coverages.extend(detail["reference_coverages"])
        prediction_coverages.extend(detail["prediction_coverages"])

    expected_frame_counts = expected_release_frame_counts(available, blocked)
    expected_overview_frames = expected_frame_counts["overview_video_frames"]
    overview = index.get("overview_video", {})
    actual_overview = _validate_video(
        overview,
        (expected_overview_frames, COMPARISON_SIZE[0], COMPARISON_SIZE[1]),
        errors,
        "all-datasets overview video",
    )
    contact = write_dataset_contact(
        candidate_documents, output / "dataset_contact.jpg"
    )
    counts = index.get("counts", {})
    expected_counts = {
        "datasets": len(representatives),
        "available_datasets": available,
        "blocked_datasets": blocked,
        "datasets_with_prediction": prediction_available,
        "datasets_missing_prediction": prediction_missing,
        "candidate_videos": available * 4 + blocked,
        "total_videos": available * 4 + blocked + 1,
    }
    for key, expected_value in expected_counts.items():
        if counts.get(key) != expected_value:
            errors.append(f"index count {key} {counts.get(key)} != {expected_value}")
    if totals["camera_frames"] != expected_frame_counts["camera_frames"]:
        errors.append(
            f"expected {expected_frame_counts['camera_frames']} frozen camera frames, "
            f"found {totals['camera_frames']}"
        )
    if (
        totals["decoded_candidate_video_frames"]
        != expected_frame_counts["decoded_candidate_video_frames"]
    ):
        errors.append(
            "expected "
            f"{expected_frame_counts['decoded_candidate_video_frames']} decoded candidate "
            f"video frames, found {totals['decoded_candidate_video_frames']}"
        )
    if not reference_coverages or min(reference_coverages) <= 0.0001:
        errors.append("reference coverage audit is empty")
    if not prediction_coverages or min(prediction_coverages) <= 0.0001:
        errors.append("prediction coverage audit is empty")

    result = {
        "schema": "genrecon.gt-calibration-visualization-release-validation",
        "schema_version": 1,
        "result": "pass" if not errors else "fail",
        "inputs": {
            "index": str(index_path),
            "index_sha256": sha256_file(index_path),
            "plan_sha256": index.get("plan_sha256"),
            "registry_sha256": index.get("registry_sha256"),
            "exporter_sha256": index.get("tool_sha256"),
            "validator_sha256": sha256_file(Path(__file__)),
        },
        "counts": {
            **expected_counts,
            "frozen_camera_frames": totals["camera_frames"],
            "decoded_candidate_video_frames": totals[
                "decoded_candidate_video_frames"
            ],
            "overview_video_frames": (
                actual_overview["decoded_frame_count"] if actual_overview else 0
            ),
            "decoded_video_frames_total": totals[
                "decoded_candidate_video_frames"
            ]
            + (actual_overview["decoded_frame_count"] if actual_overview else 0),
            "video_bytes": totals["video_bytes"]
            + (actual_overview["size_bytes"] if actual_overview else 0),
        },
        "coverage": {
            "reference_minimum": min(reference_coverages) if reference_coverages else None,
            "reference_median": (
                float(np.median(reference_coverages)) if reference_coverages else None
            ),
            "reference_maximum": max(reference_coverages) if reference_coverages else None,
            "prediction_minimum": min(prediction_coverages) if prediction_coverages else None,
            "prediction_median": (
                float(np.median(prediction_coverages)) if prediction_coverages else None
            ),
            "prediction_maximum": max(prediction_coverages) if prediction_coverages else None,
        },
        "contact_sheet": contact,
        "errors": errors,
    }
    write_json(output / "release_validation.json", result)
    print(f"[gt-visualization-release] {result['result']} {result['counts']}")
    if errors:
        raise VisualizationError(
            "Release validation failed: " + "; ".join(errors[:12])
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    validate_release(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
