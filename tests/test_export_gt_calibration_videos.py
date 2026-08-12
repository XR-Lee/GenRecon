from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.export_gt_calibration_videos import (
    COMPARISON_SIZE,
    PANEL_SIZE,
    _status_panel,
    _candidate_reusable,
    panel_camera,
    point_allocations,
)


def test_panel_camera_matches_letterboxed_source_geometry() -> None:
    intrinsic = np.asarray(
        [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
    )
    output, box = panel_camera(intrinsic, 1920, 1080)

    assert box == (0, 60, 640, 360)
    np.testing.assert_allclose(
        output,
        [[1000.0 / 3.0, 0.0, 320.0], [0.0, 1000.0 / 3.0, 240.0], [0.0, 0.0, 1.0]],
    )


def test_point_allocations_are_proportional_and_exact() -> None:
    allocation = point_allocations([10, 30, 60], 17)

    assert allocation == [2, 5, 10]
    assert sum(allocation) == 17
    assert point_allocations([2, 3], 10) == [2, 3]


def test_status_panels_have_stable_dimensions() -> None:
    panel = _status_panel("NO PREDICTION", ["Explicit status, not a render."])
    blocker = _status_panel("BLOCKED", ["No authorized source."], size=COMPARISON_SIZE)

    assert panel.size == PANEL_SIZE
    assert blocker.size == COMPARISON_SIZE
    assert np.asarray(panel).std() > 5.0
    assert np.asarray(blocker).std() > 5.0


def test_candidate_reuse_requires_current_contract_and_video_hash(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    registry = tmp_path / "registry.json"
    video = tmp_path / "video.mp4"
    manifest = tmp_path / "manifest.json"
    plan.write_text("{}\n")
    registry.write_text("{}\n")
    video.write_bytes(b"video")
    import hashlib

    tool = Path(__file__).resolve().parents[1] / "tools/export_gt_calibration_videos.py"
    contract = {
        "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
        "registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
        "tool_sha256": hashlib.sha256(tool.read_bytes()).hexdigest(),
    }
    manifest.write_text(
        json.dumps(
            {
                "build_contract": contract,
                "videos": {
                    "comparison": {
                        "path": str(video),
                        "size_bytes": video.stat().st_size,
                        "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                    }
                },
            }
        )
    )

    assert _candidate_reusable(manifest, plan_path=plan, registry_path=registry)
    video.write_bytes(b"changed")
    assert not _candidate_reusable(manifest, plan_path=plan, registry_path=registry)


def test_visualization_plan_covers_each_registry_dataset_once() -> None:
    root = Path(__file__).resolve().parents[1]
    plan = json.loads(
        (root / "configs/eval/gt_calibration_visualization_v1.json").read_text()
    )
    registry = json.loads((root / "data/gt-calibration-v1/registry.json").read_text())
    registry_units = {item["unit_id"]: item for item in registry["units"]}

    entries = plan["datasets"]
    assert len(entries) == 7
    assert len({entry["dataset"] for entry in entries}) == 7
    assert len({entry["unit_id"] for entry in entries}) == 7
    for entry in entries:
        row = registry_units[entry["unit_id"]]
        assert row["dataset"] == entry["dataset"]
        if entry.get("expected_status"):
            assert row["status"] == entry["expected_status"]
        else:
            assert row["status"] == "prepared"
