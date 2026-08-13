from __future__ import annotations

import pytest

from tools.validate_gt_calibration_videos import (
    AVAILABLE_STATUS,
    BLOCKED_STATUS,
    COMPARISON_SIZE,
    PANEL_SIZE,
    expected_release_frame_counts,
    expected_role_order,
    expected_video_contract,
)


def test_available_video_contract_requires_four_exact_sixteen_frame_streams() -> None:
    contract = expected_video_contract(AVAILABLE_STATUS)

    assert contract == {
        "source": (16, PANEL_SIZE[0], PANEL_SIZE[1]),
        "reference": (16, PANEL_SIZE[0], PANEL_SIZE[1]),
        "prediction": (16, PANEL_SIZE[0], PANEL_SIZE[1]),
        "comparison": (16, COMPARISON_SIZE[0], COMPARISON_SIZE[1]),
    }


def test_blocked_video_contract_does_not_claim_source_or_geometry() -> None:
    assert expected_video_contract(BLOCKED_STATUS) == {
        "status": (8, COMPARISON_SIZE[0], COMPARISON_SIZE[1])
    }
    with pytest.raises(ValueError, match="Unsupported visualization status"):
        expected_video_contract("prepared")


def test_release_frame_totals_follow_candidate_status_contracts() -> None:
    assert expected_release_frame_counts(available=7, blocked=0) == {
        "camera_frames": 112,
        "decoded_candidate_video_frames": 448,
        "overview_video_frames": 112,
    }
    assert expected_release_frame_counts(available=6, blocked=1) == {
        "camera_frames": 96,
        "decoded_candidate_video_frames": 392,
        "overview_video_frames": 104,
    }


def test_frozen_role_order_is_conditioning_then_heldout() -> None:
    assert expected_role_order() == [
        *(('conditioning', index) for index in range(8)),
        *(('heldout', index) for index in range(8)),
    ]
