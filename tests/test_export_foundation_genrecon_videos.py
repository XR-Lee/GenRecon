from __future__ import annotations

from pathlib import Path

from PIL import Image

from tools.export_foundation_genrecon_videos import (
    compose_comparison_frame,
    conditioning_roles,
    encode_video,
    ffmpeg_executable,
    probe_video,
    selected_source_records,
    write_json,
)


def test_selected_source_records_preserve_metadata_while_sorting_timestamps() -> None:
    foundation = {
        "selection": {
            "selected": ["frame_000003.jpg", "frame_000001.jpg"],
            "selected_dynamic_fraction": [0.3, 0.1],
        }
    }
    frames = {
        "frames": [
            {"name": "frame_000001.jpg", "timestamp_s": 1.5, "index": 0},
            {"name": "frame_000003.jpg", "timestamp_s": 2.5, "index": 2},
        ]
    }

    records = selected_source_records(foundation, frames)

    assert [item["stem"] for item in records] == ["frame_000001", "frame_000003"]
    assert [item["dynamic_fraction"] for item in records] == [0.1, 0.3]
    assert [item["selection_index"] for item in records] == [1, 0]


def test_conditioning_roles_prefer_scene_over_chunk(tmp_path: Path) -> None:
    cameras = tmp_path / "cameras.json"
    write_json(
        cameras,
        {
            "scene": [{"img_path": "/rgb/frame_000001.png"}],
            "chunks": [
                {
                    "cond2d_view": {
                        "img_path": "/rgb/frame_000001.png",
                    }
                },
                {
                    "cond2d_view": {
                        "img_path": "/rgb/frame_000002.png",
                    }
                },
            ],
        },
    )

    assert conditioning_roles(cameras) == {
        "frame_000001": "GenRecon scene conditioning",
        "frame_000002": "GenRecon chunk conditioning",
    }


def test_compose_comparison_frame_has_stable_panel_geometry() -> None:
    original = Image.new("RGB", (320, 180), (220, 30, 30))
    reconstruction = Image.new("RGB", (320, 180), (30, 80, 220))

    result = compose_comparison_frame(
        original,
        reconstruction,
        candidate_id="raw-001-test",
        frame_name="frame_000001.jpg",
        source_timestamp_s=12.5,
        role="Foundation geometry only",
        coverage=0.75,
        frame_index=0,
        frame_count=2,
    )

    assert result.size == (640, 276)
    assert result.getpixel((10, 60)) == (220, 30, 30)
    assert result.getpixel((330, 60)) == (30, 80, 220)


def test_encode_video_round_trip_preserves_frame_count(tmp_path: Path) -> None:
    frame_root = tmp_path / "frames"
    frame_root.mkdir()
    for index, color in enumerate(((200, 20, 20), (20, 200, 20), (20, 20, 200))):
        Image.new("RGB", (320, 180), color).save(frame_root / f"{index:06d}.jpg", quality=95)
    destination = tmp_path / "video.mp4"

    encode_video(
        ffmpeg=ffmpeg_executable(),
        source_pattern=frame_root / "%06d.jpg",
        destination=destination,
        fps=2.0,
        report_log=tmp_path / "ffmpeg.log",
    )
    probe = probe_video(destination)

    assert probe["decoded_frame_count"] == 3
    assert probe["reported_frame_count"] == 3
    assert probe["width"] == 320
    assert probe["height"] == 180
    assert probe["fps"] == 2.0
    assert probe["size_bytes"] > 0
    assert len(probe["sha256"]) == 64
