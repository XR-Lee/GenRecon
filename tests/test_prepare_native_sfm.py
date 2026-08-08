from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from tools.prepare_native_sfm import (
    camera_calibration,
    dynamic_fraction_lookup,
    read_images,
    read_points,
    transform_observations,
    undistort_rgba,
)


def test_native_colmap_parsers_keep_tracks_and_observations(tmp_path: Path) -> None:
    images = tmp_path / "images.txt"
    images.write_text(
        "# images\n"
        "7 1 0 0 0 1 2 3 4 frame 001.jpg\n"
        "10.5 20.5 9 30.5 40.5 -1\n"
    )
    points = tmp_path / "points3D.txt"
    points.write_text("9 1 2 3 10 20 30 0.75 7 0 8 2\n")

    image_records = read_images(images)
    point_records = read_points(points)

    assert image_records[0]["name"] == "frame 001.jpg"
    assert image_records[0]["camera_id"] == 4
    assert image_records[0]["points2d"] == [[10.5, 20.5, 9], [30.5, 40.5, -1]]
    assert point_records[0]["point_id"] == 9
    assert point_records[0]["track"] == [7, 0, 8, 2]
    assert point_records[0]["error"] == 0.75


def test_zero_distortion_observations_are_unchanged() -> None:
    calibration = camera_calibration(
        {
            "camera_id": 1,
            "model": "PINHOLE",
            "width": 640,
            "height": 480,
            "params": [500.0, 510.0, 320.0, 240.0],
        }
    )
    points = [[100.25, 200.5, 7], [300.0, 220.0, -1]]

    transformed = transform_observations(points, calibration)

    assert transformed == points
    assert calibration["roi"] == (0, 0, 640, 480)
    assert np.allclose(calibration["intrinsic"], [[500, 0, 320], [0, 510, 240], [0, 0, 1]])


def test_dynamic_fraction_lookup_accepts_pipeline_image_field() -> None:
    assert dynamic_fraction_lookup(
        {"frames": [{"image": "frame_000001.jpg", "dynamic_fraction": 0.25}]}
    ) == {"frame_000001": 0.25}


def test_undistorted_rgba_preserves_rgb_and_inverts_dynamic_mask(tmp_path: Path) -> None:
    rgb_path = tmp_path / "rgb.png"
    mask_path = tmp_path / "mask.png"
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[..., 0] = 120
    rgb[..., 1] = 80
    dynamic = np.zeros((4, 6), dtype=np.uint8)
    dynamic[1:3, 2:4] = 255
    Image.fromarray(rgb).save(rgb_path)
    Image.fromarray(dynamic).save(mask_path)
    calibration = camera_calibration(
        {
            "camera_id": 1,
            "model": "PINHOLE",
            "width": 6,
            "height": 4,
            "params": [5.0, 5.0, 3.0, 2.0],
        }
    )

    rgba, output_dynamic = undistort_rgba(rgb_path, mask_path, calibration)

    assert rgba.shape == (4, 6, 4)
    assert np.array_equal(rgba[..., :3], rgb)
    assert np.array_equal(output_dynamic, dynamic)
    assert np.array_equal(rgba[..., 3], 255 - dynamic)


def test_radial_conversion_produces_finite_pinhole_camera() -> None:
    calibration = camera_calibration(
        {
            "camera_id": 1,
            "model": "SIMPLE_RADIAL",
            "width": 1600,
            "height": 900,
            "params": [1100.0, 800.0, 450.0, -0.04],
        }
    )

    assert calibration["source_model"] == "SIMPLE_RADIAL"
    assert calibration["width"] > 0
    assert calibration["height"] > 0
    assert np.isfinite(calibration["intrinsic"]).all()
    assert calibration["map_x"].shape == (900, 1600)
    assert calibration["map_y"].shape == (900, 1600)
