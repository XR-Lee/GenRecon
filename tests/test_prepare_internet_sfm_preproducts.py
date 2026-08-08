import unittest
from types import SimpleNamespace

import numpy as np
import torch

from tools.prepare_internet_sfm_preproducts import (
    detect_scene_boundaries,
    dynamic_mask_from_prediction,
    generate_shot_windows,
    grade_sfm_quality,
    parse_scene_metadata,
    percentile_summary,
    ratio_test_matches,
)


class InternetSfmPreproductsTests(unittest.TestCase):
    def test_scene_metadata_parser_pairs_timestamp_and_score(self) -> None:
        text = (
            "frame:0 pts:0 pts_time:0\n"
            "lavfi.scene_score=0.000000\n"
            "frame:1 pts:1 pts_time:0.5\n"
            "lavfi.scene_score=0.420000\n"
        )
        self.assertEqual(
            parse_scene_metadata(text),
            [{"time_s": 0.0, "score": 0.0}, {"time_s": 0.5, "score": 0.42}],
        )

    def test_scene_boundaries_collapse_adjacent_peaks(self) -> None:
        samples = [
            {"time_s": 2.0, "score": 0.40},
            {"time_s": 2.5, "score": 0.62},
            {"time_s": 9.0, "score": 0.50},
            {"time_s": 12.0, "score": 0.20},
        ]
        self.assertEqual(
            detect_scene_boundaries(samples),
            [{"time_s": 2.5, "score": 0.62}, {"time_s": 9.0, "score": 0.5}],
        )

    def test_long_continuous_segment_is_split_into_bounded_windows(self) -> None:
        samples = [
            {"time_s": value / 2, "score": 0.02}
            for value in range(0, 481)
        ]
        windows = generate_shot_windows(
            240.0,
            samples,
            [],
            maximum_duration_s=90.0,
            long_window_stride_s=45.0,
            maximum_candidates=14,
        )
        self.assertGreaterEqual(len(windows), 4)
        self.assertTrue(all(8.0 <= item["duration_s"] <= 90.0 for item in windows))
        self.assertEqual(len({(item["start_s"], item["end_s"]) for item in windows}), len(windows))

    def test_dynamic_mask_keeps_only_selected_semantic_classes(self) -> None:
        categories = ["__background__", "person", "chair"]
        masks = torch.zeros((3, 1, 8, 10), dtype=torch.float32)
        masks[0, 0, 1:4, 2:5] = 1
        masks[1, 0, 5:7, 7:9] = 1
        masks[2, 0, :, :] = 1
        prediction = {
            "scores": torch.tensor([0.9, 0.8, 0.1]),
            "labels": torch.tensor([1, 2, 1]),
            "masks": masks,
            "boxes": torch.tensor([[2, 1, 5, 4], [7, 5, 9, 7], [0, 0, 10, 8]]),
        }
        dynamic, instances = dynamic_mask_from_prediction(
            prediction,
            categories,
            score_threshold=0.35,
            dilation_fraction=0.0,
            shape=(8, 10),
        )
        self.assertEqual(int(dynamic.sum()), 9)
        self.assertEqual([item["category"] for item in instances], ["person"])

    def test_quality_gate_passes_strong_model(self) -> None:
        metrics = {
            "largest_model": {
                "registered_fraction": 0.95,
                "points3D": 12000,
                "quality_points_error_le_2px_track_ge_3": 9000,
                "mean_reprojection_error_px": 0.9,
                "track_length": {"mean": 5.2},
                "viewpoint_geometry": {
                    "baseline_to_observed_depth": 0.8,
                    "triangulation_angle_deg": {"median": 6.0},
                },
                "cameras": [{"has_bogus_params": False}],
            },
            "mask_application": {"keypoint_centers_on_blocked_pixels": 0},
            "dynamic_masks": {"dynamic_fraction": {"mean": 0.1}},
            "largest_model_registered_dominance": 0.98,
        }
        self.assertEqual(grade_sfm_quality(metrics)["grade"], "A")

    def test_quality_gate_fails_when_mask_was_not_respected(self) -> None:
        metrics = {
            "largest_model": {
                "registered_fraction": 0.95,
                "points3D": 12000,
                "quality_points_error_le_2px_track_ge_3": 9000,
                "mean_reprojection_error_px": 0.9,
                "track_length": {"mean": 5.2},
                "viewpoint_geometry": {
                    "baseline_to_observed_depth": 0.8,
                    "triangulation_angle_deg": {"median": 6.0},
                },
                "cameras": [{"has_bogus_params": False}],
            },
            "mask_application": {"keypoint_centers_on_blocked_pixels": 1},
            "dynamic_masks": {"dynamic_fraction": {"mean": 0.1}},
            "largest_model_registered_dominance": 0.98,
        }
        result = grade_sfm_quality(metrics)
        self.assertEqual(result["grade"], "F")
        self.assertIn("keypoints_found_on_blocked_mask_pixels", result["reasons"])

    def test_strong_registration_with_low_baseline_is_only_marginal(self) -> None:
        metrics = {
            "largest_model": {
                "registered_fraction": 1.0,
                "points3D": 12000,
                "quality_points_error_le_2px_track_ge_3": 9000,
                "mean_reprojection_error_px": 0.8,
                "track_length": {"mean": 8.0},
                "viewpoint_geometry": {
                    "baseline_to_observed_depth": 0.2,
                    "triangulation_angle_deg": {"median": 3.2},
                },
                "cameras": [{"has_bogus_params": False}],
            },
            "mask_application": {"keypoint_centers_on_blocked_pixels": 0},
            "dynamic_masks": {"dynamic_fraction": {"mean": 0.3}},
            "largest_model_registered_dominance": 1.0,
        }
        result = grade_sfm_quality(metrics)
        self.assertEqual(result["grade"], "C")
        self.assertIn("low_viewpoint_baseline", result["warnings"])
        self.assertIn("weak_triangulation_angle", result["warnings"])

    def test_ratio_test_skips_queries_with_only_one_neighbor(self) -> None:
        accepted = SimpleNamespace(distance=10.0)
        rejected = SimpleNamespace(distance=30.0)
        matches = ratio_test_matches(
            [
                [SimpleNamespace(distance=5.0)],
                [accepted, SimpleNamespace(distance=20.0)],
                [rejected, SimpleNamespace(distance=35.0)],
            ]
        )
        self.assertEqual(matches, [accepted])

    def test_empty_percentiles_are_json_safe(self) -> None:
        self.assertEqual(
            percentile_summary([float("nan")]),
            {"count": 0, "mean": None, "median": None, "p90": None, "max": None},
        )


if __name__ == "__main__":
    unittest.main()
