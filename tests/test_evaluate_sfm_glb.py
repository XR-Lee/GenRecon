import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.evaluate_sfm_glb import (
    _write_json,
    distance_summary,
    match_clean_points_to_colmap_ids,
    points_inside_bounds,
)


class SfmGlbEvaluationTests(unittest.TestCase):
    def test_json_writer_replaces_non_finite_numbers_with_null(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.json"
            _write_json(
                output,
                {
                    "finite": np.float64(3.5),
                    "missing": float("nan"),
                    "array": np.asarray([np.inf, -np.inf, 4.0]),
                },
            )
            raw = output.read_text()
            self.assertNotIn("NaN", raw)
            self.assertNotIn("Infinity", raw)
            self.assertEqual(
                json.loads(raw),
                {"array": [None, None, 4.0], "finite": 3.5, "missing": None},
            )

    def test_distance_summary_reports_quantiles_and_threshold_recall(self) -> None:
        metrics = distance_summary(np.array([0.01, 0.03, 0.08, 0.30]))
        self.assertEqual(metrics["count"], 4)
        self.assertAlmostEqual(metrics["median_m"], 0.055)
        self.assertAlmostEqual(metrics["threshold_recall"]["0.020"], 0.25)
        self.assertAlmostEqual(metrics["threshold_recall"]["0.050"], 0.50)
        self.assertAlmostEqual(metrics["threshold_recall"]["0.100"], 0.75)

    def test_points_inside_bounds_applies_margin_on_all_axes(self) -> None:
        points = np.array([[0.5, 0.5, 0.5], [-0.05, 0.5, 0.5], [1.2, 0.5, 0.5]])
        bounds = np.array([[0, 0, 0], [1, 1, 1]])
        np.testing.assert_array_equal(points_inside_bounds(points, bounds, 0.1), [True, True, False])

    def test_clean_points_match_quality_point_ids_with_tolerance(self) -> None:
        clean = np.array([[1.0, 2.0, 3.0], [9.0, 9.0, 9.0]])
        quality = np.array([[1.0, 2.0, 3.00001], [4.0, 5.0, 6.0]])
        quality_ids = np.array([17, 23])
        matched, distances = match_clean_points_to_colmap_ids(clean, quality, quality_ids)
        np.testing.assert_array_equal(matched, [17, -1])
        self.assertLess(distances[0], 1e-4)
        self.assertGreater(distances[1], 1e-4)


if __name__ == "__main__":
    unittest.main()
