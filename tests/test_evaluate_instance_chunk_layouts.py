import unittest

import numpy as np

from tools.evaluate_instance_chunk_layouts import (
    aligned_axis_values,
    best_face_margins,
    layout_metrics,
    select_adaptive,
)


class InstanceChunkLayoutTests(unittest.TestCase):
    def test_best_face_margin_uses_most_interior_covering_chunk(self) -> None:
        points = np.asarray([[0.9, 0.0, 0.0], [3.0, 0.0, 0.0]])
        centers = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        margins = best_face_margins(points, centers, chunk_size=2.0)

        np.testing.assert_allclose(margins, [0.9, -1.0])

    def test_layout_metrics_report_interior_thresholds(self) -> None:
        points = np.asarray([[0.0, 0.0, 0.0], [0.85, 0.0, 0.0]])
        centers = np.asarray([[0.0, 0.0, 0.0]])

        metrics = layout_metrics(points, centers, chunk_size=2.0)

        self.assertEqual(metrics.coverage, 1.0)
        self.assertEqual(metrics.margin_10cm, 1.0)
        self.assertEqual(metrics.margin_20cm, 0.5)
        self.assertAlmostEqual(metrics.margin_q10_m, 0.235)

    def test_aligned_axis_values_preserve_reference_lattice(self) -> None:
        values = aligned_axis_values(-0.3, 0.35, origin=0.1, quantum=0.2)

        np.testing.assert_allclose(values, [-0.3, -0.1, 0.1, 0.3])

    def test_select_adaptive_prioritizes_lower_tail_margin(self) -> None:
        points = np.asarray([[-0.8, 0.0, 0.0], [0.0, 0.0, 0.0], [0.8, 0.0, 0.0]])
        candidates = []
        for center_x in (0.0, 0.2):
            centers = np.asarray([[center_x, 0.0, 0.0]])
            candidates.append(
                {
                    "first_x": center_x,
                    "center_y": 0.0,
                    "centers": centers,
                    "metrics": layout_metrics(points, centers, chunk_size=2.0),
                }
            )

        selected = select_adaptive(candidates)

        self.assertEqual(selected["first_x"], 0.0)


if __name__ == "__main__":
    unittest.main()
