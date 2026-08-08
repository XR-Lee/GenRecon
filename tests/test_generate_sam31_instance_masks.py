import unittest

import numpy as np

from tools.generate_sam31_instance_masks import binary_mask_metrics, collect_output_masks


class Sam31InstanceMaskTests(unittest.TestCase):
    def test_collect_output_masks_keys_nonempty_masks_by_object_id(self):
        masks = np.zeros((2, 3, 4), dtype=bool)
        masks[0, 1, 2] = True
        outputs = {
            "out_obj_ids": np.asarray([7, 9]),
            "out_binary_masks": masks,
        }

        result = collect_output_masks(outputs, (3, 4))

        self.assertEqual(list(result), [7])
        self.assertTrue(result[7][1, 2])

    def test_collect_output_masks_accepts_empty_predictor_output(self):
        result = collect_output_masks(
            {
                "out_obj_ids": np.empty(0, dtype=np.int64),
                "out_binary_masks": np.empty((0, 5, 6), dtype=bool),
            },
            (5, 6),
        )

        self.assertEqual(result, {})

    def test_binary_metrics_report_iou_and_directional_coverage(self):
        candidate = np.asarray([[1, 1, 0], [0, 0, 0]], dtype=bool)
        reference = np.asarray([[0, 1, 1], [0, 0, 0]], dtype=bool)

        metrics = binary_mask_metrics(candidate, reference)

        self.assertEqual(metrics["intersection"], 1)
        self.assertEqual(metrics["union"], 3)
        self.assertAlmostEqual(metrics["iou"], 1 / 3)
        self.assertAlmostEqual(metrics["candidate_recall_of_reference"], 0.5)
        self.assertAlmostEqual(metrics["reference_recall_of_candidate"], 0.5)


if __name__ == "__main__":
    unittest.main()
