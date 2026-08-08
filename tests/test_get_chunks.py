import unittest

from inference.get_chunks import IphoneChunker


class IphoneChunkerTests(unittest.TestCase):
    def test_colmap_quality_and_spatial_cleaning_are_configurable(self) -> None:
        default = IphoneChunker()
        self.assertFalse(default.skip_point_cleaning)
        self.assertIsNone(default.max_reproj_error)
        self.assertIsNone(default.min_track_len)

        configured = IphoneChunker(
            max_reproj_error=2.0,
            min_track_len=3,
            manual_z_bounds=(-1.0, 1.8),
            skip_point_cleaning=True,
        )
        self.assertEqual(configured.max_reproj_error, 2.0)
        self.assertEqual(configured.min_track_len, 3)
        self.assertEqual(configured.manual_z_bounds, (-1.0, 1.8))
        self.assertTrue(configured.skip_point_cleaning)


if __name__ == "__main__":
    unittest.main()
