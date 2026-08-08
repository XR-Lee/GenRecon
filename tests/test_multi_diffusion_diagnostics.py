import unittest

import torch

from genrecon.modules.sparse import SparseTensor
from genrecon.pipelines.samplers.multi_diff_orchestrator import MultiDiffusionOrchestrator


class MultiDiffusionDiagnosticsTests(unittest.TestCase):
    def test_dense_overlap_pairs_and_disagreement_are_measured(self):
        chunks = [torch.ones(1, 2, 4, 4, 4), torch.full((1, 2, 4, 4, 4), 2.0)]
        pairs = MultiDiffusionOrchestrator._prepare_overlap_diagnostic_pairs(
            chunks,
            [torch.zeros(3), torch.tensor([0.5, 0.0, 0.0])],
            resolution=4,
            chunk_ids=[3, 4],
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["overlap_voxels"], 32)
        self.assertEqual((pairs[0]["chunk_a"], pairs[0]["chunk_b"]), (3, 4))
        metrics = MultiDiffusionOrchestrator._measure_overlap_pair(chunks, pairs[0], step_size=0.25)
        self.assertAlmostEqual(metrics["state_rmse"], 1.0)
        self.assertAlmostEqual(metrics["state_mae"], 1.0)
        self.assertAlmostEqual(metrics["velocity_disagreement_rmse"], 4.0)
        self.assertAlmostEqual(metrics["cosine_similarity"], 1.0, places=6)

    def test_dense_roi_metrics_select_world_aligned_overlap_voxels(self):
        chunks = [torch.ones(1, 1, 4, 4, 4), torch.full((1, 1, 4, 4, 4), 3.0)]
        pairs = MultiDiffusionOrchestrator._prepare_overlap_diagnostic_pairs(
            chunks,
            [torch.zeros(3), torch.tensor([0.5, 0.0, 0.0])],
            resolution=4,
            chunk_ids=[3, 4],
            roi_bounds=[((0.1, -0.5, -0.5), (0.15, 0.5, 0.5))],
        )

        self.assertEqual(pairs[0]["roi_overlap_voxels"], 16)
        metrics = MultiDiffusionOrchestrator._measure_overlap_pair(chunks, pairs[0], step_size=0.5)
        self.assertAlmostEqual(metrics["roi_state_rmse"], 2.0)
        self.assertAlmostEqual(metrics["roi_velocity_disagreement_rmse"], 4.0)

    def test_sparse_overlap_matches_global_voxel_coordinates(self):
        coords_a = torch.tensor([[0, 2, 1, 1], [0, 3, 1, 1], [0, 0, 0, 0]], dtype=torch.int32)
        coords_b = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1], [0, 3, 3, 3]], dtype=torch.int32)
        chunks = [
            SparseTensor(feats=torch.ones(3, 2), coords=coords_a),
            SparseTensor(feats=torch.ones(3, 2), coords=coords_b),
        ]
        pairs = MultiDiffusionOrchestrator._prepare_overlap_diagnostic_pairs(
            chunks,
            [torch.zeros(3), torch.tensor([0.5, 0.0, 0.0])],
            resolution=4,
            chunk_ids=[8, 9],
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["overlap_voxels"], 2)
        self.assertEqual(pairs[0]["rows_a"].tolist(), [0, 1])
        self.assertEqual(pairs[0]["rows_b"].tolist(), [0, 1])

    def test_sparse_roi_mask_uses_common_chunk0_lattice(self):
        coords_a = torch.tensor([[0, 2, 1, 1], [0, 3, 1, 1]], dtype=torch.int32)
        coords_b = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]], dtype=torch.int32)
        chunks = [
            SparseTensor(feats=torch.ones(2, 1), coords=coords_a),
            SparseTensor(feats=torch.full((2, 1), 2.0), coords=coords_b),
        ]
        pairs = MultiDiffusionOrchestrator._prepare_overlap_diagnostic_pairs(
            chunks,
            [torch.zeros(3), torch.tensor([0.5, 0.0, 0.0])],
            resolution=4,
            chunk_ids=[3, 4],
            roi_bounds=[((0.1, -0.2, -0.2), (0.15, 0.2, 0.2))],
        )

        self.assertEqual(pairs[0]["roi_overlap_voxels"], 1)
        metrics = MultiDiffusionOrchestrator._measure_overlap_pair(chunks, pairs[0], step_size=0.25)
        self.assertAlmostEqual(metrics["roi_state_rmse"], 1.0)


if __name__ == "__main__":
    unittest.main()
