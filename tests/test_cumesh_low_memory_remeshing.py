from __future__ import annotations

import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from genrecon.vendor import cumesh_remeshing as low_memory


REPO_ROOT = Path(__file__).resolve().parents[1]


def _vectorized_reference(
    vertices: torch.Tensor,
    quads: torch.Tensor,
    directions: torch.Tensor,
) -> torch.Tensor:
    split_1_n, split_1_p, split_2_n, split_2_p = low_memory._quad_split_tables(vertices.device)
    attempt_0 = torch.where(
        (directions == 1).unsqueeze(1),
        quads[:, split_1_p],
        quads[:, split_1_n],
    )
    normals_0a = torch.cross(
        vertices[attempt_0[:, 1]] - vertices[attempt_0[:, 0]],
        vertices[attempt_0[:, 2]] - vertices[attempt_0[:, 0]],
        dim=-1,
    )
    normals_0b = torch.cross(
        vertices[attempt_0[:, 2]] - vertices[attempt_0[:, 1]],
        vertices[attempt_0[:, 3]] - vertices[attempt_0[:, 1]],
        dim=-1,
    )
    alignment_0 = (normals_0a * normals_0b).sum(dim=1).abs()

    attempt_1 = torch.where(
        (directions == 1).unsqueeze(1),
        quads[:, split_2_p],
        quads[:, split_2_n],
    )
    normals_1a = torch.cross(
        vertices[attempt_1[:, 1]] - vertices[attempt_1[:, 0]],
        vertices[attempt_1[:, 2]] - vertices[attempt_1[:, 0]],
        dim=-1,
    )
    normals_1b = torch.cross(
        vertices[attempt_1[:, 2]] - vertices[attempt_1[:, 1]],
        vertices[attempt_1[:, 3]] - vertices[attempt_1[:, 1]],
        dim=-1,
    )
    alignment_1 = (normals_1a * normals_1b).sum(dim=1).abs()
    return torch.where(
        (alignment_0 > alignment_1).unsqueeze(1),
        attempt_0,
        attempt_1,
    ).reshape(-1, 3).int()


class CuMeshLowMemoryRemeshingTests(unittest.TestCase):
    def test_full_remesh_control_flow_runs_with_cpu_extension_mocks(self) -> None:
        calls: list[str] = []
        hashmap_generations: list[list[weakref.ReferenceType[torch.Tensor]]] = []
        original_init_hashmap = low_memory._init_hashmap

        def tracked_init_hashmap(*args, **kwargs):
            if hashmap_generations:
                self.assertTrue(all(ref() is None for ref in hashmap_generations[-1]))
            hashmap = original_init_hashmap(*args, **kwargs)
            hashmap_generations.append([weakref.ref(tensor) for tensor in hashmap])
            return hashmap

        class FakeBVH:
            @staticmethod
            def unsigned_distance(points: torch.Tensor, return_uvw: bool = False):
                self.assertFalse(return_uvw)
                return (torch.zeros((points.shape[0],), dtype=torch.float32),)

        def insert(*args):
            del args
            calls.append("insert")

        def active_vertices(*args):
            del args
            calls.append("active_vertices")
            return torch.tensor([[0, 0, 0]], dtype=torch.int32)

        def dual_contour(*args):
            del args
            calls.append("dual_contour")
            return (
                torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32),
                torch.tensor([[1, 0, 0]], dtype=torch.int32),
            )

        def lookup(*args):
            keys = args[2]
            calls.append("lookup")
            return torch.zeros((keys.shape[0],), dtype=torch.int32)

        fake_extension = SimpleNamespace(
            hashmap_insert_3d_cuda=insert,
            get_sparse_voxel_grid_active_vertices=active_vertices,
            simple_dual_contour=dual_contour,
            hashmap_lookup_3d_cuda=lookup,
        )

        with (
            mock.patch.object(low_memory, "_C", fake_extension),
            mock.patch.object(low_memory, "_init_hashmap", tracked_init_hashmap),
        ):
            vertices, faces = low_memory.remesh_narrow_band_dc(
                vertices=torch.tensor(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    dtype=torch.float32,
                ),
                faces=torch.tensor([[0, 1, 2]], dtype=torch.int32),
                center=torch.zeros((3,), dtype=torch.float32),
                scale=1.0,
                resolution=1,
                band=0.0,
                project_back=0.0,
                bvh=FakeBVH(),
            )

        self.assertEqual(
            calls,
            ["insert", "active_vertices", "insert", "dual_contour", "insert", "lookup"],
        )
        self.assertEqual(len(hashmap_generations), 3)
        self.assertTrue(all(ref() is None for ref in hashmap_generations[-1]))
        torch.testing.assert_close(vertices, torch.zeros((1, 3)), rtol=0, atol=0)
        torch.testing.assert_close(faces, torch.zeros((2, 3), dtype=torch.int32), rtol=0, atol=0)

    def test_batched_hash_insert_preserves_global_row_values(self) -> None:
        calls: list[tuple[torch.Tensor, torch.Tensor, tuple[int, int, int]]] = []

        def insert(keys, values, coords, indices, width, height, depth):
            del keys, values
            calls.append((coords.clone(), indices.clone(), (width, height, depth)))

        coords = torch.tensor(
            [[4, 1, 7], [2, 8, 1], [5, 9, 3], [0, 2, 6], [8, 8, 8]],
            dtype=torch.int32,
        )
        hashmap = (torch.zeros(1, dtype=torch.uint32), torch.zeros(1, dtype=torch.uint32))
        with mock.patch.object(
            low_memory,
            "_C",
            SimpleNamespace(hashmap_insert_3d_cuda=insert),
        ):
            low_memory._insert_indexed_coords_batched(
                hashmap,
                coords,
                resolution=10,
                batch_size=2,
            )

        self.assertEqual([call[0].shape[0] for call in calls], [2, 2, 1])
        padded = torch.cat([call[0] for call in calls])
        indices = torch.cat([call[1] for call in calls])
        torch.testing.assert_close(padded[:, 0], torch.zeros(5, dtype=torch.int32), rtol=0, atol=0)
        torch.testing.assert_close(padded[:, 1:], coords, rtol=0, atol=0)
        torch.testing.assert_close(
            indices,
            torch.tensor([0, 1, 2, 3, 4], dtype=torch.uint32),
            rtol=0,
            atol=0,
        )
        self.assertTrue(all(call[2] == (10, 10, 10) for call in calls))

    def test_batched_grid_distances_match_full_expression(self) -> None:
        calls: list[int] = []

        class FakeBVH:
            @staticmethod
            def unsigned_distance(points: torch.Tensor):
                calls.append(points.shape[0])
                return (points.square().sum(dim=1).sqrt(),)

        grid = torch.tensor(
            [[0, 0, 0], [1, 2, 3], [4, 1, 0], [2, 2, 2], [3, 4, 1]],
            dtype=torch.int32,
        )
        center = torch.tensor([0.25, -0.5, 1.0])
        points = (grid.float() / 8 - 0.5) * 2.5 + center
        expected = points.square().sum(dim=1).sqrt() - 0.125

        actual = low_memory._grid_distances_batched(
            grid,
            FakeBVH(),
            center,
            scale=2.5,
            resolution=8,
            epsilon=0.125,
            batch_size=2,
        )

        self.assertEqual(calls, [2, 2, 1])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_batched_topology_matches_original_vectorized_expression(self) -> None:
        coords = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            dtype=torch.int32,
        )
        intersected = torch.tensor(
            [
                [1, -1, 1],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [-1, 1, -1],
            ],
            dtype=torch.int32,
        )
        offsets = torch.tensor(
            [
                [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
                [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
                [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
            ],
            dtype=torch.int32,
        ).unsqueeze(0)
        mapping = {tuple(row.tolist()): index for index, row in enumerate(coords)}

        def lookup(keys, values, padded, width, height, depth):
            del keys, values, width, height, depth
            result = [mapping.get(tuple(row[1:].tolist()), 0xFFFFFFFF) for row in padded]
            return torch.tensor(result, dtype=torch.uint32)

        full_neighbors = coords.reshape(-1, 1, 1, 3) + offsets
        full_mask = intersected != 0
        full_connected = full_neighbors[full_mask]
        full_directions = intersected[full_mask]
        padded = torch.cat(
            [
                torch.zeros((full_connected.shape[0] * 4, 1), dtype=torch.int32),
                full_connected.reshape(-1, 3),
            ],
            dim=1,
        )
        full_indices = lookup(None, None, padded, 3, 3, 3).reshape(-1, 4).int()
        full_valid = (full_indices != 0xFFFFFFFF).all(dim=1)
        expected_quads = full_indices[full_valid].int()
        expected_directions = full_directions[full_valid].int()

        fake_extension = SimpleNamespace(hashmap_lookup_3d_cuda=lookup)
        hashmap = (torch.zeros(1), torch.zeros(1))
        with mock.patch.object(low_memory, "_C", fake_extension):
            for batch_size in (1, 2, 3, 8, 20):
                with self.subTest(batch_size=batch_size):
                    quads, directions = low_memory._build_topology_batched(
                        coords,
                        intersected,
                        hashmap,
                        offsets,
                        resolution=3,
                        batch_size=batch_size,
                    )
                    torch.testing.assert_close(quads, expected_quads, rtol=0, atol=0)
                    torch.testing.assert_close(directions, expected_directions, rtol=0, atol=0)

    def test_batched_triangulation_matches_vectorized_cumesh_formula(self) -> None:
        generator = torch.Generator().manual_seed(314159)
        vertices = torch.randn((16, 3), generator=generator, dtype=torch.float32)
        quads = torch.tensor(
            [
                [0, 1, 2, 3],
                [3, 2, 4, 5],
                [5, 4, 6, 7],
                [1, 8, 9, 2],
                [8, 10, 11, 9],
                [7, 6, 12, 13],
                [10, 14, 15, 11],
            ],
            dtype=torch.int32,
        )
        directions = torch.tensor([1, 0, -1, 1, 2, 1, 0], dtype=torch.int32)
        expected = _vectorized_reference(vertices, quads, directions)

        for batch_size in (1, 2, 3, 7, 100):
            with self.subTest(batch_size=batch_size):
                actual = low_memory._triangulate_quads_batched(
                    vertices,
                    quads,
                    directions,
                    batch_size=batch_size,
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(actual.dtype, torch.int32)

    def test_empty_topology_is_supported(self) -> None:
        triangles = low_memory._triangulate_quads_batched(
            torch.zeros((0, 3), dtype=torch.float32),
            torch.empty((0, 4), dtype=torch.int32),
            torch.empty((0,), dtype=torch.int32),
            batch_size=2,
        )
        self.assertEqual(tuple(triangles.shape), (0, 3))
        self.assertEqual(triangles.dtype, torch.int32)

    def test_vertex_compaction_matches_full_gather_and_remap(self) -> None:
        vertices = torch.arange(30, dtype=torch.float32).reshape(10, 3)
        quads = torch.tensor(
            [[9, 2, 5, 7], [7, 5, 2, 1], [9, 7, 1, 2]],
            dtype=torch.int32,
        )
        unique = torch.unique(quads.reshape(-1))
        bounded_unique = low_memory._unique_referenced_vertices(
            quads,
            voxel_count=10,
            batch_size=1,
        )
        reference_map = torch.zeros((10,), dtype=torch.int32)
        reference_map[unique] = torch.arange(unique.shape[0], dtype=torch.int32)
        expected_quads = reference_map[quads]

        torch.testing.assert_close(bounded_unique, unique, rtol=0, atol=0)
        compact = vertices[bounded_unique]
        remapped = low_memory._remap_quads_batched(
            quads.clone(),
            bounded_unique,
            voxel_count=10,
            batch_size=1,
        )

        torch.testing.assert_close(compact, vertices[unique], rtol=0, atol=0)
        torch.testing.assert_close(remapped, expected_quads, rtol=0, atol=0)

    def test_batch_size_and_cpu_residency_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            low_memory._remap_quads_batched(
                torch.zeros((1, 4), dtype=torch.int32),
                torch.zeros((1,), dtype=torch.int32),
                1,
                batch_size=0,
            )
        with self.assertRaisesRegex(ValueError, "must have shape"):
            low_memory._triangulate_quads_batched(
                torch.zeros((1, 3)),
                torch.zeros((1, 3), dtype=torch.int32),
                torch.zeros((1,), dtype=torch.int32),
            )

    def test_cache_release_is_noop_on_cpu_and_synchronizes_cuda(self) -> None:
        with (
            mock.patch.object(low_memory.torch.cuda, "synchronize") as synchronize,
            mock.patch.object(low_memory.torch.cuda, "empty_cache") as empty_cache,
        ):
            low_memory._release_device_memory(torch.device("cpu"))
            synchronize.assert_not_called()
            empty_cache.assert_not_called()

            cuda_device = torch.device("cuda:0")
            low_memory._release_device_memory(cuda_device)
            synchronize.assert_called_once_with(cuda_device)
            empty_cache.assert_called_once_with()

    def test_cumesh_revision_is_pinned_everywhere(self) -> None:
        setup = (REPO_ROOT / "setup.sh").read_text(encoding="utf-8")
        lock = (REPO_ROOT / "requirements-repro.lock").read_text(encoding="utf-8")
        self.assertIn(f"CUMESH_COMMIT={low_memory.CUMESH_COMMIT}", setup)
        self.assertIn(f"CuMesh.git@{low_memory.CUMESH_COMMIT}", lock)
        self.assertIn("MIT License", (REPO_ROOT / "genrecon/vendor/cumesh-LICENSE").read_text())


if __name__ == "__main__":
    unittest.main()
