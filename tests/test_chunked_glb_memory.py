from __future__ import annotations

import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest import mock

import torch
import trimesh

import chunked_to_glb as cli
import inference.chunked_glb as chunked


def _attributes() -> tuple[torch.Tensor, torch.Tensor, dict[str, slice]]:
    attrs = torch.zeros((1, 6), dtype=torch.float32)
    coords = torch.zeros((1, 3), dtype=torch.int32)
    layout = {
        "base_color": slice(0, 3),
        "metallic": slice(3, 4),
        "roughness": slice(4, 5),
        "alpha": slice(5, 6),
    }
    return attrs, coords, layout


class ChunkedGlbMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        # These are lifecycle tests, not CUDA integration tests.  Keep every
        # tensor on CPU even when the host running the suite has a visible GPU.
        cuda_available = mock.patch.object(chunked.torch.cuda, "is_available", return_value=False)
        cuda_available.start()
        self.addCleanup(cuda_available.stop)

    def test_project_back_is_batched_and_matches_reference_expression(self) -> None:
        calls: list[int] = []

        class FakeBVH:
            def unsigned_distance(self, vertices: torch.Tensor, return_uvw: bool):
                if not return_uvw:
                    raise AssertionError("project-back requires barycentric coordinates")
                calls.append(vertices.shape[0])
                face_id = torch.zeros(vertices.shape[0], dtype=torch.long)
                uvw = torch.tensor([0.25, 0.25, 0.5]).repeat(vertices.shape[0], 1)
                return torch.zeros(vertices.shape[0]), face_id, uvw

        source_vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            dtype=torch.float32,
        )
        source_faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        vertices = torch.tensor(
            [[1.0, 1.0, 2.0], [2.0, 2.0, 2.0], [3.0, 3.0, 2.0]],
            dtype=torch.float32,
        )
        original = vertices.clone()
        projected = torch.tensor([0.5, 1.0, 0.0]).repeat(3, 1)
        expected = original - 0.9 * (original - projected)

        result = chunked._project_back_in_batches(
            vertices,
            source_vertices,
            source_faces,
            FakeBVH(),
            0.9,
            batch_size=2,
        )

        self.assertIs(result, vertices)
        self.assertEqual(calls, [2, 1])
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_face_assignment_is_batched_and_matches_unbatched_cdist(self) -> None:
        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [5.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [5.0, 1.0, 0.0],
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
                [10.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        faces = torch.tensor(
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [3, 5, 4], [0, 2, 1]],
            dtype=torch.int32,
        )
        centers = torch.tensor(
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        expected = torch.cdist(vertices[faces.long()].mean(dim=1), centers).argmin(dim=1)
        original_cdist = torch.cdist
        batch_sizes: list[int] = []

        def recording_cdist(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            batch_sizes.append(left.shape[0])
            return original_cdist(left, right)

        with mock.patch.object(chunked.torch, "cdist", side_effect=recording_cdist):
            result = chunked._assign_faces_to_nearest_chunk(
                vertices,
                faces,
                centers,
                batch_size=2,
            )

        self.assertEqual(batch_sizes, [2, 2, 1])
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_atomic_chunk_export_preserves_previous_cache_on_failure(self) -> None:
        class FailingScene:
            def export(self, path: str) -> None:
                Path(path).write_bytes(b"partial")
                raise RuntimeError("interrupted export")

        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "chunk_000.glb"
            destination.write_bytes(b"known-good")

            with self.assertRaisesRegex(RuntimeError, "interrupted export"):
                chunked._export_scene_atomic(FailingScene(), destination)

            self.assertEqual(destination.read_bytes(), b"known-good")
            self.assertEqual(list(Path(tmp).glob(".chunk_000.*.glb")), [])

    def test_global_cumesh_and_bvh_are_collected_before_chunk_bake(self) -> None:
        resource_refs: list[weakref.ReferenceType] = []

        class FakeCuMesh:
            def __init__(self) -> None:
                resource_refs.append(weakref.ref(self))

            def init(self, vertices: torch.Tensor, faces: torch.Tensor) -> None:
                self.vertices = vertices
                self.faces = faces

            @property
            def num_vertices(self) -> int:
                return self.vertices.shape[0]

            @property
            def num_faces(self) -> int:
                return self.faces.shape[0]

            def fill_holes(self, max_hole_perimeter: float) -> None:
                del max_hole_perimeter

            def read(self) -> tuple[torch.Tensor, torch.Tensor]:
                return self.vertices, self.faces

            def clear_cache(self) -> None:
                pass

        class FakeBVH:
            def __init__(self, vertices: torch.Tensor, faces: torch.Tensor) -> None:
                del vertices, faces
                resource_refs.append(weakref.ref(self))

        def fake_remesh(vertices: torch.Tensor, faces: torch.Tensor, **kwargs):
            del kwargs
            return vertices.clone(), faces.clone()

        def fake_bake(**kwargs):
            self.assertTrue(resource_refs)
            self.assertTrue(all(ref() is None for ref in resource_refs))
            return trimesh.Trimesh(
                vertices=kwargs["vertices"].detach().cpu().numpy(),
                faces=kwargs["faces"].detach().cpu().numpy(),
                process=False,
            )

        vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        faces = torch.tensor(
            [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]],
            dtype=torch.int32,
        )
        attrs, coords, layout = _attributes()

        with (
            mock.patch.object(chunked.cumesh, "CuMesh", FakeCuMesh),
            mock.patch.object(chunked.cumesh, "cuBVH", FakeBVH),
            mock.patch.object(chunked, "remesh_narrow_band_dc", new=fake_remesh),
            mock.patch.object(chunked.op, "to_glb", new=fake_bake),
        ):
            scene = chunked.chunked_to_glb(
                vertices_world=vertices,
                faces=faces,
                attr_volume=attrs,
                coords=coords,
                attr_layout=layout,
                aabb_world=[[-0.5, -0.5, -0.5], [1.5, 1.5, 1.5]],
                voxel_size_world=0.25,
                chunk_centers_world=torch.tensor([[0.5, 0.5, 0.5]]),
                chunk_size_world=2.0,
                chunk_indices=[0],
                remesh_res=8,
                remesh_project=0.0,
                chunks_save_dir=None,
                verbose=False,
            )

        self.assertEqual(len(scene.geometry), 1)

    def test_pre_remesh_resources_are_collected_before_output_cumesh_allocation(self) -> None:
        old_resource_refs: list[weakref.ReferenceType] = []
        cumesh_constructions = 0

        class FakeCuMesh:
            def __init__(self) -> None:
                nonlocal cumesh_constructions
                cumesh_constructions += 1
                if cumesh_constructions == 1:
                    old_resource_refs.append(weakref.ref(self))
                else:
                    self.assert_old_resources_released()

            @staticmethod
            def assert_old_resources_released() -> None:
                self.assertTrue(old_resource_refs)
                self.assertTrue(all(ref() is None for ref in old_resource_refs))

            def init(self, vertices: torch.Tensor, faces: torch.Tensor) -> None:
                self.vertices = vertices
                self.faces = faces

            def fill_holes(self, max_hole_perimeter: float) -> None:
                del max_hole_perimeter
                # Model CuMesh producing a distinct post-fill generation.
                self.vertices = self.vertices.clone()
                self.faces = self.faces.clone()
                old_resource_refs.extend((weakref.ref(self.vertices), weakref.ref(self.faces)))

            def read(self) -> tuple[torch.Tensor, torch.Tensor]:
                return self.vertices, self.faces

            def clear_cache(self) -> None:
                pass

        class FakeBVH:
            def __init__(self, vertices: torch.Tensor, faces: torch.Tensor) -> None:
                self.vertices = vertices
                self.faces = faces
                old_resource_refs.append(weakref.ref(self))

        def fake_remesh(vertices: torch.Tensor, faces: torch.Tensor, **kwargs):
            del kwargs
            return vertices.clone(), faces.clone()

        def fake_bake(**kwargs):
            return trimesh.Trimesh(
                vertices=kwargs["vertices"].detach().cpu().numpy(),
                faces=kwargs["faces"].detach().cpu().numpy(),
                process=False,
            )

        attrs, coords, layout = _attributes()
        with (
            mock.patch.object(chunked.cumesh, "CuMesh", new=FakeCuMesh),
            mock.patch.object(chunked.cumesh, "cuBVH", new=FakeBVH),
            mock.patch.object(chunked, "remesh_narrow_band_dc", new=fake_remesh),
            mock.patch.object(chunked.op, "to_glb", new=fake_bake),
        ):
            scene = chunked.chunked_to_glb(
                vertices_world=torch.tensor(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    dtype=torch.float32,
                ),
                faces=torch.tensor(
                    [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]],
                    dtype=torch.int32,
                ),
                attr_volume=attrs,
                coords=coords,
                attr_layout=layout,
                aabb_world=[[-0.5, -0.5, -0.5], [1.5, 1.5, 1.5]],
                voxel_size_world=0.25,
                chunk_centers_world=torch.tensor([[0.5, 0.5, 0.5]]),
                chunk_size_world=2.0,
                chunk_indices=[0],
                remesh_res=8,
                remesh_project=0.0,
                chunks_save_dir=None,
                verbose=False,
            )

        self.assertEqual(cumesh_constructions, 2)
        self.assertEqual(len(scene.geometry), 1)

    def test_geometry_lifetime_for_all_fill_and_remesh_flag_combinations(self) -> None:
        class FakeCuMesh:
            def init(self, vertices: torch.Tensor, faces: torch.Tensor) -> None:
                self.vertices = vertices
                self.faces = faces

            @property
            def num_vertices(self) -> int:
                return self.vertices.shape[0]

            @property
            def num_faces(self) -> int:
                return self.faces.shape[0]

            def fill_holes(self, max_hole_perimeter: float) -> None:
                del max_hole_perimeter
                self.vertices = self.vertices.clone()
                self.faces = self.faces.clone()

            def read(self) -> tuple[torch.Tensor, torch.Tensor]:
                return self.vertices, self.faces

            def clear_cache(self) -> None:
                pass

        class FakeBVH:
            def __init__(self, vertices: torch.Tensor, faces: torch.Tensor) -> None:
                del vertices, faces

        def fake_remesh(vertices: torch.Tensor, faces: torch.Tensor, **kwargs):
            del kwargs
            return vertices.clone(), faces.clone()

        baked_face_counts: list[int] = []

        def fake_bake(**kwargs):
            baked_face_counts.append(kwargs["faces"].shape[0])
            return trimesh.Trimesh(
                vertices=kwargs["vertices"].detach().cpu().numpy(),
                faces=kwargs["faces"].detach().cpu().numpy(),
                process=False,
            )

        vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        faces = torch.tensor(
            [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]],
            dtype=torch.int32,
        )
        attrs, coords, layout = _attributes()

        with (
            mock.patch.object(chunked.cumesh, "CuMesh", new=FakeCuMesh),
            mock.patch.object(chunked.cumesh, "cuBVH", new=FakeBVH),
            mock.patch.object(chunked, "remesh_narrow_band_dc", new=fake_remesh),
            mock.patch.object(chunked.op, "to_glb", new=fake_bake),
        ):
            for do_fill_holes in (False, True):
                for do_remesh in (False, True):
                    with self.subTest(do_fill_holes=do_fill_holes, do_remesh=do_remesh):
                        scene = chunked.chunked_to_glb(
                            vertices_world=vertices,
                            faces=faces,
                            attr_volume=attrs,
                            coords=coords,
                            attr_layout=layout,
                            aabb_world=[[-0.5, -0.5, -0.5], [1.5, 1.5, 1.5]],
                            voxel_size_world=0.25,
                            chunk_centers_world=torch.tensor([[0.5, 0.5, 0.5]]),
                            chunk_size_world=2.0,
                            chunk_indices=[0],
                            remesh_res=8,
                            do_fill_holes=do_fill_holes,
                            do_remesh=do_remesh,
                            remesh_project=0.0,
                            chunks_save_dir=None,
                            verbose=False,
                        )
                        self.assertEqual(len(scene.geometry), 1)

        self.assertEqual(baked_face_counts, [4, 4, 4, 4])

    def test_existing_chunk_is_resumed_and_new_chunk_is_atomically_cached(self) -> None:
        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
                [10.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32)
        attrs, coords, layout = _attributes()

        with tempfile.TemporaryDirectory() as tmp:
            chunks_dir = Path(tmp)
            cached = trimesh.Scene(
                [trimesh.Trimesh(vertices=vertices[:3].numpy(), faces=[[0, 1, 2]], process=False)]
            )
            cached.export(str(chunks_dir / "chunk_000.glb"))

            baked_means: list[float] = []

            def fake_bake(**kwargs):
                baked_means.append(float(kwargs["vertices"][:, 0].mean().item()))
                return trimesh.Trimesh(
                    vertices=kwargs["vertices"].detach().cpu().numpy(),
                    faces=kwargs["faces"].detach().cpu().numpy(),
                    process=False,
                )

            with (
                mock.patch.object(chunked.op, "to_glb", side_effect=fake_bake),
                mock.patch.object(chunked, "_release_cuda_memory") as release,
            ):
                scene = chunked.chunked_to_glb(
                    vertices_world=vertices,
                    faces=faces,
                    attr_volume=attrs,
                    coords=coords,
                    attr_layout=layout,
                    aabb_world=[[-1.0, -1.0, -1.0], [12.0, 2.0, 1.0]],
                    voxel_size_world=1.0,
                    chunk_centers_world=torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
                    chunk_size_world=2.0,
                    chunk_indices=[0, 1],
                    do_fill_holes=False,
                    do_remesh=False,
                    chunks_save_dir=chunks_dir,
                    verbose=False,
                )

            self.assertEqual(len(baked_means), 1)
            self.assertGreater(baked_means[0], 5.0)
            self.assertTrue((chunks_dir / "chunk_001.glb").is_file())
            self.assertFalse(list(chunks_dir.glob(".*.glb")))
            self.assertEqual(len(scene.geometry), 2)
            self.assertGreaterEqual(release.call_count, 4)

    def test_cli_exits_nonzero_when_conversion_fails(self) -> None:
        attrs, coords, layout = _attributes()
        inputs = {
            "vertices": torch.zeros((1, 3)),
            "faces": torch.zeros((1, 3), dtype=torch.int32),
            "attr_volume": attrs,
            "coords": coords,
            "attr_layout": layout,
            "aabb": torch.zeros((2, 3)),
            "voxel_size": 1.0,
        }
        chunk_inputs = {
            "chunk_centers_world": torch.zeros((1, 3)),
            "chunk_size_world": 1.0,
            "chunk_indices": [0],
        }

        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                "chunked_to_glb.py",
                "--inputs",
                "inputs.pt",
                "--chunk_inputs",
                "chunks.pt",
                "--output_dir",
                tmp,
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(cli, "load_pt", side_effect=[inputs, chunk_inputs]),
                mock.patch.object(cli, "chunked_to_glb", side_effect=RuntimeError("expected failure")),
                mock.patch.object(cli.torch.cuda, "is_available", return_value=False),
                mock.patch.object(cli.traceback, "print_exc"),
            ):
                with self.assertRaises(SystemExit) as raised:
                    cli.main()

        self.assertEqual(raised.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
