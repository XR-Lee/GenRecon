from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from genrecon.modules.sparse import SparseTensor
from genrecon.pipelines.full_scene_images_to_3d import FullSceneImagesTo3DPipeline
from genrecon.pipelines.joint_decode import (
    JointDecodeContext,
    _sparse_cpu_clean,
    _make_groups,
    _sparse_to_device_clean,
    joint_decode_shape,
    joint_decode_tex,
)
from genrecon.pipelines import joint_decode as joint_decode_module


class FullScenePipelineTests(unittest.TestCase):
    def test_object_token_mask_zeros_background_and_rejected_view_globals(self) -> None:
        features = torch.ones(1, 2, 7, 3)
        patch_masks = torch.tensor([[True, False], [False, False]])

        masked, record = FullSceneImagesTo3DPipeline._mask_dino_tokens(
            features, patch_masks, global_tokens=5
        )

        self.assertTrue(bool((masked[0, 0, :6] == 1).all()))
        self.assertTrue(bool((masked[0, 0, 6] == 0).all()))
        self.assertTrue(bool((masked[0, 1] == 0).all()))
        self.assertEqual(record["active_views"], 1)
        self.assertEqual(record["kept_patch_tokens"], 1)
        self.assertEqual(record["kept_global_tokens"], 5)

        patches_only, patches_record = FullSceneImagesTo3DPipeline._mask_dino_tokens(
            features,
            patch_masks,
            global_tokens=5,
            keep_global_tokens=False,
        )
        self.assertTrue(bool((patches_only[0, :, :5] == 0).all()))
        self.assertTrue(bool((patches_only[0, 0, 5] == 1).all()))
        self.assertEqual(patches_record["kept_global_tokens"], 0)
        self.assertFalse(patches_record["global_tokens_enabled"])

    def test_object_support_crop_compacts_vertices_and_faces(self) -> None:
        mesh = SimpleNamespace(
            vertices=torch.tensor(
                [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [2.0, 0.0, 0.0]]
            ),
            faces=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int32),
        )
        bounds = torch.tensor([[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]])

        record = FullSceneImagesTo3DPipeline._crop_mesh_to_support(mesh, bounds)

        self.assertEqual(mesh.vertices.shape, (3, 3))
        self.assertEqual(mesh.faces.tolist(), [[0, 1, 2]])
        self.assertEqual(record["faces_removed"], 1)
        self.assertEqual(record["vertices_outside_support_before_crop"], 1)

    def test_shared_anchor_tokens_only_change_target_chunk_sequence_lengths(self) -> None:
        features = torch.arange(1 * 3 * 2 * 4, dtype=torch.float32).reshape(1, 3, 2, 4)
        shared = torch.full((1, 1, 4), 99.0)

        result = FullSceneImagesTo3DPipeline._per_chunk_cond2d(
            features,
            chunk_indices=[3, 4, 5],
            shared_tokens=shared,
            target_chunk_ids={3, 5},
        )

        self.assertEqual([item.shape[1] for item in result], [3, 2, 3])
        torch.testing.assert_close(result[0][:, -1], shared[:, 0])
        torch.testing.assert_close(result[1], features[:, 1])
        torch.testing.assert_close(result[2][:, -1], shared[:, 0])

    def test_anchor_tokens_only_pool_true_track_observation_views(self) -> None:
        pipeline = FullSceneImagesTo3DPipeline.__new__(FullSceneImagesTo3DPipeline)
        pipeline._device = torch.device("cpu")
        projected = torch.tensor(
            [[[[1.0, 0.0], [3.0, 0.0]], [[9.0, 0.0], [5.0, 0.0]]]]
        )  # [1, 2 views, 2 anchors, 2 channels]
        valid = torch.ones(1, 2, 2, dtype=torch.bool)
        view_mask = torch.tensor([[True, False], [False, False]])
        flow_model = SimpleNamespace(projection=object())

        with mock.patch(
            "genrecon.pipelines.full_scene_images_to_3d.project_features_on_points",
            return_value=(projected, valid, None),
        ):
            tokens, counts = pipeline._build_shared_anchor_tokens(
                flow_model,
                scene_feats=torch.zeros(1, 2, 1, 2),
                scene_ext_c0=torch.eye(4).repeat(1, 2, 1, 1),
                scene_intr=torch.eye(3).repeat(1, 2, 1, 1),
                anchor_points_chunk0=torch.zeros(2, 3),
                anchor_view_mask=view_mask,
            )

        self.assertEqual(tokens.shape, (1, 1, 2))
        self.assertEqual(counts.tolist(), [1])
        self.assertGreater(float(tokens[0, 0, 0]), 0.0)

    def test_sparse_noise_uses_flow_latent_resolution_not_decoder_resolution(self) -> None:
        flow_model = SimpleNamespace(
            resolution=16,
            in_channels=8,
            dtype=torch.bfloat16,
        )

        noise = FullSceneImagesTo3DPipeline._sample_sparse_noise(
            flow_model,
            num_chunks=2,
            device=torch.device("cpu"),
        )

        self.assertEqual(len(noise), 2)
        self.assertTrue(all(item.shape == (1, 8, 16, 16, 16) for item in noise))
        self.assertTrue(all(item.dtype == torch.bfloat16 for item in noise))

    def test_joint_decode_limits_preserve_small_512_single_pass(self) -> None:
        limits = FullSceneImagesTo3DPipeline._joint_decode_limits("512", 16, True)
        self.assertEqual(limits, (None, None))

        limits = FullSceneImagesTo3DPipeline._joint_decode_limits("512", 17, True)
        self.assertEqual(limits, (8, 80_000))

        limits = FullSceneImagesTo3DPipeline._joint_decode_limits("512", 20, False)
        self.assertEqual(limits, (None, None))

        limits = FullSceneImagesTo3DPipeline._joint_decode_limits("1024", 1, False)
        self.assertEqual(limits, (10, 100_000))

    def test_chunked_decode_owned_partitions_are_disjoint_and_complete(self) -> None:
        relative = [
            torch.tensor([0.0, 0.0, 0.0]),
            torch.tensor([0.625, 0.0, 0.0]),
            torch.tensor([0.0, 0.625, 0.0]),
            torch.tensor([0.625, 0.625, 0.0]),
        ]
        sparse = SparseTensor(
            feats=torch.ones(4, 1),
            coords=torch.tensor(
                [[0, 0, 0, 0], [0, 20, 0, 0], [0, 0, 20, 0], [0, 20, 20, 0]],
                dtype=torch.int32,
            ),
        )

        groups, counts, owned = _make_groups(
            relative,
            R_in=32,
            max_chunks_per_group=1,
            joint_slat=sparse,
            overlap_r_in=16,
            max_inflated_voxels=10,
        )

        self.assertEqual(len(groups), 4)
        self.assertEqual(len(counts), 4)
        root_volume = 52 * 52 * 32
        volumes = [int(torch.prod(box[1] - box[0]).item()) for box in owned]
        self.assertEqual(sum(volumes), root_volume)
        for i, left in enumerate(owned):
            for right in owned[i + 1 :]:
                overlap = torch.minimum(left[1], right[1]) - torch.maximum(left[0], right[0])
                self.assertTrue(bool((overlap <= 0).any()))

    def test_chunked_shape_decode_runs_partition_and_merge_path(self) -> None:
        class FakeDecoder:
            resolution = 4
            num_blocks = [1, 1]
            voxel_margin = 0.0
            low_vram = False

        def fake_forward(decoder, sparse, return_subs=False):
            feats = torch.zeros(sparse.feats.shape[0], 7)
            output = SparseTensor(feats=feats, coords=sparse.coords)
            guide = SparseTensor(feats=torch.ones(sparse.feats.shape[0], 1), coords=sparse.coords)
            return output, [guide]

        def fake_mesh(coords, vertex_feats, intersected, quad_lerp, **kwargs):
            vertices = coords.float() / 2.0 - 0.5
            faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
            return vertices, faces

        local_coords = torch.tensor(
            [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]],
            dtype=torch.int32,
        )
        slats = [
            SparseTensor(feats=torch.ones(3, 1), coords=local_coords.clone()),
            SparseTensor(feats=torch.ones(3, 1), coords=local_coords.clone()),
        ]
        relative = [torch.zeros(3), torch.tensor([1.0, 0.0, 0.0])]

        with (
            mock.patch.object(
                joint_decode_module.SparseUnetVaeDecoder,
                "forward",
                new=fake_forward,
            ),
            mock.patch.object(
                joint_decode_module,
                "flexible_dual_grid_to_mesh",
                new=fake_mesh,
            ),
        ):
            mesh, _, _, ctx, _ = joint_decode_shape(
                FakeDecoder(),
                slats,
                relative,
                max_chunks_per_group=1,
                overlap_r_in=0,
                consume_inputs=True,
            )

        self.assertEqual(slats, [])
        self.assertEqual(mesh.vertices.shape, (6, 3))
        self.assertEqual(mesh.faces.shape, (2, 3))
        self.assertEqual(len(ctx.owned_aabbs), 2)
        self.assertEqual(len(ctx.subs_inflated), 2)

    def test_chunked_texture_decode_filters_and_merges_owned_outputs(self) -> None:
        class IdentityDecoder:
            def __call__(self, sparse, guide_subs=None):
                return sparse

        local_coords = torch.tensor(
            [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]],
            dtype=torch.int32,
        )
        slats = [
            SparseTensor(feats=torch.full((3, 1), 2.0), coords=local_coords.clone()),
            SparseTensor(feats=torch.full((3, 1), 4.0), coords=local_coords.clone()),
        ]
        owned = [
            torch.tensor([[0, 0, 0], [2, 2, 2]]),
            torch.tensor([[2, 0, 0], [4, 2, 2]]),
        ]
        ctx = JointDecodeContext(
            inflated_aabbs=[box.clone() for box in owned],
            owned_aabbs=owned,
            subs_inflated=[[], []],
            R_in=2,
            upscale=1,
        )

        decoded = joint_decode_tex(
            IdentityDecoder(),
            slats,
            ctx,
            [torch.zeros(3), torch.tensor([1.0, 0.0, 0.0])],
            consume_inputs=True,
        )

        self.assertEqual(slats, [])
        self.assertEqual(decoded.feats.shape, (6, 1))
        self.assertTrue(torch.equal(decoded.feats[:3], torch.full((3, 1), 1.5)))
        self.assertTrue(torch.equal(decoded.feats[3:], torch.full((3, 1), 2.5)))

    def test_without_sparse_caches_preserves_payload_and_drops_cache(self) -> None:
        sparse = SparseTensor(
            feats=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            coords=torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=torch.int32),
        )
        sparse.register_spatial_cache("temporary", torch.tensor([7]))

        clean = FullSceneImagesTo3DPipeline._without_sparse_caches(sparse)

        self.assertTrue(torch.equal(clean.feats, sparse.feats))
        self.assertTrue(torch.equal(clean.coords, sparse.coords))
        self.assertIsNone(clean.get_spatial_cache("temporary"))

    def test_joint_texture_decode_can_release_consumed_chunk_inputs(self) -> None:
        class IdentityDecoder:
            def __call__(self, sparse, guide_subs=None):
                return sparse

        slats = [
            SparseTensor(
                feats=torch.tensor([[2.0]]),
                coords=torch.tensor([[0, 0, 0, 0]], dtype=torch.int32),
            )
        ]
        ctx = JointDecodeContext(
            inflated_aabbs=None,
            owned_aabbs=None,
            subs_inflated=[[]],
            R_in=1,
            upscale=1,
        )

        decoded = joint_decode_tex(
            IdentityDecoder(),
            slats,
            ctx,
            [torch.zeros(3)],
            consume_inputs=True,
        )

        self.assertEqual(slats, [])
        self.assertTrue(torch.equal(decoded.feats, torch.tensor([[1.5]])))

    def test_decode_context_cpu_roundtrip_is_exact_and_cache_free(self) -> None:
        guide = SparseTensor(
            feats=torch.tensor([[1.0, -1.0], [0.5, 2.0]]),
            coords=torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.int32),
        )
        guide.register_spatial_cache("shape_decoder_neighbor_map", torch.arange(8))

        stashed = _sparse_cpu_clean(guide)
        restored = _sparse_to_device_clean(stashed, torch.device("cpu"))

        torch.testing.assert_close(restored.feats, guide.feats, rtol=0, atol=0)
        self.assertTrue(torch.equal(restored.coords, guide.coords))
        self.assertIsNone(stashed.get_spatial_cache("shape_decoder_neighbor_map"))
        self.assertIsNone(restored.get_spatial_cache("shape_decoder_neighbor_map"))


if __name__ == "__main__":
    unittest.main()
