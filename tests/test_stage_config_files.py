from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from genrecon.pipelines.images_to_3d import ImagesTo3DPipeline
from genrecon.pipelines.setup_utils import LoadedStage


class StageConfigFilesTests(unittest.TestCase):
    @patch("genrecon.pipelines.images_to_3d.init_runtime_assets")
    @patch("genrecon.pipelines.images_to_3d.load_pipeline_args", return_value={})
    @patch("genrecon.pipelines.images_to_3d.load_slat_stage")
    @patch("genrecon.pipelines.images_to_3d.load_sparse_flow_stage")
    def test_routes_explicit_configs_to_each_finetuned_stage(
        self,
        load_sparse: Mock,
        load_slat: Mock,
        _load_pipeline_args: Mock,
        _init_runtime_assets: Mock,
    ) -> None:
        load_sparse.return_value = LoadedStage(Mock(), {}, num_cond_views=16)
        load_slat.return_value = LoadedStage(
            Mock(), {}, output_resolution=512, num_cond_views=16
        )
        stages = {
            "sparse_structure_flow_model": "weights/sparse_structure.pt",
            "shape_slat_flow_model_512": "weights/shape_slat.pt",
            "tex_slat_flow_model_512": "weights/texture_slat.pt",
        }
        configs = {
            "sparse_structure_flow_model": "configs/ss.json",
            "shape_slat_flow_model_512": "configs/shape.json",
            "tex_slat_flow_model_512": "configs/texture.json",
        }

        ImagesTo3DPipeline.from_finetuned(stages, stage_config_files=configs)

        load_sparse.assert_called_once_with(
            ckpt_path="weights/sparse_structure.pt",
            train_config_path="configs/ss.json",
        )
        self.assertEqual(
            [call.kwargs["train_config_path"] for call in load_slat.call_args_list],
            ["configs/shape.json", "configs/texture.json"],
        )

    @patch("genrecon.pipelines.images_to_3d.init_runtime_assets")
    @patch("genrecon.pipelines.images_to_3d.load_pipeline_args", return_value={})
    def test_rejects_config_for_unrequested_stage(
        self, _load_pipeline_args: Mock, _init_runtime_assets: Mock
    ) -> None:
        with self.assertRaisesRegex(ValueError, "unexpected keys"):
            ImagesTo3DPipeline.from_finetuned(
                {"sparse_structure_flow_model": None},
                stage_config_files={"shape_slat_flow_model_512": "shape.json"},
            )


if __name__ == "__main__":
    unittest.main()
