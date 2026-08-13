<div align="center">

# GenRecon: Bridging Generative Priors for Multi-View 3D Scene Reconstruction

[Katharina Schmid](https://kasothaphie.github.io/)<sup>1</sup>, &nbsp;
[Nicolas von Lützow](https://nicolasvonluetzow.github.io/)<sup>1</sup>, &nbsp;
[Jozef Hladký](https://scholar.google.com/citations?user=CDy95WwAAAAJ&hl=en)<sup>2</sup>, &nbsp;
[Angela Dai](https://www.3dunderstanding.org/team.html)<sup>1</sup>, &nbsp;
[Matthias Nießner](https://niessnerlab.org/members/matthias_niessner/profile.html)<sup>1</sup>

<sup>1</sup> Technical University of Munich &nbsp;&nbsp; <sup>2</sup> Computing Systems Lab, Huawei Technologies, Switzerland

[![Project Page](https://img.shields.io/badge/Project-Page-blue?logo=googlechrome&logoColor=white)](https://kasothaphie.github.io/GenRecon/)
[![arXiv](https://img.shields.io/badge/arXiv-2605.23888-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.23888)
[![PDF](https://img.shields.io/badge/Paper-PDF-green)](https://arxiv.org/pdf/2605.23888)
[![Video](https://img.shields.io/badge/Video-YouTube-red?logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=Tp-i06DPXa0)

![Teaser](assets/teaser_short.png)

</div>

## ✨ Abstract

We introduce a new approach to high-fidelity 3D scene reconstruction from multi-view RGB images that tightly couples reconstruction with a strong generative 3D prior. We cast scene reconstruction as conditional 3D generation over a set of spatially-localized, overlapping chunks that together tile the scene, scaling generation to large scene extents. Crucially, we inherit the fidelity and completeness of state-of-the-art generative shape models -- we use Trellis.2 as an example -- which we generalize to the scene level. To this end, we propose a projection-based conditioning mechanism that lifts posed multi-view image features into a coherent 3D representation aligned with the generative model, independent of view ordering and spatially anchored to the scene, yielding high-fidelity, multi-view consistent generated geometry. This enables lifting the strong object-level prior of Trellis.2 to multi-view, scene-scale generation, producing faithful, editable PBR mesh reconstructions of indoor environments. As a result, we obtain high-fidelity results that outperform cutting-edge reconstruction methods by 16%.

## 📅 Timeline

✅ Paper release (22.05.2026)   
✅ Code release (29.06.2026)  
✅ Checkpoint release (29.06.2026)  

## 🛠️ Installation

1. Clone the repo:
```sh
git clone -b main https://github.com/kasothaphie/GenRecon.git --recursive
cd GenRecon
```

2. Set up environment

The simplest path is the bundled setup script, which creates the conda env and
installs PyTorch, Flash-Attention, and all CUDA extensions. It expects a working
CUDA toolkit on your system, so set `CUDA_HOME` (and your GPU architecture)
beforehand. For background on the script and troubleshooting, refer to
[microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2).

```sh
export CUDA_HOME=/usr/local/cuda    # path to your CUDA 12.x toolkit
export TORCH_CUDA_ARCH_LIST="9.0"   # adjust for your GPU
. ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
```

<details>
<summary>Alternative: create the environment manually (e.g. if you don't have a system CUDA toolkit)</summary>

Create the env yourself and install CUDA (and ninja) into it via conda.

```sh
conda create -n genrecon python=3.10 nvidia::cuda-toolkit=12.6 ninja
conda activate genrecon
conda env config vars set CUDA_HOME=$CONDA_PREFIX
conda deactivate && conda activate genrecon # to set the env variable from above
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126

conda install -c conda-forge libjpeg-turbo xorg-libx11

export TORCH_CUDA_ARCH_LIST="9.0"   # adjust for your GPU
. ./setup.sh --basic --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
```

Then install Flash-Attention manually. Pick the wheel matching your
Python/torch/CUDA/ABI from the [v2.7.3 releases](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.3);
the example below requires an Ampere or newer GPU (sm_80+).
```sh
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```
</details>

#### CuMesh reproducibility note

The global GLB remesh uses the repository-owned
`genrecon/vendor/cumesh_remeshing.py`, a memory-bounded derivative of CuMesh
commit `12289e1062f0603f2f0d0771b02e1395d247f26f`. The setup script pins that
revision because the vendored implementation calls its extension entry points.
No manual edit under `site-packages` is required (or captured by `pip freeze`).

## 🗂️ Data

We are grateful to the authors of the following datasets, whose data made this work
possible. Please refer to the respective sources for licensing and download instructions.

| Dataset | Source data |
| --- | --- |
| SAGE-10k | [original data](https://huggingface.co/datasets/nvidia/SAGE-10k) |
| 3D-FRONT | [original data](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset) (no longer available) |
| ScanNet++| [original data](https://scannetpp.mlsg.cit.tum.de/scannetpp/) |

**3D-FRONT note:** the original source data is no longer available, but more recent
re-releases (e.g. [this one](https://huggingface.co/datasets/huanngzh/3D-Front)) should
work similarly.

## 📦 Pretrained Weights

We provide finetuned checkpoints for the 3 generative models. Please refer to [https://kaldir.vc.cit.tum.de/genrecon/README.md](https://kaldir.vc.cit.tum.de/genrecon/README.md) for download instructions. Alternatively, just run:

```sh
wget https://kaldir.vc.cit.tum.de/genrecon/sparse_structure.pt
wget https://kaldir.vc.cit.tum.de/genrecon/shape_slat.pt
wget https://kaldir.vc.cit.tum.de/genrecon/texture_slat.pt
```


## 🏋️ Training

### Data Preparation

Before training, we need to render the 3D indoor scenes, create chunks, convert them into the O-Voxel representation and compute the latent representation.

Please refer to [data_toolkit_scenes/README.md](data_toolkit_scenes/README.md) for detailed instructions.


### Sparse Structure

```sh
python train.py \
    --config configs/gen/ss_flow_img/genrecon.json \
    --output_dir results/ss_gen \
    --data_dir "{\"SAGE\": {\"base\": \"${DATA_ROOT}/SAGE\", \"ss_latent\": \"${DATA_ROOT}/SAGE/ss_latents/ss_enc_conv3d_16l8_fp16_64\"}}" \
    --val_data_dir "{\"SAGE\": {\"base\": \"${DATA_ROOT}/SAGE_val\", \"ss_latent\": \"${DATA_ROOT}/SAGE_val/ss_latents/ss_enc_conv3d_16l8_fp16_64\"}}"
```

### Shape SLat

```sh
python train.py \
    --config configs/gen/slat_flow_img2shape/genrecon_512.json \
    --output_dir results/shape_gen \
    --data_dir "{\"SAGE\": {\"base\": \"${DATA_ROOT}/SAGE\", \"shape_latent\": \"${DATA_ROOT}/SAGE/shape_latents/shape_enc_next_dc_f16c32_fp16_512\"}}" \
    --val_data_dir "{\"SAGE\": {\"base\": \"${DATA_ROOT}/SAGE_val\", \"shape_latent\": \"${DATA_ROOT}/SAGE_val/shape_latents/shape_enc_next_dc_f16c32_fp16_512\"}}"
```

### Texture SLat

```sh
python train.py \
    --config configs/gen/slat_flow_imgshape2tex/genrecon_512.json \
    --output_dir results/tex_gen \
    --data_dir "{\"SAGE\": {\"base\": \"${DATA_ROOT}/SAGE\", \"shape_latent\": \"${DATA_ROOT}/SAGE/shape_latents/shape_enc_next_dc_f16c32_fp16_512\", \"pbr_latent\": \"${DATA_ROOT}/SAGE/pbr_latents/tex_enc_next_dc_f16c32_fp16_512\"}}" \
    --val_data_dir "{\"SAGE\": {\"base\": \"${DATA_ROOT}/SAGE_val\", \"shape_latent\": \"${DATA_ROOT}/SAGE_val/shape_latents/shape_enc_next_dc_f16c32_fp16_512\", \"pbr_latent\": \"${DATA_ROOT}/SAGE_val/pbr_latents/tex_enc_next_dc_f16c32_fp16_512\"}}"

```

## 🚀 Inference

Make sure you have downloaded the checkpoints or trained the models yourself.

### Reconstruct ScanNet++ scenes
```sh
python reconstruct_scene.py \
    --mode Scannet_colmap \
    --path "${PATH_TO_SCANNETPP_SCENE}" \
    --output_path "${OUT_DIR}" \
    --ss_ckpt "${SS_CKPT}" \
    --shape_ckpt "${SHAPE_CKPT}" \
    --tex_ckpt "${TEX_CKPT}" \
    --num_imgs_per_scene 32
```
To reconstruct from ScanNet++ iPhone captures instead, pass `--mode Scannet_iphone`.


### Reconstruct scenes from smartphone videos
First, you need to compute the camera paramters. We recommend [COLMAP](https://github.com/colmap/colmap).
```sh
python reconstruct_scene.py \
  --mode Iphone \
  --path "${WORK_ROOT}" \
  --output_path "${OUT_DIR}" \
  --ss_ckpt    "${SS_CKPT}" \
  --shape_ckpt "${SHAPE_CKPT}" \
  --tex_ckpt   "${TEX_CKPT}" \
  --num_imgs_per_scene 999 \
  --chunk_size_factor 1.08 \
  --stat_std_ratio 3.0 \
  --radius_nb_points 7 \
  --radius_m 0.2 \
  --pipeline_config configs/pipelines/texture.json \
  --proj_batch_voxels 2048
```

For internet-video scenes that pass the native masked-COLMAP A gate, prepare
undistorted RGBA plus a z-up, proxy-scaled native-SfM package with:

```sh
.venv/bin/python tools/prepare_native_sfm.py all
```

The three native-SfM scenes, their complete GenRecon PLY/PBR assets, and all 303
registered-camera comparison frames are documented in
`reports/NATIVE_SFM_GENRECON_REPORT_zh.md`. Review the videos at
`outputs/internet-zero-shot/native-sfm-video-comparisons-v1/index.html`. These
inputs retain native COLMAP point IDs, errors, and tracks; only scale and gravity
are proxies.

When masked COLMAP cannot provide a usable scene model, the research-only VGGT
fallback can emit proxy-scaled, z-up pseudo geometry in the same `Iphone`
layout:

```sh
PYTHONPATH=/tmp/pycolmap-wheel .venv/bin/python \
  tools/prepare_foundation_sfm.py all
```

Review `data/internet-zero-shot/foundation-sfm-v1/index.html` before using any
output. A GenRecon-consumable point cloud is not ground truth and can still
represent a slideshow, a fixed-camera person shot, an outdoor scene, or more
than one physical room. The public VGGT-1B checkpoint used by this tool is
CC-BY-NC-4.0. See `reports/FOUNDATION_SFM_FALLBACK_REPORT_zh.md` for the fixed
revisions, quality gates, and measured results. The 17-scene current-version
GenRecon mesh/PBR asset run and its input-to-output directory map are documented
in `reports/FOUNDATION_GENRECON_MESH_ASSETS_REPORT_zh.md`. Frame-exact original,
reconstruction, and side-by-side H.264 review videos are documented in
`reports/FOUNDATION_GENRECON_VIDEO_COMPARISON_REPORT_zh.md`; open
`outputs/internet-zero-shot/foundation-video-comparisons-v1/index.html` to review
all 17 candidates.

### Ground-truth calibration suite

The frozen mixed-provenance calibration plan contains 76 scene/object units
across ScanNet++, ETH3D, Tanks and Temples, 7-Scenes, Redwood, DTU, and
OmniObject3D. Build and validate the locally available packages with:

```sh
PYTHONPATH=/tmp/pycolmap-wheel .venv/bin/python \
  tools/calibrate_tnt_meetingroom.py
.venv/bin/python tools/prepare_gt_calibration_datasets.py all
.venv/bin/python tools/validate_gt_calibration_sources.py
```

The current local build has all 76 calibration packages prepared. The 24
OmniObject3D units use official object IDs and the official 100-view
reconstruction assets: white-background RGB and cameras are exported, the
professional scan is normalized into the audited render frame, and depth EXRs
remain audit-only rather than GenRecon conditioning. Tanks and Temples
Meetingroom includes the official individual scans/alignment, fixed-pose
intrinsics calibration, undistorted 8+8 inputs, and an official-crop 1 cm laser
reference. The registry contains 26 evaluated predictions: 22 legacy G0
predictions plus four representative RGB-only runs for Meetingroom, 7-Scenes
`chess`, Redwood `livingroom`, and DTU `scan24`. The remaining 50 prepared units,
including all 24 OmniObject3D objects, are explicitly `missing-prediction`.
Those four runs use only the frozen eight conditioning RGB images and
conditioning camera records; VGGT-1B pseudo geometry is aligned by a
conditioning-camera-only Sim(3), with no heldout RGB/depth, reference geometry,
or GT ICP. Protocol scope, GT tier, scene/instance unit type, and
prediction-generation track are never merged into one score. See
`reports/GT_CALIBRATION_V1_REPORT_zh.md` and
`reports/GT_REPRESENTATIVE_GENRECON_V1_REPORT_zh.md` for source checksums,
metrics, limitations, and reproduction commands.

Representative same-camera review videos are available locally at
`outputs/gt-calibration-v1/visualizations-v1/index.html`. Each prepared scene
shows frozen reference RGB beside evaluation-only GT/reference geometry. The
third panel contains the registry-backed GenRecon prediction. Build and
release-validate the review set with:

```sh
EGL_PLATFORM=surfaceless .venv/bin/python \
  tools/export_gt_calibration_videos.py all
.venv/bin/python tools/validate_gt_calibration_videos.py
```

The current review has seven prepared source/reference representatives, six
registry-backed GenRecon prediction renders, one explicit OmniObject3D
`missing-prediction` panel, and no source blocker. See
`reports/GT_CALIBRATION_VIDEO_VISUALIZATION_V1_REPORT_zh.md` for the per-dataset
selection, provenance boundaries, hashes, and validation counts.

### GLB conversion
Bake the reconstructed scene into a single textured `scene.glb`. This reads the
`to_glb_inputs.pt` and `chunk_inputs.pt` written by `reconstruct_scene.py` into
`--output_path`, and writes `scene.glb` to the same directory.
```sh
python chunked_to_glb.py \
    --inputs "${OUT_DIR}/to_glb_inputs.pt" \
    --chunk_inputs "${OUT_DIR}/chunk_inputs.pt" \
    --output_dir "${OUT_DIR}"
```


## 🙏 Acknowledgements

This work would not have been possible without the following open-source projects, and we thank their authors and contributors.

- [Trellis.2](https://github.com/microsoft/TRELLIS.2/tree/main)
- [CuMesh](https://github.com/JeffreyXiang/CuMesh)
- [FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM)
- [O-Voxel](https://github.com/microsoft/TRELLIS.2/tree/main/o-voxel)
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
- [nvdiffrec](https://github.com/NVlabs/nvdiffrec)
- [Flash-Attention](https://github.com/Dao-AILab/flash-attention)

## ⚖️ License

This model and code are released under the **[MIT License](LICENSE)**.

Please note that certain dependencies operate under separate license terms:

- [**nvdiffrast**](https://github.com/NVlabs/nvdiffrast): Utilized for rendering generated 3D assets. This package is governed by its own [License](https://github.com/NVlabs/nvdiffrast/blob/main/LICENSE.txt).

- [**nvdiffrec**](https://github.com/NVlabs/nvdiffrec): Implements the split-sum renderer for PBR materials. This package is governed by its own [License](https://github.com/NVlabs/nvdiffrec/blob/main/LICENSE.txt).

## 📚 Citation

If you find GenRecon useful, please consider citing:

```bibtex
@article{schmid2026genreconbridginggenerativepriors,
  author={Schmid, Katharina and von Lützow, Nicolas and Hladký, Jozef and Dai, Angela and Nießner, Matthias},
  title={GenRecon: Bridging Generative Priors for Multi-View 3D Scene Reconstruction},
  year={2026},
  eprint={2605.23888},
  archivePrefix={arXiv}
}
```
