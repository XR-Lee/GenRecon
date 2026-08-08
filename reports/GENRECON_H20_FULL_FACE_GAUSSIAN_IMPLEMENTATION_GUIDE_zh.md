# GenRecon 全面一基元 Gaussian：H20 实现完整指南

## 0. 文档目标

本文定义一个可直接实施的工程方案：在单张 NVIDIA H20 上，为 GenRecon `lecture_room` 正式 PLY 的每个三角面绑定一个叶级 Gaussian，并从原始 COLMAP 相机和 RGB 图像训练其外观参数。

最终不可变目标：

- 正式 mesh 有 `26,854,772` 个有效三角面。
- 叶级 Gaussian 数量严格为 `26,854,772`。
- 每个 face 恰好对应一个叶级 Gaussian。
- 正式 mesh 顶点、面、顺序和哈希不改变。
- Gaussian 只作为 appearance / novel-view rendering layer。
- Exact 模式使用所有当前视角可见叶节点。
- Realtime 模式可以使用 LOD 父节点，但资产中保留全部叶节点。
- 16 个 input 相机参与训练；7 个 heldout 相机只用于最终评测。

本文不把 Gaussian 当作新的几何真值，也不允许它替代正式 PLY/GLB。

## 1. 固定输入与不可变约束

### 1.1 正式资产

```text
Mesh:
/home/campus.ncl.ac.uk/nxl51/genrecon/outputs/eth3d/lecture_room/final/mesh.ply

PBR GLB:
/home/campus.ncl.ac.uk/nxl51/genrecon/outputs/eth3d/lecture_room/final/scene.glb

COLMAP:
/home/campus.ncl.ac.uk/nxl51/genrecon/data/eth3d/lecture_room/colmap/

RGB:
/home/campus.ncl.ac.uk/nxl51/genrecon/data/eth3d/lecture_room/rgb/
```

固定哈希：

```text
mesh.ply
b24fe5628305e120b945dd72503620ab6984016b31a70a1d26f3b8aba4afcd83

scene.glb
6fd4000afd4287ea947d53de6a4dda92ae483136693946efb78b19df67db4a7d
```

任何实现阶段都不得写入 `outputs/eth3d/lecture_room/final/`。

### 1.2 Mesh 实测规模

```text
vertices:             12,732,795
faces:                26,854,772
zero-area faces:      0
total area:           212.937271 m^2
face area P10:        5.030 mm^2
face area median:     8.635 mm^2
face area P90:        9.403 mm^2
max edge P10:         4.743 mm
max edge median:      5.884 mm
max edge P90:         6.151 mm
max edge maximum:    16.946 mm
```

这些三角面尺度较均匀。一面一个 Gaussian 是有意义的叶级表面采样，而不是在大三角形上进行过稀采样。

### 1.3 相机拆分

训练相机只能是以下 16 张：

```text
DSC_0899.JPG
DSC_0901.JPG
DSC_0902.JPG
DSC_0904.JPG
DSC_0905.JPG
DSC_0907.JPG
DSC_0914.JPG
DSC_0915.JPG
DSC_0917.JPG
DSC_0919.JPG
DSC_0920.JPG
DSC_0922.JPG
DSC_0923.JPG
DSC_0925.JPG
DSC_0926.JPG
DSC_0927.JPG
```

以下 7 张必须严格 heldout：

```text
DSC_0900.JPG
DSC_0903.JPG
DSC_0906.JPG
DSC_0916.JPG
DSC_0918.JPG
DSC_0921.JPG
DSC_0924.JPG
```

Heldout 相机不得参与：

- loss；
- visibility cache；
- SH degree 选择；
- scale/opacity 调参；
- LOD 构建或蒸馏视角；
- early stopping；
- hyperparameter selection；
- exposure/white-balance fitting。

## 2. 总体架构

完整系统分为两条并行路径：

```text
                    +------------------------------+
                    | formal mesh.ply              |
                    | authoritative geometry       |
                    +---------------+--------------+
                                    |
                  +-----------------+------------------+
                  |                                    |
          mesh raster pass                     face Gaussian builder
      face-ID / depth / mask              mean / rotation / base scale
                  |                                    |
                  +-----------------+------------------+
                                    |
                          active visible face IDs
                                    |
                          indexed appearance gather
                                    |
                         gsplat Gaussian rasterizer
                                    |
                       RGB / alpha / expected depth
                                    |
                       mesh-depth-aware compositing
```

核心原则：

1. Mesh 决定几何、first hit、遮挡和 face identity。
2. Gaussian 决定训练照明下的颜色、透明度和视角相关外观。
3. 每次训练只 rasterize 当前 image/tile 可能贡献的 faces。
4. 全部 26.85M 叶级参数保存在 H20 或分片 checkpoint 中。
5. 不使用自由 densification，不创建脱离 face 的普通 3DGS 点。

## 3. Face-bound Gaussian 数学定义

### 3.1 一对一身份

令正式 mesh face 顺序为：

$$
F = \{f_i\}_{i=0}^{N_f-1}, \qquad N_f=26{,}854{,}772.
$$

定义：

$$
g_i \leftrightarrow f_i.
$$

第一版直接要求 `gaussian_id == face_id`。如果后续为了空间排序对 Gaussian 重排，必须保存双向置换：

```text
face_to_gaussian[int32, N_f]
gaussian_to_face[int32, N_f]
```

不得依赖隐式、不可复现的排序。

### 3.2 Mean

对三角面顶点 $v_0,v_1,v_2$，基础中心固定为质心：

$$
\mu=\frac{v_0+v_1+v_2}{3}.
$$

V1 不学习 barycentric offset，也不学习 normal offset。这样 Gaussian mean 严格由 mesh 派生。

### 3.3 Tangent covariance

定义面法向：

$$
n=\frac{(v_1-v_0)\times(v_2-v_0)}{\|(v_1-v_0)\times(v_2-v_0)\|}.
$$

均匀三角形面积分布的协方差为：

$$
C_f=\frac{1}{12}\sum_{j=0}^{2}(v_j-\mu)(v_j-\mu)^T.
$$

由于 $C_f$ 在法向方向退化，添加固定法向厚度：

$$
\Sigma_f=\gamma^2 C_f+\epsilon_n^2nn^T.
$$

推荐初始化：

```text
coverage_gain gamma: 1.0
normal_scale epsilon_n:
    clamp(0.02 * min(tangent_sigma_1, tangent_sigma_2), 0.00002, 0.00020) meters
```

`gamma=1.0` 时，等边三角形顶点约位于 Gaussian 的 $2.83\sigma$ 位置，接近标准 3DGS 的 $3\sigma$ 截断。邻接 faces 的 splats 会在边缘重叠。

不要直接把最长边长度当成 $1\sigma$，否则会产生严重跨轮廓 bleed。

### 3.4 Quaternion 和 scale

对 $C_f$ 的切向 $2\times2$ 子空间进行特征分解，得到主方向 $u_1,u_2$ 和特征值 $\lambda_1,\lambda_2$。

构造右手旋转矩阵：

$$
R=[u_1,u_2,n], \qquad \det(R)>0.
$$

基础 scale：

$$
s_1=\gamma\sqrt{\lambda_1},\qquad
s_2=\gamma\sqrt{\lambda_2},\qquad
s_3=\epsilon_n.
$$

转换为 `gsplat` quaternion 时使用 **wxyz** 顺序。转换为 glTF `KHR_gaussian_splatting` 时使用 **xyzw** 顺序。

稳定性要求：

- 特征值最小值 clamp 到 `1e-12 m^2`。
- quaternion 归一化。
- quaternion 符号规范化为 `w >= 0`，保证导出可复现。
- 如果 `cross(u1, u2) dot n < 0`，翻转 `u2`。

### 3.5 可学习 footprint

V1 可以学习两个切向 scale correction，但不改变 mean 和 rotation：

$$
\hat{s}_k=s_k\exp(0.45\tanh(a_k)), \qquad k\in\{1,2\}.
$$

这把每个切向 scale 限制在基础值的大约 `0.638x` 至 `1.568x`。法向 scale 固定。

如果出现轮廓 bleed，先收紧 correction 范围，不要引入自由 mean。

### 3.6 Appearance

每个 face 存储：

```text
opacity_logit:      [1]
SH coefficients:    [(degree + 1)^2, 3]
scale_delta:        [2]
trained flag:       [1 bit or uint8]
view mask:          [uint16]
```

源 PLY 的 `red/green/blue` 顶点属性用于初始化 face RGB：

$$
c_f=\frac{c(v_0)+c(v_1)+c(v_2)}{3}.
$$

按照 3DGS / gsplat 约定初始化 degree-0 SH：

$$
\mathrm{SH}_0=\frac{c_f-0.5}{C_0},
\qquad C_0=0.28209479177387814.
$$

其余 SH 初始化为零。

Opacity 初始化：

```text
initial activated opacity = clamp(face_alpha * 0.95, 0.05, 0.995)
opacity_logit = log(p / (1 - p))
```

### 3.7 自适应 SH degree

场景只有 16 个训练视角。对单个 face 直接拟合 degree-3 的 16 个 SH basis 很容易欠约束。

根据准确 first-hit 训练视角数量选择有效 degree：

```text
view_count 0..3:  degree 0
view_count 4..8:  degree 1
view_count 9..16: degree 2
```

这只是必要的 observation-count gate。Degree 1/2 还必须检查对应观察方向 SH design matrix 的条件数；条件数过大时降低一级。Degree 3 默认关闭。只有满足以下全部条件时才开启：

- face 在 16 个训练视角中全部有有效 first-hit observation；
- 观察方向设计矩阵条件数低于预设阈值，例如 `1e3`；
- degree-3 在 input-only validation split 上优于 degree-2；
- heldout 不参与该选择。

内部可以统一分配 degree-3 tensor，但通过 gradient mask 把不允许的系数固定为零。

## 4. H20 容量设计

### 4.1 静态参数估算

对 26,854,772 个 Gaussians：

| 表示 | 每基元 | 总量 |
|---|---:|---:|
| 显式 FP32 SH0 | 56B | 1.40GiB |
| 显式 FP32 SH3 | 236B | 5.90GiB |
| FP16 SH3 近似内部布局 | 112B | 2.80GiB |
| 32B 量化运行时 | 32B | 0.80GiB |
| 48B 量化运行时 | 48B | 1.20GiB |

### 4.2 Optimizer 估算

若训练 SH3、opacity 和两个 scale delta，共约 51 个标量：

| 状态 | 总量 |
|---|---:|
| 参数 + gradient + FP32 Adam m/v | 20.41GiB |
| 混合精度参数/gradient + FP32 m/v | 15.31GiB |

这些数字不含 rasterizer intermediate、排序、图像和 framebuffer。

### 4.3 H20 推荐模式

首先执行：

```bash
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
```

以下配置按单张 96GB H20 设计：

- 全量 appearance master parameters 常驻 GPU。
- 全量 Adam moments 常驻 GPU。
- mean/quaternion/base scale 可以常驻 GPU，约额外 1GiB 量级。
- 每个 iteration 只 gather 当前 view/tile active face rows。
- 只为 active rows 构造 autograd graph。
- 使用 indexed Adam 将 active gradients 写回全量 master tensors。
- `gsplat.rasterization(..., packed=True)`。

即使 H20 能容纳全量 model，也不要每一步投影全部 26.85M Gaussians。主要原因是投影、tile intersection 和排序成本，而不是模型参数本身。

### 4.4 主机和磁盘建议

```text
Host RAM:     >= 128GB recommended
Local SSD:    >= 200GB free
GPU VRAM:     full H20 instance, MIG disabled if possible
```

磁盘预算包括：

- 正式输入只读副本或链接；
- geometry cache；
- visibility caches；
- appearance shards；
- Adam states；
- 至少两个原子 checkpoint generations；
- 23 视角 full-resolution renders；
- KHR/SPZ exports。

## 5. 推荐代码结构

在不改动现有 GenRecon inference 的前提下新增独立模块：

```text
mesh_gaussian/
  __init__.py
  config.py
  colmap.py
  ply_io.py
  face_geometry.py
  face_visibility.py
  active_set.py
  indexed_adam.py
  model.py
  rasterizer.py
  compositor.py
  losses.py
  checkpoint.py
  lod.py
  export_khr.py
  export_spz.py
  metrics.py

 tools/mesh_gaussian/
  preflight.py
  prepare.py
  precompute_visibility.py
  render_initial.py
  train.py
  render.py
  evaluate.py
  build_lod.py
  export.py
  validate.py

 tests/mesh_gaussian/
  test_face_geometry.py
  test_camera_projection.py
  test_visibility.py
  test_indexed_adam.py
  test_checkpoint.py
  test_export.py
  test_small_scene.py
```

不要把该分支直接塞进 `reconstruct_scene.py`。它是 GenRecon mesh 的后处理/appearance optimization pipeline，应有独立入口和独立环境。

## 6. 环境

### 6.1 建议独立环境

示例：

```bash
cd /home/campus.ncl.ac.uk/nxl51/genrecon
python3.11 -m venv data/mesh-gaussian-h20-env
source data/mesh-gaussian-h20-env/bin/activate

python -m pip install --upgrade pip wheel setuptools ninja
python -m pip install \
  torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu126

python -m pip install gsplat==1.5.3
python -m pip install \
  numpy scipy pillow opencv-python-headless imageio zstandard \
  plyfile trimesh safetensors pyyaml tqdm tensorboard pytest \
  scikit-image pytorch-msssim lpips pygltflib

python -m pip install \
  'git+https://github.com/NVlabs/nvdiffrast.git'
```

如果 H20 机器使用其他驱动/CUDA，保持 PyTorch、CUDA 和 gsplat wheel/JIT 一致，不要机械复制版本。

安装完成后固定环境：

```bash
python -m pip freeze > outputs/eth3d/lecture_room/mesh_gaussian_full/environment.freeze.txt
python - <<'PY'
import torch, gsplat
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name())
print(torch.cuda.get_device_properties(0).total_memory)
print(gsplat.__version__)
PY
```

记录 gsplat tag/commit。本文 API 以 `gsplat v1.5.3` 为基线。

### 6.2 运行环境变量

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export OMP_NUM_THREADS=16
```

第一轮关闭 `torch.compile` 和 CUDA graph。正确性稳定后再单独 benchmark。

## 7. 配置文件

建议配置：

```yaml
experiment:
  name: lecture_room_full_face_gaussian
  seed: 42
  output_root: outputs/eth3d/lecture_room/mesh_gaussian_full

inputs:
  mesh: outputs/eth3d/lecture_room/final/mesh.ply
  mesh_sha256: b24fe5628305e120b945dd72503620ab6984016b31a70a1d26f3b8aba4afcd83
  glb: outputs/eth3d/lecture_room/final/scene.glb
  glb_sha256: 6fd4000afd4287ea947d53de6a4dda92ae483136693946efb78b19df67db4a7d
  images_root: data/eth3d/lecture_room/rgb
  cameras_txt: data/eth3d/lecture_room/colmap/cameras.txt
  images_txt: data/eth3d/lecture_room/colmap/images.txt
  points3D_txt: data/eth3d/lecture_room/colmap/points3D.txt

split:
  source: outputs/eth3d/lecture_room/final/cameras.json
  deduplicate_by_basename: true
  expected_input_views: 16
  expected_heldout_views: 7
  development_validation: farthest_camera_centers_3_seed42
  final_training: all_16_input_views

geometry:
  one_gaussian_per_face: true
  center: centroid
  coverage_gain: 1.0
  normal_scale_ratio: 0.02
  normal_scale_min_m: 0.00002
  normal_scale_max_m: 0.00020
  learn_mean: false
  learn_rotation: false
  learn_normal_offset: false
  learn_tangent_scale: true
  tangent_scale_log_range: 0.45

appearance:
  color_space: srgb_rec709_display
  max_sh_degree: 2
  experimental_sh3: false
  opacity_init_multiplier: 0.95
  pbr_fallback_for_untrained: true

visibility:
  face_id_render_widths: [1600, 2400, 6211]
  fullres_tile_size: 1024
  tile_halo: 64
  store_face_view_mask_uint16: true
  use_heldout: false

rasterizer:
  backend: gsplat
  packed: true
  sparse_grad: false
  rasterize_mode: classic
  near_plane_m: auto
  far_plane_m: auto
  radius_clip_coarse_px: 0.25
  radius_clip_exact_px: 0.0
  render_mode: RGB+ED

training:
  indexed_adam: true
  betas: [0.9, 0.999]
  eps: 1.0e-8
  sh0_lr: 0.0025
  shN_lr: 0.000125
  opacity_lr: 0.001
  scale_lr: 0.0002
  exposure_lr: 0.001
  lr_final_multiplier: 0.1
  checkpoint_every_steps: 1000
  checkpoint_every_minutes: 20

stages:
  - name: sh0_coarse
    steps: 8000
    width: 1600
    tile_size: null
    sh_degree: 0
  - name: sh12_medium
    steps: 15000
    width: 2400
    tile_size: null
    sh_degree: adaptive
  - name: fullres_tiles
    steps: 10000
    width: 6211
    tile_size: 1024
    tile_halo: 64
    sh_degree: adaptive

loss:
  l1: 0.8
  dssim: 0.2
  alpha: 0.10
  depth: 0.05
  scale_reg: 0.001
  opacity_entropy: 0.0001
  sh_high_order_reg: 0.0001
  exposure_reg: 0.001

checkpoint:
  faces_per_shard: 1048576
  keep_last: 2
  atomic: true

lod:
  enabled_after_exact_validation: true
  leaf_faces_per_page: 131072
  target_fanout: 8
  exact_mode_uses_leaves: true
```

所有配置、脚本 commit 和环境 freeze 必须进入最终 manifest。

## 8. Stage 0：Preflight

目标：在分配大 tensor 前拒绝错误输入。

必须检查：

1. 正式 PLY/GLB SHA-256。
2. PLY header 的 vertex/face count。
3. 顶点和 face index 有限、合法。
4. 面积大于零。
5. PLY face 顺序可稳定复现。
6. COLMAP 为 undistorted PINHOLE/SIMPLE_PINHOLE。
7. RGB basename 与 `images.txt` 完全匹配。
8. 输入/heldout split 精确为 16/7。
9. H20 可见，显存满足配置。
10. 输出目录不是 `final/`。
11. 磁盘空间满足预算。

推荐 CLI 契约：

```bash
python -m tools.mesh_gaussian.preflight \
  --config configs/mesh_gaussian/lecture_room_h20.yaml
```

输出：

```text
mesh_gaussian_full/preflight.json
mesh_gaussian_full/environment.freeze.txt
mesh_gaussian_full/input_manifest.json
```

`preflight.json` 至少包含：

```json
{
  "mesh_sha256": "...",
  "glb_sha256": "...",
  "vertices": 12732795,
  "faces": 26854772,
  "zero_area_faces": 0,
  "input_views": 16,
  "heldout_views": 7,
  "gpu_name": "NVIDIA H20",
  "gpu_memory_bytes": 0,
  "disk_free_bytes": 0
}
```

## 9. Stage 1：Geometry cache

### 9.1 保持 PLY face 顺序

不要使用会合并顶点、修复法向或重排 faces 的 mesh processing。

推荐使用 `plyfile` 直接读取：

```python
ply = PlyData.read(mesh_path)
vertices = np.stack([
    ply["vertex"]["x"],
    ply["vertex"]["y"],
    ply["vertex"]["z"],
], axis=-1).astype(np.float32)
faces = np.vstack(ply["face"]["vertex_indices"]).astype(np.int32)
```

断言：

```python
assert vertices.shape == (12_732_795, 3)
assert faces.shape == (26_854_772, 3)
assert faces.min() >= 0
assert faces.max() < len(vertices)
```

### 9.2 分批构建

每批建议 `250K` 至 `1M` faces。输出：

```text
geometry/
  vertices.f32
  faces.i32
  means.f32
  quats_wxyz.f32
  base_scales.f32
  face_rgb.u8
  face_alpha.u8
  face_area.f32
  page_id.i32
  geometry_manifest.json
```

可以使用 NumPy memmap，避免 Python pickle 和单个超大 `torch.save`。

### 9.3 参考伪代码

```python
for start in range(0, num_faces, batch_faces):
    ids = slice(start, min(start + batch_faces, num_faces))
    tri = vertices[faces[ids]]                  # [B, 3, 3]
    center = tri.mean(axis=1)                   # [B, 3]

    e01 = tri[:, 1] - tri[:, 0]
    e02 = tri[:, 2] - tri[:, 0]
    normal = normalize(np.cross(e01, e02))

    t0 = normalize(e01)
    t1 = normalize(np.cross(normal, t0))

    centered = tri - center[:, None, :]
    x = np.sum(centered * t0[:, None, :], axis=-1)
    y = np.sum(centered * t1[:, None, :], axis=-1)

    c00 = np.sum(x * x, axis=1) / 12.0
    c01 = np.sum(x * y, axis=1) / 12.0
    c11 = np.sum(y * y, axis=1) / 12.0

    # Closed-form eigendecomposition of symmetric 2x2 matrix.
    angle = 0.5 * np.arctan2(2.0 * c01, c00 - c11)
    lambda_max, lambda_min = eigenvalues_2x2(c00, c01, c11)

    u0 = np.cos(angle)[:, None] * t0 + np.sin(angle)[:, None] * t1
    u1 = -np.sin(angle)[:, None] * t0 + np.cos(angle)[:, None] * t1
    enforce_right_handed(u0, u1, normal)

    rotation = np.stack([u0, u1, normal], axis=-1)
    quat_wxyz = rotation_matrix_to_quaternion(rotation)

    tangent0 = coverage_gain * np.sqrt(np.maximum(lambda_max, 1e-12))
    tangent1 = coverage_gain * np.sqrt(np.maximum(lambda_min, 1e-12))
    normal_s = np.clip(
        normal_scale_ratio * np.minimum(tangent0, tangent1),
        normal_scale_min_m,
        normal_scale_max_m,
    )

    scales = np.stack([tangent0, tangent1, normal_s], axis=-1)
    write_batch(ids, center, quat_wxyz, scales)
```

### 9.4 Geometry cache 验收

随机和分层采样至少一百万 faces，验证：

- mean 等于原 face centroid，绝对误差不超过 `1e-6m`。
- quaternion finite 且归一化误差不超过 `1e-5`。
- scale 全正。
- reconstructed covariance 与目标 covariance 相对误差不超过 `1e-4`。
- `gaussian_id == face_id`。
- 全量输出数量严格为 `26,854,772`。

## 10. Stage 2：训练视角 First-hit visibility

### 10.1 为什么必须使用 mesh ID pass

普通 3DGS 只做正深度和 frustum projection，不能确定某个 Gaussian 是否位于第一表面。这里已有权威 mesh，因此应先 rasterize mesh：

```text
mesh -> per-pixel face ID + z depth + mask
```

然后只让这些 first-hit faces 对该相机或 tile 参与 Gaussian rasterization。

这会显著降低 active Gaussian 数，也抑制后表面透出。

### 10.2 Camera conventions

训练始终使用正式 PLY 的 COLMAP world coordinates。

COLMAP `images.txt` 提供：

$$
x_{cam}=R x_{world}+t.
$$

`gsplat.rasterization` 接受 world-to-camera `viewmats`，因此可以直接使用该 $4\times4$ 矩阵。不要使用 GLB 轴变换参与训练。

相机内参：

$$
K=\begin{bmatrix}
f_x&0&c_x\\
0&f_y&c_y\\
0&0&1
\end{bmatrix}.
$$

对缩放图像，按宽高比例缩放 $K$ 的前两行。对 crop/tile，必须：

$$
c'_x=c_x-x_0,\qquad c'_y=c_y-y_0.
$$

`nvdiffrast`需要clip-space坐标。对COLMAP/OpenCV的$+z$向前、图像$+y$向下约定，可以使用$w_{clip}=z_{cam}$和下式投影：

$$
P=
\begin{bmatrix}
2f_x/W & 0 & 2c_x/W-1 & 0\\
0 & -2f_y/H & 1-2c_y/H & 0\\
0 & 0 & (f+n)/(f-n) & -2fn/(f-n)\\
0 & 0 & 1 & 0
\end{bmatrix}.
$$

其中$n/f$分别为near/far。最终：

$$
x_{clip}=P\,T_{world\rightarrow camera}\,x_{world}.
$$

必须用已知3D点做CPU像素投影与nvdiffrast结果的逐像素测试，特别检查图像$y$轴、pixel-center约定和矩阵乘法方向。可直接复用`tools/evaluate_view_fidelity.py`中的COLMAP解析和当前renderer已经验证过的相机约定，避免重新解释quaternion。

### 10.3 nvdiffrast face ID

使用 `nvdiffrast` rasterize 正式 mesh。其 raster 输出最后通道保存 `triangle_id + 1`；背景为零。必须通过小场景单元测试确认当前版本行为。

输出每个训练相机、每个分辨率的：

```text
visibility/<width>/<image>/
  face_id.u32.zst
  depth.f32.zst
  mask.png
  camera.json
```

### 10.4 Face view mask

训练视角刚好为 16 个，可以给每个 face 保存一个 `uint16`：

```python
face_view_mask[face_id] |= np.uint16(1 << train_view_index)
```

定义：

```python
trained_mask = face_view_mask != 0
view_count = popcount(face_view_mask)
```

大小约 `51.2MiB`，非常适合全量保存。

### 10.5 Tile active index

Full-resolution 阶段使用 `1024x1024` central tile 和 `64px` halo。

每个 tile 保存扩展区域中的 unique face IDs：

```text
visibility/6211/tile_index/
  offsets.i64
  face_ids.i32
  tile_meta.json
```

Loss 只在 central tile 计算；halo 仅用于收集可能跨入 central tile 的 splats。

不要求构建全 mesh 邻接图。图像空间 halo 已经覆盖大多数边界贡献。

### 10.6 Near/far

从 16 个 input 相机下的 mesh positive z-depth 分布自动计算：

```text
near = max(0.01m, 0.5 * positive_depth_P0.1)
far  = 1.2 * positive_depth_P99.9
```

把最终数值写入 manifest，不在训练过程中变化。

### 10.7 Visibility 验收

- 只读取 16 个 input views。
- `face_view_mask` 中不得有第 16 bit 之外的值。
- heldout basename 不得出现在 visibility manifest。
- 每个 face ID 小于 `26,854,772`。
- face-ID mask 与 mesh depth mask 完全一致。
- 将 face ID 伪彩色渲染并人工检查至少 6 个关键视角。

## 11. Stage 3：未训练 Gaussian 初始化渲染

训练前必须先证明 geometry、camera 和 rasterizer 一致。

### 11.1 Active gather

```python
face_ids = load_active_face_ids(view_or_tile)
means = means_master[face_ids]
quats = quats_master[face_ids]
scales = base_scales_master[face_ids]
sh = sh_master[face_ids]
opacities = sigmoid(opacity_logits_master[face_ids])
```

### 11.2 gsplat 调用

基于 `gsplat v1.5.3`：

```python
render, alpha, meta = gsplat.rasterization(
    means=means,
    quats=quats,                  # wxyz
    scales=scales,
    opacities=opacities,
    colors=sh,
    viewmats=world_to_camera[None],
    Ks=K[None],
    width=width,
    height=height,
    sh_degree=0,
    packed=True,
    sparse_grad=False,
    rasterize_mode="classic",
    render_mode="RGB+ED",
    near_plane=near,
    far_plane=far,
    radius_clip=0.0,
)
```

输出最后一通道为 expected depth。确认当前 gsplat 版本的返回维度后封装，不在业务代码到处直接索引。

### 11.3 Mesh-depth-aware compositing

第一版至少执行：

```text
final_alpha = gaussian_alpha * mesh_mask
final_rgb   = gaussian_rgb * mesh_mask
```

并报告：

```text
abs(gaussian_expected_depth - mesh_depth)
```

更严格版本应在 Gaussian raster kernel 中读取 mesh depth，逐贡献拒绝：

$$
|z_g-z_{mesh}|>\tau(z).
$$

建议：

$$
\tau(z)=\max(0.005m, 0.002z).
$$

在没有 kernel depth gate 前，必须只 rasterize first-hit face IDs，不能传入整个 frustum 的所有 faces。

### 11.4 初始化渲染 gate

在 width `1600` 的 16 个 input views 上：

- Gaussian alpha 与 mesh mask IoU 至少 `0.98`。
- mesh mask 内无系统性棋盘孔洞。
- expected-depth 对 mesh depth 的中位差不超过 `5mm`。
- P90 不超过 `15mm`。
- 无 NaN/Inf。
- face count 和 mapping 不变化。

如果未通过，不开始训练。先修复 covariance、camera、active set 或 compositing。

## 12. Stage 4：训练实现

### 12.1 不使用普通 densification

明确禁用：

- clone；
- split；
- prune；
- opacity-based deletion；
- mean position update；
- free Gaussian creation。

每个 face 的叶级 Gaussian 从开始到结束始终存在。

### 12.2 H20 上的 master tensors

建议全量常驻：

```text
fixed:
  means             float32 [N, 3]
  quats_wxyz        float32 [N, 4]
  base_scales       float32 [N, 3]
  face_view_mask    uint16  [N]

learned:
  sh0               float32 [N, 1, 3]
  shN               float32 [N, K-1, 3]
  opacity_logits    float32 [N]
  scale_delta       float32 [N, 2]

optimizer:
  m/v per learned tensor
  row_step_sh0      uint32 [N]
  row_step_shN      uint32 [N]
  row_step_opacity  uint32 [N]
  row_step_scale    uint32 [N]
```

不要让每个 tile 的 gather 对全量 parameter 自动分配 dense gradient。使用 active leaf tensors：

```python
active = master.index_select(0, face_ids).detach().requires_grad_(True)
```

反向后通过 `IndexedAdam` 只更新 `face_ids` 对应行。

### 12.3 IndexedAdam

每行维护自己的 step，避免低可见 face 和高可见 face 共用错误 bias correction：

```python
@torch.no_grad()
def indexed_adam_update(param, m, v, row_step, ids, grad, lr, beta1, beta2, eps):
    ids, grad = coalesce_duplicate_rows(ids, grad)
    row_step[ids] += 1
    t = row_step[ids].float()

    m_i = m[ids].mul(beta1).add(grad, alpha=1.0 - beta1)
    v_i = v[ids].mul(beta2).addcmul(grad, grad, value=1.0 - beta2)

    m[ids] = m_i
    v[ids] = v_i

    m_hat = m_i / (1.0 - beta1 ** t).reshape(broadcast_shape)
    v_hat = v_i / (1.0 - beta2 ** t).reshape(broadcast_shape)
    param[ids] -= lr * m_hat / (torch.sqrt(v_hat) + eps)
```

要求：

- `ids` 更新前排序并去重。
- 重复 ID 的 gradient 求和。
- 对不允许的高阶 SH 系数把 gradient 置零。
- 每个 optimizer parameter group 使用独立的 per-row step；SHN 后期开启时不能继承 SH0 的 step。
- 未训练 face 的各组 row step 始终为零。
- scale update 后通过有界参数化，不做硬截断 master raw value。

先用 CPU/小 tensor 与 `torch.optim.Adam` 做逐步数值对照。

### 12.4 Camera exposure

可选每训练相机学习 RGB gain/bias：

$$
\hat{I}=\exp(g_c)\odot I+b_c.
$$

初始化均为零。强正则到 identity。Heldout 不创建可学习 exposure；评测时使用 identity。

Exposure 是否启用只能根据 input 内部 development validation 判断。不得在看到 heldout 结果后关闭、重调或重新训练同一候选。

### 12.5 Loss mask

定义：

- $M$：mesh valid mask。
- $A$：Gaussian alpha。
- $I$：原图。
- $\hat{I}$：Gaussian render。
- $D_m$：mesh z-depth。
- $D_g$：Gaussian expected depth。

RGB 主 loss 只在 mesh valid mask 内计算。另在 mesh silhouette 外侧小 ring 计算 alpha 抑制，防止 splat bleed。

### 12.6 Loss

建议：

$$
\mathcal{L}_{rgb}=0.8\,\|M\odot(\hat{I}-I)\|_1
+0.2\,(1-\mathrm{SSIM}(M\odot\hat{I},M\odot I)).
$$

Alpha 只在 mesh foreground 与其外侧约 8px silhouette ring 内计算，并对正负像素做类别平衡，避免大面积背景支配 loss：

$$
\mathcal{L}_{alpha}=\mathrm{BalancedBCE}(A,M).
$$

Depth 只在 $M=1$ 且 $A>0.1$ 的像素使用 robust normalized error：

$$
\mathcal{L}_{depth}=\operatorname{mean}_{M}
\left[\sqrt{\left(\frac{D_g-D_m}{0.05}\right)^2+10^{-6}}\right].
$$

Scale regularization：

$$
\mathcal{L}_{scale}=\|a_1\|_2^2+\|a_2\|_2^2.
$$

高阶 SH regularization：

$$
\mathcal{L}_{SH}=\sum_{l=1}^{L}l^2\|c_l\|_2^2.
$$

总 loss：

$$
\mathcal{L}=\mathcal{L}_{rgb}
+0.10\mathcal{L}_{alpha}
+0.05\mathcal{L}_{depth}
+10^{-3}\mathcal{L}_{scale}
+10^{-4}\mathcal{L}_{opacity\_entropy}
+10^{-4}\mathcal{L}_{SH}
+10^{-3}\mathcal{L}_{exposure}.
$$

这些是起点，不是无需验证的常数。调权重只能使用 input views 内部拆分，不能看 heldout 后回调。

### 12.7 Development 与最终训练协议

从 16 个 input camera centers 中，用 seed 42 的确定性 farthest-point sampling 选择 3 个 development-validation views，其余 13 个作为 development-train。将实际 basename 写入 manifest，不只记录算法名称。

开发阶段：

- 只用 13 个 development-train views 更新参数。
- 只用 3 个 development-validation views选择 loss 权重、SH degree、early stopping、exposure 和 scale 范围。
- Heldout 7 views 完全不读取。

配置冻结后：

1. 写入 frozen config hash。
2. 从相同 PBR/SH0 初始化重新开始。
3. 使用全部 16 个 input views 做最终训练。
4. 按预先冻结的 steps/loss/degree 结束，不用 heldout early stopping。
5. 最终 checkpoint 固定后才第一次渲染 7 个 heldout views。

Heldout 是终局审计。若结果失败，可以提出新的预注册实验，但不能根据 heldout 结果回调当前候选并仍把同一 7 views 称为 heldout。

### 12.8 训练阶段

#### Stage A：SH0 coarse

```text
resolution width: 1600
steps:            8,000
train:            SH0, opacity, bounded tangent scale
freeze:           SH higher orders, geometry
radius_clip:      0.25 px
```

前 500 steps 可固定 opacity 和 per-face scale，只训练 SH0，先稳定颜色。

#### Stage B：Adaptive SH1/SH2

```text
resolution width: 2400
steps:            15,000
train:            SH0, allowed SH1/2, opacity, tangent scale
radius_clip:      0.10 or 0.0 px
```

每个 face 的有效 degree 由 input-only `face_view_mask` 固定，训练期间不变化。

#### Stage C：Full-resolution tiled refinement

```text
resolution:       6211 x 4139
central tile:     1024 x 1024
halo:             64 px
steps:            10,000
radius_clip:      0.0
```

Tile sampling 应按以下混合：

- 50% 均匀覆盖训练图像/tiles；
- 30% mesh/PBR RGB 高误差 tile；
- 20% silhouette、深度边缘和小物体 tile。

误差分布只能由 input images 构建。

### 12.9 Learning rates

推荐初始值：

```text
SH0:             2.5e-3
SH higher:       1.25e-4
opacity logit:   1.0e-3
scale raw:       2.0e-4
camera exposure: 1.0e-3
```

每阶段使用 cosine decay 到初始值的 `0.1x`。不要对固定 mean/quaternion/base scale 创建 optimizer state。

### 12.10 训练日志

每 100 steps：

```text
loss components
active face count
visible splat count from gsplat meta
alpha mean/P10/P90
scale multiplier P1/P50/P99
SH norm by degree
mesh/Gaussian depth median/P90
GPU allocated/reserved/peak
iteration time
```

每 1,000 steps：

- 保存 checkpoint；
- 渲染固定 4 个 input diagnostic views；
- 不渲染 heldout；
- 检查 NaN/Inf 和 face identity。

## 13. Checkpoint 与恢复

### 13.1 输出结构

```text
outputs/eth3d/lecture_room/mesh_gaussian_full/
  config.resolved.yaml
  input_manifest.json
  preflight.json
  environment.freeze.txt
  geometry/
  visibility/
  checkpoints/
    step_00001000/
      manifest.json
      appearance/
        shard_00000.safetensors
        ...
      optimizer/
        shard_00000.safetensors
        ...
      camera.safetensors
      rng_state.pt
    latest -> step_XXXXXXXX
  renders/
  evaluation/
  lod/
  export/
```

### 13.2 Sharding

建议 `1,048,576` faces/shard，共约 26 shards。每个 shard 使用固定 face ID 区间：

```text
[start_face_id, end_face_id)
```

禁止按当前可见性动态重分片。

### 13.3 原子 checkpoint

写入：

```text
step_XXXXXXXX.tmp/
```

全部文件写完、hash 完成后：

```text
fsync -> rename to step_XXXXXXXX -> atomically update latest symlink
```

恢复时验证：

- mesh hash；
- config hash；
- face ranges；
- tensor shape/dtype；
- shard hash；
- step；
- RNG state；
- environment provenance。

不允许把半写 checkpoint 当作可恢复点。

## 14. Stage 5：Exact rendering

### 14.1 新视角流程

每个相机执行：

1. Rasterize mesh face ID/depth/mask。
2. 提取 first-hit face IDs。
3. 加载或 gather 对应叶级 Gaussians。
4. 使用 `radius_clip=0` 渲染。
5. 使用 mesh mask/depth gate 合成。
6. 输出 RGB、alpha、expected depth 和诊断。

Exact 模式禁止使用 LOD parent 代替叶节点。

### 14.2 Full-resolution tile render

如果单次 6211×4139 raster 超时或内存波动，使用 tile render：

- central tile `1024`；
- halo `64`；
- tile K 修正 principal point；
- 只把 central tile 写入最终图；
- 重叠区域不得平均两次；
- 所有 tile 使用同一背景和颜色空间。

### 14.3 输出

每个相机：

```text
renders/exact/<group>/<image>/
  original.jpg
  gaussian_rgb.png
  gaussian_alpha.png
  gaussian_depth_mm.png
  mesh_depth_mm.png
  depth_difference.png
  comparison.jpg
  metrics.json
```

保留 input/heldout 分组，但训练期间不生成 heldout。

## 15. Stage 6：评测

### 15.1 RGB 指标

与当前项目一致：

- RGB 指标只在有效 mesh mask 内计算。
- Coverage 单独报告。
- 分 input / heldout。
- 指标包括 PSNR、SSIM、LPIPS、MAE。
- 不能用黑背景区域抬高指标。

### 15.2 Coverage

报告：

```text
mesh coverage
gaussian alpha coverage
intersection coverage
union coverage
```

推荐 alpha threshold 同时报告 `0.01/0.1/0.5`，避免单阈值误导。

### 15.3 Depth

比较：

- Gaussian expected depth vs mesh z-depth；
- Gaussian expected depth vs COLMAP sparse observations；
- mesh z-depth vs COLMAP sparse observations。

Gaussian depth 不是独立 surface GT。它用于检查外观层是否漂离绑定mesh。

建议 guardrail：

```text
heldout Gaussian-vs-mesh depth median <= 5 mm
heldout Gaussian-vs-mesh depth P90    <= 15 mm
coverage relative to mesh drop        <= 0.5 percentage point
```

如果使用严格 per-contribution mesh depth gate，误差应更低。

### 15.4 渲染成功标准

至少满足：

- heldout median PSNR 相对 PBR mesh 提高 `>= 0.5dB`，或 heldout median LPIPS 降低 `>= 5%`；
- 不出现两个以上 heldout 视角的明显大面积退化；
- coverage guardrail 通过；
- depth guardrail 通过；
- mesh hash 不变；
- 26.85M face mapping 完整。

即使 RGB 未提升，完整一面一Gaussian资产仍可作为实验结果保留，但不能替代推荐 PBR renderer。

### 15.5 工作台专项

复用现有 worktop/cabinet/anchor heldout 区域：

- worktop RGB/coverage；
- cabinet RGB/coverage；
- Gaussian-vs-mesh depth；
- silhouette edge error；
- 反光台面视角相关改善；
- 柜门孔洞是否被错误透明层掩盖。

## 16. Stage 7：LOD

LOD 是部署优化，不改变叶级目标。

### 16.1 叶级 pages

按 face centroid Morton code 建立 pages：

```text
131,072 faces/page
about 205 leaf pages
```

每个 page 保存：

- world AABB；
- face ID range/list；
- Gaussian parameter range；
- child/parent relation；
- content hash。

### 16.2 Parent hierarchy

使用 octree 或近似 `8:1` fanout。Parent Gaussian 不能取代 leaf identity，只是运行时代理。

初始 parent moment matching：

$$
\mu_p=\frac{\sum_i w_i\mu_i}{\sum_i w_i},
$$

$$
\Sigma_p=\frac{\sum_i w_i\left[\Sigma_i+(\mu_i-\mu_p)(\mu_i-\mu_p)^T\right]}{\sum_i w_i}.
$$

权重可用：

$$
w_i=A_i\alpha_i,
$$

其中 $A_i$ 为 face area。

Parent appearance 需要用 leaf exact render 进行蒸馏。只能使用 input camera poses或在input camera hull内生成的合成视角，不能使用 heldout pose。

### 16.3 LOD 选择

按 parent projected radius / screen-space error 选择：

```text
projected error > threshold: expand children
projected error <= threshold: render parent
```

建议初始 threshold：`0.75px`，并提供 `0.5/1.0/1.5px` 质量档。

### 16.4 两种模式必须共存

```text
--render-mode exact-leaves
--render-mode realtime-lod
```

Exact 用于 fidelity 和最终离线图。LOD 用于交互 viewer/FPS。

### 16.5 相关实现参考

- Hierarchical 3D Gaussians：chunk training、hierarchy merge、VRAM budget viewer。
- CityGaussian：large-scene partition/merge。
- Octree-GS：LOD-structured Gaussian anchors。

这些项目可参考结构，不应直接复制其非商业代码到生产路径而不审查许可证。

## 17. Stage 8：导出

### 17.1 内部推荐格式

最高保真内部资产：

```text
scene.glb                 # 正式 PBR mesh，不变
appearance/pages/*.bin    # 量化或FP16 Gaussian pages
appearance/manifest.json
attachment/face_order.bin # 如果 gaussian order 不等于 face order
lod/hierarchy.bin
```

内部 renderer 可从 mesh face 直接派生 mean/quaternion/base scale，因此无需重复存储所有几何属性。

### 17.2 SPZ

SPZ 适合作为紧凑 Gaussian viewer 资产，但通常不保存 face binding。必须同时保留：

- mesh hash；
- Gaussian-to-face mapping；
- trained/untrained mask；
- coordinate convention；
- SH degree；
- quantization settings。

### 17.3 `KHR_gaussian_splatting`

Khronos Release Candidate 扩展支持：

- `POSITION`；
- rotation；
- scale；
- opacity；
- degree 0..3 SH。

注意单个 GLB chunk 长度受 32-bit 限制。26.85M 个显式 FP32 SH3 Gaussians 约 `5.90GiB`，加上原mesh后不能放入单个标准 GLB。

可选方案：

1. SH0/SH1 的单 hybrid GLB，先实测最终字节数是否低于 4GiB。
2. 多个空间 page GLB + manifest。
3. `.gltf` + 多个外部 buffer。
4. 正式 `scene.glb` + `appearance.spz` 双资产。

推荐优先级：

```text
internal paged format > GLB + SPZ > paged KHR GLBs > monolithic hybrid GLB
```

### 17.4 坐标转换

训练 PLY 在 COLMAP world frame。现有 GLB 使用：

$$
(x,y,z)_{world}\mapsto(x,z,-y)_{gltf}.
$$

这是绕 $x$ 轴的纯旋转，determinant 为正。

导出时必须一致转换：

- mean；
- quaternion；
- covariance/scale frame；
- SH coefficients。

Higher-order SH 不能只转 XYZ；需要对应 Wigner-D rotation。初始 SH0 导出不受该问题影响。SH1/2/3 导出必须用同相机做 world-frame renderer 与 glTF-frame renderer 逐像素对照。

KHR quaternion 顺序是 `xyzw`，gsplat 是 `wxyz`。

### 17.5 激活值

内部可以保存：

- opacity logits；
- raw scale correction；
- raw SH。

导出 KHR 时必须保存：

- linear positive scale；
- normalized quaternion；
- activated opacity $[0,1]$；
- 规范要求的 SH coefficients。

不要把原始 3DGS 的 log-scale/logit opacity 直接写入 KHR 属性。

## 18. 建议 CLI Runbook

以下是建议实现的完整 CLI 契约：

```bash
CONFIG=configs/mesh_gaussian/lecture_room_h20.yaml

python -m tools.mesh_gaussian.preflight \
  --config "$CONFIG"

python -m tools.mesh_gaussian.prepare \
  --config "$CONFIG"

python -m tools.mesh_gaussian.precompute_visibility \
  --config "$CONFIG" \
  --split input

python -m tools.mesh_gaussian.render_initial \
  --config "$CONFIG" \
  --width 1600

python -m tools.mesh_gaussian.train \
  --config "$CONFIG" \
  --stage sh0_coarse

python -m tools.mesh_gaussian.train \
  --config "$CONFIG" \
  --stage sh12_medium \
  --resume latest

python -m tools.mesh_gaussian.train \
  --config "$CONFIG" \
  --stage fullres_tiles \
  --resume latest

python -m tools.mesh_gaussian.render \
  --config "$CONFIG" \
  --checkpoint latest \
  --split all \
  --render-mode exact-leaves

python -m tools.mesh_gaussian.evaluate \
  --config "$CONFIG" \
  --checkpoint latest

python -m tools.mesh_gaussian.build_lod \
  --config "$CONFIG" \
  --checkpoint latest

python -m tools.mesh_gaussian.render \
  --config "$CONFIG" \
  --checkpoint latest \
  --split all \
  --render-mode realtime-lod

python -m tools.mesh_gaussian.export \
  --config "$CONFIG" \
  --checkpoint latest \
  --formats internal,spz,khr-pages

python -m tools.mesh_gaussian.validate \
  --config "$CONFIG" \
  --checkpoint latest \
  --require-full-face-mapping
```

这些命令应支持：

- `--dry-run`；
- `--profile`；
- `--max-steps`；
- `--resume`；
- `--views`；
- `--tiles`；
- `--output-override`；
- SIGTERM 安全 checkpoint。

## 19. 测试计划

### 19.1 单元测试

#### Geometry

- 单个等边三角形的 mean/covariance。
- 任意旋转和平移后 covariance 等变。
- quaternion round trip。
- skinny triangle 数值稳定。
- batch 与 scalar reference 一致。

#### Camera

- COLMAP point 投影与现有 fidelity 工具一致。
- resize 后内参正确。
- tile crop 后 principal point 正确。
- PLY world frame 不经过 GLB axis map。

#### Visibility

- nvdiffrast face ID 与合成三角形预期一致。
- 后表面不进入 first-hit active set。
- heldout 不写入 face view mask。
- uint16 popcount 正确。

#### Optimizer

- IndexedAdam 与普通 Adam 在相同唯一 rows 上数值一致。
- duplicate ID 梯度正确 coalesce。
- row-specific step 正确。
- checkpoint/resume 后下一步 bitwise 或 tolerance 一致。

#### Export

- wxyz/xyzw 转换。
- world/glTF coordinate round trip。
- opacity/scale activation。
- SH0 render round trip。
- GLB/page size上限检查。

### 19.2 集成测试

按规模逐步运行：

```text
1 triangle
2 overlapping triangles
synthetic cube
1K real faces
100K worktop faces
1M scene faces
26.85M full scene
```

每级都必须先通过：

- face mapping；
- initial render；
- 10-step overfit；
- checkpoint/resume；
- render/evaluate；
- finite parameter validation。

### 19.3 Full-scale smoke

在全 26.85M 参数已构建后，先只执行：

```text
1 input view
width 800
10 optimization steps
1 checkpoint
resume 10 more steps
```

确认 H20 显存、active set、optimizer 和 checkpoint 后再开始正式 33K steps。

## 20. 常见故障与定位

### 20.1 黑色孔洞

优先检查：

1. active face IDs 是否遗漏。
2. tile halo 是否不足。
3. covariance $1\sigma$ 是否错误当作整面范围。
4. opacity 激活是否错误。
5. quaternion axis/顺序是否错误。
6. mesh mask 是否被错误腐蚀。

不要先放大到 3x scale；这通常会把孔洞变成轮廓 bleed。

### 20.2 轮廓发光或穿帮

优先：

- mesh mask compositing；
- first-hit-only active set；
- per-contribution mesh depth gate；
- 收紧 scale correction；
- alpha ring loss。

### 20.3 视角切换闪烁

检查：

- SH degree 是否超出 view support；
- page color calibration；
- Gaussian sorting；
- LOD parent/child切换；
- exposure optimization；
- trained/untrained face边界。

### 20.4 Input 很好、heldout 很差

这是过拟合，不是成功。依次：

- 降低 SH degree；
- 提高高阶 SH regularization；
- 关闭 per-camera exposure；
- 固定 opacity/scale；
- 检查相机和颜色空间。

### 20.5 H20 OOM

依次执行：

1. 确认 `packed=True`。
2. 减小 full-resolution tile。
3. 减小 halo，但不低于 32px。
4. 限制每 tile active face count，拆分 subtiles。
5. 保持 master/moments 常驻，只降低 active graph。
6. 把 SH3 降到 SH2。
7. 最后才把 optimizer moments page 到 host。

不要删除叶级 Gaussians 来解决 OOM。

### 20.6 Checkpoint 太慢

- 使用 1M face 固定 shards。
- 后台异步写 CPU snapshot，但 snapshot 完成前不得复用 buffer。
- 只保留最近两个完整 checkpoint。
- 不在每 100 steps 写全量状态。
- appearance-only preview checkpoint 与可恢复 full checkpoint 分开命名。

## 21. 验收清单

### Representation

- [ ] `valid_face_count == 26,854,772`
- [ ] `leaf_gaussian_count == 26,854,772`
- [ ] 每个 face 恰好映射一个 leaf Gaussian
- [ ] 无遗漏、无重复、无非法 face ID
- [ ] 未训练 face 仍存在并有 PBR fallback

### Geometry

- [ ] 正式 PLY hash 不变
- [ ] 正式 GLB hash 不变
- [ ] mean 严格由 face centroid 派生
- [ ] 无 mean/rotation/normal offset 训练
- [ ] scale 有界且 finite

### Split

- [ ] 仅 16 input views 出现在训练 manifest
- [ ] 7 heldout views 未出现在 visibility/cache/optimizer 调参记录
- [ ] heldout 只在 final evaluation 阶段读取

### Training

- [ ] IndexedAdam 与 reference 测试通过
- [ ] 全量 smoke/resume 通过
- [ ] 无 NaN/Inf
- [ ] 每阶段资源 profile 完整
- [ ] checkpoint 原子且可恢复

### Rendering

- [ ] Exact 模式只使用叶节点
- [ ] Realtime 模式报告 LOD 选择
- [ ] RGB/alpha/depth 产物完整
- [ ] mesh-depth gate 或等效保护启用
- [ ] 23 相机均成功渲染

### Evaluation

- [ ] input/heldout 分开报告
- [ ] RGB 只在有效 mesh mask 内计算
- [ ] coverage 单独报告
- [ ] Gaussian-vs-mesh depth 单独报告
- [ ] PBR mesh baseline 同相机对比
- [ ] 没有把更好 RGB 表述成更好几何

### Export

- [ ] 内部 paged asset 可加载
- [ ] SPZ/KHR 导出记录坐标与SH约定
- [ ] 超过 GLB 4GiB 时自动拒绝 monolithic export
- [ ] page manifest 与 face mapping hash 完整
- [ ] 导出渲染与内部渲染通过同相机对照

## 22. 实施顺序与停止条件

### Milestone 1：正确性

范围：`1K -> 100K faces`。

通过条件：

- camera、covariance、face mapping、initial render 全部正确；
- 100-step overfit 能稳定下降；
- 无轮廓严重 bleed。

### Milestone 2：全量资产构建

范围：全部 `26.85M faces`，不正式训练。

通过条件：

- geometry cache 完整；
- face identity 完整；
- H20 全量常驻成功；
- 单视角 width 800 smoke 成功；
- checkpoint/resume 成功。

### Milestone 3：SH0 full training

先完成 SH0，不直接开启 SH2/3。

通过条件：

- input RGB 收敛；
- alpha/depth guardrail 通过；
- full-resolution input render 无结构性孔洞。

### Milestone 4：自适应 SH 与 heldout

先在 input 的 13/3 development split 上完成选择，再从初始化使用全部 16 input views 重训冻结配置。最终 checkpoint 固定后第一次打开 heldout；该结果是终局审计，不用于回调当前候选。

通过条件：

- heldout RGB 相对 PBR mesh 有净收益；
- coverage/depth 不退化；
- 没有明显训练视角过拟合。

### Milestone 5：LOD 和部署

只有 Exact 模式通过后构建。

通过条件：

- 多质量档可控；
- 近景展开到叶节点；
- LOD render 与 Exact 的误差/FPS/显存可量化。

## 23. 预期时间

不要在代码完成前承诺固定总时长。建议在 H20 上用以下方式估算：

1. 全量 width 800，100 steps。
2. 全量 width 1600，100 steps。
3. Full-resolution tile，100 steps。
4. 分别记录 median/P90 step time。
5. 用实际 stage tile/view sampling 计算总预算。

初始工程预算可按：

```text
环境与CUDA编译:            0.5-1.5h
geometry/visibility cache:  1-3h
全量smoke与修正:            1-4h
正式训练:                   4-12h，必须由profile校正
23-view exact render:        1-4h
LOD/export/evaluation:       2-6h
```

这里最大的不确定性是每个视角/高分辨率tile的 first-hit active face 数和 gsplat 排序吞吐。

## 24. 最终交付物

```text
outputs/eth3d/lecture_room/mesh_gaussian_full/
  README.md
  config.resolved.yaml
  input_manifest.json
  preflight.json
  environment.freeze.txt
  geometry/
  visibility/
  checkpoints/latest
  renders/exact/
  renders/lod/
  evaluation/summary.json
  evaluation/summary.csv
  evaluation/index.html
  export/internal/
  export/spz/
  export/khr_pages/
  export/manifest.json
  validation.json
```

README 必须明确：

- Gaussian 是 appearance layer。
- Mesh 是权威几何。
- 训练只用了 16 input views。
- 7 heldout views 只用于评测。
- 全部 26,854,772 faces 均有叶级 Gaussian。
- Exact 与 LOD 结果分开。
- 正式 PLY/GLB hash 未改变。
- 许可证与第三方依赖。

## 25. 参考实现与规范

- SuGaR: <https://github.com/Anttwo/SuGaR>
- GaMeS: <https://github.com/waczjoan/gaussian-mesh-splatting>
- Gaussian Frosting: <https://github.com/Anttwo/Frosting>
- gsplat: <https://github.com/nerfstudio-project/gsplat>
- Hierarchical 3D Gaussians: <https://github.com/graphdeco-inria/hierarchical-3d-gaussians>
- CityGaussian: <https://github.com/Linketic/CityGaussian>
- Octree-GS: <https://github.com/city-super/Octree-GS>
- Khronos `KHR_gaussian_splatting`: <https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_gaussian_splatting>
- Niantic SPZ: <https://github.com/nianticlabs/spz>

许可证注意：SuGaR、GaMeS、Frosting和Hierarchical 3D Gaussians官方实现继承原始3DGS的非商业研究限制。`gsplat`是Apache-2.0，但落地前仍需审查整个依赖链、数据许可和目标用途。
