# GenRecon 在 DA3-BENCH ScanNet++ 子集上的复现报告

> 完成日期：2026-07-30。结论：公开代码与 GenRecon checkpoint 已在一张 RTX A4000 16 GB 上完成真实单场景推理，产出了可解析的 PLY mesh 和带 PBR 纹理的 GLB。本文只报告由本地文件、日志或 JSON 验证的结果。

## 1. 结论与边界

“GenRecon 能否在 Hugging Face 的 DA3 ScanNet++ 子集上运行”的答案是**可以**。本次对场景 `286b55a2bf` 使用 8 张带 COLMAP 位姿的 iPhone 图像，完整执行了 sparse structure、shape、texture、mesh 提取、GLB 烘焙和 200,000 点几何评测。

这不是论文 ScanNet++ Table 2 的严格整表复现，原因有三项：

1. [论文](https://arxiv.org/html/2605.23888)报告 25 个 ScanNet++ 场景；[DA3-BENCH](https://huggingface.co/datasets/depth-anything/DA3-BENCH) 的公开压缩包只有 20 个场景，且没有论文的场景名单映射。
2. 本次从可注册的 iPhone 帧中确定性选取 8 帧。论文只说明使用 provided COLMAP poses 的 8 views，没有公开逐场景帧 ID 或明确相机流；仓库 README 的 ScanNet++ 主示例是 `Scannet_colmap`，iPhone 仅作为备选模式。因此本次 iPhone 输入不能称为论文输入协议复现。
3. 论文先用完整 scanner fusion 构造 1.1 cm observation envelope，再膨胀 15 cm 并裁掉预测 mesh 的未观测部分；该 envelope 及构造代码不在公开仓库或 DA3 子集中。

因此，本文把“代码、权重、输入适配和 mesh 产出成功”判定为工程复现成功；几何数字标为 `paper-like-unclipped` 单场景代理结果，不把它冒充论文汇总值。

## 2. 固定版本与硬件

- GenRecon commit：`eaf1468118d20469d17079a4a19737297d2ef87b`
- Eigen submodule：`21e4582d1739107337a03460c81412981130373e`
- DA3-BENCH revision：`cb98821d5d4f721704f11bafde0403826b82b638`
- Python：3.10.12
- PyTorch / torchvision：2.6.0+cu126 / 0.21.0+cu126
- CUDA toolkit：12.6；Flash-Attention：2.7.3
- GPU：NVIDIA RTX A4000，16,376 MiB
- CPU：Intel Core i9-12900；系统内存 32 GB（另有 swap）
- 本次 Python 环境的完整 `pip freeze` 快照：`requirements-repro.lock`（其中本地 O-Voxel 路径需随仓库位置调整）

启用了 GenRecon 的 `low_vram` 整模串行卸载路径，并把 projection batch 设为 256 voxels。采样仍为每阶段 12 steps、seed 42、512 decoder、8 views；没有通过减视角、减步数或提高 occupancy threshold 来换取成功。

## 3. 数据与输入协议

DA3-BENCH `scannetpp.zip`：

- 大小：10,830,641,822 bytes
- SHA256：`01f8ce1996ac64ba1dc9770c4f52db139f91c5c1a0d8a98107ef38fe45490d6c`
- ZIP CRC 全包检查：通过
- 内容：20 scenes，29,673 entries

适配清单位于 `data/da3-adapted/286b55a2bf/selection.json`。源 COLMAP 模型有 353 张注册图像，其中 159 张为 iPhone；按图像名排序并用含端点的等间隔索引选择 `[0, 23, 45, 68, 90, 113, 135, 158]`。`--center_crop` 保证 GenRecon 最终接收恰好 8 个视角。

适配清单记录源稀疏模型有 32,593 点；实际重建日志记录 `Initial number of points: 26883`，GenRecon 清理后为 26,022 点，并产生 2 个重叠 chunks。适配器把 DA3 的二进制、混合 DSLR/iPhone COLMAP 模型转换为 GenRecon 所需的文本 iPhone 子模型，同时检查相机、图像、POINTS2D 与 point tracks 的双向一致性。

## 4. 模型资产

| 阶段 | 文件大小 | SHA256 | 严格载入结果 |
| --- | ---: | --- | --- |
| sparse structure | 2,862,971,400 B | `e18c1caddb2357dbf5839f0f7e1569c50d855fcb47e0871483d99c91a14e2bb7` | 1011/1011；无 missing、unexpected 或 shape mismatch |
| shape SLat 512 | 5,410,532,174 B | `d9e13be151a213bf67565d2a17341fe97328e455814fb2328a59366344e122fb` | 1010/1010；无 missing、unexpected 或 shape mismatch |
| texture SLat 512 | 5,410,728,782 B | `28f99217a4fbcd04f36a5f576905975ae8b874ad63548d6ab8afdb96cf03ed47` | 1010/1010；无 missing、unexpected 或 shape mismatch |

TRELLIS decoder 使用 `microsoft/TRELLIS.2-4B` revision `af44b45f2e35a493886929c6d786e563ec68364d` 与 `microsoft/TRELLIS-image-large` revision `25e0d31ffbebe4b5a97464dd851910efc3002d96` 的本地缓存。

### DINOv3 条件编码器说明

GenRecon 配置要求 `facebook/dinov3-vitl16-pretrain-lvd1689m`，该仓库需先接受 Meta 许可并登录。本次正式运行时本机没有 Hugging Face 登录态，因此实际使用公开的 [`timm/vit_large_patch16_dinov3_qkvb.lvd1689m`](https://huggingface.co/timm/vit_large_patch16_dinov3_qkvb.lvd1689m)，固定 revision `3653d393df9fd16a9c1bae8a6c5a514f683b70ce`：

- 源 safetensors：366 tensors，303,128,576 parameters，SHA256 `56df97c299afea2ffa1eb90f866e8a290f452e227e732fef32a9eb9e8b52027f`
- 转换后：303,129,600 parameters，SHA256 `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`
- 唯一新增参数是 1,024 个值的全零 mask token；GenRecon 调用 `bool_masked_pos=None`，该 token 不参与本次推理
- 按正式推理分辨率，固定 seed 42 的随机 512x512 输入逐 token 对照：shape `[1, 1029, 1024]`，max absolute error `8.5831e-06`，mean absolute error `2.5990e-07`，RMSE `3.4990e-07`，`allclose(atol=2e-5, rtol=2e-5)` 通过

上述对照验证了 timm 到 Transformers 键映射和 GenRecon 实际 feature path 的数值等价性。由于没有访问 gated Meta 文件，本文不声称两家发布文件逐字节相同；这仍是本次结果相对论文原环境的一项明确偏差。

## 5. 可重复执行命令

### 5.1 数据适配

```bash
HF_HOME=$PWD/data/hf-cache .venv/bin/hf download \
  depth-anything/DA3-BENCH scannetpp.zip \
  --repo-type dataset \
  --revision cb98821d5d4f721704f11bafde0403826b82b638 \
  --local-dir data/da3-bench

unzip -q data/da3-bench/scannetpp.zip -d data/da3-bench/extracted

.venv/bin/python tools/prepare_da3_scannetpp.py \
  --input-root data/da3-bench/extracted/scannetpp \
  --output-root data/da3-adapted \
  --scene 286b55a2bf \
  --num-views 8
```

GenRecon 三个官方 checkpoint 放在以下位置：

```text
weights/ss/checkpoints/sparse_structure.pt
weights/shape/checkpoints/shape_slat.pt
weights/texture/checkpoints/texture_slat.pt
```

可按官方 README 的地址下载并用上表哈希校验：

```bash
mkdir -p weights/ss/checkpoints weights/shape/checkpoints weights/texture/checkpoints
curl -fL https://kaldir.vc.cit.tum.de/genrecon/sparse_structure.pt \
  -o weights/ss/checkpoints/sparse_structure.pt
curl -fL https://kaldir.vc.cit.tum.de/genrecon/shape_slat.pt \
  -o weights/shape/checkpoints/shape_slat.pt
curl -fL https://kaldir.vc.cit.tum.de/genrecon/texture_slat.pt \
  -o weights/texture/checkpoints/texture_slat.pt
sha256sum \
  weights/ss/checkpoints/sparse_structure.pt \
  weights/shape/checkpoints/shape_slat.pt \
  weights/texture/checkpoints/texture_slat.pt
```

### 5.2 官方 DINOv3 路径

先在 [`facebook/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) 接受许可，然后只在本机登录：

```bash
HF_HOME=$PWD/data/hf-cache .venv/bin/hf auth login
bash scripts/run_smoke.sh
```

### 5.3 本次实际使用的公开 timm 路径

```bash
HF_HOME=$PWD/data/hf-cache .venv/bin/hf download \
  timm/vit_large_patch16_dinov3_qkvb.lvd1689m model.safetensors \
  --revision 3653d393df9fd16a9c1bae8a6c5a514f683b70ce \
  --local-dir data/hf-cache/downloads/dinov3-timm

.venv/bin/python tools/convert_timm_dinov3.py \
  data/hf-cache/downloads/dinov3-timm/model.safetensors \
  data/hf-cache/converted/dinov3-vitl16-timm-qkvb \
  --source-revision 3653d393df9fd16a9c1bae8a6c5a514f683b70ce

.venv/bin/python tools/verify_dinov3_conversion.py \
  data/hf-cache/downloads/dinov3-timm/model.safetensors \
  data/hf-cache/converted/dinov3-vitl16-timm-qkvb \
  --device cuda \
  --output reports/generated/286b55a2bf/dinov3_equivalence.json

PIPELINE_CONFIG=data/hf-cache/converted/dinov3-vitl16-timm-qkvb/pipeline.json \
bash scripts/run_smoke.sh
```

首次运行需联网缓存公共 TRELLIS decoders 等运行资产；缓存完成后，本次审计运行额外设置了 `HF_HUB_OFFLINE=1`，以证明没有在推理中漂移到其他 revision。`run_smoke.sh` 顺序执行重建、GLB 烘焙和 mesh 评测，并拒绝把旧文件误报成本次产物。完整命令与日志保存在 `reports/generated/286b55a2bf/`。

## 6. Mesh 产物

| 产物 | 大小 | 内容校验 | SHA256 |
| --- | ---: | --- | --- |
| `outputs/286b55a2bf/reconstruction/mesh.ply` | 156,610,587 B | 3,460,671 vertices；7,255,246 faces；坐标有限；索引合法；含 vertex colors | `3bc44dd7b60acd1e234df5dc3adca69b4428bdddcb7ca12b867d83095f91636c` |
| `outputs/286b55a2bf/reconstruction/scene.glb` | 166,652,400 B | 2 个 geometry nodes；坐标/UV 有限；索引合法；每个 chunk 均有 4096x4096 base-color 与 metallic-roughness 纹理 | `0211ab329dd3fbe77f762938d4cdffde250e82a6297a30fb0529b1ed86502b05` |

PLY bounds 为 `[(-0.091, -0.163, 0.012), (2.951, 1.507, 2.651)]` m；GT bounds 为 `[(0, 0, 0.003), (2.695, 1.438, 2.611)]` m，二者处于同一世界坐标系。GLB 的两个烘焙 chunk 分别含 1,192,321 / 1,112,838 vertices 和 1,951,609 / 1,891,136 faces。

实际 PLY 的两张无头渲染预览：

- [view 00](generated/286b55a2bf/mesh_preview_view_00.png)：可辨认马桶、红色柜体、瓶罐、地面与墙体
- [view 03](generated/286b55a2bf/mesh_preview_view_03.png)：可辨认天花灯、墙柜、搁架和房间布局

两张预览分别为 1280x960，像素标准差 47.83 / 39.74，不是空白渲染。与对应输入图像目视比较，主体语义和空间关系一致；同时能看到生成式补全带来的额外薄片/表面，这与后述较低 precision 和未执行论文 envelope 裁剪相符。

结构化校验记录：`reports/generated/286b55a2bf/artifact_validation.json`；DINOv3 数值对照记录：`reports/generated/286b55a2bf/dinov3_equivalence.json`。

## 7. 实测性能

| 阶段 | Wall time | 进程 GPU 采样峰值 | 全局 GPU 采样峰值 | 进程树 RSS 采样峰值 |
| --- | ---: | ---: | ---: | ---: |
| GenRecon 到 PLY | 225.36 s | 5,362 MiB | 7,119 MiB | 17.29 GiB |
| PLY 到 PBR GLB | 253.53 s | 7,348 MiB | 9,092 MiB | 4.18 GiB |

两个 GPU 阶段合计 478.89 s，即约 7.98 分钟，不含数据下载、首次模型缓存与 CPU mesh 指标计算。外部 profiler 约每秒采样一次，因此表中资源峰值是观测下界；它记录到 100% GPU utilization，重建采样峰值 140.38 W / 84 C，GLB 采样峰值 138.61 W / 72 C。

程序内部 CUDA 统计分别为 sparse decode 0.35 GB、shape decode 4.06 GB、texture decode 4.36 GB、fill holes 1.15 GB。它们是局部 PyTorch 峰值，不能替代外部的整进程显存值。实测证明本场景在 16 GB A4000 上可完成；不据此保证更大场景或更多 chunks 也能在 16 GB 内运行。

原始性能 JSON：`reconstruct_profile.json` 与 `glb_profile.json`，两者 `return_code=0`、`success=true`，且目标文件均为该次运行新建或更新。

## 8. 几何指标与论文对照

预测和 GT 各按三角形面积均匀采样 200,000 点，seed 42。Chamfer 是双向平均距离的算术均值；F-score 按论文公式使用 precision 与 recall 的算术均值，同时额外保留常见 harmonic 值作为诊断；normal consistency 使用双向绝对法向点积，最近邻距离大于 20 cm 的对应记零。

| 指标 | 本次单场景、未裁剪 | 论文 25 场景 GenRecon 汇总 |
| --- | ---: | ---: |
| pred -> GT mean | 0.12834 m | 未单列 |
| GT -> pred mean | 0.05964 m | 未单列 |
| Chamfer symmetric mean | **0.09399 m** | **0.0688 m** |
| Precision @ 10 cm | 0.44818 | 未单列 |
| Recall @ 10 cm | 0.79338 | 未单列 |
| F-score @ 10 cm，论文算术定义 | **0.62078** | **0.7771** |
| F-score @ 10 cm，harmonic 诊断 | 0.57279 | 不适用 |
| Normal consistency | **0.67984** | **0.7860** |

预测表面积为 106.91 m²，GT 为 27.41 m²。较高 recall（0.793）说明已覆盖多数 GT，较低 precision（0.448）则表明预测包含大量离 GT 超过 10 cm 的表面。论文 observation envelope 正是会显著影响这类未观测/外延预测的步骤，因此不能把表中差值全部归因于模型复现误差，也不能用 GT AABB 裁剪伪装论文 envelope。本次保留未裁剪主结果，避免事后调指标。

结构化指标：`reports/generated/286b55a2bf/mesh_metrics_unclipped.json`。

## 9. 为复现加入的代码

- `tools/prepare_da3_scannetpp.py`：DA3 ScanNet++ 二进制 COLMAP 到 GenRecon iPhone 输入的确定性适配器。
- `tools/evaluate_mesh.py`：论文公开几何公式的 200k 点评测，并显式标注缺失 envelope。
- `tools/profile_command.py`：采样 wall time、进程树 RSS、GPU 显存/利用率/功耗/温度，并拒绝陈旧产物。
- `tools/convert_timm_dinov3.py` 与 `tools/verify_dinov3_conversion.py`：公开 DINOv3 fallback 的可审计转换与数值验证。
- `scripts/run_smoke.sh`：固定单场景端到端命令。
- 推理代码：显式路由三阶段训练 config；修复首次 undistortion cache 建目录；修复 sparse flow 应在 latent resolution 16 采样噪声、再由 decoder 输出 occupancy resolution 32 的维度错误。

最终审计重新执行了完整回归测试，18/18 通过；所有本地大数据、权重和生成产物均由 `.gitignore` 排除，源码与报告可独立审查。

## 10. 最终判断

本次已经达到“在 DA3-BENCH ScanNet++ 子集上把 GenRecon 跑通并看到真实 mesh、性能和量化结果”的目标。可复用的工程链路、产物和日志齐全，16 GB GPU 对该两-chunk 场景足够。

尚未达到、也不应宣称达到的是论文 ScanNet++ Table 2 的严格复现。要完成那一步，仍需要论文使用的 25 个 scene IDs、每场景相机流与 8 帧清单、scanner observation envelope/裁剪实现，以及最好使用授权后的原始 Meta DINOv3 artifact 对 fallback 做同权重哈希或输出对照。
