# ETH3D pipes / delivery_area GenRecon 高精度推理报告

> 本报告只记录本机实际生成、流式校验和激光点云代理评测得到的结果。GPU 为单张 NVIDIA RTX A4000 16GB，GenRecon 生成分辨率为 512，随机种子为 42。

## 结论

两个场景均完成 RGB + COLMAP 到原始 PLY mesh 的三阶段推理，并生成可解析的 PBR GLB。

- `pipes`：结果稳定，管道、阀门、柜体、帘门和灭火器均可辨认。最终选择 `occ_threshold=-1` 的 6-chunk 版本。
- `delivery_area`：完整扫描高度对应的 6.37m 单层 chunk 明显超出模型训练尺度，几何误差较大。最终选择由密集 COLMAP 点自动估计高度的 4.82m、21-chunk `model_scale` 版本。
- 原始 PLY 是保留最多几何细节的结果。GLB 为便于查看和传输的 PBR 版本，`delivery_area` 每 chunk 限制到约 100 万面。

## 最终产物

| 场景 | 原始高精度 PLY | PBR GLB | 预览与指标 |
|---|---|---|---|
| pipes | `outputs/eth3d/pipes/high_precision/mesh.ply` | `outputs/eth3d/pipes/high_precision/scene.glb` | `reports/generated/eth3d/pipes/high_precision/` |
| delivery_area | `outputs/eth3d/delivery_area/model_scale/mesh.ply` | `outputs/eth3d/delivery_area/model_scale/scene.glb` | `reports/generated/eth3d/delivery_area/model_scale/` |

每个输出目录同时保留 `args.json`、`cameras.json`、chunk 变换、逐块 occupancy 坐标、`to_glb_inputs.pt` 和可恢复的逐块 GLB。

## 输入与策略

| 项目 | pipes | delivery_area |
|---|---:|---:|
| COLMAP 注册图像 | 14 / 14 | 44 / 44 |
| 原始稀疏点 | 2,473 | 31,978 |
| 空间清理后点数 | 1,246 | 5,730 |
| 推理所用物理相机 | 14 | 16（沿序列均匀采样） |
| 左右方形裁剪后视图 | 28 | 32 |
| chunk 数 | 6 | 21 |
| chunk 边长 | 2.8756m | 4.8246m |
| chunk 重叠下限 | 25% | 25% |
| 占据阈值 | -1.0 | -1.0 |
| 投影批大小 | 256 voxels | 256 voxels |

`pipes` 使用高置信 COLMAP 点估计的竖直边界 `[-1.0, 1.765]m`，保留地面与顶部管线。`delivery_area` 不使用完整激光扫描高度，因为该范围会把单个 chunk 扩大到 6.37m；最终版本使用相机密集观察区域的点云高度。

## 网格完整性

流式读取二进制 PLY，核对文件长度、有限坐标、三角面计数和全部索引范围：

| 场景 | 顶点 | 三角面 | PLY 大小 | 世界坐标边界 |
|---|---:|---:|---:|---|
| pipes | 5,879,189 | 12,763,818 | 271,755,348 B | `[-3.187,-0.253,-1.000]` 至 `[2.566,2.712,1.783]` |
| delivery_area | 14,637,780 | 30,067,986 | 654,364,171 B | `[-8.897,-6.313,-4.961]` 至 `[9.198,15.099,-0.143]` |

两者坐标均有限，全部 face list 长度为 3，索引最小值为 0，最大值恰好为 `vertex_count - 1`。

## 激光 ROI 代理指标

ETH3D `dslr_scan_eval` 只提供激光点云，没有三角面，因此不能使用仓库原有 mesh-to-mesh 法向指标。本次使用明确标记为 `roi-pointcloud-proxy` 的协议：

- 从预测 mesh 按面积采样 200,000 点。
- 应用官方 `scan_alignment.mlp`，将激光点对齐到 COLMAP 世界坐标。
- 激光点裁到预测 AABB 外扩 5cm，最多固定种子采样 1,000,000 点。
- 计算双向最近邻距离以及 2/5/10cm precision、recall 和调和 F-score。

| 场景/策略 | 双向均值 | mesh→laser | laser→mesh | P@10cm | R@10cm | F@10cm |
|---|---:|---:|---:|---:|---:|---:|
| pipes，最终 `threshold=-1` | **0.1068m** | **0.1749m** | **0.0387m** | **0.4365** | **0.9462** | **0.5974** |
| pipes，对照 `threshold=0` | 0.1100m | 0.1812m | 0.0388m | 0.4231 | 0.9441 | 0.5843 |
| delivery_area，完整高度 6.37m | 0.5743m | 0.6950m | 0.4537m | 0.2229 | 0.3682 | 0.2776 |
| delivery_area，最终 4.82m | **0.3047m** | **0.4205m** | **0.1889m** | **0.3639** | **0.5575** | **0.4404** |

`delivery_area` 的尺度调整使代理 Chamfer 降低约 47%，F@10cm 提高约 59%。高置信 COLMAP 点到激光点的中位距离为 `pipes 4.5mm`、`delivery_area 13.1mm`，证明较大的生成误差不是坐标对齐错误。

这些数字不是 ETH3D 官方深度图评测分数：激光采样密度不均匀，ROI 随预测范围变化，也没有官方遮挡处理。它们仅用于本次策略 A/B 和误差诊断。

## 性能

| 场景 | 重建耗时 | 进程显存峰值 | 进程树 RSS 峰值 | shape / texture 解码峰值 |
|---|---:|---:|---:|---:|
| pipes | 366.3s | 7,718 MiB | 19.28GB | 6.20 / 6.14GB |
| delivery_area | 729.5s | 7,812 MiB | 19.31GB | 6.97 / 6.88GB |

`delivery_area` 的 21 chunks 被确定性分成 4 组，组大小为 `[5,5,5,6]`，膨胀体素数为 `[23198,13475,18149,19054]`。生成分辨率、视图和采样参数没有因分组而降低。

## GLB

| 场景 | mesh primitives | GLB 顶点 | GLB 三角面 | 内嵌纹理 | 大小 | 成功段耗时 / 显存峰值 |
|---|---:|---:|---:|---:|---:|---:|
| pipes | 6 | 7,747,513 | 12,180,328 | 12 PNG | 569,535,480 B | 354.3s / 5,878 MiB |
| delivery_area | 21 | 11,742,298 | 17,393,979 | 42 PNG | 1,073,427,416 B | 992.4s / 6,284 MiB |

GLB 顶点包含 UV seam 复制，因此可能多于 PLY 顶点。两个文件均通过 GLB 2.0 magic/version、声明长度、JSON/BIN chunk、内嵌 buffer、accessor 边界有限性和节点/材质/纹理数量检查。

SHA-256：

```text
pipes mesh.ply  0db26137a3ccfdf8a24fc3baf7ddcbef1d0c367dc5a54d6e1f8b65f3d9bd3723
pipes scene.glb 4ede5b2623c28486e004ce9a5847b3cae3e800bc265a407df5c9ae0ac7874f78
delivery mesh.ply 17328a06ec517340b05ce5565c8137cd5292f7be241bf580aff5898382d49c93
delivery scene.glb 6214530c8bf571271e61d687147aae48c936932fa67105e31e59ca689cfb6561
```

## 代码调整与验证

为支持本次稳健调参，做了以下向后兼容调整：

- 修复通用 `IphoneChunker` 未接收已有 `--skip_point_cleaning` 参数的问题。
- 增加可选 `--max_reproj_error`、`--min_track_len` COLMAP 点质量过滤。
- 增加可选 `--manual_z_bounds FLOOR CEILING`，允许只覆盖竖直边界而保留点云推导的 XY ROI。
- 新增 `tools/evaluate_mesh_pointcloud.py`，用于明确标记的 mesh-to-laser ROI 代理评测。

默认参数和现有 ScanNet++ 行为不变。相关 3 个单元测试通过，修改文件也通过 `py_compile`。

## 局限与后续精度方向

- GenRecon 是生成式重建；在 `delivery_area` 这类高顶、开阔工业大厅中，单层 chunk 仍明显偏离训练分布。当前结果可浏览且结构可辨认，但不能视为测量级 mesh。
- 若目标是厘米级工程测量，应把 COLMAP/MVS 或激光重建作为几何主干，再用 GenRecon 补纹理和局部缺失区域。
- 若继续改 GenRecon，最值得实现的是多层竖直 chunk、基于图像可见性的 mesh 裁剪，以及复用 sparse-structure 中间结果进行阈值/随机种子搜索；继续增加面数或盲目加入更多远距离 chunks 不会解决当前误差。
