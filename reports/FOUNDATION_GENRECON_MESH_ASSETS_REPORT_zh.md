# Foundation-SfM 到 GenRecon 完整 Mesh 资产报告

## 1. 结论

已对 foundation-SfM v1 中所有 17 个 consumer preflight 为 `passed` 或 `marginal` 的候选，使用当前工作区版本的 GenRecon 完整运行一次 shape + texture 推理，并导出高精度 PLY 与嵌入 PBR 纹理的 GLB。

- 17/17 GenRecon reconstruction 完成。
- 17/17 `mesh.ply` 完成并通过 binary PLY payload 校验。
- 17/17 `scene.glb` 完成并通过 glTF 2.0、mesh/material、base-color 与 metallic/roughness texture 校验。
- 空间 chunks 共 126 个；生成 125 个非空 PBR primitives，Cologne 的 chunk 5 明确为空并由转换器跳过。
- 125/125 chunk cache GLB 均可解析且包含一套 PBR material。
- Hilltop chunk 11 与 Lincoln chunk 2 各发生一次 closest-camera fallback，其余为零 fallback。
- 高精度 PLY 合计 100,098,825 vertices、208,516,328 faces、4,512,496,412 B。
- 最终 PBR GLB 合计 3,884,501,864 B。
- 17 对 PLY/GLB 共 8,396,998,276 B 已重新计算 SHA256，并与生成时 profile 完全一致。
- 批处理墙钟时间 11,012.96 秒，约 3 小时 3 分 33 秒。
- GenRecon 阶段累计 5,310.77 秒，GLB 阶段累计 5,693.36 秒。
- GenRecon 进程显存峰值 12,748 MiB；GLB 峰值 5,538 MiB。
- 完整输出目录约 19 GiB。

这里的“完整”表示 shape、texture、PLY、PBR GLB 和审计资产齐全，不表示 source shot、foundation geometry 或生成表面真实可信。只有 Wawasee 和 Unity House 仍建议直接进入后续 pilot。

## 2. 固定运行协议

本次运行固定在 detached commit `eaf1468118d20469d17079a4a19737297d2ef87b` 的当前 dirty worktree；完整 dirty 文件列表和代码/模型哈希位于输出 `config.json`。

GenRecon 参数：

```text
mode = Iphone
colmap_subdir = colmap_vggt
num_imgs_per_scene = 8
center_crop = false
seed = 42
chunk_size_factor = 1.08
min_overlap_factor = 4
proj_batch_voxels = 256
```

没有加入统一 `manual_z_bounds`，避免把 Cologne 等高空间裁到固定房间高度。点云清理沿用 foundation preflight 的 `IphoneChunker` 默认值。

PBR GLB 参数：

```text
texture_size = 4096
simplify_threshold = 300000 faces per nonempty chunk
skip_fill_holes = true
skip_remesh = true
chunks_dir = reconstruction/chunks_300k
```

高精度几何保留在 `mesh.ply`；`scene.glb` 是浏览、交换和 PBR 检查资产，不替代 PLY。每个 GLB primitive 包含 base-color texture 和 metallic/roughness texture，纹理嵌入 GLB。

官方 GenRecon checkpoint SHA256：

- sparse structure：`e18c1caddb2357dbf5839f0f7e1569c50d855fcb47e0871483d99c91a14e2bb7`
- shape SLat：`d9e13be151a213bf67565d2a17341fe97328e455814fb2328a59366344e122fb`
- texture SLat：`28f99217a4fbcd04f36a5f576905975ae8b874ad63548d6ab8afdb96cf03ed47`

DINOv3 继续使用此前验证过的 timm-converted local pipeline，`pipeline.json` SHA256 为 `34c2f320a8c856c796c04061f45158401ddad3b2965b79650a160fa7da409f23`。

## 3. 逐场景结果

`nonempty/space chunks` 后括号为明确 empty chunk ID。耗时为 `GenRecon/GLB`。

| Candidate | Foundation | Visual disposition | Nonempty/space chunks | Fallback | Vertices/Faces | PLY | GLB | Time |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| raw-009-wawasee-performing-arts | P-A | proceed_foundation_pilot | 3/3 (-) | 0 | 3.91M/8.08M | 167.3 MiB | 91.0 MiB | 3.1/1.8 min |
| raw-014-unity-house | P-B | proceed_foundation_pilot | 7/7 (-) | 0 | 7.99M/16.71M | 344.3 MiB | 208.1 MiB | 5.5/3.9 min |
| raw-006-evel-knievel-museum | P-A | refine_then_retry | 4/4 (-) | 0 | 4.02M/8.54M | 174.9 MiB | 123.4 MiB | 4.3/2.7 min |
| raw-007-burlington-high-tech | P-B | refine_then_retry | 6/6 (-) | 0 | 10.08M/20.72M | 430.0 MiB | 183.9 MiB | 5.9/2.8 min |
| raw-019-crockett-middle-school | P-B | refine_then_retry | 16/16 (-) | 0 | 9.16M/18.94M | 392.1 MiB | 456.9 MiB | 8.3/10.7 min |
| raw-002-cologne-cathedral | P-C | refine_then_retry | 5/6 (5) | 0 | 5.48M/11.83M | 240.7 MiB | 156.2 MiB | 5.6/3.6 min |
| raw-008-hilltop-school-arts | P-C | refine_then_retry | 12/12 (-) | 1 | 9.66M/19.83M | 411.7 MiB | 375.0 MiB | 8.4/11.6 min |
| raw-005-beverly-theater | P-A | reject_current_shot | 12/12 (-) | 0 | 10.14M/21.26M | 437.6 MiB | 371.6 MiB | 7.8/7.8 min |
| raw-018-jabal-restaurant | P-A | reject_current_shot | 7/7 (-) | 0 | 6.23M/13.32M | 272.1 MiB | 234.0 MiB | 5.2/4.9 min |
| raw-013-lincoln-health-security | P-C | reject_current_shot | 15/15 (-) | 1 | 5.18M/10.46M | 218.5 MiB | 375.2 MiB | 7.2/16.7 min |
| raw-017-frederick-funeral-home | P-C | reject_current_shot | 3/3 (-) | 0 | 0.92M/1.91M | 39.5 MiB | 56.0 MiB | 2.1/0.8 min |
| raw-003-castleton-house-tour | P-F | reject_current_shot | 6/6 (-) | 0 | 3.26M/6.96M | 142.2 MiB | 198.0 MiB | 4.5/8.2 min |
| raw-004-blue-rose-ballroom | P-F | reject_current_shot | 5/5 (-) | 0 | 3.57M/7.40M | 153.1 MiB | 159.8 MiB | 3.7/3.7 min |
| raw-011-lake-johanna-fire-station | P-F | reject_current_shot | 9/9 (-) | 0 | 6.91M/14.33M | 296.3 MiB | 271.2 MiB | 5.9/6.3 min |
| raw-012-bartlett-bay-wastewater | P-F | reject_current_shot | 11/11 (-) | 0 | 8.07M/16.42M | 342.0 MiB | 313.5 MiB | 6.3/6.2 min |
| raw-015-moses-myers-house | P-F | reject_current_shot | 2/2 (-) | 0 | 1.94M/4.03M | 83.3 MiB | 60.4 MiB | 2.2/1.9 min |
| raw-020-bethel-lutheran-home | P-F | reject_current_shot | 2/2 (-) | 0 | 3.57M/7.77M | 157.7 MiB | 70.2 MiB | 2.4/1.0 min |

## 4. 目录关系

每个 candidate 使用相同 ID 串联 source、foundation input、GenRecon output 和 profile report：

```text
data/internet-zero-shot/sfm-preproducts-v1/candidates/<candidate_id>/
  rgb/ + masks_dynamic/ + masked COLMAP fragments
        |
        v
data/internet-zero-shot/foundation-sfm-v1/candidates/<candidate_id>/
  rgb/                       # 原 RGB + dynamic alpha
  colmap_vggt/               # GenRecon 相机与 pseudo points
  foundation_points.ply      # foundation pseudo geometry
  predictions.npz
  manifest.json
  genrecon_preflight/
        |
        v
outputs/internet-zero-shot/foundation-genrecon-v1/candidates/<candidate_id>/
  reconstruction/
        |
        +-- mesh.ply                 # 高精度完整 mesh，主要几何交付物
        +-- scene.glb                # 合并 PBR GLB，主要浏览/交换交付物
        +-- chunks_300k/
        |     +-- chunk_000.glb      # 可恢复的单 chunk PBR 资产
        |     +-- ...
        +-- to_glb_inputs.pt         # 可重新烘焙 GLB 的 mesh/attribute volume
        +-- chunk_inputs.pt          # chunk metadata
        +-- coords_NNN.ply           # 每 chunk sparse structure coords
        +-- clean_points.ply         # GenRecon 实际使用的清理后 pseudo points
        +-- cameras.json             # 选中 conditioning cameras
        +-- chunk_transforms.json
        +-- chunk_layout.png
        +-- args.json
        +-- scene/view_NNN.png       # 8 个 source views 双裁剪后最多 16 个 conditioning crops
        +-- chunk_NNN/cond2d.png     # 每 chunk conditioning contact
        |
        v
reports/generated/internet-zero-shot/foundation-genrecon-v1/candidates/<candidate_id>/
  reconstruct.log
  reconstruct_profile.json          # command、hash、耗时、RSS、GPU
  glb.log
  glb_profile.json                   # command、scene.glb hash、耗时、GPU
```

批次根目录：

```text
outputs/internet-zero-shot/foundation-genrecon-v1/
  config.json       # 固定协议、commit、dirty worktree、模型/代码 hash
  index.json        # 17 场景机器可读关系、尺寸、SHA256、PBR/mesh 统计
  summary.csv       # 扁平目录索引
  validation.json   # 独立最终验证结果
  candidates/

reports/generated/internet-zero-shot/foundation-genrecon-v1/
  batch_run.json
  batch_driver.log
  summarize.log
  validate.log
  candidates/
```

例如 Wawasee 的核心交付物：

```text
outputs/internet-zero-shot/foundation-genrecon-v1/candidates/
  raw-009-wawasee-performing-arts/reconstruction/mesh.ply
  raw-009-wawasee-performing-arts/reconstruction/scene.glb
```

Wawasee PLY SHA256 为 `171a2a4154b799e41d5f60d56b196b8dd506254e7293e3de47c299a23ddbb20f`，GLB SHA256 为 `68e5dd6154db2e7f9893fd0275cf3b5c08f5616bf056f6e596ed6f3d236d37c5`。

Unity House PLY SHA256 为 `efc1d529c01b6980da45dec51bf2f1e06e1b0575f2fbfe1aa38b39d244c71666`，GLB SHA256 为 `3a402b5705f40bf23785f870d0f27584c4fd80219b2dc0b9783f5fe0b0a87a83`。

全部 34 个交付物的 SHA256 位于 `index.json`，不在本文重复展开。

## 5. 验证内容

最终 `validation.json` 为 `pass`，包含：

1. 17 个 reconstruct profile 均成功且中间件存在。
2. 17 个 binary little-endian PLY 的 vertex/face header、固定 vertex record 和 triangle face payload 字节数精确匹配。
3. 17 个最终 GLB 的 magic、version、总长度、scene、mesh/material 和 PBR texture 绑定有效。
4. 125 个 chunk GLB 全部重新解析，每个恰有一个 mesh/material 和两张嵌入纹理。
5. 126 个空间 chunks 由 125 个非空缓存加 Cologne 明确 empty chunk 5 完整覆盖。
6. 17 对 PLY/GLB 重新计算的 SHA256 与生成 profile 一致。
7. 126 个输出、preview 与 profile JSON 均为严格 JSON，无 `NaN` 或 `Infinity`。
8. 相关回归测试为 47 passed。

## 6. 解释边界

- GenRecon 成功生成 mesh 不会把 foundation pseudo geometry 变成真实 SfM 或 ground truth。
- 所有场景尺度来自 1.6 m 相机高度假设，只是 proxy metric scale。
- `P-A/P-B` 是初始化可用等级，不是几何准确率等级。
- Beverly、Jabal 等技术 P-A 仍被视觉 gate 拒绝，证明技术可消费和场景可用必须分开。
- Castleton、Blue Rose、Lincoln、Moses 等外景/幻灯片也生成了大型 mesh，证明文件大小和三角形数不能作为质量指标。
- Hilltop 与 Lincoln 各有一次 closest-camera fallback，不满足正式 8-view zero-fallback gate。
- Cologne 的第 6 个空间块在生成 mesh 中为空；最终 GLB 有 5 个有效 primitive，不是缺文件。
- 源视频许可、隐私、拍摄日期和 foundation/GenRecon 训练污染仍由独立 gate 决定。

### 6.1 推荐场景的相机位姿渲染

对 Wawasee 和 Unity House 的 16 个 foundation 相机分别以 320 px 宽度实际渲染最终 GLB，并逐像素检查 alpha coverage、非黑像素比例和 RGB 方差：

| Candidate | Views | Coverage min/median/max | Min nonblack fraction | Min RGB std | Result |
|---|---:|---:|---:|---:|---|
| Wawasee | 16 | 11.3% / 74.3% / 87.4% | 93.7% | 7.27 | nonblank，但表面/外观偏暗且不完整 |
| Unity House | 16 | 41.7% / 52.0% / 79.6% | 86.3% | 30.86 | nonblank，床架可辨但房间外壳不完整 |

固定 source/GLB contact：

- `reports/generated/internet-zero-shot/foundation-genrecon-v1/candidates/raw-009-wawasee-performing-arts/glb_preview/overview.jpg`
- `reports/generated/internet-zero-shot/foundation-genrecon-v1/candidates/raw-014-unity-house/glb_preview/overview.jpg`
- 机器验证：`reports/generated/internet-zero-shot/foundation-genrecon-v1/preview_validation.json`

因此“完整 mesh 资产”只表示交付链路和文件结构完整，不表示视觉上形成封闭、完整或高质量的房间。Wawasee 主要保留窗口/服务口结构，Unity House 主要保留床架与局部表面；两者可继续做 fidelity stress test，而不是作为成品房间资产。

当前只建议继续量化评测 Wawasee 与 Unity House。Cologne、Evel、Burlington、Hilltop、Crockett 应先重新切为单一物理空间；其余 10 条当前 shot 保留为失败/负对照资产。

17 场景的原始选帧、相机位姿重建和左右逐帧对比视频见 [Foundation GenRecon 原始帧与重建视频逐帧对比报告](FOUNDATION_GENRECON_VIDEO_COMPARISON_REPORT_zh.md)，本地审查入口为 `outputs/internet-zero-shot/foundation-video-comparisons-v1/index.html`。
