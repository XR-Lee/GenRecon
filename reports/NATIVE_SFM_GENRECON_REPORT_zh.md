# Native COLMAP-SfM 到 GenRecon 完整产物报告

## 1. 结论

原 masked-COLMAP v2 gate 为 A 的 3 个候选已作为独立 `native-sfm` 轨道完成输入适配、GenRecon consumer preflight、shape/texture 推理、高精度 PLY、PBR GLB、所有注册相机逐帧视频和独立验证。

- 场景：3/3。
- 原生注册相机：303。
- 原生 COLMAP 点：78,101；保留原 point ID、RGB、ERROR 和跨视图 TRACK。
- GenRecon 清理后真实 SfM 点：22,783。
- 空间 chunks：19；preflight fallback：0。
- 完整 GenRecon：3/3。
- 最终 `mesh.ply`：3/3。
- 最终 PBR `scene.glb`：3/3。
- 非空 chunk GLB：19/19；empty chunks：0。
- GenRecon runtime fallback：0。
- 高精度 mesh：24,621,583 vertices、52,622,420 faces、1,127,280,891 B。
- PBR GLB：640,532,028 B。
- PLY + GLB：1,767,812,919 B，SHA256 与 profile 全部一致。
- 重建累计 1,077.81 秒；GLB 累计 684.13 秒。
- 批次墙钟约 29 分 24 秒。
- GenRecon 进程显存峰值 11,556 MiB；GLB 峰值 5,658 MiB。
- 注册相机逐帧对比：303 帧、9 个 H.264 视频、151.5 秒。

这 3 条不是 foundation fallback，也没有使用 VGGT pseudo points。它们的相机、tracks、误差和几何来自 masked COLMAP；只有尺度与重力方向是后处理 proxy。

## 2. 场景分流

20 条互联网候选现在完整分为两个互斥轨道：

```text
masked COLMAP v2 gate
├── A/B: native-sfm track                  3 scenes, 本报告
└── C/F: foundation-sfm fallback track    17 scenes
```

当前没有 B 场景。两个轨道不得把 `SfM-A` 与 foundation `P-A/P-B` 混为同一等级，也不得把 proxy scale 当作真实测量。

## 3. Native 输入适配

源输入：

```text
data/internet-zero-shot/sfm-preproducts-v1/candidates/<candidate_id>/
  rgb/*.jpg
  masks_dynamic/*.png
  colmap/{cameras,images,points3D}.txt
  quality.json
```

统一 adapter 输出：

```text
data/internet-zero-shot/native-sfm-v1/candidates/<candidate_id>/
  rgb/*.png                     # 无畸变 RGBA，RGB 保留，动态区 alpha=0
  masks_dynamic/*.png           # 无畸变二值动态 mask
  colmap_sfm/
    cameras.txt                 # PINHOLE
    images.txt                  # 变换后的原生相机与无畸变 2D observations
    points3D.txt                # 原生 ID/RGB/ERROR/TRACK，仅 XYZ 做同一 Sim(3)
  sfm_points.ply
  frames.json
  manifest.json
  genrecon_preflight/
```

固定协议：

1. `SIMPLE_RADIAL` 使用 OpenCV camera model 转为裁边后的 `PINHOLE`。
2. RGB 使用线性 remap；动态 mask 和有效像素使用 nearest-neighbor remap。
3. 输出 alpha 精确等于 `255 - masks_dynamic`，不修改动态区域 RGB。
4. 2D feature observations 使用同一无畸变相机变换，point IDs 和 observation 顺序不变。
5. `z-up` 使用原生 COLMAP 相机平均 up direction。
6. floor proxy 使用质量点 z 的 2nd percentile。
7. 尺度令相机中位高度为 1.6 m；状态固定为 `proxy-not-ground-truth-metric`。
8. GenRecon 点过滤固定为 `ERROR <= 2 px` 且 `track length >= 4`。
9. consumer preflight 和正式推理均要求零 closest-camera fallback。

Adapter 后 reprojection P90 为 1.788–2.044 px，固定 gate 为 2.5 px。该 gate 验证无畸变和 Sim(3) 没有破坏原观测，不把 source COLMAP error 重新解释为绝对几何精度。

## 4. 逐场景结果

| Candidate | Registered | Native points / clean | Chunks | Fallback | Vertices / Faces | PLY | GLB | Reconstruction / GLB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw-001-copped-hall | 96/99 | 39,675 / 6,882 | 9 | 0 | 10.91M / 23.25M | 475.5 MiB | 289.2 MiB | 462.9 / 315.3 s |
| raw-010-waco-fire-station | 37/37 | 7,926 / 3,381 | 4 | 0 | 5.14M / 10.90M | 223.4 MiB | 131.6 MiB | 262.8 / 128.5 s |
| raw-016-harnden-tavern | 170/170 | 30,500 / 12,520 | 6 | 0 | 8.58M / 18.47M | 376.2 MiB | 190.1 MiB | 352.0 / 240.3 s |

核心 SHA256：

| Candidate | `mesh.ply` | `scene.glb` |
|---|---|---|
| Copped Hall | `e3d3d1d4bbee0e35ac7cc1663fc90b24c1acd26916e8b463d716b84f29335860` | `090deab2080fb6470df9b4198cd9108aea273646cff4c652f3936014f2d91789` |
| Waco | `53151590ad1b2e45e8bba58d291f9f3e6ab6d9acb9921cdc60c4db8e49e8c63d` | `23818ae521daa53d5393bb86d96d8c929ba6291e8c9cc5973cb7cfce5f2776cf` |
| Harnden | `94023fc870f19c4c29c863a487e06efc39a40dbe9f38b2cb65f707476039449e` | `384f4a141ffaee93787ef8327df80917bf491e26da2a68d4707923b80ab97987` |

## 5. 逐帧视频

所有 303 个注册相机都按完全相同内外参渲染最终 GLB；没有插值轨迹。

| Candidate | Frames | Playback | Coverage min / median / max | Videos total |
|---|---:|---:|---:|---:|
| Copped Hall | 96 | 48.0 s | 13.4% / 54.7% / 94.2% | 16.48 MB |
| Waco | 37 | 18.5 s | 78.6% / 97.7% / 100.0% | 5.12 MB |
| Harnden | 170 | 85.0 s | 24.1% / 77.5% / 97.8% | 23.77 MB |

每场景包含：

```text
original.mp4
reconstruction.mp4
side_by_side.mp4
frames/{original,reconstruction,reconstruction_mask,comparison}/
contacts/contact_NNN.jpg
```

`overview.jpg` 均匀采样 16 帧；完整逐帧 contact 每页最多 20 帧。9 个 MP4 均由 OpenCV 检查帧数/FPS/尺寸，并通过 bundled FFmpeg 7.0.2 全流解码。

## 6. 目录关系

```text
data/internet-zero-shot/native-sfm-v1/
  config.json
  index.json
  summary.csv
  validation.json
  candidates/
        |
        v
outputs/internet-zero-shot/native-sfm-genrecon-v1/
  config.json
  index.json
  summary.csv
  validation.json
  candidates/<candidate_id>/reconstruction/{mesh.ply,scene.glb,...}
        |
        v
outputs/internet-zero-shot/native-sfm-video-comparisons-v1/
  index.html
  index.json
  summary.csv
  validation.json
  candidates/<candidate_id>/{original.mp4,reconstruction.mp4,side_by_side.mp4,...}
        |
        v
reports/generated/internet-zero-shot/
  native-sfm-genrecon-v1/
  native-sfm-video-comparisons-v1/
```

## 7. 验证

输入 adapter `validation.json`：

- 3/3 preflight passed，19 chunks，零 fallback。
- 303 个 RGBA 与动态 mask 逐像素互补。
- 435,562,803 个 alpha pixels 已检查。
- 78,101 个原生 COLMAP points 已重新解析。
- 所有 adapter JSON 为严格 JSON。

Mesh/GLB `validation.json`：

- 3/3 PLY binary payload 精确匹配 header。
- 3/3 GLB 为 glTF 2.0 且包含嵌入 base-color 与 metallic/roughness textures。
- 19/19 chunk GLB 有效。
- 6 个最终 PLY/GLB 的 SHA256 与 profile 一致。

视频 `validation.json`：

- 303 对 original/reconstruction/mask/comparison frames 有效。
- 原图与记录源 PNG 对应；重建与 Open3D render cache 逐像素一致。
- 9/9 H.264 MP4 解码到准确帧数和 2 fps。
- 3 个连续原片共 839,857,921 B，SHA256 3/3 匹配。
- 1440x1200 与 390x844 审查页已人工检查，无布局重叠。

## 8. 视觉与权利边界

- Copped Hall：进入 `proceed_engineering_pilot`。原生单-shot 结果与旧 140 帧双片段 pilot 不是同一输入；当前 mesh 仍有缺面和暗区，不能替代真实扫描。
- Waco：保持 `reject_current_shot_person_dominated`。高 coverage 主要说明洗衣设备和房间表面被 mesh 覆盖，不解决持续人物、隐私和未声明许可。
- Harnden：保持 `reject_current_shot_person_dominated`。人物 mask 不完整，重建中仍可见人物状伪影；许可未声明。
- `SfM-A` 只表示 masked COLMAP 几何 gate 通过，不覆盖场景语义、隐私、许可、日期或 zero-shot 污染。
- 1.6 m camera-height scale 和 camera-up gravity 均为 proxy，不支持厘米级声明。

## 9. 复现命令

```bash
.venv/bin/python tools/prepare_native_sfm.py all

.venv/bin/python tools/run_foundation_genrecon_batch.py all \
  --input-root data/internet-zero-shot/native-sfm-v1 \
  --output-root outputs/internet-zero-shot/native-sfm-genrecon-v1 \
  --report-root reports/generated/internet-zero-shot/native-sfm-genrecon-v1 \
  --track-name native-sfm \
  --colmap-subdir colmap_sfm \
  --max-reproj-error 2 \
  --min-track-len 4

.venv/bin/python tools/export_foundation_genrecon_videos.py all \
  --scene-root data/internet-zero-shot/native-sfm-v1 \
  --preproducts-root data/internet-zero-shot/native-sfm-v1 \
  --genrecon-root outputs/internet-zero-shot/native-sfm-genrecon-v1 \
  --output-root outputs/internet-zero-shot/native-sfm-video-comparisons-v1 \
  --report-root reports/generated/internet-zero-shot/native-sfm-video-comparisons-v1 \
  --track-name native-sfm \
  --colmap-subdir colmap_sfm
```
