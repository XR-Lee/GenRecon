# Foundation GenRecon 原始帧与重建视频逐帧对比报告

## 1. 结论

已为 foundation-GenRecon v1 的全部 17 个可消费候选导出原始选帧视频、重建渲染视频和左右逐帧对比视频。

- 候选：17/17。
- 有显式 foundation 预测相机的帧：235/235。
- MP4：51 个，每场景 3 个。
- 总检查播放时长：117.5 秒。
- MP4 总大小：39,242,025 B，约 37.4 MiB。
- 17 个连续原始视频：3,288,935,666 B，已全部重新计算 SHA256 并与 source manifest 匹配。
- 完整逐帧资产目录：约 318 MiB。
- 视频编码：H.264/libx264、CRF 18、`yuv420p`、无音频。
- 固定检查播放速度：2 fps。
- 面板宽度：640 px；左右对比宽度为 1280 px。
- 最终独立验证：`pass`，`errors=[]`。

审查入口：

- `outputs/internet-zero-shot/foundation-video-comparisons-v1/index.html`
- `outputs/internet-zero-shot/foundation-video-comparisons-v1/index.json`
- `outputs/internet-zero-shot/foundation-video-comparisons-v1/summary.csv`
- `outputs/internet-zero-shot/foundation-video-comparisons-v1/validation.json`

## 2. “逐帧”的准确范围

这里的逐帧是指：对 foundation-SfM 输入中每一个具有显式 VGGT 预测相机的原始帧，使用同一个相机内外参渲染最终 `scene.glb`，再按源时间戳一一配对。

它不是原始连续视频中每一个编码帧的重建。当前 17 个场景的原 shot 通常包含最多 180 个 2 fps 预处理帧，但只有 5 至 16 个冻结视图进入本轮 foundation-SfM，并拥有可直接审计的预测相机。对其余帧没有插值或伪造轨迹。

因此：

- `original.mp4` 是有相机位姿的原始选帧按时间排序后的固定 2 fps 检查视频。
- `reconstruction.mp4` 是完全相同帧序和相机的 PBR GLB 渲染。
- `side_by_side.mp4` 左侧为原始选帧，右侧为重建渲染。
- 底栏保存帧名、原视频时间戳、conditioning 角色和 mesh coverage。
- 原视频实际时间间隔可能不均匀；2 fps 仅为检查播放速度，不表示真实时间流速。
- 每个 frame manifest 保留原始 JPEG 绝对路径和 SHA256，可返回连续原片定位该帧。

每场景通常有 8 个 `GenRecon scene conditioning` 视图；其余标记为 `Foundation geometry only`。后者没有直接作为 GenRecon scene image，但已经参与 VGGT pseudo geometry 估计，因此不能称为独立几何 heldout。

## 3. 目录关系

```text
data/internet-zero-shot/raw-candidates-v1/candidates/<candidate_id>/
  source.*                         # 连续原始视频
        |
        v
data/internet-zero-shot/sfm-preproducts-v1/candidates/<candidate_id>/
  frames.json                      # shot、源时间戳和原片 SHA256
  rgb/frame_NNNNNN.jpg             # 从原片精确抽取的 RGB
        |
        v
data/internet-zero-shot/foundation-sfm-v1/candidates/<candidate_id>/
  manifest.json                    # 冻结视图选择
  colmap_vggt/                     # 每个选帧的预测相机
        |
        v
outputs/internet-zero-shot/foundation-genrecon-v1/candidates/<candidate_id>/
  reconstruction/scene.glb         # 当前 GenRecon PBR 资产
        |
        v
outputs/internet-zero-shot/foundation-video-comparisons-v1/candidates/<candidate_id>/
  original.mp4
  reconstruction.mp4
  side_by_side.mp4
  poster.jpg
  overview.jpg
  manifest.json
  render/
    render_glb.json
    views/<group>/<frame>/
      original.jpg
      glb_render.png
      glb_mask.png
      glb_depth_mm.png
      camera.json
  frames/
    original/NNNNNN.jpg
    reconstruction/NNNNNN.png
    reconstruction_mask/NNNNNN.png
    comparison/NNNNNN.jpg
        |
        v
reports/generated/internet-zero-shot/foundation-video-comparisons-v1/
  batch.log
  resume_check.log
  candidates/<candidate_id>/
    render.log
    encode_original.log
    encode_reconstruction.log
    encode_side_by_side.log
```

## 4. 固定协议

```text
renderer = Open3D OffscreenRenderer
GLB coordinate map = exported (x, z, -y) -> COLMAP world
shader = defaultUnlit baked PBR albedo
panel width = 640 px
playback = 2 fps, ascending source timestamp
codec = H.264/libx264, CRF 18, yuv420p
missing surface/background = black
camera interpolation = disabled
audio = none
```

原始视频面板使用 `sfm-preproducts-v1/rgb/` 中由原片抽取的 JPEG，不使用重建后的颜色作为参考。独立 `original.mp4` 和 `reconstruction.mp4` 不添加文字；只有 `side_by_side.mp4` 添加固定 header/footer。

复现命令：

```bash
.venv/bin/python tools/export_foundation_genrecon_videos.py all
```

在已有完整批次中可单场景恢复，再重建全局索引和验证：

```bash
.venv/bin/python tools/export_foundation_genrecon_videos.py run \
  --candidate raw-009-wawasee-performing-arts
.venv/bin/python tools/export_foundation_genrecon_videos.py summarize
.venv/bin/python tools/export_foundation_genrecon_videos.py validate
```

## 5. 逐场景结果

Coverage 只表示当前相机下有多少像素命中生成 mesh，不衡量命中的表面是否几何正确、颜色正确或属于同一真实房间。

| Candidate | Grade | Disposition | Frames | Source timestamps | Playback | Coverage min/median/max |
|---|---:|---|---:|---:|---:|---:|
| raw-002-cologne-cathedral | P-C | refine_then_retry | 16 | 197.000–225.500 s | 8.0 s | 96.7% / 100.0% / 100.0% |
| raw-003-castleton-house-tour | P-F | reject_current_shot | 16 | 237.000–244.500 s | 8.0 s | 78.4% / 78.5% / 78.5% |
| raw-004-blue-rose-ballroom | P-F | reject_current_shot | 16 | 624.000–631.500 s | 8.0 s | 51.9% / 52.0% / 52.0% |
| raw-005-beverly-theater | P-A | reject_current_shot | 16 | 215.700–223.700 s | 8.0 s | 69.2% / 91.0% / 91.8% |
| raw-006-evel-knievel-museum | P-A | refine_then_retry | 16 | 254.500–269.500 s | 8.0 s | 59.1% / 59.7% / 76.1% |
| raw-007-burlington-high-tech | P-B | refine_then_retry | 16 | 1758.500–1766.000 s | 8.0 s | 53.4% / 90.5% / 100.0% |
| raw-008-hilltop-school-arts | P-C | refine_then_retry | 16 | 11.675–19.175 s | 8.0 s | 62.7% / 68.9% / 74.0% |
| raw-009-wawasee-performing-arts | P-A | proceed_foundation_pilot | 16 | 262.000–309.500 s | 8.0 s | 11.3% / 74.3% / 87.4% |
| raw-011-lake-johanna-fire-station | P-F | reject_current_shot | 5 | 31.725–33.725 s | 2.5 s | 61.1% / 63.3% / 64.6% |
| raw-012-bartlett-bay-wastewater | P-F | reject_current_shot | 16 | 974.500–982.000 s | 8.0 s | 49.4% / 50.4% / 50.8% |
| raw-013-lincoln-health-security | P-C | reject_current_shot | 9 | 33.000–37.000 s | 4.5 s | 41.5% / 74.1% / 74.5% |
| raw-014-unity-house | P-B | proceed_foundation_pilot | 16 | 65.000–72.500 s | 8.0 s | 41.7% / 52.0% / 79.6% |
| raw-015-moses-myers-house | P-F | reject_current_shot | 16 | 1693.500–1701.000 s | 8.0 s | 52.7% / 52.7% / 52.9% |
| raw-017-frederick-funeral-home | P-C | reject_current_shot | 7 | 621.350–624.350 s | 3.5 s | 29.9% / 38.1% / 39.2% |
| raw-018-jabal-restaurant | P-A | reject_current_shot | 16 | 180.725–193.225 s | 8.0 s | 37.5% / 39.3% / 49.5% |
| raw-019-crockett-middle-school | P-B | refine_then_retry | 16 | 15.050–22.550 s | 8.0 s | 42.6% / 61.2% / 75.5% |
| raw-020-bethel-lutheran-home | P-F | reject_current_shot | 6 | 24.475–26.975 s | 3.0 s | 61.0% / 64.0% / 65.3% |

## 6. 独立验证

`validation.json` 当前为 `pass`，验证范围包括：

1. 17 个 candidate manifest 均为 complete。
2. 17 个连续原始视频全部存在；共 3,288,935,666 B，重新计算的 SHA256 与 source manifest 17/17 匹配，见 `source_video_hash_validation.json`。
3. 235 个原始输出帧均可解码、尺寸一致，并与记录的源 JPEG SHA256 对应。
4. 原始输出帧与源 JPEG 按同一 camera resolution 缩放后的平均像素误差不超过 JPEG 容差。
5. 235 个重建 PNG 与 Open3D `glb_render.png` cache 逐像素一致。
6. 235 个 reconstruction mask 为二值图，并与 render alpha 逐像素一致。
7. 235 个 comparison frame 与记录尺寸一致，时间戳单调递增。
8. 51 个 MP4 SHA256 与 manifest 一致。
9. 51 个 MP4 均由 OpenCV 重新解码到准确帧数、尺寸和 2 fps，并再次通过 bundled FFmpeg 7.0.2 全流解码；日志为 `reports/generated/internet-zero-shot/foundation-video-comparisons-v1/ffmpeg_decode_validation.log`。
10. 119 个 HTML 本地资源引用全部存在，包括 17 个连续原始视频文件链接。
11. 1440×1200 和 390×844 审查页截图尺寸正确且人工检查无文本/视频重叠。
12. 290 个输出及报告 JSON 均为严格 JSON，无 `NaN` 或 `Infinity`。

## 7. 视觉解释

视频使若干此前仅靠技术指标不明显的问题直接可见：

- Cologne 的 coverage 接近 100%，但生成表面仍有明显形变和纹理拖拽，说明 coverage 不能替代几何/视觉判断。
- Castleton、Beverly 等外景同样能产生高 coverage；它们仍不属于室内单房间有效样本。
- Wawasee 个别视角只有约 11% coverage，窗口/服务口结构较稳定，但房间表面不完整。
- Unity House 的床架可辨，房间外壳和许多表面缺失，适合作为 foundation-pseudo-geometry stress test，不是成品资产。
- P-F 或 `reject_current_shot` 场景的视频是失败/负对照，不因能够播放而升级。

## 8. 完整连续原片的下一层协议

若要求对原始 shot 中每一个 2 fps 帧，甚至原视频 25/30 fps 每个编码帧，都生成严格对齐的重建视图，必须先为这些帧获得可审计相机。可行路径为：

1. 对真实 COLMAP 已注册帧直接使用 SfM 相机，并把 GenRecon GLB 保持在同一世界坐标。
2. 对 foundation-only 场景以重叠窗口运行相机模型，再用共享帧做全局 Sim(3) 对齐和轨迹连续性 gate。
3. 无法求得相机的帧标记为 `unposed`，不进入逐帧 fidelity 对比。
4. 相机插值只可单列为 synthetic fly-through，不能冒充原视频视角重建。

当前导出选择了最保守、可复核的范围：只比较具有明确相机的 235 个真实源帧。
