# Internet 室内视频动态 Mask 与 COLMAP 预产物报告

## 1. 结论

本轮对 `raw-candidates-v1` 的 20 条互联网视频执行了统一的 shot 选择、抽帧、动态实例 mask、masked COLMAP 和 SfM 质检。原始视频目录未被修改。

最终使用 v2 质检协议得到：

- `A / pass`：3 条，分别为 Copped Hall、Waco Fire Station、Harnden Tavern。
- `B / pass`：0 条。
- `C / marginal`：5 条，分别为 Beverly Theater、Evel Knievel Museum、Hilltop School、Wawasee Performing Arts、Jabal Restaurant。
- `F / fail`：12 条。
- COLMAP 正常完成 19 条；Bartlett Bay Wastewater 因反复 bundle-adjustment 线性求解失败，在约 408 秒后按资源 gate 终止。

这是 SfM 技术分级，不是正式数据集接纳结论。`A` 条目仍需人工确认房间身份、静态性、隐私、许可、尺度和重力方向。

## 2. 入口

- 预产物目录：`data/internet-zero-shot/sfm-preproducts-v1/`
- 可视化审查页：`data/internet-zero-shot/sfm-preproducts-v1/index.html`
- 机器索引：`data/internet-zero-shot/sfm-preproducts-v1/index.json`
- CSV 汇总：`data/internet-zero-shot/sfm-preproducts-v1/summary.csv`
- 独立验证：`data/internet-zero-shot/sfm-preproducts-v1/validation.json`
- 处理工具：`tools/prepare_internet_sfm_preproducts.py`

## 3. 预处理协议

### 3.1 Shot 和抽帧

- 处理单位是一条自动选择的连续候选区间，而不是整条长视频。
- FFmpeg 以 2 fps 计算 scene score，并用硬切、区间长度、背景 ORB 特征、背景单应运动、匹配连续性和动态占比进行确定性排序。
- 每条选择 `8-90 s` 区间；抽帧 2 fps，最多 180 帧。
- RGB 长边最多 1600 像素，低分辨率来源不放大。
- 自动选择结果完整保存在每条 `selection.json`，并明确标记需要人工复核。

自动选择不能可靠理解“同一物理房间”。Castleton、Blue Rose、Moses Myers 等条目找到了有视觉变化的区间，但最终无法形成有效 SfM，这些失败被保留。

### 3.2 动态 Mask

模型为 torchvision Mask R-CNN：

- 架构：`maskrcnn_resnet50_fpn_v2`
- 权重：`MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1`
- torchvision：`0.21.0+cu126`
- checkpoint SHA256：`73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e`
- 实例分数阈值：`0.35`
- mask probability 阈值：`0.5`
- mask 膨胀：图像长边的 `0.8%`

屏蔽类别包括 person、车辆、动物、tv、laptop 和 cell phone。室内场景的主要目标是人物；模型偶尔将装饰或设备误识别为 airplane/train 等类别，因此类别计数只作诊断，不能当作真实物体统计。

每帧保存两种 mask：

- `masks_dynamic/frame_XXXXXX.png`：`255` 表示动态像素。
- `masks_colmap/frame_XXXXXX.jpg.png`：`255` 表示允许提取 SIFT，`0` 表示忽略。

原始 RGB 不做涂黑或 inpainting。当前“去除”仅指在 COLMAP 特征提取和匹配中排除动态区域，避免把粗糙修补图像误当成真实观测。

## 4. COLMAP 配置

- pycolmap：`3.13.0`，CPU-only。
- 相机共享方式：`CameraMode.SINGLE`。
- 相机模型：`SIMPLE_RADIAL`。
- SIFT 工作长边：1280 像素。
- SIFT 基础特征上限：4096；多方向复制后数据库实际数量可以更高。
- 60 帧及以下使用 exhaustive matching；其余使用 sequential quadratic matching。
- sequential overlap：10。
- guided matching：启用。
- mapper seed：42。
- mapper 单条运行上限：300 秒；Wastewater 的首次运行由外部资源 gate 在约 408 秒终止并保留数据库。

每条最佳模型同时保存 binary、COLMAP text 和 sparse PLY。所有模型仍为单目任意尺度，且没有完成 `z-up` 对齐。

## 5. v2 质检协议

质检以最大模型为主，同时统计所有模型的注册并集和最大模型 dominance。

`F / fail` 的核心条件包括：

- 没有重建模型；
- 最大模型注册率低于 40%；
- 稀疏点少于 500；
- 平均重投影误差缺失或高于 4 px；
- COLMAP 在黑色 mask 像素上产生 keypoint；
- 相机基线/观测深度低于 0.10；
- 中位三角化角低于 2 度；
- mapper 异常或资源超时。

`A / pass` 要求同时满足：

- 注册率至少 90%；
- 至少 5,000 个稀疏点和 3,000 个质量点；
- 平均重投影误差不高于 1.5 px；
- 平均 track 长度至少 4；
- 最大模型 dominance 至少 90%；
- 基线/观测深度至少 0.50；
- 中位三角化角至少 4 度；
- 平均动态占比不高于 25%；
- 无 mask 泄漏和异常内参。

`B / pass` 放宽注册率、点数、误差和 track 要求，但保持相同视点几何和动态占比要求。本轮没有 B。未触发 F、但达不到 A/B 的条目标为 `C / marginal`。

加入视点几何检查后，Jabal 从粗略 A 降为 C。它虽然 `26/26` 注册，但基线/观测深度仅 `0.190`、中位三角化角 `3.29` 度、动态占比 `31.1%`，实际是近固定机位的人物进食镜头。

## 6. 全量结果

| ID | Shot (s) | 动态 | 最大模型注册 | 点数 | 误差 px | Track | 基线/深度 | 三角化角 | 等级 | 主要原因 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| raw-001 Copped Hall | 254.5-304.0 | 1.9% | 96/99 | 39,675 | 0.899 | 7.47 | 2.93 | 6.23 | A | 通过 |
| raw-002 Cologne Cathedral | 136.0-226.0 | 10.6% | 58/180 | 11,379 | 1.279 | 5.79 | 1.11 | 6.10 | F | 6 个模型，最大模型注册率 32.2% |
| raw-003 Castleton | 185.0-275.0 | 1.3% | 0/180 | 0 | - | - | - | - | F | 幻灯片/外景，无模型 |
| raw-004 Blue Rose | 581.5-641.5 | 22.8% | 0/120 | 0 | - | - | - | - | F | 无模型 |
| raw-005 Beverly Theater | 211.7-224.3 | 1.0% | 17/25 | 7,295 | 0.676 | 6.88 | 0.47 | 4.65 | C | 两个碎片，注册率 68% |
| raw-006 Evel Museum | 242.5-270.4 | 27.2% | 33/56 | 7,092 | 1.117 | 11.38 | 0.71 | 4.91 | C | 注册率 58.9%，动态占比高 |
| raw-007 Burlington High | 1677.0-1767.0 | 21.0% | 0/180 | 0 | - | - | - | - | F | 无模型 |
| raw-008 Hilltop School | 11.7-23.8 | 10.7% | 16/24 | 9,095 | 0.977 | 4.22 | 0.32 | 4.58 | C | 注册率 66.7%，低基线 |
| raw-009 Wawasee Arts | 221.5-311.5 | 12.1% | 102/180 | 9,259 | 0.932 | 8.40 | 1.51 | 5.45 | C | 3 个模型，dominance 56.7% |
| raw-010 Waco Fire Station | 165.5-184.0 | 15.7% | 37/37 | 7,926 | 0.667 | 9.75 | 2.01 | 5.18 | A | 通过 |
| raw-011 Lake Johanna | 27.2-40.3 | 35.9% | 5/26 | 1,386 | 0.873 | 2.34 | 0.47 | 11.00 | F | 注册率低、动态高、内参异常 |
| raw-012 Wastewater | 905.5-995.5 | 28.6% | 0/180 | 0 | - | - | - | - | F | mapper timeout，154 次线性求解失败 |
| raw-013 Lincoln Institute | 5.5-37.3 | 5.7% | 9/64 | 0 | 0.000 | - | - | - | F | 270x480，零有效点、内参异常 |
| raw-014 Unity House | 1.0-91.0 | 21.5% | 2/180 | 274 | 0.059 | 2.00 | 0.34 | 19.38 | F | 270x480，注册率和点数过低 |
| raw-015 Moses Myers | 1664.0-1754.0 | 4.4% | 0/180 | 0 | - | - | - | - | F | 幻灯片段，无模型 |
| raw-016 Harnden Tavern | 1553.0-1638.0 | 15.1% | 170/170 | 30,500 | 0.932 | 17.52 | 0.88 | 4.97 | A | 通过 |
| raw-017 Frederick Funeral | 609.4-624.6 | 6.8% | 7/31 | 3,149 | 0.505 | 4.38 | 0.61 | 17.63 | F | 注册率 22.6%、内参异常 |
| raw-018 Jabal Restaurant | 180.7-193.8 | 31.1% | 26/26 | 6,858 | 0.792 | 11.54 | 0.19 | 3.29 | C | 近固定机位、动态高、三角化弱 |
| raw-019 Crockett School | 13.1-22.9 | 21.6% | 3/20 | 4 | 0.000 | 2.00 | 0.27 | 5.47 | F | 镜头剪辑，注册率和点数过低 |
| raw-020 Bethel Home | 24.5-33.0 | 37.9% | 6/17 | 855 | 0.426 | 4.35 | 0.13 | 4.88 | F | 注册率低、动态高、低基线 |

## 7. 产物结构

每条候选目录包含：

```text
candidates/<candidate-id>/
├── selection.json             # 全部候选区间、自动评分和最终选择
├── scene_scores.txt           # FFmpeg scene score 原始记录
├── frames.json
├── rgb/                       # 未修改的抽取 RGB
├── masks_dynamic/             # 255 = 动态
├── masks_colmap/              # 255 = 允许 SIFT，0 = 忽略
├── mask_summary.json
├── mask_contact.jpg           # 动态 mask 叠图
├── database.db                # masked COLMAP 数据库
├── sparse/                    # 全部 binary 模型
├── sparse_best/               # 最大模型 binary
├── colmap/                    # 最大模型 cameras/images/points3D.txt
├── sparse_points.ply
├── qc_contact.jpg             # 绿色注册、红色未注册
├── trajectory.png             # 相机与稀疏点 PCA 投影
└── quality.json               # v2 完整质检
```

无模型的条目不会伪造 `colmap/`。Wastewater 保留完整数据库和失败诊断，但没有 partial sparse model。

## 8. 完整性验证

独立 validator 的结果为 `pass`：

- 20/20 条完成预处理，共 1,975 张 RGB。
- 1,975 个动态 mask 和 1,975 个 COLMAP mask 全部可解码、二值、尺寸一致且逐像素互补。
- 20 个数据库共 6,691,454 个 keypoint；落在 COLMAP 黑色 mask 上的 keypoint 为 0。
- 19 条 COLMAP 正常完成，1 条资源超时。
- 15 个最佳模型可重新加载，注册帧和点数与 `quality.json` 一致。
- 最佳模型合计 134,747 个稀疏点。
- 102 个验证前 JSON 全部拒绝 `NaN/Infinity` 并严格解析。
- Mask R-CNN checkpoint 哈希与 manifest 一致。
- 相关测试：33 passed。

记录的逐条预处理耗时合计约 707 秒，SfM/质检耗时合计约 1,735 秒。最终预产物目录约 1.9 GiB。

## 9. 限制与后续处置

1. `A` 只表示当前 shot 的 masked SfM 技术通过。必须人工打开 `qc_contact.jpg` 和原视频，确认没有跨房间、镜面主导、残留动态实例或不可发布人物。
2. Mask R-CNN 是 COCO 语义实例模型，不检测所有真实运动。窗外运动、反射、投影画面和未覆盖类别仍可能残留。
3. 预产物 RGB 本身仍未做生成式去物体或 inpainting。后续 native adapter 已无畸变为 RGBA，保持 RGB 不变并令动态区域 alpha=0；人物漏检仍会进入条件图和生成结果。
4. 原 SfM 模型仍为任意单目尺度且未对齐重力。后续 native adapter 的 `z-up` 与 1.6 m 相机高度只属于 proxy normalization，仍不能声明真实米制几何质量。
5. `C` 条目应优先人工切出更短且单房间的子 shot，再运行相同参数；不应直接进入 GenRecon 主榜。
6. `F` 条目在当前自动区间下停止。除非重新选择原视频中的另一段连续镜头，否则不继续消耗 GenRecon 资源。
7. Copped Hall、Waco Fire Station 和 Harnden Tavern 已完成统一 native adapter、8-view consumer preflight、完整 GenRecon、PLY/PBR GLB 和所有注册相机视频：3/3 preflight passed、19 chunks、零 fallback。Copped 仅作 engineering pilot；Waco/Harnden 因人物与许可继续拒绝。详见 [Native COLMAP-SfM 到 GenRecon 完整产物报告](NATIVE_SFM_GENRECON_REPORT_zh.md)。
