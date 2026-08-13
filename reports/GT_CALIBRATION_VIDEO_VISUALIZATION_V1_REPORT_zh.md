# GT Calibration 代表场景视频可视化 v1

## 1. 交付结论

已经为冻结 GT calibration suite 的 7 个数据集建立统一视频审查入口：

- 真实 prepared 数据集代表：7 个，source blocker 为 0；
- 每个代表使用冻结的 8 conditioning + 8 heldout 相机，共 112 个真实相机帧；
- 每个代表输出 `source.mp4`、`reference.mp4`、`prediction.mp4`、`comparison.mp4`；
- 7 个代表均有 registry-backed GenRecon prediction render；OmniObject3D `bottle_045` 第三栏为最终 RGB-masked prediction，不再是 status card；
- 五个 representative prediction 只由 8 张 conditioning RGB 和 conditioning camera records 构建，不读取 heldout RGB/depth、conditioning depth 或 reference geometry；
- Omni reference 使用 official professional scan 在 audited normalized-object render frame 中的同相机渲染，绝不冒充 prediction 或 conditioning；
- 总计 29 个 H.264/yuv420p 视频、54,679,055 bytes，完整解码 560 帧；
- 内部 validation 与独立 release validation 均为 `pass`，`errors=[]`。

浏览入口：

- `outputs/gt-calibration-v1/visualizations-v1/index.html`
- `outputs/gt-calibration-v1/visualizations-v1/all_datasets_overview.mp4`
- `outputs/gt-calibration-v1/visualizations-v1/dataset_contact.jpg`

## 2. 三栏协议

每个 prepared 代表场景的 `comparison.mp4` 固定为三栏：

1. `REFERENCE RGB`：calibration package 中冻结的原始 conditioning/heldout 图像；
2. `GT / REFERENCE GEOMETRY`：按同一冻结相机渲染的评测 reference，只用于可视化与评测；
3. `GENRECON PREDICTION`：只渲染 registry 中具有完整 prediction package provenance 的 mesh；registry membership、路径、SHA 或 package source chain 不一致时 exporter/validator 失败。若 registry 明确没有 prediction，则该栏只能输出 `missing-prediction` status card，不能渲染 reference geometry。

所有视频遵循：

- conditioning 0–7 在前，heldout 0–7 在后；
- 2 fps 是检查播放速度，不表示源序列真实时间；
- 不插值相机，不生成 synthetic fly-through；
- reference geometry 从未作为 GenRecon conditioning；
- point-cloud reference 最多确定性抽样 2.5M 点，仅用于交互渲染，不改 GT 评测文件；
- 黑色背景表示该相机射线上无可渲染几何；
- 非黑画面、coverage 高或视觉对应不等于几何指标优秀。

ScanNet++ 的 frozen RGB 使用 manifest 中 OPENCV 参数去畸变后进入展示，reference/prediction 使用相同 PINHOLE 投影。ETH3D 的两个 laser scans 分别应用官方 MeshLab `.mlp` transform 后再渲染。T&T 使用已经审计的官方 COLMAP-to-laser Sim(3) 与 fixed-pose 恢复内参。五个 representative predictions 使用 conditioning-camera-only Sim(3) 从 GenRecon work frame 映射到各自 official frame；GT/reference geometry 和 GT ICP 均未参与该映射。

## 3. 代表场景

| 数据集 | 代表场景 | GT 层级 | Reference render | Prediction | Reference min / median | Prediction min / median | 主视频 SHA256 |
|---|---|---|---|---|---:|---:|---|
| ScanNet++ | `286b55a2bf` | G0 | laser scan mesh | available | 0.7527 / 0.9147 | 0.9954 / 1.0000 | `be5eba47787e28fa8784dbb4b398a8fed49d0161ab2b72819cf8e792069aa6d7` |
| ETH3D | `delivery_area` | G0 | 2 aligned laser point clouds | available | 0.7403 / 0.8432 | 0.2746 / 0.5320 | `468823d0a822c34d85d1e19bec45cce990c1d4eb3d0a455082fb9a164aca0b00` |
| Tanks and Temples | `Meetingroom` | G0 | official-crop 1 cm laser points | available | 0.8974 / 0.9427 | 0.7057 / 0.8327 | `c026f4b34e1f52f6ee808544707339b998849608e7b978b29aad074f1ee26410` |
| 7-Scenes | `chess` | G1 | clean-depth fusion points | available | 0.5823 / 0.7938 | 0.3106 / 0.6872 | `40a74e0fcc8b8470136aca389540f713470b92670a7a8dfe836fd7eff8e7b351` |
| Redwood | `livingroom` | G2 | exact synthetic points | available | 0.8285 / 0.9466 | 0.3345 / 0.8313 | `784dfd11e9b032ae3f5922dc8436df74b6e41a3b002858481c7037c3a38388c9` |
| DTU MVS | `scan24` | O0 | structured-light points | available | 0.4111 / 0.7790 | 0.4634 / 0.6322 | `eff0e77c7170a536ed1ffa567b49524a4168c613df4d6ed58abc6ad84eeb5e3a` |
| OmniObject3D | `bottle_045` | O0 | normalized professional scan mesh | available | 0.0460 / 0.1162 | 0.0078 / 0.0236 | `f95b8d8a2a06808027103eeba6081228de632f6e69a71a164ef8dbce66f2ec29` |

这里的 coverage 是 reference/prediction render mask 在 640x480 panel 中的像素占比，只用于发现空帧、相机错位或异常包围，不是 F-score、recall 或完整度指标。所有 coverage 均原样保留，未用 GT AABB 裁掉 prediction。全 release 的 reference coverage 为 0.0460/0.8414/0.9988（min/median/max），prediction 为 0.0078/0.7339/1.0000。Omni 的 prediction coverage 极低，和其 fragmented topology 与 2% recall 3.387% 一致，但 coverage 本身仍不是几何指标。

## 4. 人工目检

对每个可用数据集检查 conditioning 首帧、heldout 首帧和 2x2 overview：

- ScanNet++：马桶、红柜、洗手盆和门框在 RGB、scan mesh、prediction 中同位；prediction 近满屏覆盖需结合定量 geometry 指标解释；
- ETH3D：警示柱、卷帘门、车辆和墙面在 RGB、laser 和 prediction 中同位；prediction 黑区暴露未覆盖表面；
- T&T：reference 中梁柱、吊灯、椅桌、门框和镜面结构同位；prediction 主体相机方向与房间布局一致，但细部与缺失区仍需结合 official-crop 指标审查；
- 7-Scenes：棋桌、显示器、红墙和白板在 RGB/reference/prediction 中保持同一视向，prediction 存在局部缺失和生成表面；
- Redwood：椅子、窗帘、灯具和墙角的视向一致；该输入是低重叠 `P-C marginal`，视频不能把模型内部自洽解释为真实精度；
- DTU：相机方向与建筑模型位置正确，但 prediction 出现大面积绿色/白色背景平面并遮挡建筑，属于真实 hallucination；F@10 cm 较高不能覆盖这一视觉失败；
- OmniObject3D：RGB-derived exact-white background mask 消除了第一次无效推理的全画面背景平面，但最终 prediction 将浅色瓶盖和绿色瓶身塌缩成两个分离的扁平 patch；conditioning 与 heldout 视角都未恢复瓶体拓扑。该项 review status 为 `severe-fragmented-incomplete-reconstruction`，normalized Chamfer 0.199096，2% F-score 0.058624、recall 0.033870。

## 5. 验证

`outputs/gt-calibration-v1/visualizations-v1/release_validation.json`：

| 检查项 | 结果 |
|---|---:|
| Dataset statuses | 7 |
| Available / blocked | 7 / 0 |
| With prediction / missing prediction | 7 / 0 |
| Frozen camera frames | 112 |
| Candidate videos | 28 |
| Overview videos | 1 |
| Candidate decoded frames | 448 |
| Overview decoded frames | 112 |
| Total decoded frames | 560 |
| Video bytes | 54,679,055 |
| Reference coverage min / median / max | 0.0460 / 0.8414 / 0.9988 |
| Prediction coverage min / median / max | 0.0078 / 0.7339 / 1.0 |
| Result | `pass` |

release validator 重新检查：

- visualization plan、GT registry 和 exporter SHA256；
- 7 个代表与冻结 plan/registry 的 membership、顺序、状态；
- 112 个源 RGB、source frame、reference frame、prediction/status frame 和 comparison frame 的 SHA256；
- reference geometry 与 7 个完整 prediction source chains：unit manifest、work-frame PLY/GLB、official-frame PLY/PBR GLB、package manifest、4x4 transform 和 input-contract SHA；
- Omni normalized-object prediction 还绑定 RGB-derived foreground mask、projected-area camera selection、reconstruct `args.json`/`cameras.json` 和 zero-fallback preflight；
- conditioning/heldout 严格 8+8 顺序；
- 每个 prepared 数据集每路视频恰好 16 帧；
- 总览视频恰好 112 帧；
- 29 个 MP4 的 SHA256、字节数、分辨率、fps、逐帧完整解码和非空画面。

关键 SHA256：

- plan：`20c6baab863dda7c23e86a535f37868589e284cbfa7242a220977b9e78c971ff`
- index：`0b860334a06c4232c259def46eb1c3655e9ac84d445773c83ad04cf13d1bafc0`
- internal validation：`8482f24e258d1e24a0a8aa88aba88f7a53254618503c355f7e7505af296d7629`
- overview MP4：`3ab8af412457a7079f07da935b99df3e0e6c83a4bded011bcdf37aae1582eac5`
- dataset contact：`dc0e9ba32fb0f5b85f45ab594c7a71b1f5e0d0ea871cb31922cfdb55342d7914`
- release validation：`f8de4ac55c6a11273bf69420fffb62d392522409417b8ef03f4d5c34cd109c68`

精简离线交付包：

- 路径：`outputs/gt-calibration-v1/genrecon-gt-calibration-offline-preview-v1.zip`；
- 大小：58,176,454 bytes；
- SHA256：`36ed25540bb5fd2f1643359cc266a851d57947fc59fbb465ebec19de32ecb244`；
- 内容：单一顶层目录、58 个 files、29 个 MP4；不含逐帧 PNG，但保留 HTML 的 50 个本地引用及其全部目标；
- 独立复验：`unzip -t`、包内 `SHA256SUMS`、路径穿越/符号链接检查、0 external references、OpenCV 与 bundled FFmpeg 双重完整解码均通过；29 个视频共 560 帧。

## 6. 复现入口

```bash
EGL_PLATFORM=surfaceless .venv/bin/python \
  tools/export_gt_calibration_videos.py all
.venv/bin/python tools/validate_gt_calibration_videos.py
```

只重建一个代表：

```bash
EGL_PLATFORM=surfaceless .venv/bin/python \
  tools/export_gt_calibration_videos.py all \
  --unit tanks-and-temples-meetingroom --force
```

固定配置：`configs/eval/gt_calibration_visualization_v1.json`。

## 7. 尚未完成与禁止声明

1. 五个 representative prediction 只能称为 8-view RGB-only、conditioning-camera-aligned 的 `GT-pose-foundation-pseudo-geometry` 结果，不能称为 estimated-pose 端到端结果。
2. 五者未读取 heldout RGB/depth、conditioning depth、GT laser/fusion/structured-light/scan reference 或 GT ICP；但 foundation-only conditioning RGB 已参与 VGGT geometry，因此不能称为 geometry-independent conditioning evidence。
3. OmniObject3D `bottle_045` 已运行最终 masked GenRecon；第三栏必须保留真实 registry-backed failure prediction，不能用 reference geometry 或旧 missing-status stream 代替。
4. Omni RGB 与 scan 来自同一对象，heldout 只用于同相机审查，不是 geometry-independent evidence；其坐标也不是米制。
5. `GT / REFERENCE GEOMETRY` 栏不是模型输出，也不进入纹理/图像质量比较。
6. 视频是 sparse frozen-camera inspection，不是连续时序、真实帧率或自由视角 fly-through。
7. 视觉同位、nonblank validator 和 render coverage 不能替代 2/5/10 cm、normalized-object、completion/hallucination、heldout full-GT-mask RGB/depth 或 pose 指标。
8. DTU 的背景平面 hallucination 必须与其几何分数同时报告，不能以 F@10 cm 或 coverage 将其判为视觉成功。
