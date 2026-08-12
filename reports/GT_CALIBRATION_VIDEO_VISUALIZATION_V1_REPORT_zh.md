# GT Calibration 代表场景视频可视化 v1

## 1. 交付结论

已经为冻结 GT calibration suite 的 7 个数据集建立统一视频审查入口：

- 可用真实数据集代表：6 个；
- 授权阻塞代表：OmniObject3D 1 个；
- 每个可用代表使用冻结的 8 conditioning + 8 heldout 相机，共 96 个真实相机帧；
- 每个可用代表输出 `source.mp4`、`reference.mp4`、`prediction.mp4`、`comparison.mp4`；
- ScanNet++ 与 ETH3D 有真实 GenRecon prediction render；
- T&T、7-Scenes、Redwood、DTU 尚无 prediction，第三栏是显式 `missing-prediction` 状态卡，不是伪造渲染；
- OmniObject3D 仍为 `blocked-auth`，只输出授权状态视频，不展示虚构 RGB、object ID、scan 或 prediction；
- 总计 26 个 H.264/yuv420p 视频、44,164,015 bytes，完整解码 496 帧；
- 独立 release validation：`pass`，`errors=[]`。

浏览入口：

- `outputs/gt-calibration-v1/visualizations-v1/index.html`
- `outputs/gt-calibration-v1/visualizations-v1/all_datasets_overview.mp4`
- `outputs/gt-calibration-v1/visualizations-v1/dataset_contact.jpg`

## 2. 三栏协议

每个 prepared 代表场景的 `comparison.mp4` 固定为三栏：

1. `REFERENCE RGB`：calibration package 中冻结的原始 conditioning/heldout 图像；
2. `GT / REFERENCE GEOMETRY`：按同一冻结相机渲染的评测 reference，只用于可视化与评测；
3. `GENRECON PREDICTION`：只有 registry 已存在 prediction mesh 时才渲染；否则显示不可误认的状态卡。

所有视频遵循：

- conditioning 0–7 在前，heldout 0–7 在后；
- 2 fps 是检查播放速度，不表示源序列真实时间；
- 不插值相机，不生成 synthetic fly-through；
- reference geometry 从未作为 GenRecon conditioning；
- point-cloud reference 最多确定性抽样 2.5M 点，仅用于交互渲染，不改 GT 评测文件；
- 黑色背景表示该相机射线上无可渲染几何；
- 非黑画面、coverage 高或视觉对应不等于几何指标优秀。

ScanNet++ 的 frozen RGB 使用 manifest 中 OPENCV 参数去畸变后进入展示，reference/prediction 使用相同 PINHOLE 投影。ETH3D 的两个 laser scans 分别应用官方 MeshLab `.mlp` transform 后再渲染。T&T 使用已经审计的官方 COLMAP-to-laser Sim(3) 与 fixed-pose 恢复内参。

## 3. 代表场景

| 数据集 | 代表场景 | GT 层级 | Reference render | Prediction | Reference coverage min / median | 主视频 SHA256 |
|---|---|---|---|---|---:|---|
| ScanNet++ | `286b55a2bf` | G0 | laser scan mesh | available | 0.7527 / 0.9218 | `be5eba47787e28fa8784dbb4b398a8fed49d0161ab2b72819cf8e792069aa6d7` |
| ETH3D | `delivery_area` | G0 | 2 aligned laser point clouds | available | 0.7403 / 0.8457 | `468823d0a822c34d85d1e19bec45cce990c1d4eb3d0a455082fb9a164aca0b00` |
| Tanks and Temples | `Meetingroom` | G0 | official-crop 1 cm laser points | missing | 0.8974 / 0.9472 | `76ad93e68d5272a878510f24f3f2eb8a23f2ea30ac6325d269d482cfe409e9e8` |
| 7-Scenes | `chess` | G1 | clean-depth fusion points | missing | 0.5823 / 0.7977 | `70cfe2110bff35f27d84f4a8d28d2580a65d4283414eaf20734540b6f09a5739` |
| Redwood | `livingroom` | G2 | exact synthetic points | missing | 0.8285 / 0.9602 | `b471e231afc8a9f13a9b6fae166f6f69ab29291f2b2cb001599d5cd2d0b910c7` |
| DTU MVS | `scan24` | O0 | structured-light points | missing | 0.4111 / 0.8074 | `430182182999c574052014c6e7a8ce4d627ca5b2635c6f89250056c0e0d9b774` |
| OmniObject3D | requested bottle slot | O0 | unavailable | blocked-auth | 不适用 | `6724cf44dc7c901bbad30ee410c74094895f0d3f77c886f6b3e399d26c9b4827` |

这里的 coverage 是 reference/prediction render mask 在 640x480 panel 中的像素占比，只用于发现空帧、相机错位或异常包围，不是 F-score、recall 或完整度指标。ScanNet++ prediction 的 coverage 为 0.9954–1.0，视频中保留这一近满屏覆盖现象，不用 GT AABB 裁掉；ETH3D prediction coverage 为 0.2746–0.8866，缺失区域同样原样保留。

## 4. 人工目检

对每个可用数据集检查 conditioning 首帧、heldout 首帧和 2x2 overview：

- ScanNet++：马桶、红柜、洗手盆和门框在 RGB、scan mesh、prediction 中同位；prediction 近满屏覆盖需结合定量 geometry 指标解释；
- ETH3D：警示柱、卷帘门、车辆和墙面在 RGB、laser 和 prediction 中同位；prediction 黑区暴露未覆盖表面；
- T&T：梁柱、吊灯、椅桌、门框和镜面结构与官方 laser reference 同位；窗/镜面的黑洞属于 scan 缺测；
- 7-Scenes：棋桌、显示器、红墙和白板与 fusion reference 轮廓对应；
- Redwood：椅子、窗帘、灯具和墙角与 exact point geometry 方位一致；reference 没有可靠 RGB 时使用中性几何色；
- DTU：建筑模型在 16 个标定视角中与 structured-light cloud 同位；白色背景/支撑面属于 official reference，不是模型补全；
- OmniObject3D：只显示授权 blocker，未将语义请求槽位冒充官方 object ID。

## 5. 验证

`outputs/gt-calibration-v1/visualizations-v1/release_validation.json`：

| 检查项 | 结果 |
|---|---:|
| Dataset statuses | 7 |
| Available / blocked | 6 / 1 |
| With prediction / missing prediction | 2 / 4 |
| Frozen camera frames | 96 |
| Candidate videos | 25 |
| Overview videos | 1 |
| Candidate decoded frames | 392 |
| Overview decoded frames | 104 |
| Total decoded frames | 496 |
| Video bytes | 44,164,015 |
| Reference coverage min / median / max | 0.4111 / 0.8713 / 0.9988 |
| Prediction coverage min / median / max | 0.2746 / 0.9410 / 1.0 |
| Result | `pass` |

release validator 重新检查：

- visualization plan、GT registry 和 exporter SHA256；
- 7 个代表与冻结 plan/registry 的 membership、顺序、状态；
- 96 个源 RGB、source frame、reference frame、prediction/status frame 和 comparison frame 的 SHA256；
- reference/prediction geometry source SHA256 与 provenance；
- conditioning/heldout 严格 8+8 顺序；
- prepared 数据集每路视频恰好 16 帧；
- Omni blocker 视频恰好 8 帧；
- 总览视频恰好 104 帧；
- 26 个 MP4 的 SHA256、字节数、分辨率、fps、逐帧完整解码和非空画面。

关键 SHA256：

- plan：`a7c4288ec63d7004288cf17735bcefa19e70665094e28c0e350def8c3e3bef3d`
- index：`18c64fc53d856da9f44fedd6ab344b799c2bf603f0f216c59ede4d683ee0cf80`
- overview MP4：`3aa003a08cb9aa168eecc1ddd2a3aea2603ec623c71f25f53059b7cea6b2dc31`
- dataset contact：`a62f5bddf617e2f4a712ea23f332da788212db7a548fd20b3863e98b81353109`
- release validation：`b8db3a12926243768debdb22eadcc0d04735d84ac4e71b9294852d9c1f252854`

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

1. T&T、7-Scenes、Redwood、DTU 当前没有 GenRecon prediction；状态卡不应被称为预测渲染。
2. 为四者补 prediction 时，只能从 8 conditioning views 构建 GenRecon 所需控制点；禁止将 heldout RGB/depth、GT laser、fusion reference 或 structured-light points作为 conditioning geometry。
3. OmniObject3D 未授权前没有代表 RGB、相机和 dense scan；当前 blocker 视频不代表数据已下载。
4. `GT / REFERENCE GEOMETRY` 栏不是模型输出，也不进入纹理/图像质量比较。
5. 视频是 sparse frozen-camera inspection，不是连续时序、真实帧率或自由视角 fly-through。
6. 视觉对应与 nonblank validator 不能替代 2/5/10 cm、normalized threshold、heldout RGB/depth 或 pose 指标。
