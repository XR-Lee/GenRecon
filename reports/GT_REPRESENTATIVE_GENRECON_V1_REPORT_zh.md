# GT Representative GenRecon v1：8-view RGB-only 推理、交付与评测报告

## 1. 结论

本轮为 GT calibration v1 中四个此前缺失 prediction 的代表单元完成统一 GenRecon 推理：

- Tanks and Temples `Meetingroom`；
- 7-Scenes `chess`；
- Redwood `livingroom`；
- DTU `scan24`。

四项均属于 `GT-pose-foundation-pseudo-geometry` prediction-generation track。每项只读取冻结的 8 张 conditioning RGB、unit manifest 和 conditioning camera records；不读取 heldout RGB、conditioning/heldout depth、evaluation reference geometry，也不执行 GT geometry ICP。VGGT-1B 从 conditioning RGB 产生 pseudo geometry，再仅以 conditioning camera centers 做 Umeyama Sim(3)，映射到 metric z-up work frame 供 GenRecon 使用。

最终结果：

- 4/4 input packages 完成；
- 4/4 zero-fallback preflight 通过；
- 40/40 GenRecon chunks 非空，0 closest-camera fallback；
- 4/4 PLY + PBR GLB official-frame packages 完整；
- 合计 33,090,484 vertices、69,432,804 faces；
- official-frame PLY 合计 1,498,256,412 bytes；
- official-frame PBR GLB 合计 1,273,596,236 bytes；
- conditioning-only inference validator 与通用 GenRecon asset validator 均为 `pass`、`errors=[]`；
- 4/4 已登记到 GT registry 并完成统一 geometry evaluation；
- 4/4 已进入同相机三栏视频 release。

技术 validator 的 `pass` 只证明输入合同、资产结构、哈希、坐标变换和审计链有效，不证明几何或视觉正确。DTU `scan24` 虽然 F@10 cm 为 0.935098，但 prediction 中存在大面积绿色/白色背景平面并遮挡建筑，属于实际 hallucination。

## 2. 冻结输入合同

配置：`configs/eval/gt_representative_inference_v1.json`。

固定协议：

- seed 42；
- 每单元 8 张 conditioning RGB；
- VGGT-1B camera + depth heads；
- confidence P70；
- relative depth tolerance 0.15；
- prefilter budget 250,000；
- 最多 100,000 pseudo points；
- conditioning-camera-only Umeyama Sim(3)；
- work frame 为 metric z-up；
- GenRecon 每 chunk 使用 8 views；
- zero closest-camera fallback；
- PBR texture size 4096；
- 每 chunk simplify threshold 300,000；
- official-frame package 同时提供 PLY 和 GLB。

每个 source audit 精确记录：

- 1 个 unit manifest；
- 1 个 camera metadata 文件；
- 8 张 conditioning RGB；
- 0 heldout RGB；
- 0 conditioning/heldout depth；
- 0 reference geometry。

每个实际 GenRecon input manifest 另冻结 11 个输入资产的 path、size 和 SHA256：

- `colmap_vggt/cameras.txt`；
- `colmap_vggt/images.txt`；
- `colmap_vggt/points3D.txt`；
- `rgb/000.png` 到 `rgb/007.png`。

Unit manifest 的 conditioning-contract SHA 排除 registry-only 的 `prediction_mesh` 和 `prediction_provenance` 字段，避免 prediction registration 形成自引用循环。规范化 GenRecon input-contract SHA 排除 runtime、GPU peak、checkpoint 本机路径和 unit registration-only 字段，但保留模型/checkpoint SHA、camera、points、work frame 和 11 个 conditioning asset hashes。

## 3. 模型与 checkpoint

VGGT：

- code revision：`a288dd0f14786c93483e45524328726ab7b1b4ce`；
- model revision：`860abec7937da0a4c03c41d3c269c366e82abdf9`；
- checkpoint SHA256：`d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`；
- license：`CC-BY-NC-4.0`，因此本轮 foundation inference 只适用于非商业研究。

GenRecon/TRELLIS 三级 checkpoint：

- Sparse Structure：`e18c1caddb2357dbf5839f0f7e1569c50d855fcb47e0871483d99c91a14e2bb7`；
- Shape SLat：`d9e13be151a213bf67565d2a17341fe97328e455814fb2328a59366344e122fb`；
- Texture SLat：`28f99217a4fbcd04f36a5f576905975ae8b874ad63548d6ab8afdb96cf03ed47`。

三者分别负责 sparse occupancy/topology、surface shape 和 PBR appearance，不是高/中/低质量的三套重复模型。

VGGT-Omega checkpoint 当前需要 gated Hugging Face 授权，本轮没有绕过授权，也没有将其写成已运行模型。

## 4. Foundation pseudo geometry 与 preflight

| Unit | Grade | GenRecon points | Verified fraction | Overlap opportunity | Conditional consistency | Official-camera visible | Center P90 / baseline | Rotation P90 | Clean points | Chunks | Fallback |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T&T Meetingroom | P-B | 100,000 | 0.801776 | 0.862192 | 0.929927 | 1.000000 | 0.009479 | 2.1067 deg | 92,470 | 20 | 0 |
| 7-Scenes chess | P-B | 99,995 | 0.970152 | 0.971036 | 0.999090 | 0.999950 | 0.017336 | 4.0993 deg | 95,760 | 4 | 0 |
| Redwood livingroom | P-C marginal | 97,515 | 0.008316 | 0.012404 | 0.670429 | 0.975150 | 0.137410 | 8.0641 deg | 95,419 | 14 | 0 |
| DTU scan24 | P-B | 100,000 | 0.936000 | 0.936144 | 0.999846 | 1.000000 | 0.002612 | 0.5502 deg | 92,567 | 2 | 0 |

Redwood 使用冻结的 low-overlap gate：overlap opportunity 不高于 5%、verified points 至少 1,000、conditional consistency 至少 50%。其 250,000 个 prefiltered points 中只有 3,101 个存在跨视图 overlap opportunity，2,079 个通过验证；因此全局 verified fraction 很低，但 conditional consistency 为 0.670429。该结果只能标为 `P-C marginal`，不能写成高精度 pseudo geometry。

`P-B/P-C` 是 foundation pseudo geometry initialization 等级，不是几何精度等级。跨视图 consistency 也是模型内部自洽检查，不是独立 GT 精度证据。

Conditioning-manifest contract SHA256：

- T&T：`102ab636aa000664db38967afa1b966221a5ddad1c0741e7df8d5bc0040de52c`；
- 7-Scenes：`b088f62a3be957238dcee2b23feca254a7182bbeeec5aa194708e73baec2cf5e`；
- Redwood：`f9be63f6ffd887e8ba2df700fcec5c401f2a886914551cff5f6b98afb32e85f4`；
- DTU：`2f70bc5f97f4795c021e7ddd53f93c7730bc87238ab642f6d930975c7e119107`。

## 5. GenRecon 与 official-frame packages

| Unit | Vertices | Faces | PLY bytes | PBR GLB bytes | Reconstruct s | GLB s | Peak reconstruct / GLB MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| T&T Meetingroom | 17,634,000 | 36,901,464 | 797,131,345 | 626,102,576 | 994.918 | 810.867 | 10,258 / 6,524 |
| 7-Scenes chess | 3,048,268 | 6,445,930 | 138,666,225 | 144,425,340 | 279.022 | 242.330 | 4,198 / 4,592 |
| Redwood livingroom | 10,692,948 | 22,579,552 | 486,007,553 | 439,503,980 | 695.795 | 1,048.528 | 9,762 / 5,780 |
| DTU scan24 | 1,715,268 | 3,505,858 | 76,451,289 | 63,564,340 | 168.998 | 167.613 | 3,256 / 4,356 |
| **总计** | **33,090,484** | **69,432,804** | **1,498,256,412** | **1,273,596,236** | **2,138.733** | **2,269.338** | **10,258 / 6,524 max** |

Official-frame PLY 实际改写 vertices；PBR GLB 则添加同一 `work_to_official` 4x4 根节点。Package validator 同时检查 work-frame PLY/GLB、official-frame PLY/GLB、矩阵、source hashes、output hashes、input-contract SHA，以及 no-reference/no-heldout/no-GT-ICP 声明。

Official output SHA256：

| Unit | PLY SHA256 | PBR GLB SHA256 | Package manifest SHA256 |
|---|---|---|---|
| T&T Meetingroom | `636f896bf1992be71f239bfeb9f9c3a623ef249a0b967f71c52869322aa94056` | `f5892fe6db5740d4a668396e88b01a0b65e2ae538e52c0e7f1ecf62d55b77ee1` | `928da03d421205c57a9a5d237be2895ece2cc23f634e4e9f39b007abee27d507` |
| 7-Scenes chess | `fdd4f45535c701dc9e1e99865ff8eacd3f1707c9cc6f5417836e78431f05f8d4` | `6b4678365fcc1d7a138625d94f836e7818722471682cedd49b1a51c10c9b24e6` | `32b437a3f1298f61e409db942b2920c5dae45b2d5446b9970c46b88186399a5f` |
| Redwood livingroom | `e41e9522b3103584ac8e4e8f5736f974bea32c1e7f02340f302d00f420be295c` | `260b673785ce1c52507fea4f8df9ba3f5efddff3e0d199631e4c94afa1b77318` | `ea048909882942bad7e71205cad03c232e12328f1bde5aae0c6f1be3682a6377` |
| DTU scan24 | `5616f9304e9bf924e099535471bc09aceea1e113f273eecd1664fdc6f1e78a39` | `0e77fb85ba2585163da31cd3b5baefc5723de17b2549a83674e0931c0f7eb1ef` | `adecc84a52260ea5ac184156dc16743ea40a6eaa86cadf3403b4a8b28167f72c` |

多轮 deterministic replay 中，4 个单元的 44 个实际 GenRecon input hashes 和 8 个 official PLY/GLB hashes 均保持不变。

## 6. 分轨 geometry evaluation

| Unit | GT tier | Scope | Accuracy m | Completeness m | Chamfer m | F@2cm | F@5cm | F@10cm |
|---|---|---|---:|---:|---:|---:|---:|---:|
| T&T Meetingroom | G0 independent scan | official-crop-global-reference | 0.402961 | 0.394587 | 0.398774 | 0.012516 | 0.090562 | 0.217938 |
| 7-Scenes chess | G1 fusion reference | raw-global | 0.223986 | 0.207923 | 0.215955 | 0.084095 | 0.200952 | 0.362664 |
| Redwood livingroom | G2 synthetic exact | raw-global | 0.169891 | 0.392519 | 0.281205 | 0.066537 | 0.192308 | 0.335949 |
| DTU scan24 | O0 instance scan | raw-global | 0.051826 | 0.047409 | 0.049617 | 0.174609 | 0.553768 | 0.935098 |

四个 reference 都是 point-cloud evaluation backend 且没有可审计 surface normals，因此 normal consistency 为 `null`，没有伪填 0。

禁止对四行直接求总均值：T&T 使用独立 official-crop protocol，四者 GT tier 不同，scene/instance unit type 也不同。统一 evaluation index 以 `protocol_groups`、`tracks_by_tier` 和 `prediction_tracks_by_tier` 分离汇总。22 个 legacy G0 predictions 明确标为 `prediction-provenance-not-recorded`，不猜测其 generation track，也不与新增四项混成同一 prediction-track 均值。

## 7. 同相机视觉审查

6 个可用代表场景均已有真实 prediction 第三栏。最终视频 release：

- 7 datasets：6 available、1 blocked-auth；
- 6/6 available datasets 有 prediction；
- 96 frozen camera frames；
- 26 H.264/yuv420p videos；
- 496 decoded frames；
- 53,315,310 video bytes；
- reference coverage min/median/max：0.4111/0.8713/0.9988；
- prediction coverage min/median/max：0.2746/0.7702/1.0000；
- release validation：`pass`、`errors=[]`。

T&T、7-Scenes 和 Redwood 的 conditioning/heldout 检查中，相机方向与主要房间结构同位。DTU 相机方向也正确，但 prediction 有大面积绿色/白色背景面，且从多个冻结视角遮挡建筑主体。这一 hallucination 必须与 F@10 cm 0.935098 同时报告。

Render coverage、nonblank frame 和视觉同位只用于发现空帧、相机错位或异常包围，不是 F-score、recall、完整度或视觉质量结论。

## 8. 验证与关键哈希

`inference_validation.json`：

- 4 units；
- 32 conditioning RGB；
- 397,510 pseudo points；
- 40 chunks；
- 4 prediction packages；
- 33,090,484 prediction vertices；
- 1,498,256,412 prediction mesh bytes；
- 1,273,596,236 prediction GLB bytes；
- 42 strict JSON files；
- result `pass`，`errors=[]`。

通用 GenRecon asset validation：

- 4/4 reconstruct complete；
- 4/4 GLB complete；
- 40 nonempty primitives；
- 0 empty chunks；
- 0 fallback；
- 69,432,804 faces；
- result `pass`，`errors=[]`。

关键 SHA256：

- inference plan：`a578dbe0597a501dad4e9d8d710a823b13ddda7b3c22ed634274c51503e07672`；
- adapter：`89117edfb6a01f8ed897b002994ef6ac60f51bbe099183b23f7bd032f0fc3ea8`；
- inference index：`ec73d3e2e6560a6bbb906545508165dc7de88e3811ffa74f0261a357575fb15f`；
- general asset summary CSV：`ff1086009ef695877eaa36c4040c9c7363178cfc18c3cf6f0a520d5abb47fe3b`；
- general asset validation：`15fa6c14f9ef7a764653362bb410098b63020bbb261edf26de7eaaab81d4d3c1`；
- inference validation：`53fc696342bb8be325b5f73a3bd3f5688985c50a7ac827a6f316c56bca061776`；
- GT registry：`9d5592a913463ce23c38e638d48558f23f9d41600696ccad9ea6ef1bf6fc6cc0`；
- evaluation index：`85fb5bb17410039f2c106d74f30735abbe71ce0799158b961cc3325da1a69e33`；
- visualization release validation：`22b55528147d89c3417a159e370bbe7524aca917827193b0dbc8c1e5c6d99b1d`。

专用 inference validator 写 `inference_validation.json`；通用 GenRecon validator 写 `validation.json`，二者职责分离，互不覆盖。通用 runner 的 deterministic `index.json` 与 `validation.json` 不写墙钟字段；连续两次真实四场景 validation 的 `index.json`、`summary.csv` 和 `validation.json` SHA256 均保持一致。

## 9. 复现入口

```bash
.venv/bin/python tools/prepare_gt_representative_genrecon.py prepare
.venv/bin/python tools/prepare_gt_representative_genrecon.py preflight
.venv/bin/python tools/run_foundation_genrecon_batch.py all \
  --input-root data/gt-calibration-v1/genrecon-inputs-v1 \
  --output-root outputs/gt-calibration-v1/representative-genrecon-v1 \
  --report-root reports/generated/gt-calibration-v1/representative-genrecon-v1 \
  --track-name GT-pose-foundation-pseudo-geometry --fail-fast
.venv/bin/python tools/prepare_gt_representative_genrecon.py package
.venv/bin/python tools/prepare_gt_representative_genrecon.py validate
.venv/bin/python tools/prepare_gt_calibration_datasets.py register-predictions
.venv/bin/python tools/evaluate_gt_suite.py batch \
  --registry data/gt-calibration-v1/registry.json \
  --output-root reports/generated/gt-calibration-v1/evaluations \
  --num-samples 200000 --max-gt-samples 1000000 --workers -1
.venv/bin/python tools/evaluate_gt_suite.py summarize \
  --output-root reports/generated/gt-calibration-v1/evaluations
EGL_PLATFORM=surfaceless .venv/bin/python \
  tools/export_gt_calibration_videos.py all --force
.venv/bin/python tools/validate_gt_calibration_videos.py
```

主要本地产物：

- `data/gt-calibration-v1/genrecon-inputs-v1/`；
- `outputs/gt-calibration-v1/representative-genrecon-v1/`；
- `outputs/gt-calibration-v1/representative-genrecon-v1/inference_validation.json`；
- `outputs/gt-calibration-v1/representative-genrecon-v1/candidates/*/prediction_official/`；
- `reports/generated/gt-calibration-v1/evaluations/units/*/evaluation.json`；
- `outputs/gt-calibration-v1/visualizations-v1/index.html`。

## 10. 尚未完成与禁止声明

1. 这些结果不是 estimated-pose 端到端 Any-Video-to-Mesh；official conditioning cameras 参与 Sim(3) 对齐。
2. Foundation pseudo point cloud 不是 SfM observation、scan 或 GT；synthetic COLMAP 文件仅用于接口兼容。
3. 8 张 conditioning RGB 已参与 VGGT geometry，因此不能称为 geometry-independent conditioning evidence。
4. T&T 的 8 heldout RGB 未参与 inference，但全部 371 帧参与过共享内参恢复，不能称为 intrinsics-independent heldout。
5. GT/reference geometry 只用于最终评测和视频第二栏，绝不能称为 GenRecon conditioning 或 prediction。
6. VGGT-1B 为 CC-BY-NC-4.0；不能据此声明商业可用。
7. Redwood 是 `P-C marginal` low-overlap case，不能将 conditional consistency 写成绝对几何精度。
8. DTU 的背景平面 hallucination 是已确认视觉失败，不能由高 F-score、coverage 或 validator pass 覆盖。
9. Observed/unobserved completion、full-GT-mask heldout RGB/depth、chunk boundary、trajectory 和 dedicated hallucination metrics 尚未实现，不能以当前空缺字段冒充已评测。
10. 剩余 ETH3D 5、7-Scenes 6、Redwood 1、DTU 14，共 26 个 prepared units 仍无 prediction。
