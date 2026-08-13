# GT Representative GenRecon v1：8-view RGB-only 推理、交付与评测报告

## 1. 结论

本轮为 GT calibration v1 的五个代表单元完成统一 GenRecon 推理：

- Tanks and Temples `Meetingroom`；
- 7-Scenes `chess`；
- Redwood `livingroom`；
- DTU `scan24`；
- OmniObject3D `bottle_045`。

五项均属于 `GT-pose-foundation-pseudo-geometry` prediction-generation track。每项只读取冻结的 8 张 conditioning RGB、unit manifest 和 conditioning camera records；不读取 heldout RGB、conditioning/heldout depth、evaluation reference geometry，也不执行 GT geometry ICP。VGGT-1B 从 conditioning RGB 产生 pseudo geometry，再仅以 conditioning camera centers 做 Umeyama Sim(3)，映射到保持声明坐标单位的 z-up work frame供 GenRecon 使用。

最终结果：

- 5/5 input packages 完成；
- 5/5 zero-fallback preflight 通过；
- 44/44 GenRecon chunks 非空，0 closest-camera fallback；
- 5/5 PLY + PBR GLB official-frame packages 完整；
- 合计 33,225,770 vertices、69,705,754 faces；
- official-frame PLY 合计 1,504,240,219 bytes；
- official-frame PBR GLB 合计 1,320,828,140 bytes；
- conditioning-only inference validator 与通用 GenRecon asset validator 均为 `pass`、`errors=[]`；
- 5/5 已登记到 GT registry 并完成统一 geometry evaluation；
- 5/5 已进入同相机三栏视频 release。

技术 validator 的 `pass` 只证明输入合同、资产结构、哈希、坐标变换和审计链有效，不证明几何或视觉正确。DTU `scan24` 虽然 F@10 cm 为 0.935098，但 prediction 中存在大面积绿色/白色背景平面。Omni `bottle_045` 的最终 masked prediction 不再生成旧的全画面平面，但瓶盖和瓶身塌缩成两个分离扁平 patch，仍是严重 topology/completion failure。

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
- work frame 保持 unit 声明的坐标单位，并由 conditioning cameras/pseudo points 构造 z-up；
- GenRecon 每 chunk 使用 8 views；
- zero closest-camera fallback；
- room/meter track 的 projected chunk area gate 为 0.4；Omni normalized-object track 为显式 0.2；
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

Omni 的白底不能作为 scene geometry。最终 adapter 只从 8 张 conditioning RGB 推导“边界连通精确白色背景”mask，同时用于 VGGT valid mask 和 GenRecon RGBA alpha；不读取 depth、heldout RGB 或 reference。Preflight/reconstruct 的每个 chunk 还记录 selected camera 的 projected chunk area、selection mode 和 threshold，package 对 `args.json` 与 `cameras.json` 做 SHA256 binding。

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
| Omni bottle_045 | P-B | 88,489 | 0.969817 | 1.000000 | 0.969817 | 1.000000 | 0.004126 | 1.2427 deg | 84,451 | 4 | 0 |

Redwood 使用冻结的 low-overlap gate：overlap opportunity 不高于 5%、verified points 至少 1,000、conditional consistency 至少 50%。其 250,000 个 prefiltered points 中只有 3,101 个存在跨视图 overlap opportunity，2,079 个通过验证；因此全局 verified fraction 很低，但 conditional consistency 为 0.670429。该结果只能标为 `P-C marginal`，不能写成高精度 pseudo geometry。

Omni 的 object-scale chunk cube 在 8 张 conditioning cameras 下 projected area 为 0.240479–0.351257，低于 room-scale 默认 gate 0.4，因此默认 selector 会错误产生 4 个 fallback。最终协议显式使用 0.2；4 个 selected chunk/camera areas 为 0.240479–0.345289，全部通过 frustum + area gate，fallback 为 0。该阈值和逐 chunk 证据在 preflight、reconstruct args/cameras 和 package source hashes 中绑定。

`P-B/P-C` 是 foundation pseudo geometry initialization 等级，不是几何精度等级。跨视图 consistency 也是模型内部自洽检查，不是独立 GT 精度证据。

Conditioning-manifest contract SHA256：

- T&T：`102ab636aa000664db38967afa1b966221a5ddad1c0741e7df8d5bc0040de52c`；
- 7-Scenes：`b088f62a3be957238dcee2b23feca254a7182bbeeec5aa194708e73baec2cf5e`；
- Redwood：`f9be63f6ffd887e8ba2df700fcec5c401f2a886914551cff5f6b98afb32e85f4`；
- DTU：`2f70bc5f97f4795c021e7ddd53f93c7730bc87238ab642f6d930975c7e119107`；
- Omni：`26de211d473c3ed852da24dd762081c7d4bc23fe73acd523f7c5cac0a98af104`。

## 5. GenRecon 与 official-frame packages

| Unit | Vertices | Faces | PLY bytes | PBR GLB bytes | Reconstruct s | GLB s | Peak reconstruct / GLB MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| T&T Meetingroom | 17,634,000 | 36,901,464 | 797,131,345 | 626,102,576 | 994.918 | 810.867 | 10,258 / 6,524 |
| 7-Scenes chess | 3,048,268 | 6,445,930 | 138,666,225 | 144,425,340 | 279.022 | 242.330 | 4,198 / 4,592 |
| Redwood livingroom | 10,692,948 | 22,579,552 | 486,007,553 | 439,503,980 | 695.795 | 1,048.528 | 9,762 / 5,780 |
| DTU scan24 | 1,715,268 | 3,505,858 | 76,451,289 | 63,564,340 | 168.998 | 167.613 | 3,256 / 4,356 |
| Omni bottle_045 | 135,286 | 272,950 | 5,983,807 | 47,231,904 | 144.919 | 59.470 | 3,216 / 4,672 |
| **总计** | **33,225,770** | **69,705,754** | **1,504,240,219** | **1,320,828,140** | **2,283.652** | **2,328.808** | **10,258 / 6,524 max** |

Official-frame PLY 实际改写 vertices；PBR GLB 则添加同一 `work_to_official` 4x4 根节点。Package validator 同时检查 work-frame PLY/GLB、official-frame PLY/GLB、矩阵、source hashes、output hashes、input-contract SHA，以及 no-reference/no-heldout/no-GT-ICP 声明。

Official output SHA256：

| Unit | PLY SHA256 | PBR GLB SHA256 | Package manifest SHA256 |
|---|---|---|---|
| T&T Meetingroom | `636f896bf1992be71f239bfeb9f9c3a623ef249a0b967f71c52869322aa94056` | `f5892fe6db5740d4a668396e88b01a0b65e2ae538e52c0e7f1ecf62d55b77ee1` | `d0e75d9a2e69baed59a7524075133120f694514e5ad333f1e20caf00efcf62d9` |
| 7-Scenes chess | `fdd4f45535c701dc9e1e99865ff8eacd3f1707c9cc6f5417836e78431f05f8d4` | `6b4678365fcc1d7a138625d94f836e7818722471682cedd49b1a51c10c9b24e6` | `a6aa6b33a51255f14625f134506dbe6f3f74704e201d0f5bba3db1830f307503` |
| Redwood livingroom | `e41e9522b3103584ac8e4e8f5736f974bea32c1e7f02340f302d00f420be295c` | `260b673785ce1c52507fea4f8df9ba3f5efddff3e0d199631e4c94afa1b77318` | `93d9d7bf3e17ca9fdb9c9f9e3d914a3f8deefc82e9aed0504b10b30261e55398` |
| DTU scan24 | `5616f9304e9bf924e099535471bc09aceea1e113f273eecd1664fdc6f1e78a39` | `0e77fb85ba2585163da31cd3b5baefc5723de17b2549a83674e0931c0f7eb1ef` | `cd8653fb684c3a8959b587ec5d21ffeb1b116b7261c3e061db35e3d5b44056cd` |
| Omni bottle_045 | `0f7159af685ff7c27065fbcc9feb027ab361b8a27593530da3c59730bab8b55d` | `b1a9ad19c585878b785d067a1ddaddeab94ec441a59f8ffb19d0daecbdb1a796` | `136b9d631697c52f46267bb0b55ce0075718f93c63fe57de29dcf542bb746914` |

五个 package 均绑定 input-contract、work mesh/GLB、reconstruct `args.json` 和 `cameras.json` hashes；normalized-object Omni 还强制 preflight/reconstruction camera JSON 完全相同。四个原有单元的 44 个实际 GenRecon input hashes 和 8 个 official PLY/GLB hashes 已在历史 deterministic replay 中保持不变；本轮仅扩展 package source-chain metadata，几何 PLY/GLB hashes 未变。5-unit resumability replay 中全部 reconstruction/GLB/package 被识别为 current 并跳过。

## 6. 分轨 geometry evaluation

| Unit | GT tier | Scope | Accuracy m | Completeness m | Chamfer m | F@2cm | F@5cm | F@10cm |
|---|---|---|---:|---:|---:|---:|---:|---:|
| T&T Meetingroom | G0 independent scan | official-crop-global-reference | 0.402961 | 0.394587 | 0.398774 | 0.012516 | 0.090562 | 0.217938 |
| 7-Scenes chess | G1 fusion reference | raw-global | 0.223986 | 0.207923 | 0.215955 | 0.084095 | 0.200952 | 0.362664 |
| Redwood livingroom | G2 synthetic exact | raw-global | 0.169891 | 0.392519 | 0.281205 | 0.066537 | 0.192308 | 0.335949 |
| DTU scan24 | O0 instance scan | raw-global | 0.051826 | 0.047409 | 0.049617 | 0.174609 | 0.553768 | 0.935098 |

四个 meter-coordinate reference 都是 point-cloud evaluation backend 且没有可审计 surface normals，因此 normal consistency 为 `null`，没有伪填 0。

禁止对四行直接求总均值：T&T 使用独立 official-crop protocol，四者 GT tier 不同，scene/instance unit type 也不同。Omni 单列 normalized-object protocol：

| Unit | GT tier | Scope | Accuracy | Completeness | Normalized Chamfer | F@bbox 0.5% | F@bbox 1% | F@bbox 2% |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Omni bottle_045 | O0 instance scan | normalized-object-global-reference | 0.108910 | 0.289282 | 0.199096 | 0.012049 | 0.025558 | 0.058624 |

Omni 不生成 meter-labelled metrics。2% precision 为 0.217820，但 recall 仅 0.033870；同相机 render 显示瓶盖和瓶身塌缩为两个分离扁平 patch。统一 evaluation index 以 `protocol_groups`、`tracks_by_tier` 和 `prediction_tracks_by_tier` 分离汇总。22 个 legacy G0 predictions 明确标为 `prediction-provenance-not-recorded`，不猜测其 generation track。

## 7. 同相机视觉审查

7 个代表场景均有真实 registry-backed prediction 第三栏。最终视频 release：

- 7 datasets：7 available、0 blocked；
- 7 个有 prediction、0 个 `missing-prediction`；
- 112 frozen camera frames；
- 29 H.264/yuv420p videos；
- 560 decoded frames；
- 54,679,055 video bytes；
- reference coverage min/median/max：0.0460/0.8414/0.9988；
- prediction coverage min/median/max：0.0078/0.7339/1.0000；
- release validation：`pass`、`errors=[]`。

T&T、7-Scenes 和 Redwood 的 conditioning/heldout 检查中，相机方向与主要房间结构同位。DTU 相机方向也正确，但 prediction 有大面积绿色/白色背景面。Omni 的旧全画面平面在 RGB-derived mask 后消失，但 16 个视角都只看到分离的 cap/body patches；其 prediction coverage 为 0.0078–0.0506，明显低于多数 reference 视角的约 0.09–0.12，与低 completion/recall 一致。

Render coverage、nonblank frame 和视觉同位只用于发现空帧、相机错位或异常包围，不是 F-score、recall、完整度或视觉质量结论。

## 8. 验证与关键哈希

`inference_validation.json`：

- 5 units；
- 40 conditioning RGB；
- 485,999 pseudo points；
- 44 chunks；
- 5 prediction packages；
- 33,225,770 prediction vertices；
- 1,504,240,219 prediction mesh bytes；
- 1,320,828,140 prediction GLB bytes；
- 51 strict JSON files；
- result `pass`，`errors=[]`。

通用 GenRecon asset validation：

- 5/5 reconstruct complete；
- 5/5 GLB complete；
- 44 nonempty primitives；
- 0 empty chunks；
- 0 fallback；
- 69,705,754 faces；
- result `pass`，`errors=[]`。

关键 SHA256：

- inference plan：`2033b18488f392e3344eda1048cf1440ecc3b487386c70fb3b3286bac5a90e89`；
- adapter：`492d7fdc47545dad9b740962894b7969406dd391f2f2c17acf6121da27f0ee76`；
- inference index：`77564e8186f8dc658f2a52c76eff23fa148c742b961745d750708f4a6034538f`；
- general asset summary CSV：`d3ebe6fa27edcf91094abf0914df949e59c9532a1b80aad302ea79ca61fef245`；
- general asset validation：`1ab985035a3b56153b916486f89768b6f7f15372c49d502f671479dd0a167506`；
- inference validation：`52983724fbe71e1c2196135e574205ec256f01b9c1ccd0f9882fb7d7a6104345`；
- GT registry：`ea5ce9627e9b20ac90ae90454b7e4fcb1955d6256a1bdc1cac3f24c4250d80e0`；
- evaluation index：`1c4ee92c5ba7ff9b0c15fa39bd1eec08b9962a2dbddc54cb43a0f1fc7e5ff707`；
- visualization release validation：`f8de4ac55c6a11273bf69420fffb62d392522409417b8ef03f4d5c34cd109c68`。

专用 inference validator 写 `inference_validation.json`；通用 GenRecon validator 写 `validation.json`，二者职责分离，互不覆盖。通用 runner 的 deterministic `index.json` 与 `validation.json` 不写墙钟字段；四个历史场景保留既有 deterministic replay 证据，最终 5-unit resumability replay 则验证了 command/profile 一致性并全部跳过重建。

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
10. 剩余 ETH3D 5、7-Scenes 6、Redwood 1、DTU 14、OmniObject3D 23，共 49 个 prepared units 仍无 prediction；Omni 必须继续使用 normalized-object protocol，不能输出米制分数。
