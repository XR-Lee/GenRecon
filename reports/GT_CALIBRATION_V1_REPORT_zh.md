# GenRecon GT Calibration v1 构建、统一评测与验证报告

## 1. 结论

本轮将此前规划的 room-scale 与 instance-scale 校准集冻结为 76 个可审计单元，并完成全部单元的数据构建、26 个 prediction 的统一评测，以及 7 个数据集代表的同相机视频审查。

- 冻结总数：76 单元，76/76 calibration packages 为 `prepared`，无 source blocker。
- 已有 prediction 并完成统一评测：26 个；其中 22 个是历史 G0 prediction，4 个是本轮新增的 RGB-only representative prediction。
- 尚无 prediction：50 个 prepared 单元，其中包括全部 24 个 OmniObject3D 单元。
- OmniObject3D 已从 OpenXLab 官方 metadata 解析真实 object IDs，并以 adapter v3 构建 24 个 100-view render/raw-scan packages；坐标固定为 `normalized-object`。
- 深 source validation 覆盖 71 个 archives/videos、80,772,051,894 bytes；构建验证、representative inference validation 和视频 release validation 均为 `pass`、`errors=[]`。

这里的 `prepared` 只表示 input split、相机元数据、GT、provenance 和 manifest 已完整落盘，不表示已经为该单元运行 GenRecon。76 个 prepared 单元中已有 26 个 prediction/result，另 50 个为 `missing-prediction`。四个新增结果只读取各单元的 8 张 conditioning RGB、unit manifest 和 conditioning camera records，不读取 heldout RGB/depth、conditioning depth 或 reference geometry，也不使用 GT ICP。

## 2. 冻结范围与状态

| 数据集 | GT 层级 | 轨道 | 冻结 | 完整准备 | 已统一计分 | 当前状态 |
|---|---|---|---:|---:|---:|---|
| DA3 ScanNet++ | G0 independent scan | scene | 20 | 20 | 20 | 完整 |
| ETH3D indoor training | G0 independent scan | scene | 7 | 7 | 2 | 5 个新场景待重建 |
| T&T Meetingroom | G0 independent scan | scene | 1 | 1 | 1 | representative prediction 已计分，official-crop scope |
| 7-Scenes | G1 fusion reference | scene | 7 | 7 | 1 | `chess` 已计分，6 个待重建 |
| Redwood reconstruction | G2 synthetic exact | scene | 2 | 2 | 1 | `livingroom` 已计分，1 个待重建 |
| DTU MVS | O0 instance scan | instance | 15 | 15 | 1 | `scan24` 已计分，14 个待重建 |
| OmniObject3D | O0 instance scan | instance | 24 | 24 | 0 | adapter v3 packages 完整，24 个待重建 |
| **总计** | | | **76** | **76** | **26** | **50 个 prepared 单元待 prediction；无 source blocker** |

冻结协议位于 `configs/eval/gt_calibration_v1.json`。汇总同时按 protocol signature、GT provenance tier、scene/instance unit track 和 prediction-generation track 分离，禁止直接求一个跨层级或跨协议总均值。

## 3. 实际构建内容

### 3.1 ScanNet++ 20 场景

保留此前 GenRecon 使用的 8 个 conditioning views，不改变已有 prediction 的输入；从同一官方注册 iPhone 序列中均匀选择 8 个不重叠 heldout views。每个场景落盘：

- 8 conditioning RGB + GT render depth；
- 8 heldout RGB + GT render depth；
- 每帧原始 COLMAP `qvec/tvec`、相机中心、camera model、intrinsics；
- 独立 laser scan mesh；
- 输入、depth、COLMAP model、selection 和 GT 的 SHA256 provenance。

heldout RGB 不属于 8 个 conditioning views，但它们参与官方全局 COLMAP model，因此不能称为 geometry-independent heldout。该边界已写入每个 manifest。

### 3.2 ETH3D 7 场景

完整覆盖：`delivery_area`、`pipes`、`kicker`、`office`、`relief`、`relief_2`、`terrains`。本轮新增下载后 5 个场景的 DSLR undistorted images、真实 COLMAP camera、laser evaluation clouds 和 MeshLab alignment。

- 6 个场景为 8 conditioning + 8 heldout。
- `pipes` 公开注册序列只有 14 帧，因此冻结为 7+7，不重复帧补足 8。
- `delivery_area` 和 `pipes` 有既有 prediction，已进入统一评测。
- 其余 5 个场景只声明 data-ready，不声明 reconstruction-ready/result-ready。

### 3.3 7-Scenes 7 场景

完整下载并解析 7 个官方 archive 及 TSDF 包；condition 使用首个官方 train sequence，heldout 使用首个官方 test sequence，各 8 帧。reference 从所有官方 train/test clean-depth 与 KinectFusion poses 构建：

- frame stride 50；
- depth pixel stride 8；
- 1 cm voxel 去重；
- 每场景最多 2,000,000 points。

reference 点数约 65,825 到 651,893。它是 clean-depth/KinectFusion 融合参考，严格标为 `G1-fusion-reference`，不是独立 laser GT。官方 raw RGB/depth 没有完整跨传感器标定，当前使用官方默认 depth intrinsics，此限制已进入 manifest。

### 3.4 Redwood 2 场景

`livingroom` 和 `office` 使用 trajectory 1 的 8 帧作为 conditioning、trajectory 2 的 8 帧作为 heldout，并保留官方 camera-to-world poses 与 RGB-D intrinsics。检查官方 PLY payload 后确认两份 reference 是 dense point-based exact synthetic surface，不是 triangle mesh，因此 manifest 使用 `kind=pointcloud`，没有伪造 faces 或 normal consistency。

### 3.5 DTU 15 scans

冻结 scans：1、4、9、10、11、12、13、15、23、24、29、32、33、34、48。每个对象包含：

- 从 49 个 calibrated views 中冻结 8 conditioning + 8 heldout；
- 每帧 intrinsics 与 world-to-camera extrinsics；
- camera translation 从毫米显式换算为米；
- 官方 structured-light PLY 从毫米显式换算为米；
- 2/5/10 cm 绝对阈值和 bbox diagonal 0.5%/1%/2% 归一化阈值。

15 个 reference 点数约 2.44M 到 6.83M。当前包没有冒充官方 DTU `ObsMask` protocol；统一 evaluator 使用 raw-global reference，官方 mask/crop 需要单列 scope。

### 3.6 Tanks and Temples Meetingroom

已经完整准备：

- 官方 4K `Meetingroom.mp4`，4,271,432,079 bytes；
- 官方 371-frame image ZIP，416,347,897 bytes；
- 官方 371-camera COLMAP log 和 crop JSON；
- 官方 individual scans ZIP，1,305,335,684 bytes，包含 11 个预对齐 binary PLY 和 12 条 scanner-position records；
- 官方 COLMAP-to-laser-GT alignment，408 bytes；
- 无畸变 8 conditioning + 8 heldout lossless PNG、PINHOLE intrinsics 和 laser-GT-frame camera-to-world poses；
- 官方 SelectionPolygonVolume crop 后的 1 cm 全局 voxel reference，10,321,864 points、154,828,197 bytes。

视频与图像 ZIP 的官方 MD5 分别为：

- `5beeb4e21ca5b8fda31235cf15972393`
- `754932b99adcfc602908c5bda917c5a3`

两者与官方 GCS metadata 精确匹配。用户提供的 scans ZIP 与 alignment 导入 SHA256 分别为：

- `2653a0849c7e8023c4280f041777977409c26f5d1c3f2f3f6238f91e8d7227f6`
- `fed12060a3f347e1eaf4a806c43966cf5972cd1dfc6032a4e412a814eda7392f`

官方 log 不发布精确内参。adapter 以官网建议的 `f=0.7W`、中心主点和零畸变为初值，对全部 371 帧做顺序 SIFT matching，并在所有官方 poses 严格固定时只优化共享 `SIMPLE_RADIAL` focal/k1：

- 2,007,483 keypoints；
- 2,828 verified pairs、899,858 geometric inliers；
- 最终 103,433 points、474,800 observations、mean track 4.590；
- 平均重投影误差 0.673695 px，point-error P90 1.215781 px；
- `f=1160.495872 px`、`k1=-0.031124605`；
- 最终 model 对官方 371 poses 的最大平移差为 0、最大旋转差 `3.22e-11°`。

reference 的流式 polygon crop 另以 Open3D 官方 `SelectionPolygonVolume` 在 250,000 个原始 scan points 上交叉验证：两者均选择 249,493 points，逐点完全相同。

两轮 Ceres 都在 100 iterations 上限处返回 `NO_CONVERGENCE`，但 solution usable、cost 均下降且所有质量 gate 通过；该 termination 原样保留，未写成“完全收敛”。matching、triangulation 和 BA 固定单线程、seed 42；连续两次完整数值 calibration JSON（规范化 source references 前）SHA256 均为 `03442a739028e99ee134dfd1b1bd7d252abedd8e01c94f99a85760cb78554693`，diff 为零。最终 artifact 将两个 clone-dependent 绝对路径规范化为 source basenames 后，冻结 SHA256 为 `efa8dcdc70f81361c8ff5b17384d63fc157fa7d523429674cf1f067d183c72b0`；除这两个路径字段外内容逐字段相同。laser GT 没有参与内参优化。旧的 2,009-byte quota HTML 仅以 `previous-quota-response.html` 保留为已解决下载历史，不再是 blocker。

该单元现已有 registry-backed representative prediction，并按 `official-crop-global-reference` scope 完成计分。相机与 GT 的关系来自官方 alignment，属于 `GT-pose-official-alignment` unit track；prediction 则属于 `GT-pose-foundation-pseudo-geometry` generation track，不是 estimated-pose 或端到端 Any-Video-to-Mesh 结果。8 个 heldout views 未参与 GenRecon conditioning，但全部 371 帧参与过共享内参恢复，因此也不是 intrinsics-independent heldout。该 scope 始终单独汇总，不能与 `raw-global` 表混报。

### 3.7 OmniObject3D 24 official-object units

OpenXLab 认证与 metadata blocker 已解除。凭据只保存在本机用户配置，不进入仓库、项目环境或日志。先获取 1,497-file metadata index（总索引 1,531,408,658,770 bytes），再只下载冻结 12 类所需资产；没有执行约 1.39 TiB 的全量下载。

Adapter v3 固定使用官方 `/raw/blender_renders/<category>.tar.gz` 100-view reconstruction track 与 `/raw/raw_scans/<category>.tar.gz` professional scans。旧 24-view GET3D track 不用于 scan calibration。24 个 official object IDs 覆盖 12 类各 2 个；`vase_002` 因官方 full-render/raw-scan pair 在 `r_0` 上只有 0.581420 visibility-aware mask IoU 而被拒绝，冻结替代为 `vase_004`，失败证据保留在 plan 的 `rejected_candidates`。

每个单元落盘：

- 100 张 RGB、100 张 normal、100 张 depth EXR 和一份 transforms 的 source-completeness audit；
- 从 100 个 official cameras 以 camera-center FPS 冻结 8 conditioning + 8 disjoint heldout；
- 用 `depth < 65504.0` 得到前景 mask，将 source RGB 合成到白底；depth 数值和 normal 均不作为 GenRecon conditioning，也不导出 depth；
- raw OBJ 先乘审计后的 per-object scale，再右乘 Blender OBJ import axis；reference 写为 `normalized-object` PLY；
- 只允许 bbox-diagonal 0.5%/1%/2% 阈值，禁止报告米制 Chamfer 或 2/5/10 cm 分数。

24 x 100 个独立 scan/render alignment views 全部通过：2,399 个 fast projected-vertex checks，`book_001` 的 1 个 view 通过 visibility-aware raycast fallback；`teapot_002` 有 1 个有界 nearest-SO(3) repair；`table_001`、`table_002` 的记录 scale 明显违反发布 renderer target，因此严格按官方 `0.99/raw_max_abs` 公式恢复。Fast gate 固定为最大 bbox 边误差 8 px、最小 bbox IoU 0.94；fallback full-mask IoU 至少 0.95。所有例外均写入 unit audit，没有放宽阈值。

这些 RGB 和 reference scan 来自同一对象的 scan-derived render domain，因此 8 heldout RGB 不是 geometry-independent evidence。Reference scan 只用于评测和可视化，绝不能作为 GenRecon conditioning。

## 4. 数据完整性验证

`data/gt-calibration-v1/validation.json`：

| 检查项 | 结果 |
|---|---:|
| Registry units | 76 |
| Prepared / blocked | 76 / 0 |
| Conditioning / heldout RGB（prepared） | 607 / 607 |
| 成功解码 RGB | 1,214 |
| 成功解码 depth | 464 |
| Omni adapter-v3 package contracts | 24 |
| Reference files | 82 |
| Reference vertices | 351,284,071 |
| Reference faces | 101,807,332 |
| Reference bytes | 5,950,057,924 |
| Source records | 918 |
| Existing prediction meshes / bytes | 26 / 12,059,314,964 |
| Strict JSON files | 197 |
| 结果 | `pass` |

`source_hash_bytes=216,842,849,236` 是 manifest 对 source records 的逻辑引用总量，DTU 和 Omni category archives 等共享源会被多个 unit 引用，不能解释为物理磁盘占用。validator 在同一进程内按路径、size、mtime 去重 SHA256 读取。

`data/gt-calibration-v1/source_validation.json` 对 71 个 archives/videos、80,772,051,894 bytes 完成深验证：

- 22 ZIP：全 archive CRC；
- 10 7z：全 archive CRC；
- 1 MP4：metadata 和首/中/尾有效帧解码；
- 38 TAR/GZip：流式完整读取、path safety、无 link/device member、无重复 file member，并逐个绑定 OpenXLab metadata size/SHA256；
- 12 个 selected full-render archives 与 12 个 selected raw-scan archives；24 个 object 各自恰好 100 RGB、100 normal、100 depth、1 transforms 和 1 个非空 `Scan.obj`；
- T&T 视频/图像 ZIP 官方 MD5 对照，以及 11 个 binary PLY、12 条 scanner positions、alignment Sim(3) 和 fixed-pose intrinsics artifact；
- 旧 quota HTML 仅作为 1 条 `historical-resolved-download-failure`，当前 declared quota blocker 为 0。

结果为 `pass`，`errors=[]`。

同一 source tree 上连续执行两次 T&T 数值 calibration，规范化 source references 前的 `Meetingroom_intrinsics.json` SHA256 均为 `03442a739028e99ee134dfd1b1bd7d252abedd8e01c94f99a85760cb78554693`；最终可移植 artifact SHA256 为 `efa8dcdc70f81361c8ff5b17384d63fc157fa7d523429674cf1f067d183c72b0`。Omni adapter-v3 authoritative build 后 registry/summary/validation 均使用无墙钟 deterministic JSON/CSV。当前关键 SHA256：plan `aac90c2147d4e6d05c6a7dc86200b50a313749c47cf6d7e0e6847863b99f0ff3`，registry `75e7f7e19dd8d305625ab9375d95a3c4ef70c36c652ed86849a9fba503079017`，summary CSV `8ed1a1f5d261b1fe41892c6ee1a1ff6793d6b14e0aa6a093403b827740bbbea3`，validation `f7dc07b989eb94b7303569d43c5b1cfcf3996c04d9677db933c318c059aabadd`，source-validation `d47c926c5f80a6ce665c7dafa495c6380711425c5da89246b855199316860a12`，evaluation index `81ba603bd41af7bee95c8b57c2548523a862e965731813f989bc45e595298fb5`。

## 5. 统一评测协议

核心 evaluator 为 `tools/evaluate_gt_suite.py`，mesh backend 为 `tools/evaluate_mesh.py`，point-cloud backend 为 `tools/evaluate_mesh_pointcloud.py`。

固定协议：

- seed 42；
- prediction mesh 按 triangle area 均匀采样 200,000 points；
- mesh GT 同样采样 200,000 points；
- point-cloud GT 最多确定性采样 1,000,000 points；
- 主 scope 为 `raw-global`；
- 绝对阈值 2/5/10 cm；
- instance 归一化阈值为 bbox diagonal 0.5%/1%/2%；
- Accuracy 和 Completeness 均报告 mean/median/P90/P95；
- symmetric Chamfer 使用双向 mean 的算术平均；
- F-score 使用 harmonic mean；
- mesh GT normal correspondence cutoff 为 20 cm；
- point-cloud GT 无可靠 surface normals 时，normal consistency 写 `null`；
- 所有 JSON 使用 `allow_nan=false`；
- 汇总按物理 unit macro average，bootstrap 2,000 次、seed 42；
- 禁止用 GT mesh ICP 为主榜修正 prediction。

旧 ScanNet++ 报告中的 `F@10 cm` 使用 `(Precision + Recall) / 2` 的 arithmetic 版本；新统一主字段使用 harmonic F-score，因此 20-scene 均值从旧 0.524253 变为 0.503604。旧兼容字段保留，但新旧 F-score 不得混用。Chamfer 和 normal consistency 定义不变。

## 6. 26 个分轨实际结果

### 6.1 22 个 legacy raw-global G0 结果

| 数据集 | n | Accuracy mean m | Completeness mean m | Chamfer m | F@2cm | F@5cm | F@10cm | NC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DA3 ScanNet++ | 20 | 0.190805 | 0.144856 | 0.167831 | 0.068151 | 0.260917 | 0.503604 | 0.552065 |
| ETH3D | 2 | 0.298992 | 0.783577 | 0.541285 | 0.072105 | 0.260567 | 0.432860 | `null` |
| G0 合并诊断 | 22 | 0.200641 | 0.202922 | 0.201781 | 0.068510 | 0.260885 | 0.497173 | 0.552065 (`n=20`) |

`G0 合并诊断` 只用于查看当前 22 个现有结果，不能掩盖 20:2 的数据集不平衡。正式报告继续保留 dataset macro 表。

Bootstrap 95% CI：

| 分组 | Chamfer mean 95% CI | F@5 mean 95% CI | F@10 mean 95% CI | NC mean 95% CI |
|---|---|---|---|---|
| ScanNet++ 20 | [0.147400, 0.190850] | [0.224470, 0.299628] | [0.468020, 0.541336] | [0.516885, 0.585567] |
| ETH3D 2 | [0.430849, 0.651720] | [0.218239, 0.302895] | [0.412304, 0.453416] | `null` |
| G0 22 | [0.159044, 0.260456] | [0.225425, 0.299575] | [0.463661, 0.532388] | [0.518183, 0.585285] (`n=20`) |

ETH3D 的 completeness 明显差于 ScanNet++，说明 full laser reference 中存在大量 prediction 未覆盖区域；统一 evaluator 没有先按 prediction AABB 裁掉这些 GT 点。该结果比旧 ROI proxy 更适合作为 raw-global 诊断，但仍不是官方 ETH3D depth-map benchmark。

### 6.2 22 个 legacy G0 逐场景结果

| Unit | Chamfer m | F@5cm | F@10cm | NC |
|---|---:|---:|---:|---:|
| scannetpp-286b55a2bf | 0.093991 | 0.392642 | 0.572785 | 0.679838 |
| scannetpp-7bc286c1b6 | 0.101005 | 0.423389 | 0.680997 | 0.664182 |
| scannetpp-3e8bba0176 | 0.119472 | 0.289169 | 0.576653 | 0.612314 |
| scannetpp-bcd2436daf | 0.120354 | 0.415931 | 0.567456 | 0.673312 |
| scannetpp-c5439f4607 | 0.127209 | 0.384322 | 0.547910 | 0.567636 |
| scannetpp-1ada7a0617 | 0.128093 | 0.302272 | 0.539299 | 0.577639 |
| scannetpp-7831862f02 | 0.129421 | 0.239852 | 0.613252 | 0.626036 |
| scannetpp-c4c04e6d6c | 0.134826 | 0.300765 | 0.593880 | 0.628552 |
| scannetpp-acd95847c5 | 0.147394 | 0.213839 | 0.436676 | 0.585058 |
| scannetpp-5f99900f09 | 0.168931 | 0.172634 | 0.446865 | 0.555270 |
| scannetpp-21d970d8de | 0.170068 | 0.191589 | 0.467886 | 0.563008 |
| scannetpp-40aec5fffa | 0.176535 | 0.328160 | 0.516641 | 0.532819 |
| scannetpp-cc5237fd77 | 0.177901 | 0.289306 | 0.550338 | 0.535180 |
| scannetpp-fb5a96b1a2 | 0.182137 | 0.183623 | 0.458244 | 0.518553 |
| scannetpp-09c1414f1b | 0.195754 | 0.209266 | 0.481680 | 0.493184 |
| scannetpp-bde1e479ad | 0.202340 | 0.157889 | 0.414883 | 0.473100 |
| scannetpp-578511c8a9 | 0.224491 | 0.214789 | 0.446723 | 0.488422 |
| scannetpp-f3d64c30f8 | 0.225223 | 0.237550 | 0.403463 | 0.461149 |
| scannetpp-9071e139d9 | 0.227843 | 0.155287 | 0.409450 | 0.378465 |
| scannetpp-38d58a7a31 | 0.303626 | 0.116067 | 0.347006 | 0.427578 |
| eth3d-pipes | 0.430849 | 0.302895 | 0.453416 | `null` |
| eth3d-delivery_area | 0.651720 | 0.218239 | 0.412304 | `null` |

### 6.3 四个 RGB-only representative 结果

四个新增 prediction 的 generation track 均为 `GT-pose-foundation-pseudo-geometry`。每项只使用 8 张 conditioning RGB 和对应 conditioning camera records；VGGT-1B 产生 pseudo geometry，再以 conditioning camera centers 的 Umeyama Sim(3) 映射到 metric z-up work frame。heldout RGB/depth、conditioning depth、reference geometry 和 GT ICP 均未使用。

| Unit | GT tier | Scope | Accuracy m | Completeness m | Chamfer m | F@2cm | F@5cm | F@10cm | NC |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| tanks-and-temples-meetingroom | G0 | official-crop-global-reference | 0.402961 | 0.394587 | 0.398774 | 0.012516 | 0.090562 | 0.217938 | `null` |
| seven-scenes-chess | G1 | raw-global | 0.223986 | 0.207923 | 0.215955 | 0.084095 | 0.200952 | 0.362664 | `null` |
| redwood-livingroom | G2 | raw-global | 0.169891 | 0.392519 | 0.281205 | 0.066537 | 0.192308 | 0.335949 | `null` |
| dtu-scan24 | O0 | raw-global | 0.051826 | 0.047409 | 0.049617 | 0.174609 | 0.553768 | 0.935098 | `null` |

这四行不能彼此求总均值，也不能并入 legacy G0 表：T&T 使用不同 scope；四者 GT tier 不同；legacy 22 项的 prediction provenance 未被历史 artifact 完整记录，而新增四项有完整 RGB-only foundation input contract。evaluation index 顶层因此固定为 `aggregate_policy=no-cross-protocol-aggregation`，包含 25 个 `raw-global` results 和 1 个 T&T `official-crop-global-reference` result；其余 50 个 prepared units 全部以 `missing-prediction` skip 明示，其中 24 个是 OmniObject3D。

DTU `scan24` 的 F@10 cm 为 0.935098，但同相机 prediction render 明确出现大面积绿色/白色背景平面，并遮挡建筑主体。这是实际 hallucination，不能由高阈值 F-score、非黑 coverage 或 technical validator 的 `pass` 覆盖。

## 7. 代码与 schema

新增或扩展：

- `configs/eval/gt_calibration_v1.json`：76-unit 冻结计划、GT 分层、阈值、聚合和 checksum policy。
- `tools/calibrate_tnt_meetingroom.py`：全部 371 帧顺序 SIFT、官方 poses 固定的共享 radial intrinsics calibration 和质量 gate。
- `tools/prepare_gt_calibration_datasets.py`：多数据集 adapter、split、T&T Sim(3)/undistortion/official-crop reference、Omni adapter v3 的 safe TAR/EXR mask/100-view alignment/normalized-object package、registry、source SHA256 和 payload validator。
- `tools/validate_gt_calibration_sources.py`：ZIP/7z CRC、TAR/GZip stream/path safety、OpenXLab metadata size/SHA binding、selected 100-view render/raw-scan payload contracts、视频采样解码、官方 MD5、T&T scans/alignment/calibration 和历史 quota 响应审计。
- `tools/evaluate_gt_suite.py`：mesh/point-cloud 统一 canonical schema、按 protocol/tier/dataset/unit track/prediction track 聚合、bootstrap CI，以及 stale registry/result 拒绝。
- `tools/evaluate_mesh.py`：2/5/10 cm、distance quantiles、bbox-normalized thresholds 和严格 JSON。
- `tools/evaluate_mesh_pointcloud.py`：同一多阈值 schema，并显式区分 legacy prediction-AABB 与 full-reference scope。
- `configs/eval/gt_representative_inference_v1.json`：四个代表单元、8-view RGB-only/no-leakage 合同和 zero-fallback gate。
- `tools/prepare_gt_representative_genrecon.py`：conditioning-only source audit、VGGT pseudo geometry、conditioning-camera Sim(3)、metric z-up、input-contract SHA、official-frame PLY/PBR GLB package 和专用 validator。
- `tools/run_foundation_genrecon_batch.py`：复用现有 resumable GenRecon/PBR runner；本轮 40/40 chunks 非空、0 fallback。
- `tools/export_gt_calibration_videos.py`：固定代表场景、冻结 8+8 相机、RGB/reference/prediction 三栏 H.264 和 HTML 审查页。
- `tools/validate_gt_calibration_videos.py`：逐帧、geometry provenance、SHA256、视频完整解码、missing-prediction/source-blocker 边界和按候选状态推导全局计数的独立 release validator。

每个 `evaluation.json` 包含 prediction/reference/manifest SHA256、GT kind、alignment、ROI、scope、sampling、阈值、canonical metrics、backend 原始结果和 limitations。`summarize` 会核对当前 registry membership、unit manifest 路径、prediction path/SHA、GT tier、protocol 和 prediction track；registry 撤销、换路径、换 manifest 或换 generation track 时直接拒绝 stale result，不会静默复用旧指标。

验证结果：

- 全仓测试：`239 passed in 7.43s`；
- 新增/修改 Python 文件全部通过 `py_compile`；
- `configs/eval/gt_calibration_v1.json` 和所有生成 JSON 通过严格 JSON 解析；
- `git diff --check` 无输出；
- authoritative Omni adapter-v3 build、GT deep validation、71-file source validation、evaluation summarize、视频内部/独立 release validation 均实际执行成功；
- 第二次 `all --force` replay 在 19:51.50 完成并再次得到 76/76 prepared；170 个 builder-owned unit JSON/CSV 与 registry/summary/validation 的 pre/post SHA256 manifests 逐行相同；
- 最终无残留 builder/evaluator/pytest/GenRecon 进程。

## 8. 未完成项与禁止声明

1. 全部 76 个数据单元已 prepared，但仍有 50 个没有 GenRecon prediction：ETH3D 5、7-Scenes 6、Redwood 1、DTU 14、OmniObject3D 24。
2. OmniObject3D 是 `normalized-object`、scan-derived 100-view domain，只允许 bbox-diagonal normalized thresholds；禁止报告米制 Chamfer/2/5/10 cm，也禁止与 DTU 的 meter protocol 合并。
3. Omni 的 conditioning/heldout RGB 与 reference scan 来自同一对象；heldout 不属于 geometry-independent evidence，reference 也不得作为 GenRecon conditioning 或 GT ICP。
4. T&T、7-Scenes、Redwood 和 DTU 新结果属于 `GT-pose-foundation-pseudo-geometry`，不是 estimated-pose 端到端 Any-Video-to-Mesh，也不是完全独立 geometry heldout。
5. 7-Scenes、Redwood、DTU 不是 G0 room-scale independent laser table；T&T 还使用独立 official-crop scope，禁止直接求总均值。
6. ScanNet++ 可能参与过 GenRecon 训练，只能作 calibration，不证明 strict zero-shot。
7. VGGT-1B checkpoint 为 `CC-BY-NC-4.0`，本轮 foundation inference 仅限非商业研究用途；pseudo geometry 的内部一致性不是独立真实精度证据。
8. DTU `scan24` prediction 存在大面积绿色/白色背景平面 hallucination；高 F@10 cm 不能覆盖该视觉失败。
9. 当前 v1 已统一核心 3D geometry schema；observed/unobserved visibility、heldout full-GT-mask RGB/depth、chunk boundary 和 camera trajectory 仍需对应 adapter 后单列，不能用空值伪装成已评测。
10. 视频中 6 个代表有真实 registry-backed prediction；Omni 代表只显示 real RGB/reference 和显式 `missing-prediction` card。`GT / REFERENCE GEOMETRY` 栏绝不能称为模型输出或 conditioning 输入。

## 9. 复现入口

```bash
PYTHONPATH=/tmp/pycolmap-wheel .venv/bin/python \
  tools/calibrate_tnt_meetingroom.py
.venv/bin/python tools/prepare_gt_calibration_datasets.py all
.venv/bin/python tools/validate_gt_calibration_sources.py
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
  tools/export_gt_calibration_videos.py all
.venv/bin/python tools/validate_gt_calibration_videos.py
```

关键产物：

- `data/gt-calibration-v1/registry.json`
- `data/gt-calibration-v1/summary.csv`
- `data/gt-calibration-v1/validation.json`
- `data/gt-calibration-v1/source_validation.json`
- `data/gt-calibration-v1/genrecon-inputs-v1/`
- `outputs/gt-calibration-v1/representative-genrecon-v1/inference_validation.json`
- `outputs/gt-calibration-v1/representative-genrecon-v1/candidates/*/prediction_official/manifest.json`
- `reports/generated/gt-calibration-v1/evaluations/index.json`
- `reports/generated/gt-calibration-v1/evaluations/units/*/evaluation.json`
