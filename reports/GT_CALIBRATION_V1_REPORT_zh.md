# GenRecon GT Calibration v1 构建、统一评测与验证报告

## 1. 结论

本轮将此前规划的 room-scale 与 instance-scale 校准集冻结为 76 个可审计单元，并实际完成公开、可无认证下载部分的构建。

- 冻结总数：76 单元。
- 完整 calibration package：51 单元。
- 部分准备但 GT 下载阻塞：Tanks and Temples `Meetingroom` 1 单元。
- 官方认证阻塞：OmniObject3D 24 个请求槽位。
- 已有 GenRecon prediction 并完成统一重算：22 个 `G0-independent-scan` 场景。
- 新增本地数据：约 41 GiB，其中 source archives 约 40 GiB、冻结 unit packages 约 1.5 GiB。
- 构建验证：`pass`，`errors=[]`。
- source archive 深验证：`pass`，`errors=[]`。

这里的 `prepared` 只表示输入 split、相机元数据、GT、provenance 和 manifest 已完整落盘，不表示已经为该单元运行 GenRecon。51 个 prepared 单元中，目前只有既有 ScanNet++ 20 场景和 ETH3D 2 场景有 prediction，因此实际指标表严格保持 `n=22`，没有为其余 29 个单元虚构结果。

## 2. 冻结范围与状态

| 数据集 | GT 层级 | 轨道 | 冻结 | 完整准备 | 已统一计分 | 当前状态 |
|---|---|---|---:|---:|---:|---|
| DA3 ScanNet++ | G0 independent scan | scene | 20 | 20 | 20 | 完整 |
| ETH3D indoor training | G0 independent scan | scene | 7 | 7 | 2 | 5 个新场景待重建 |
| T&T Meetingroom | G0 independent scan | scene | 1 | 0 | 0 | RGB/位姿完成，laser/alignment quota 阻塞 |
| 7-Scenes | G1 fusion reference | scene | 7 | 7 | 0 | 完整 calibration package |
| Redwood reconstruction | G2 synthetic exact | scene | 2 | 2 | 0 | 完整 calibration package |
| DTU MVS | O0 instance scan | instance | 15 | 15 | 0 | 完整 calibration package |
| OmniObject3D | O0 instance scan | instance | 24 | 0 | 0 | OpenXLab 登录与 AK/SK 阻塞 |
| **总计** | | | **76** | **51** | **22** | **25 个声明式 blocker** |

冻结协议位于 `configs/eval/gt_calibration_v1.json`。所有 GT provenance 层级和 scene/instance 轨道都独立汇总，禁止直接求一个跨层级总均值。

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

已经成功准备：

- 官方 4K `Meetingroom.mp4`，4,271,432,079 bytes；
- 官方 371-frame image ZIP，416,347,897 bytes；
- 官方 371-camera COLMAP log；
- 官方 crop JSON；
- 冻结 8 conditioning + 8 heldout 及 camera-to-world poses。

视频与图像 ZIP 的 MD5 分别为：

- `5beeb4e21ca5b8fda31235cf15972393`
- `754932b99adcfc602908c5bda917c5a3`

两者与官方 GCS metadata 精确匹配。官方 individual laser scans 与 alignment 的 Google Drive 请求返回 2,009-byte `Quota exceeded` HTML。该文件保存为 `Meetingroom_individual_scans.quota-blocked.html`，禁止使用 `.zip` 后缀，也不进入 GT 列表。场景状态为 `blocked-reference-download`，不是 `prepared`。

### 3.7 OmniObject3D 24 请求槽位

官方 OpenXLab 需要登录和 AK/SK；本机没有授权凭据，也未绕过认证。24 个条目均为 `blocked-auth`。当前 object 名称是冻结的语义请求槽位，未授权前不宣称它们已经映射到官方 object IDs。

## 4. 数据完整性验证

`data/gt-calibration-v1/validation.json`：

| 检查项 | 结果 |
|---|---:|
| Registry units | 76 |
| Prepared / blocked-reference / blocked-auth | 51 / 1 / 24 |
| Conditioning / heldout RGB（prepared） | 407 / 407 |
| 成功解码 RGB | 814 |
| 成功解码 depth | 464 |
| Reference files | 57 |
| Reference vertices | 329,475,528 |
| Reference faces | 78,818,470 |
| Reference bytes | 5,358,529,068 |
| Source records | 843 |
| Existing prediction meshes / bytes | 22 / 10,561,058,552 |
| Strict JSON files | 123 |
| 结果 | `pass` |

`source_hash_bytes=146,935,884,518` 是 manifest 对 source records 的逻辑引用总量，DTU 等共享 archive 会被多个 unit 引用，不能解释为物理磁盘占用。validator 在同一进程内按路径、size、mtime 去重 SHA256 读取。

`data/gt-calibration-v1/source_validation.json` 对 32 个 archives/videos、37,808,231,702 bytes 完成深验证：

- 21 ZIP：全 archive CRC；
- 10 7z：全 archive CRC；
- 1 MP4：metadata 和首/中/尾有效帧解码；
- T&T 官方 MD5 对照；
- 1 个 quota HTML 作为声明式 blocker 单列。

结果为 `pass`，`errors=[]`。

同一 source tree 上连续执行两次 `prepare_gt_calibration_datasets.py all`，对 125 个生成 JSON/CSV 逐文件 SHA256 比较，diff 为零。最终 registry SHA256 为 `06ca7920870f43864ab1cea086ca37deffb0e33aea3e3a3030297aa3b5722b82`，validation SHA256 为 `bdc9d938a013c0c9d7800ebd366925ac322b2af87433d668607e14c9fb5f5e0f`。构建 manifest 不写墙钟时间，因此同输入不会仅因运行时间不同而产生 provenance 漂移。

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

## 6. 22 场景实际结果

### 6.1 分数据集宏平均

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

### 6.2 逐场景结果

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

## 7. 代码与 schema

新增或扩展：

- `configs/eval/gt_calibration_v1.json`：76-unit 冻结计划、GT 分层、阈值、聚合和 checksum policy。
- `tools/prepare_gt_calibration_datasets.py`：多数据集 adapter、split、reference 生成、registry、source SHA256 和 payload validator。
- `tools/validate_gt_calibration_sources.py`：ZIP/7z CRC、视频采样解码、官方 MD5 与 quota blocker 审计。
- `tools/evaluate_gt_suite.py`：mesh/point-cloud 统一 canonical schema、按 tier/dataset/track 聚合和 bootstrap CI。
- `tools/evaluate_mesh.py`：2/5/10 cm、distance quantiles、bbox-normalized thresholds 和严格 JSON。
- `tools/evaluate_mesh_pointcloud.py`：同一多阈值 schema，并显式区分 legacy prediction-AABB 与 full-reference scope。

每个 `evaluation.json` 包含 prediction/reference/manifest SHA256、GT kind、alignment、ROI、scope、sampling、阈值、canonical metrics、backend 原始结果和 limitations。`summarize` 在只刷新 metadata 前重新核对 prediction/reference SHA256；几何输入变化时会直接失败，不会静默复用旧指标。

验证结果：

- 全仓库：`188 passed in 7.65s`；
- 新增/修改 Python 文件全部通过 `py_compile`；
- `configs/eval/gt_calibration_v1.json` 通过严格 JSON 解析；
- `git diff --check` 无输出；
- 完整 `prepare ... all`、深 source validation 和 evaluation summarize 均实际执行成功；
- 最终无残留 builder/evaluator/pytest/GenRecon 进程。

## 8. 未完成项与禁止声明

1. T&T `Meetingroom` 仍缺官方 laser scans 和 alignment，因此不能进入 G0 数值表。
2. OmniObject3D 仍缺官方授权，24 个语义请求槽位不能称为已下载对象。
3. 新增 ETH3D 5、7-Scenes 7、Redwood 2、DTU 15 尚无 GenRecon prediction；当前只完成 calibration package，不报告模型成绩。
4. 7-Scenes、Redwood、DTU 不是 G0 room-scale independent laser table，禁止与 G0 直接求总均值。
5. ScanNet++ 可能参与过 GenRecon 训练，只能作 calibration，不证明 strict zero-shot。
6. 当前 v1 已统一核心 3D geometry schema；observed/unobserved visibility、heldout full-GT-mask RGB/depth、chunk boundary 和 camera trajectory 仍需对应 prediction/render adapter 后单列，不能用空值伪装成已评测。
7. 文件、hash 和 schema `pass` 不等于几何视觉正确；仍需逐场景渲染和人工检查。

## 9. 复现入口

```bash
.venv/bin/python tools/prepare_gt_calibration_datasets.py all
.venv/bin/python tools/validate_gt_calibration_sources.py
.venv/bin/python tools/evaluate_gt_suite.py batch \
  --registry data/gt-calibration-v1/registry.json \
  --output-root reports/generated/gt-calibration-v1/evaluations \
  --num-samples 200000 --max-gt-samples 1000000 --workers -1
.venv/bin/python tools/evaluate_gt_suite.py summarize \
  --output-root reports/generated/gt-calibration-v1/evaluations
```

关键产物：

- `data/gt-calibration-v1/registry.json`
- `data/gt-calibration-v1/summary.csv`
- `data/gt-calibration-v1/validation.json`
- `data/gt-calibration-v1/source_validation.json`
- `reports/generated/gt-calibration-v1/evaluations/index.json`
- `reports/generated/gt-calibration-v1/evaluations/units/*/evaluation.json`
