# GenRecon 互联网室内 zero-shot 测试集采集与验收方案

> 调研与首个工程 pilot 日期：2026-08-06。本文区分“COLMAP 可恢复”“GenRecon 可运行”“zero-shot 可主张”和“有真值可量化”四件不同的事，不用其中一项替代另一项。

## 1. 结论

可以从互联网视频或同一房间的静态照片集构造 GenRecon 测试场景，但采集单位必须是**同一物理房间内的一段连续 shot**，不是整条 house tour，也不是网页上的任意若干室内图片。

建议采用两级数据集：

1. **公开校准集**：ETH3D、Tanks and Temples、DL3DV-Evaluation 等已有相机或真值的数据，只用于验证适配器、指标和资源参数。
2. **wild zero-shot 主集**：checkpoint 训练结束后新拍摄或新发布、许可可审计的 CC/作者授权视频，重新跑 COLMAP，并保留全部失败样本。

正式主协议固定为 8 个物理输入视角，另报 16-view 辅助结果；每个场景至少保留 8 个未输入模型的注册相机。真正的一张照片不能运行 COLMAP，应单列为 `single-view stress`，不能混入多视图主榜。

本次已经用一条 Wikimedia Commons 4K 室内视频完成实际验证：两个同房间片段抽取 140 帧后，COLMAP 注册 `140/140`，得到 153,017 点；经过坐标归一化与 lower-room ROI 后，GenRecon 在 RTX A4000 16 GB 上完成 20-chunk、8-view、512 推理，并生成通过结构检查和原相机渲染检查的高精度 PLY 与 PBR GLB。它证明链路可行，但该素材拍摄于 checkpoint 发布前、门洞远处有人、尺度只是相机高度 proxy，且 8-view 有 3 个边缘 chunk 覆盖不足，因此只能算工程 pilot，不能进入正式主集。

## 2. GenRecon 对输入的真实约束

仓库 `Iphone` 模式期望以下目录：

```text
<scene>/
  rgb/                         # 原始或抽取帧，文件名必须与 images.txt 一致
  colmap/
    cameras.txt
    images.txt
    points3D.txt
```

当前图像加载器支持 `PINHOLE`、`SIMPLE_PINHOLE`、`SIMPLE_RADIAL`、`RADIAL` 和 `OPENCV`。其他模型应先用 COLMAP undistorter 转成 pinhole；`OPENCV_FISHEYE` 当前不支持。

还有四个容易被忽略的约束：

- chunker 把世界坐标 `z` 当竖直方向。普通单目 COLMAP 没有重力方向，必须先做 `z-up` 对齐。
- 普通单目 COLMAP 没有米制尺度。当前点云去噪使用固定米制半径，输出也按米解释，因此必须显式记录尺度来源。
- chunk 边长由房间高度决定。过高大厅会形成超出训练分布的大 chunk；仓库现有 ETH3D 实验已经观察到明显退化。
- 图像选择器假设 `images.txt` 的记录顺序近似采集顺序，再做 `linspace` 抽样。输入适配器必须确定性重排，不能依赖数据库偶然分配的 image ID。

训练配置在每个 chunk 随机使用 1 到 16 个条件视角，故 8-view 是适合主比较的固定预算，16-view 是合理上限；把视频全部帧作为模型输入既增加显存，也改变协议。

## 3. 互联网来源优先级

| 优先级 | 来源 | 用法 | 主要风险 |
|---|---|---|---|
| A | 自行或委托新拍摄，随后以 CC BY/CC BY-SA 发布 | 最强 temporal holdout；可取得原片、许可、房间身份和拍摄时间 | 需要组织采集，不是现成素材 |
| A | Wikimedia Commons 原始视频 | 可直接下载，许可和修订时间可通过 API 固化 | 合适的连续室内 walk-through 数量少；需检查人物权利 |
| B | Internet Archive 中逐条带明确 CC 许可的原文件 | 可用 API 搜索并下载原文件 | 很多素材只有 240p/360p；元数据许可由上传者填写，仍需人工核验 |
| B | Vimeo/创作者网站明确启用下载的 CC 视频 | 画质通常较高，可联系作者确认 | 平台标注不保证上传者拥有全部权利 |
| C | ETH3D、Tanks and Temples、DL3DV-Evaluation | 校准 COLMAP、坐标和评测实现 | 知名数据可能进入基础模型预训练，不能作为最强 zero-shot 证据 |
| C | RealEstate10K test | 约 8 万 clips 已有相机轨迹，可高效发现可恢复房产视频 | 来源是 YouTube，链接失效、平台条款和基础模型污染风险较高 |
| 禁止默认抓取 | 房产 listing、Matterport/Zillow showcase、酒店预订页 | 只有取得书面授权后才考虑 | 服务条款、版权、隐私、虚拟布置、全景投影和不可公开再分发 |

YouTube 虽提供 CC BY 标记，但其服务条款同时限制未经服务明确允许或权利人书面许可的下载和自动化访问。因此正式可发布测试集不应把 `yt-dlp` 当作默认法律依据；优先使用作者提供的下载、Commons/Archive 镜像或书面许可。

参考入口：

- Wikimedia Commons `Videos of interiors of buildings`：<https://commons.wikimedia.org/wiki/Category:Videos_of_interiors_of_buildings>
- Internet Archive Advanced Search API：<https://archive.org/advancedsearch.php>
- YouTube license 说明：<https://support.google.com/youtube/answer/2797468>
- Vimeo CC 风险说明：<https://help.vimeo.com/hc/en-us/articles/12427604972305>
- RealEstate10K：<https://google.github.io/realestate10k/>
- DL3DV-10K：<https://github.com/DL3DV-10K/Dataset>
- Tanks and Temples：<https://www.tanksandtemples.org/download/>
- ETH3D：<https://www.eth3d.net/datasets>

所有链接均在 2026-08-06 核验；许可应按下载当天再次归档。本轮也检索到若干 2026-06-29 之后上传到 Internet Archive 的 1080p 室内 walk-through，但条目没有 `licenseurl`/`rights`，其中一条标题还表明实际拍摄可能早于上传一年。它们只能进入“联系作者取得书面授权与拍摄日期证明”的候选池，不能因 Archive 可下载或上传日期较新就进入正式集。

## 4. 素材粗筛

### 4.1 视频硬条件

- 同一 shot 连续 20 到 90 秒；较长 house tour 先按房间和剪辑切段。
- 环境基本静止；短暂路人可剔除对应时段，但大面积人群、移动家具、开关门或变化屏幕应拒绝。
- 相机必须平移，不能只是原地摇摄。纯旋转或同一中心的全景切片没有三角化视差。
- 优先 1080p 及以上，最低边不低于 720 px；避免重度运动模糊、景深虚化、数字变焦和变焦镜头。
- 画面之间保持约 60% 到 80% 重叠，并从不同高度或侧向观察主要物体。
- 同一 shot 内不切镜头、不慢动作、不插入 B-roll、不做交叉淡化。
- 优先普通层高、单房间、1 到 12 个 chunks 的场景；大房间最多接受 24 chunks。高大厅单列 stress split。

视频描述中的 “static indoor” 应指场景静态，**不是相机静止**。COLMAP 恰恰需要相机移动产生基线。

### 4.2 静态照片集硬条件

网页照片只有在满足以下条件时才能走 COLMAP：

- 至少 12 张，建议 20 到 60 张，且明确属于同一物理房间、同一时刻附近。
- 相邻照片存在可见重叠和相机平移；每张 listing 只拍一个不同房间通常无法注册。
- 不混入 HDR 版本、虚拟布置前后对比、广角拼接、不同装修时期或重复缩略图。
- 无法确认同一房间身份时直接拒绝，不通过语义相似度强行组场景。

同一张照片的多个 crop、由一个 360 全景导出的 cubemap 面、或扩散模型生成的邻近视角都不是真实多视图数据。

## 5. 视频到 COLMAP

### 5.1 归档源证据

每条候选先保存：源页面 URL、直链、作者、许可名/URL、拍摄时间、上传时间、下载时间、原文件 SHA-256、原始分辨率/FPS/时长、隐私审查和所选时间段。页面截图或 API JSON 与数据 manifest 一起保存。

不要只保存下载后的视频；链接失效或许可变化时，没有页面快照就无法审计来源。

### 5.2 切 shot 与抽帧

普通步行视频从 1 到 2 fps 起步。相邻帧过密只会增加匹配成本和动态模糊，不会等比例增加几何信息。

```bash
ffmpeg -ss 90 -to 130 -i source.webm \
  -vf 'fps=2,scale=1920:-2:flags=lanczos' -q:v 2 \
  rgb/a_%06d.jpg
```

先保留连续帧，完成 COLMAP 后再选 8/16 个模型输入。不要在 SfM 前只剩 8 张图，否则相机注册和稀疏几何会显著变脆弱。

### 5.3 COLMAP 配置

单个固定镜头的视频使用一个 `SIMPLE_RADIAL` 或 `OPENCV` camera，按时间顺序做 sequential matching；多个同相机 shot 可以用不同前缀放入一个场景，并加入跨 shot 匹配。无序照片集使用 exhaustive matching，并按真实相机情况选择 shared/per-image camera。

本次 pilot 使用 `pycolmap==3.13.0`、单一 `SIMPLE_RADIAL` camera、每帧最多 12,000 个 SIFT 特征、sequential overlap 30、guided matching 和固定 mapper seed 42。对应 CLI 骨架为：

```bash
colmap feature_extractor \
  --database_path work/database.db \
  --image_path rgb \
  --ImageReader.camera_model SIMPLE_RADIAL \
  --ImageReader.single_camera 1

colmap sequential_matcher \
  --database_path work/database.db \
  --SequentialMatching.overlap 30 \
  --SiftMatching.guided_matching 1

mkdir -p work/sparse
colmap mapper \
  --database_path work/database.db \
  --image_path rgb \
  --output_path work/sparse

colmap model_converter \
  --input_path work/sparse/0 \
  --output_path colmap \
  --output_type TXT
```

具体 option 名随 COLMAP 版本变化，数据集必须固定版本和完整命令。若 mapper 产生多个模型，只能保留与目标房间对应且覆盖主要轨迹的单一模型；不能事后把不一致模型按视觉近似拼接。

## 6. 世界坐标归一化

COLMAP 成功后仍不能直接进入 GenRecon。建议固定以下过程，并把 4x4 similarity matrix 写入 manifest：

1. 对每个注册相机计算世界坐标中的 camera-up，稳健平均后旋转到 `+z`。
2. 检查相机 up 与均值的角度分布；P90 大于 10 度通常意味着剪辑、滚转或姿态不稳定。
3. 在水平面搜索使稀疏点 robust XY AABB 最小的 yaw，减少斜放房间导致的空 chunks。
4. 确定尺度。优先级为已知测量/AR 元数据、已知相机高度、人工标注墙高；完全未知时只能标记为 proxy scale。
5. 清除窗外、镜面后方和相邻房间的 coherent points，或给出人工 ROI。所有裁剪必须在推理前冻结并记录。
6. 重新写出相机、图像观测和 point tracks 一致的 COLMAP 模型。

用“假设相机高度 1.6 m”恢复尺度只能用于浏览和模型尺度归一化，不能据此报告厘米级几何误差。

## 7. 三层验收门槛

以下是首版保守阈值，应只在 dev split 上修订一次，然后冻结到 sealed test。

### 7.1 COLMAP gate

| 指标 | 建议门槛 |
|---|---:|
| 抽取帧 | 40 到 180 |
| 最大单一模型注册图像 | 至少 24 |
| 注册率 | 至少 70% |
| 最大模型占全部已注册图像 | 至少 90% |
| `error <= 2 px && track >= 3` 点数 | 至少 5,000 |
| 质量点占全部稀疏点 | 至少 50% |
| 点重投影误差中位数 | 不高于 1.5 px |
| track 长度中位数 | 至少 3 |
| camera-up 角度 P90 | 不高于 10 度 |
| 输入相机模型 | 当前 loader 支持，或已确定性 undistort |

此外必须人工查看稀疏点和相机轨迹，排除纯旋转、重复纹理错配、两个房间错误闭环、镜面复制和尺度漂移。

### 7.2 GenRecon preflight gate

- 普通主集层高建议在 2.2 到 3.8 m；更高场景进入 `tall-room` stress split。
- 清理后每个保留 chunk 至少 500 点；稀疏数据可降阈值，但要在协议中固定。
- 主协议固定 8 个物理相机并使用 `--center_crop`，保证不是 8 帧扩成 16 个左右 crop。
- 8 个选中相机必须覆盖每个 chunk；任何 `falling back to closest camera` 都记为 gate failure。
- 16-view 辅助协议也固定 frame IDs，不能按输出质量临时改帧。
- 16 GB 档建议不超过 24 chunks，并设置 `--proj_batch_voxels 256`；资源回退只允许确定性的 joint-decode grouping。

### 7.3 产物 gate

- 命令返回 0，且 `mesh.ply`、`to_glb_inputs.pt`、`chunk_inputs.pt` 都是本次运行新产物。
- PLY header/body 长度一致，坐标有限，全部 face 为三角形，索引在范围内。
- 至少 4 个原相机视角渲染非空且无明显坐标翻转。
- GLB 通过 header/chunk/accessor/有限坐标检查，每个预期 chunk 有 geometry、UV 和 PBR 纹理。
- Manifest、profile 和评测 summary 必须是严格 JSON；缺失指标写 `null`，禁止输出 `NaN`/`Infinity`。
- GLB 简化只影响浏览产物，不能替代高精度 PLY；必须记录每 chunk 面数上限、纹理尺寸、缓存目录和失败重试。若统一上限触发 UV 展开超时，只能使用预先定义的全场景确定性回退，不能逐块按视觉质量手调。
- 同时发布 PLY/GLB 时，在至少 4 个协议相机中检查两者 mask/depth；建议 mask IoU 中位数至少 0.95，且重叠像素中至少 90% 深度差不超过 0.10 m，以发现坐标轴或导出错误。该检查只验证产物一致性，不代表对真实场景的精度。
- 所有失败场景保留状态和失败原因；不能只发布通过 COLMAP 和视觉挑选后的成功子集而不报告筛选率。

## 8. Zero-shot 定义与防泄漏

建议不要笼统写 “zero-shot”，而是标级：

- `Z0 pipeline-calibration`：已知公开 3D benchmark，只验证工程链路。
- `Z1 genrecon-finetune-unseen`：场景不属于 SAGE-10k、3D-FRONT、ScanNet++ 或本项目调参集；只主张未用于 GenRecon scene finetuning。
- `Z2 temporal-holdout`：场景在 checkpoint 冻结后才拍摄，且来源、拍摄日期和哈希可验证。这是对 DINOv3/TRELLIS 污染更强的防线。
- `Z3 strict-foundation-unseen`：需要 DINOv3、TRELLIS 和 GenRecon 全部训练清单或可证明的后训练拍摄日期。公开信息不足时不要声称达到。

防泄漏规则：

- 按物理建筑/房产分组，不按 clip 随机切分；同一 house tour 的不同房间只能落在同一 split。
- 同一上传者/系列尽量不跨 dev/test，避免拍摄风格和地点泄漏。
- 保存 pHash，并用图像 embedding 对 SAGE render、3D-FRONT render、ScanNet++、ETH3D、Tanks and Temples、RealEstate10K/DL3DV 可访问索引做近重复检索。
- duplicate search 只能发现相似项，不能证明基础模型未见过。
- 用 3 到 5 个 dev scenes 调抽帧、COLMAP、ROI 和资源参数；sealed test 不按最终 mesh 反向调参数。
- checkpoint、DINO artifact、COLMAP 版本、源文件和 split manifest 都固定哈希。

GenRecon README 显示论文发布于 2026-05-22、公开 checkpoint 于 2026-06-29。最强的新采集应优先使用 2026-06-29 之后拍摄的场景，而不是只看上传日期。

## 9. 建议的数据集组成

### 9.1 Pilot 12

- 6 个普通住宅房间：卧室、客厅、厨房、书房等。
- 3 个工作空间：办公室、教室、工作台/实验室。
- 3 个 OOD stress：反光/玻璃、狭长走廊、挑高大厅。
- 至少 4 个独立创作者或来源；单一来源最多 3 scenes。
- 每场景保存 8 input + 8 heldout，相机均来自同一完整 COLMAP 模型。

预计需要先收集 30 到 50 条候选，才能得到 12 条同时通过许可、静态性、COLMAP 和 chunk 覆盖的场景。通过率本身应报告。

### 9.2 V1 30

- 18 个 standard rooms，作为主表。
- 6 个 appearance stress：弱纹理、镜面、玻璃、强曝光差、压缩严重。
- 6 个 geometry stress：挑高、狭长、大开间、严重遮挡。
- 主表固定 8-view；16-view 只做输入预算曲线。
- 另设 10 张 literal single images，只报告单视图方法/伪深度分支，不进入 COLMAP/GenRecon 多视图平均。

## 10. 无 GT 场景如何评测

互联网 wild 场景通常没有激光或 mesh GT，可报告：

- 数据与工程：许可通过率、COLMAP 注册率、质量点数、chunk 数、完整流程成功率、耗时和资源峰值。
- 输入/heldout 原相机渲染：coverage、PSNR、曝光补偿 PSNR、SSIM、masked LPIPS、Edge F1。
- 稀疏几何：质量 COLMAP 点到 mesh/GLB 的距离和相机射线深度一致性。
- 产物一致性：PLY 与 GLB 的 mask IoU 和深度差。
- 人工盲评：布局、物体身份、额外/幻觉表面、跨 chunk 断裂，且评审看不到方法名。

不能报告：

- sparse SfM 点并不是均匀表面真值，不能用 GLB 到 SfM 的反向距离冒充 precision。
- heldout RGB 指标受曝光、生成纹理和遮挡影响，不等于绝对几何精度。
- 用假设相机高度得到的尺度不能支持厘米阈值的跨场景比较。

因此主集应搭配一个有 GT 的公开校准集，分别呈现“wild 泛化”和“绝对几何”。

## 11. Manifest 最小字段

```json
{
  "scene_id": "source-room-shot",
  "source": {
    "page_url": "...",
    "media_url": "...",
    "creator": "...",
    "license": "CC BY-SA 4.0",
    "license_url": "...",
    "capture_utc": "...",
    "upload_utc": "...",
    "retrieved_utc": "...",
    "sha256": "..."
  },
  "shot": {"start_s": 90.0, "end_s": 130.0, "room_key": "building/room"},
  "privacy": {"people": false, "faces_redacted": false, "reviewer": "..."},
  "sfm": {
    "colmap_version": "3.13.0",
    "registered": 140,
    "source_frames": 140,
    "quality_points": 96721,
    "median_error_px": 0.85,
    "median_track": 4
  },
  "world": {
    "new_from_old": [[1, 0, 0, 0]],
    "scale_status": "proxy",
    "scale_evidence": "assumed camera height 1.6m",
    "roi_m": [[-5.5, -4.5, 0.0], [5.5, 4.5, 3.2]]
  },
  "protocol": {
    "input_frame_ids": ["..."],
    "heldout_frame_ids": ["..."],
    "view_budget": 8,
    "seed": 42,
    "checkpoint_hashes": {"ss": "...", "shape": "...", "texture": "..."}
  }
}
```

## 12. Copped Hall 实证

源视频：Wikimedia Commons，`Copped Hall 1st floor interior walk-around May Open Day 2026`，3840x2160、29.97 fps、305.028 s、CC BY-SA 4.0；拍摄于 2026-05-31，上传于 2026-06-10。源文件 SHA-256：

```text
8e9836fede465e78e47af95b90f45136a72667dc6e1974594141f67219e99c1d
```

同时保存 Wikimedia API 原始响应 `source/commons_api_snapshot.json`，其 SHA-256 为 `7d56c583a5e166c1d8224879928fbdee3203127573a7d3362edbd95f861380bb`；快照固定了 page ID `193625827`、revision `1248347582`、作者、CC BY-SA 4.0、拍摄/上传时间、原文件 URL、尺寸和 MediaWiki SHA-1。

整条视频包含多个房间、人物、剪辑、交叉淡化和片尾黑场，不能作为一个 scene。实际只取同一静态餐厅的 `90-130 s` 与 `270-300 s`，2 fps、1920x1080，共 140 帧。逐帧复查发现门洞远处仍有访客可见且未脱敏，因此该 pilot 同时触发隐私 gate failure，不能进入正式发布集。

COLMAP 3.13 实测：

| 项目 | 结果 |
|---|---:|
| 注册 | 140/140，单一模型 |
| 稀疏点 | 153,017 |
| mean track length | 6.263 |
| mean reprojection error | 0.889 px |
| ROI 后点数 | 105,386 |
| ROI 后质量点 `error<=2, track>=3` | 96,721 |
| 质量点误差中位数 | 0.848 px |
| 质量点 track 中位数 | 4 |

坐标使用平均 camera-up 对齐 `+z`，camera-up 偏差 P90 为 4.18 度；以 2% 低分位近似地面，并把相机高度 proxy 设为 1.6 m。由于完整房间高度约 6 m，本次只重建 `x=[-5.5,5.5] m, y=[-4.5,4.5] m, z=[0,3.2] m` 的 lower-room ROI。

GenRecon 主重建结果：

| 项目 | 结果 |
|---|---:|
| 协议 | 8 physical views，seed 42，512，20 chunks |
| 点清理后 | 93,648 |
| chunk 可见性 | 17/20 严格通过；3 个 fallback，故不进正式主集 |
| 重建耗时 | 740.84 s |
| GPU 进程采样峰值 | 10,908 MiB |
| PLY | 16,691,548 vertices；35,678,188 faces；764,264,621 B |
| PLY bounds | `[-6.300,-5.079,-0.001]` 到 `[6.427,5.290,3.456] m` |
| PLY 校验 | 坐标有限；三角面/索引/文件长度全部合法 |

PBR GLB 使用 4096 纹理、`--skip_fill_holes --skip_remesh`。最初的每 chunk 100 万面尝试在 8/20 后遇到病理性 xatlas chart 计算，运行 1,782.66 s 后人工终止，失败 profile 和 8 个缓存块均保留。随后没有混用缓存，而是把**全部 20 块**统一按每块 30 万面重烘焙：

| 项目 | 结果 |
|---|---:|
| GLB 烘焙 | 20/20 chunks，1,127.32 s，return code 0 |
| 烘焙资源 | process-tree RSS 峰值 8,405,749,760 B；进程显存峰值 6,416 MiB |
| `scene.glb` | 668,657,068 B；SHA-256 `0c555e5bccbb5317d611c61555d5044eb1c8118cf6e9bd3162c9f7770bcc06e0` |
| GLB geometry | 20 meshes/materials；4,285,448 vertices；5,751,398 triangles |
| GLB textures | 40 张可解码 4096x4096 PNG；每个 material 有 base-color 与 metallic-roughness texture |
| GLB 校验 | header/声明长度/BIN buffer/accessor/有限坐标/索引/UV 全部通过 |

16 个冻结相机（8 input + 8 heldout）的 640x360 渲染结果为：

| 指标 | 全部 | input | heldout |
|---|---:|---:|---:|
| GLB coverage | 85.30% | 86.19% | 84.41% |
| 原图 masked PSNR | 11.53 dB | 12.06 dB | 11.00 dB |
| 曝光补偿 masked PSNR | 13.50 dB | 13.36 dB | 13.63 dB |
| 曝光补偿 masked SSIM | 0.353 | 0.358 | 0.349 |
| masked LPIPS（128x128 patches） | 0.535 | 0.519 | 0.552 |
| Edge F1，3 px | 0.430 | 0.454 | 0.407 |
| PLY/GLB mask IoU | 0.9942 | 0.9946 | 0.9939 |
| PLY/GLB 平均绝对深度差 | 0.0200 m | 0.0171 m | 0.0230 m |
| PLY/GLB 深度差不超过 0.10 m | 97.31% | 97.37% | 97.24% |

对 93,648 个清理后的 SfM 点做精确 point-to-GLB triangle BVH 查询，距离中位数为 0.0903 m、P90 为 0.4006 m；53.99% 在 0.10 m 内、93.72% 在 0.50 m 内。这个单向指标说明生成表面与 SfM 支撑的一致性，但 SfM 稀疏点不是均匀真值，且本场景尺度来自 1.6 m 相机高度假设，因此不能把这些数值写成真实世界厘米精度。

视觉上桌椅、壁炉、砖墙和房间布局可辨认，GLB 坐标与 COLMAP 相机一致；窗户、门洞、天花板和未观察到的高处存在明显空洞。该结果证明互联网 CC 视频能走完下载、COLMAP、归一化、GenRecon、PLY、PBR GLB 和 heldout 评测链路，同时也证明“COLMAP 100% 注册”不足以保证隐私、尺度、temporal holdout 和 8-view chunk 覆盖。

可直接审计的结果包括 [GLB/原图总览](generated/internet-zero-shot/copped-hall-dining-room/fidelity/overview.jpg)、[图像 fidelity 报告](generated/internet-zero-shot/copped-hall-dining-room/fidelity/README.md)、[SfM-to-GLB 报告](generated/internet-zero-shot/copped-hall-dining-room/sfm_glb/README.md) 和 [产物结构校验 JSON](generated/internet-zero-shot/copped-hall-dining-room/artifact_validation.json)。生成数据、日志和产物位于被 `.gitignore` 排除的：

```text
data/internet-zero-shot/copped-hall-dining-room/
outputs/internet-zero-shot/copped-hall-dining-room/reconstruction/
reports/generated/internet-zero-shot/copped-hall-dining-room/
```

## 13. Foundation-SfM fallback 实证

对 raw pool 中 17 个原 masked-COLMAP `C/F` 条目，已使用公开、无需 token 的 VGGT-1B checkpoint 执行 zero-shot camera + depth 推理，并导出 GenRecon `Iphone` mode 可读取的 `colmap_vggt/`。VGGT-Omega 是 2026 年更新的官方模型，但 checkpoint gated；当前机器没有获批 Hugging Face 凭据，因此未绕过授权运行。

固定协议包括动态区域白底推理、RGBA alpha 输出、confidence P70、至少一个其他视图的 15% 相对深度一致性、最多 100,000 点、相机方向 `z-up` 和 1.6 m 相机高度 proxy scale。结果为：

- 17/17 生成 COLMAP text 和 PLY，共 1,700,000 点、235 个相机。
- GenRecon 默认清理后共 1,496,635 点和 126 个 chunks。
- 15 条 consumer preflight 零 fallback；Hilltop、Lincoln 各 1 个 fallback。
- pseudo geometry 分级为 `P-A=4、P-B=3、P-C=4、P-F=6`。
- 视觉复核后仅 Wawasee 和 Unity House 直接进入 foundation pilot；5 条需重新切 shot，10 条拒绝当前 shot。

Beverly 和 Jabal 是关键反例：两者技术上为 P-A，但前者实际是室外露台，后者是固定机位人物进食。由此规定：foundation P-A/P-B 只表示初始化可用性，不能与真实 SfM A/B 混报，也不能覆盖 shot/domain、权利、隐私、尺度和真实几何 gate。

完整模型 provenance、逐场景指标、消费验证和命令见 [VGGT Foundation-SfM Fallback 实验报告](FOUNDATION_SFM_FALLBACK_REPORT_zh.md)。审查入口为 `data/internet-zero-shot/foundation-sfm-v1/index.html`。

## 14. Native-SfM GenRecon 实证

原 masked-COLMAP v2 gate 为 A 的 Copped Hall、Waco Fire Station 和 Harnden Tavern 已单列为 `native-sfm` 轨道，不经过 VGGT。统一 adapter 将 `SIMPLE_RADIAL` RGB、动态 mask 和 2D observations 无畸变为 PINHOLE RGBA，保留真实 point ID、RGB、ERROR 和 TRACK；随后仅施加相机平均 up 的 `z-up` 与 1.6 m 相机高度 proxy scale。

- 303 个注册相机、78,101 个原生 COLMAP 点。
- GenRecon 清理后 22,783 个真实 SfM 点、19 chunks。
- 3/3 consumer preflight passed，零 fallback。
- 3/3 完成 shape、texture、PLY 与 PBR GLB。
- 共 24,621,583 vertices、52,622,420 faces；PLY+GLB 为 1,767,812,919 B。
- 303 个注册相机全部导出原图/重建/左右对比，共 9 个 H.264 视频。
- Copped 仅为 engineering pilot；Waco、Harnden 因持续人物和许可未声明保持 reject。

完整协议、逐场景 SHA256、目录和验证见 [Native COLMAP-SfM 到 GenRecon 完整产物报告](NATIVE_SFM_GENRECON_REPORT_zh.md)。视频入口为 `outputs/internet-zero-shot/native-sfm-video-comparisons-v1/index.html`。

20 条候选因此形成互斥且完整的技术分流：`3 native SfM + 17 foundation fallback`。两条轨道分榜，`SfM-A` 不与 `P-A/P-B` 混报。

## 15. 下一步

1. 17 个可消费 foundation 候选的完整 GenRecon/PLY/PBR GLB 已全部生成；先检查 Wawasee 和 Unity House 资产，再分别冻结 8 input + 8 heldout 做正式 fidelity 评测。结果单列为 foundation-pseudo-geometry track，不进入真实 SfM 主榜。
2. 对 Cologne、Evel Museum、Burlington、Hilltop 和 Crockett 重新切成单一物理空间，再运行完全相同的 VGGT 协议，不针对结果调阈值。
3. 继续收集普通层高、许可明确且优先在 2026-06-29 后拍摄的候选；只有上传日期而无拍摄日期证明，或只有可下载文件而无明确许可的条目，不进入正式候选。
4. 正式 `Z2` 场景继续要求真实 COLMAP、可审计尺度、`z-up`、8-view 零 fallback、许可和隐私通过；foundation 输出只作 fallback/diagnostic。
5. 取得官方 VGGT-Omega checkpoint 授权后，在同一冻结视图上做 VGGT-1B/Omega 对照；MASt3R-SfM 作为第二模型交叉验证，不把模型间一致性误写成真实精度。
6. 同时用 ETH3D/Tanks and Temples 做有 GT 校准，避免仅凭互联网 heldout RGB 或 foundation 自洽指标得出绝对几何结论。
