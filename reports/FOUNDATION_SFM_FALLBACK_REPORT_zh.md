# VGGT Foundation-SfM Fallback 实验报告

## 1. 目标与结论

本实验针对传统 masked COLMAP 未通过的互联网室内视频，验证开源/公开权重的多视图 foundation model 是否可以 zero-shot 生成可被 GenRecon 消费的基础点云、相机位姿和 COLMAP text。

结果是：**可以生成并被 GenRecon 实际读取，但“能生成”与“有场景参考价值”必须分开判定。**

- 17/17 个原 SfM `C/F` 条目均生成了 VGGT 点云和 `colmap_vggt/`。
- 17/17 个 COLMAP text 均可由 pycolmap 重新加载。
- 17/17 个场景均被 GenRecon `IphoneChunker` 和 `IphoneImageSelecter` 实际读取并产生 chunks。
- 15 个 preflight 零 fallback；Hilltop 和 Lincoln 各有 1 个 closest-camera fallback，记为 marginal。
- foundation 技术分级为 `P-A=4、P-B=3、P-C=4、P-F=6`。
- 视觉复核后，只有 Wawasee 和 Unity House 可直接进入下一轮 foundation pilot；5 条应重新切分，10 条当前 shot 应拒绝。

因此，foundation point cloud 适合作为 GenRecon 初始化、空间分块和失败诊断，但不能替代真实 SfM/扫描，也不能自动修复错误 shot、幻灯片、室外内容或固定机位动态人物。

## 2. 模型选择

### 2.1 候选比较

| 模型 | 状态 | 权重可获得性 | 工程特征 | 本轮决策 |
|---|---|---|---|---|
| VGGT-Omega-1B-512 | 2026 官方最新，静态/动态场景 | gated；当前机器请求返回 `401 GatedRepo` | 官方报告约为原 VGGT 30% 显存，适合长序列 | 代码已审查，因无获批 HF 凭据而未运行 |
| VGGT-1B | CVPR 2025，2026 commit 修复冗余中间张量 | 公开、无需 token | 联合预测相机、depth、point map；无 CUDA 扩展编译 | **本轮采用** |
| MASt3R / MASt3R-SfM | ECCV 2024 / 3DV 2025 | checkpoint 公开 | pair matching + sparse global alignment；依赖 DUSt3R submodule，可选 ASMK/CUDA RoPE | 保留为后续对照，不作为第一实现 |

选择 VGGT-1B 的原因：

1. 它直接联合输出 camera、intrinsics 和 depth，适合 COLMAP 已经没有全局位姿的 fallback。
2. 官方代码能直接转换为 COLMAP；当前仓库也已有 `Iphone`/`colmap_mast3r` 风格消费接口。
3. 在 RTX A4000 16 GB 上，16 个 518x518 输入的实测峰值为 9,287.9 MiB。
4. MASt3R 当前官方 README 明确称 GLOMAP/COLMAP 脚本为未充分测试的 toy；完整 MASt3R-SfM 还需要 DUSt3R、global alignment 和更多依赖。

### 2.2 固定 provenance

- VGGT code repository：`https://github.com/facebookresearch/vggt`
- code revision：`a288dd0f14786c93483e45524328726ab7b1b4ce`
- model repository：`facebook/VGGT-1B`
- model revision：`860abec7937da0a4c03c41d3c269c366e82abdf9`
- checkpoint：`model.pt`
- checkpoint 大小：5,026,874,952 B
- checkpoint SHA256：`d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`
- checkpoint license：CC-BY-NC-4.0，仅限非商业使用

MASt3R 代码为 CC BY-NC-SA 4.0，checkpoint 还继承多项训练数据许可，官方 `CHECKPOINTS_NOTICE` 特别提示 MapFree 等限制。两条路线都不能被笼统描述为无条件商用开源。

## 3. 处理协议

### 3.1 视图选择

每条最多选择 16 个视图：

- 若 masked COLMAP 有至少 4 个最大模型注册视图，从该 fragment 按时间分箱，并以低动态占比作局部 tie-break。
- 若没有可用模型，从 COLMAP database 的 verified pair inlier 图中选择得分最高的连续 16 帧窗口。
- 这一策略优先得到一个可重建子场景，不声称连接原视频中的所有房间或全部模型碎片。

### 3.2 动态 mask

- VGGT 输入将 Mask R-CNN 动态区域精确置白；resize 后再次用 nearest-neighbor mask 覆盖，避免 bicubic 边界重新混入 RGB。
- 输出给 GenRecon 的 `rgb/*.png` 保存原 RGB，并把动态 mask 写入 alpha；不做生成式 inpainting。
- foundation 点只从静态、有效画幅像素导出。

### 3.3 点过滤

- 输入保持长宽比，pad 到 518x518，patch multiple 为 14。
- 在所有静态有效像素中使用 scene confidence P70 作为阈值。
- 每场景最多预采样 250,000 点。
- 每个点投影到其他预测相机；要求至少一个其他视图的相对深度误差不超过 15%。
- 最终按 source frame、4x4 pixel cell、support 和 confidence 稀疏化到 100,000 点。
- PLY 保存 `confidence`、`support` 和 `source_frame`；COLMAP text 每点保留一个 source observation。

跨视图深度检查仍是同一个 foundation model 的内部自洽性，不是独立真实精度证据。

### 3.4 坐标、尺度与 COLMAP 输出

- `z-up`：取预测相机平均 `-camera-y` 为世界 up。
- 水平朝向：取平均 camera-forward 在水平面的投影。
- proxy scale：将相机中心到点云低部 P02 的中位高度设为 1.6 m；异常时回退为点云 P02-P98 高度 2.7 m。
- 原点：相机中心水平中位数和 proxy floor。
- 相机模型：每视图独立 `PINHOLE`，恢复到原始 RGB 分辨率。
- `points3D.txt` 的 `ERROR=0` 仅为兼容占位，不能作为真实 reprojection error。

## 4. 总体结果

- 输入候选：17
- 选择视图：235
- VGGT 纯推理时间合计：44.23 s
- 16-view 单场景推理：约 3.0-3.14 s
- GPU 峰值：9,287.9 MiB
- 静态有效预测像素：31,460,439
- 跨视图预检点：4,140,048
- 通过跨视图深度一致性的点：3,823,458
- 候选平均跨视图通过率：92.48%
- 导出点：1,700,000
- GenRecon 默认空间清理后点：1,496,635
- GenRecon chunks：126
- closest-camera fallback：2
- 产物目录：约 644 MiB

所有 235 张 GenRecon 输入均为可解码 RGBA；1,700,000 个 PLY 点和 COLMAP 点数完全一致。源 observation 在导出坐标变换后的最大数值 reprojection residual 为 0.110 px。

## 5. 逐场景结果

`P-A/P-B` 只表示 pseudo geometry initialization pass，`P-C` 为 marginal，`P-F` 为 foundation geometry fail。

| ID | 原 SfM | Foundation | Preflight | Clean points | Chunks/Fallback | 视觉处置 |
|---|---:|---:|---:|---:|---:|---|
| Cologne Cathedral | F | P-C | pass | 92,952 | 6/0 | 缩短子段后重试；相机旋转与 COLMAP 不一致 |
| Castleton | F | P-F | pass | 95,548 | 6/0 | 拒绝；重复室外立面、近零基线 |
| Blue Rose | F | P-F | pass | 98,598 | 5/0 | 拒绝；标题/幻灯片 |
| Beverly Theater | C | P-A | pass | 93,756 | 12/0 | 拒绝；实际为室外露台，不是目标室内房间 |
| Evel Museum | C | P-A | pass | 97,511 | 4/0 | 清理人物主导视图后重试 |
| Burlington High | F | P-B | pass | 52,000 | 6/0 | 按 kitchenette/corridor/meeting room 分段 |
| Hilltop Arts | C | P-C | marginal | 96,114 | 12/1 | 单 auditorium 有潜力，缩短后重试 |
| Wawasee Arts | C | P-A | pass | 92,913 | 3/0 | **进入 foundation pilot** |
| Lake Johanna | F | P-F | pass | 94,239 | 9/0 | 拒绝；动态仪式、近零基线 |
| Wastewater | F | P-F | pass | 67,369 | 11/0 | 拒绝；室外主持人、近零基线 |
| Lincoln Institute | F | P-C | marginal | 92,388 | 15/1 | 拒绝；低分辨率标题卡/幻灯片 |
| Unity House | F | P-B | pass | 96,099 | 7/0 | **进入 foundation pilot**；无 SfM anchor |
| Moses Myers | F | P-F | pass | 96,821 | 2/0 | 拒绝；幻灯片、近零基线 |
| Frederick Funeral | F | P-C | pass | 93,689 | 3/0 | 拒绝；海报特写，不是房间覆盖 |
| Jabal Restaurant | C | P-A | pass | 95,573 | 7/0 | 拒绝；固定机位人物进食，高动态 |
| Crockett School | F | P-B | pass | 43,985 | 16/0 | lobby 与 stage 分开后重试 |
| Bethel Home | F | P-F | pass | 97,080 | 2/0 | 拒绝；人物活动、近零基线 |

这里出现了重要反例：Beverly 和 Jabal 都得到 `P-A`，但视觉语义不满足目标。前者是室外，后者主要重建人物周围的固定视角。说明 foundation 几何自洽和 COLMAP fragment agreement 仍不足以替代 shot/domain 人审。

## 6. Wawasee Pilot

Wawasee 是当前最完整的端到端样例：

- 从最大 COLMAP fragment 的 102 张注册图像中选择 16 张。
- 输入动态占比约 0-10.6%。
- 250,000 个预采样点中 224,142 个通过跨视图深度检查，通过率 89.66%。
- 导出 100,000 点，全部为 cross-view verified。
- 点 support 中位数 9 个其他视图。
- GenRecon 默认清理后 92,913 点。
- 3 个 chunks，零 fallback。
- 与 masked COLMAP fragment 做 Sim(3) 后，相机中心误差 P90 为 fragment baseline 的 17.56%。
- 相机旋转误差 P90 为 5.25 度。

该场景仍包含人物，需要沿用 alpha mask；尺度是 1.6 m 相机高度 proxy，不能用于厘米级评测。

## 7. Unity House Pilot

Unity House 是“传统 SfM 无法形成有效模型，但 foundation 有参考价值”的代表：

- 原输入为 270x480 竖屏视频。
- 当前连续窗口是一个 bunk bedroom，动态内容很少。
- VGGT 输出 `P-B`，没有足够 COLMAP camera anchor，因此最高不能记为 P-A。
- GenRecon 默认清理后 96,099 点。
- 7 个 chunks，零 fallback。

它适合测试 GenRecon 在低分辨率 foundation pseudo geometry 上的鲁棒性，但不适合做真实精度基准。

## 8. GenRecon 消费验证

每个场景实际执行：

1. `IphoneChunker(colmap_subdir="colmap_vggt")`；
2. 默认 statistical + radius point cleaning；
3. 依据 proxy `z-up` 点云生成 chunks；
4. `IphoneImageSelecter(center_crop=False)` 读取 PINHOLE、RGBA 和相机位姿；
5. 选择 8 个原始视图，并按现有双裁剪策略生成最多 16 个 2D/3D conditioning crops；
6. 记录 `cameras.json`、chunk transforms、clean PLY 和 closest-camera fallback。

这验证了数据接口和 chunk/image 前处理，尚未运行 GenRecon diffusion/mesh 生成。

## 9. 复现命令

```bash
# 固定 VGGT source revision
git clone https://github.com/facebookresearch/vggt data/model-sources/vggt
git -C data/model-sources/vggt checkout a288dd0f14786c93483e45524328726ab7b1b4ce

# checkpoint 会按固定 HF revision 自动下载并校验 SHA256
PYTHONPATH=/tmp/pycolmap-wheel .venv/bin/python \
  tools/prepare_foundation_sfm.py run

PYTHONPATH=/tmp/pycolmap-wheel .venv/bin/python \
  tools/prepare_foundation_sfm.py quality

PYTHONPATH=/tmp/pycolmap-wheel .venv/bin/python \
  tools/prepare_foundation_sfm.py preflight

PYTHONPATH=/tmp/pycolmap-wheel .venv/bin/python \
  tools/prepare_foundation_sfm.py summarize

PYTHONPATH=/tmp/pycolmap-wheel .venv/bin/python \
  tools/prepare_foundation_sfm.py validate
```

单场景 GenRecon 的输入方式为：

```bash
.venv/bin/python reconstruct_scene.py \
  --path data/internet-zero-shot/foundation-sfm-v1/candidates/raw-009-wawasee-performing-arts \
  --mode Iphone \
  --colmap_subdir colmap_vggt \
  --num_imgs_per_scene 8 \
  --output_path outputs/internet-zero-shot/wawasee-vggt \
  --ss_ckpt "${SS_CKPT}" \
  --shape_ckpt "${SHAPE_CKPT}" \
  --tex_ckpt "${TEX_CKPT}"
```

本报告在 foundation-SfM 阶段完成了真实 GenRecon chunk/image consumer preflight。随后已对 17 个可消费候选全部运行 diffusion、mesh、texture 和 PBR GLB；完整结果见 [Foundation-SfM 到 GenRecon 完整 Mesh 资产报告](FOUNDATION_GENRECON_MESH_ASSETS_REPORT_zh.md)。

## 10. 产物与验证入口

- 工具：`tools/prepare_foundation_sfm.py`
- 单元测试：`tests/test_prepare_foundation_sfm.py`
- 根目录：`data/internet-zero-shot/foundation-sfm-v1/`
- 审查页：`data/internet-zero-shot/foundation-sfm-v1/index.html`
- 固定总览：`data/internet-zero-shot/foundation-sfm-v1/overview.jpg`
- 机器索引：`data/internet-zero-shot/foundation-sfm-v1/index.json`
- CSV：`data/internet-zero-shot/foundation-sfm-v1/summary.csv`
- provisional visual review：`data/internet-zero-shot/foundation-sfm-v1/visual_review.json`
- 独立验证：`data/internet-zero-shot/foundation-sfm-v1/validation.json`

独立验证结果为 pass：235 个 RGBA 的 297,777,600 个 alpha 像素全部匹配动态 mask，1,700,000 个 PLY/COLMAP 点、235 个 cameras/images、全部 GenRecon preflight 和 89 个验证前 JSON 均通过；相关测试为 42 passed。

## 11. 不能越过的边界

1. foundation 点是模型预测，不是 feature triangulation，也不是扫描 GT。
2. proxy scale 和自动 `z-up` 只满足 GenRecon 分块需要，不能支持米制精度声明。
3. 同模型跨视图 depth consistency 可能共同自洽地出错。
4. 与小型或退化 COLMAP fragment 一致，也不能证明真实几何正确。
5. foundation 模型可能训练过相似建筑/视频，严格训练污染仍未知。
6. 权利、隐私、人物发布和拍摄日期 gate 完全独立；本实验不改变原候选的发布资格。
7. P-A/P-B 与原 SfM A/B 不同，不能在同一主榜中混报。
8. 当前 VGGT-1B 权重为非商业许可；公开发布产物前需要再次审查模型输出许可和源视频许可。

后续完整推理确认 17/17 均能生成 PLY 与 PBR GLB，但这一结果不改变视觉 gate：只有 Wawasee 和 Unity House 建议直接继续 heldout 评测；Cologne、Evel、Burlington、Hilltop、Crockett 应先重新切成单物理空间。VGGT-Omega 只有在取得官方 checkpoint 授权后才应加入同场景对照。
