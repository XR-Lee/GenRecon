# ETH3D `lecture_room` 独立 Object-Centered Chunk 实验

## 重要修正：局部Mask索引

后续strict-object可视化发现：SAM masks以`a0_current/cameras.json`为canonical顺序，而局部三chunk selector产生了不同的scene-camera列表。旧局部N1/N2/N2b按数组下标套用mask，只有24/32 crops实际对应，且第三主视图local 12为`DSC_0916`、canonical mask 12为`DSC_0925`。

因此本文旧局部N1/N2/N2b表格及“depth ownership相对N1的因果增量”结论已撤回，不应引用。完整11-chunk `full_scene_obj_center_n2b`的32/32 camera keys与canonical一致，完整场景表格仍有效。

修正后的局部对照为`n2c_contextual_canonical`，strict object-only对照为`o3c_strict_sam31_canonical`。详见：

- `reports/ETH3D_LECTURE_ROOM_STRICT_OBJECT_CENTRIC_EXPERIMENT_zh.md`
- `outputs/eth3d/lecture_room/fallback_instance_experiment/independent_object_chunks/MASK_INDEX_CORRECTION.md`

## Goal

在不训练新权重、不修改正式 `lecture_room/final` 的条件下，建立SAM3.1引导的独立工作台chunk分支，并验证：

1. 独立object branch是否比原scene-wide chunks更一致。
2. SAM像素归属是否足够。
3. COLMAP track depth和free/surface/occluded ownership是否是必要增量。
4. 在局部迭代没有明显全局退化后，完整生成一次11-chunk object-centered全场景，并提供全部23个原COLMAP相机的效果可视化。

原强标准要求shape disagreement AUC下降25%且heldout几何不退化。本轮允许放宽为：工作台主要几何和全局指标不能出现明显净退化，同时必须把coverage损失、局部反例和后处理单独报告。

## 固定条件

- ETH3D `lecture_room`，16个输入物理相机、32个左右crops、7个heldout相机。
- Seed 42、512模型、12步Euler、`occ_threshold=-1`。
- 相同模型权重、scene crops、物理`cond_2D`视图和联合解码上限。
- 原主视图固定为scene views 1/24/12。
- 709个COLMAP训练point IDs用于mask/depth；304个point IDs保留为anchor-bank holdout。
- Heldout相机不参与mask提示、track depth、chunk布局或融合规则。
- SAM3.1 checkpoint SHA-256：`0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`。
- 正式重建 `outputs/eth3d/lecture_room/final` 未被替换。

## 实现

### SAM part masks

SAM3.1为32个crops持久化union mask以及`tabletop`、`cabinet_front`逐part mask。Union mask与先前E-SAM3.1使用的32张mask逐像素完全相同。

当前预训练GenRecon没有part-ID输入通道，因此逐part mask用于诊断和ROI所有权；送入DINO条件的RGB仍使用工作台union mask，mask外保留15%亮度。

### Mask-aware `cond_3D`

对DINO 32x32 patch网格应用SAM所有权：

- ROI外保持原投影规则。
- ROI内只有投影到工作台mask patch的视图可参与聚合。
- 不再允许背景patch仅因正深度和图像范围检查就进入工作台体素。

### Track-depth ownership

只使用COLMAP `points3D` 中真实观测该point ID的输入物理相机。每个crop将训练tracks投到32x32 patch网格并取patch内深度中位数；最近填充半径为2 patches。

- 工作台mask patches：9,998。
- 具有track depth的patches：5,061。
- Mask内depth覆盖：50.62%。
- 无可靠depth的mask patches保留为`unknown`，没有伪造稠密深度。

对候选体素分类：

- `free`：位于相机和track表面之前。
- `surface`：位于8cm或12cm surface band内。
- `occluded`：位于track表面之后。
- `unknown`：mask有效但无邻近track depth。

`surface/unknown`可进入特征聚合，已知`free/occluded`不参与该视图聚合。Sparse occupancy阶段要求至少两个free votes且没有surface vote才执行硬抑制。

### 独立分支与融合

N0-N2b只运行原003/004/005物理范围对应的3 chunks。它们使用独立MultiDiffusion采样图，但N0-N2b之间固定布局、seed和初始噪声协议。

融合只在预注册工作台/柜体ROI内替换baseline三角面。纯替换会在新分支缺失处产生空洞，因此最终另提供5cm coverage safeguard：只有新对象表面在baseline face附近5cm内有支持时才删除baseline face；无支持处保留baseline。

Safeguard是显式后处理，不是纯GenRecon生成结果。

## 独立分支消融

| 条件 | Sparse AUC | Shape AUC | Texture AUC | Heldout中央台面中位/P90 | 10cm内 | 平面P90 |
|---|---:|---:|---:|---:|---:|---:|
| N0 独立raw branch | 1.219 | 0.909 | 0.244 | 4.33 / 7.48cm | 93.22% | 6.23mm |
| N1 + SAM mask ownership | 1.250 | 0.947 | 0.264 | 4.14 / 21.45cm | 80.79% | 9.12mm |
| N2 + 8cm track depth | 1.124 | 0.890 | 0.237 | 1.67 / 3.81cm | 98.87% | 1.30mm |
| N2b + 12cm track depth | 1.058 | 0.948 | 0.222 | 1.58 / 3.55cm | 100.00% | 1.17mm |

结论：

- N1相对N0三个AUC全部恶化，且P90大幅退化。独立branch加SAM像素归属不是充分条件。
- N2相对N0三个AUC方向一致改善，中央台面P90下降约49%，证明track depth ownership有实际增量。
- N2b进一步改善sparse/texture和几何，但shape AUC回升。选择N2b是以heldout几何为主的折中，不是所有latent指标同时最优。
- N2的8cm版本删除208个已有occupancy的hard-free cells；该约束规模有限，没有把ROI大面积清空。

独立3-chunk分支只覆盖场景局部，因此其全局coverage不能与11-chunk全场景直接比较；独立分支AUC也只应在N0-N2b的相同pair graph内比较。

## 全场景 Object-Centered 尝试

完整运行保留11 chunks和原模型尺度，只将003/004/005从规则中心：

- `x=-1.522/0.108/1.738m, y=-1.669m`

改为COLMAP对象中心：

- `x=-1.114/0.516/2.146m, y=-1.262m`

并应用SAM3.1 union mask、12cm track-depth ownership和hard-free sparse gating。

运行结果：

- 时间：463.2s。
- 峰值进程显存：10,000MiB。
- 纯生成PLY：10,634,848顶点、22,493,559面、483,843,844 bytes。

### Heldout几何

| 条件 | 完整台面中位/P90 | 台面10cm内 | 柜体中位/P90 | Anchor-holdout中位/P90 | 全局中位 | 全局depth覆盖 |
|---|---:|---:|---:|---:|---:|---:|
| A baseline | 2.36 / 11.33cm | 87.03% | 1.35 / 1.77cm | 1.44 / 5.05cm | 6.59cm | 77.37% |
| A0 object-center raw | 2.55 / 13.20cm | 80.39% | 1.28 / 1.68cm | 1.49 / 9.25cm | 6.20cm | 80.07% |
| Full object-center N2b | 1.59 / 7.33cm | 93.60% | 0.42 / 1.10cm | 0.62 / 3.57cm | 5.93cm | 76.06% |
| Full safeguarded 5cm | 1.65 / 7.93cm | 92.49% | 0.42 / 1.16cm | 0.64 / 4.72cm | 6.59cm | 77.32% |

相对baseline，纯生成版改善完整台面、柜体、anchor-holdout和全局中位，但全局depth覆盖下降1.31个百分点，关键视图可见局部柜门空洞。

5cm safeguard将全局coverage损失压到约0.05个百分点，并保留大部分工作台收益：完整台面中位/P90约改善30%/30%，柜体约改善68%/35%。

### 局部反例

本实验重新定义的中央台面core为世界坐标`x=[-0.70,0.70]`与完整台面相同的y/z范围。相对baseline：

- 中位：0.64cm增加到1.07cm。
- P90：3.41cm增加到3.69cm。
- 10cm内：97.15%增加到98.78%（纯生成）或98.37%（5cm safeguard）。

因此不能表述为“台面所有子区域都改善”。收益主要来自完整台面的长尾、柜体和anchor-holdout；中央台面的典型误差略有增加。

### Cross-chunk disagreement

| 条件 | Sparse AUC | Shape AUC | Texture AUC |
|---|---:|---:|---:|
| A baseline | 0.826 | 0.890 | 0.231 |
| A0 object-center raw | 0.881 | 0.977 | 0.245 |
| Full object-center N2b | 0.915 | 0.851 | 0.226 |

相对raw object-center布局，N2b将shape/texture AUC降低约12.9%/7.5%，但sparse AUC增加约3.8%。相对baseline，shape/texture改善约4.5%/2.0%，sparse恶化约10.8%。强25% AUC标准仍未达到。

## 判定

本轮为**放宽标准下的部分成功**：

1. SAM-only独立分支被明确否定。
2. Track depth ownership相对SAM-only和raw独立分支同时改善多个heldout几何指标。
3. 完整object-centered生成显著修复了原A0自适应布局的工作台退化。
4. 纯生成版存在约1.3个百分点coverage损失和可见局部空洞。
5. 5cm safeguard将全局coverage恢复到baseline尺度，同时保留完整台面和柜体的主要收益。
6. 中央台面中位/P90仍略差于baseline，且shape/sparse AUC没有统一改善，所以不能称为完整几何成功或工业测量级重建。

支持的因果结论是：

> 独立object chunks只有在加入COLMAP track depth ownership后才产生稳定几何收益。SAM解决像素归属，但depth ownership决定这些特征能否分配到正确三维表面。

## 限制

- 只有seed 42。
- ETH3D test split没有dense/laser GT。
- Track depth只覆盖约50.6%的SAM mask patches，其余仍依赖生成先验。
- 当前part masks没有作为独立learned token/channel输入预训练模型。
- Free-space证据是推理期硬约束，不是经过训练的occupancy likelihood。
- ROI融合按face centroid和最近表面距离执行，没有拓扑焊接或watertight保证。
- 5cm safeguard可能保留靠近新表面的旧几何；纯生成版和safeguarded版必须分开解释。
- 纯生成PLY保留GenRecon PBR顶点属性；safeguarded流式融合PLY只保留几何、RGB和alpha，不保留metallic/roughness，不能直接替代正式PBR/GLB资产。
- 直接效果图仍显示纹理漂移、小物体形变、黑色缺失区和局部对象边界痕迹。

## 主要产物

- 纯全场景生成：`outputs/eth3d/lecture_room/fallback_instance_experiment/independent_object_chunks/full_scene_obj_center_n2b/mesh.ply`
- 推荐coverage-safeguarded版：`outputs/eth3d/lecture_room/fallback_instance_experiment/independent_object_chunks/full_scene_obj_center_n2b_safeguarded5/mesh.ply`
- 最终可视化：`outputs/eth3d/lecture_room/fallback_instance_experiment/independent_object_chunks/final_visualization/index.html`
- 关键视角：`outputs/eth3d/lecture_room/fallback_instance_experiment/independent_object_chunks/final_visualization/key_views_comparison.jpg`
- 全23视角：`outputs/eth3d/lecture_room/fallback_instance_experiment/independent_object_chunks/final_visualization/all_views_contact.jpg`
- 布局图：`outputs/eth3d/lecture_room/fallback_instance_experiment/independent_object_chunks/final_visualization/chunk_layout_comparison.png`
- 统一结果：`outputs/eth3d/lecture_room/fallback_instance_experiment/independent_object_chunks/final_analysis/summary.json`
