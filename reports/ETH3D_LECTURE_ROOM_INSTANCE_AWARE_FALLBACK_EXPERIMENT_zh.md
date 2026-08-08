# ETH3D `lecture_room` Instance-Aware 降级与 SAM3.1 对照实验

## Goal

原始降级Goal是在不使用SAM3.1、也不训练GenRecon新权重的条件下，验证 `COLMAP实例锚点 + 对象感知条件 + 自适应chunk + SAM2.1 mask` 是否能降低工作台ROI的cross-chunk denoiser预测冲突，并改善未参与条件构建的几何观测。

补充对照Goal是在其他条件不变时，用本地官方SAM3.1 Object Multiplex checkpoint替换SAM2.1 mask生成器，分别比较mask点召回、时间、显存、denoiser disagreement和heldout几何。

预设强成功标准始终是：ROI shape disagreement AUC至少下降25%，同时heldout工作台几何不退化。SAM2.1和SAM3.1均未达到。

## 固定条件

- ETH3D `lecture_room`，16个输入物理相机，7个heldout相机。
- 512模型、seed 42、12步Euler、`occ_threshold=-1`。
- 相同scene-wide 32 crops和`cond_3D`。
- 相同联合解码与分组上限。
- 所有实验只生成PLY，不经过GLB简化和烘焙。
- 工作台ROI由台面和柜体正面两个世界坐标box组成。
- 1,013个高置信实例点中709个用于条件构建，304个point IDs留作anchor-bank holdout。
- SAM3.1固定官方仓库commit `96914d2425f90a64f45ca977c2b5165418099543`。
- 本地checkpoint为 `sam3.1_multiplex.pt`，SHA-256 `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`。
- SAM3.1正式性能在独立Python 3.12.13、PyTorch 2.7.1+cu126、CUDA 12.6环境测量；没有升级GenRecon环境。

## A0：坐标相位

穷举的几何预检显示，自适应3-chunk布局将工作台最佳边界裕量P10从30.9cm提高到38.4cm；但实际生成结果更差：

| 布局 | Shape AUC | Texture AUC | 台面核心中位 | 平面残差P90 |
|---|---:|---:|---:|---:|
| 当前布局 | 0.803 | 0.207 | 1.12cm | 1.12mm |
| 自适应3 chunks | 0.838 | 0.204 | 2.90cm | 6.22mm |

三个固定随机相位也表现出不同的shape/texture权衡。纯几何interior margin不能预测denoiser一致性或最终表面质量，因此后续阶段保留当前布局。

## B：共享实例锚点

从709个训练实例点中用farthest-point sampling选择64个三维anchors。每个anchor token聚合对应三维点的DINO投影特征并恢复到典型token范数；仅chunks 003/004/005追加相同tokens。

B1使用当前in-frustum规则聚合所有正深度和图像范围内的crops：每个token有5–12个候选视图，中位8个。结果shape/texture AUC分别变差约15.2%/11.4%，说明visibility-unaware token bank重新引入了遮挡污染。

B2严格限制到points3D track中真实观测该点的物理相机，再用crop UV筛选：

- 64个tokens全部有输入观测，每个对应2–12个crops，中位4个。
- Sparse/shape/texture AUC相对A分别改善约1.2%/1.9%/5.8%。
- Shape final从0.205升到0.220，仍然变差。
- Heldout台面核心中位误差从1.66cm增加到2.36cm。
- 台面平面残差P90从1.12mm增加到5.30mm。

真实track过滤彻底消除了B1的shape崩坏，但未经训练直接追加tokens仍未形成几何成功。该结果不能否定经过adapter/LoRA训练的instance token。

## C：对象感知选图

不使用B的失败tokens，只根据训练实例点在每个chunk和scene crop中的台面/柜体覆盖率选择主视图：

- Chunk 003：view 2 `DSC_0917 left`
- Chunk 004：view 0 `DSC_0927 left`
- Chunk 005：保留view 12 `DSC_0925 left`

结果：

- Sparse AUC改善约3.2%。
- Shape AUC升至0.979，变差约22.0%。
- Texture AUC升至0.217，变差约4.4%。
- Heldout台面核心中位/P90为1.64cm/12.09cm，接近当前1.66cm/12.48cm。
- `DSC_0903`平面拟合残差P90改善到0.55mm。

最大化稀疏实例点覆盖不是可靠的`cond_2D`视图目标；构图、观察方向和训练图像分布仍然重要。

## E：SAM2.1 降级自动mask

使用非gated官方 `facebook/sam2.1-hiera-small`。COLMAP训练点分别提示台面和柜体，SAM2生成两个mask后取并集；不把相机序列当视频，跨视角ID完全由COLMAP负责。

32个crop聚合mask质量：

- 训练点召回：86.24%
- Anchor-bank heldout点召回：86.44%
- 平均mask面积：25.31%
- 原主视图1/24/12的heldout召回：94.34% / 90.63% / 66.67%

保持原始主视图和全部其他条件不变，只在chunks 003/004/005的mask外保留15%图像亮度。

## F：SAM3.1 Object Multiplex 对照

SAM3.1使用相同32个crops、相同709/304 point-ID拆分、相同两个part ROI和每part最多8个COLMAP正提示。官方multiplex交互接口不支持与points同时传入box，因此另外补跑SAM2的point-only、single-mask配置，分离模型差异和提示协议差异。

| Mask条件 | 提示协议 | 训练点召回 | Heldout point-ID召回 | 平均mask面积 | 32-crop总时间 | 峰值进程显存 |
|---|---|---:|---:|---:|---:|---:|
| SAM2 deployed | points + boxes，multimask | 86.24% | 86.44% | 25.31% | 9.57s | 774MiB |
| SAM2 matched | point-only，single-mask | 86.50% | 87.45% | 28.79% | 8.95s | 774MiB |
| SAM3.1 | point-only，multiplex single-mask | 88.36% | 88.46% | 28.99% | 28.44s | 6404MiB |

严格提示对照中，SAM3.1的训练/heldout点召回提高1.86/1.01个百分点，但总时间和峰值进程显存分别为SAM2 matched的3.18倍和8.27倍。这里的召回只检查稀疏COLMAP点是否落入mask，不是dense mask GT IoU。

相对已部署SAM2 mask，SAM3.1全局像素IoU为0.850；三个实际主视图1/24/12的IoU为0.875/0.969/0.794。其heldout点召回为94.34%/89.06%/70.18%。

当前官方multiplex实现对每个active crop的首对象第一次调用都返回空mask；用完全相同points和`clear_old_points=True`重试一次后，23/23 active crops均成功。重试没有增加提示信息，并已计入时间。Python 3.10/PyTorch 2.6兼容预检与独立官方兼容环境生成的32张mask逐像素完全相同，因此下游重建无需重复。

## Latent一致性

| 条件 | Sparse AUC | Shape AUC | Texture AUC | Shape final | Texture final |
|---|---:|---:|---:|---:|---:|
| A current | 0.749 | 0.803 | 0.207 | 0.205 | 0.193 |
| B1 in-frustum anchors | 0.737 | 0.925 | 0.231 | 0.216 | 0.189 |
| B2 observed anchors | 0.739 | 0.787 | 0.195 | 0.220 | 0.184 |
| C object views | 0.725 | 0.979 | 0.217 | 0.209 | 0.189 |
| E SAM2 masks | 0.734 | 0.755 | 0.198 | 0.211 | 0.189 |
| E SAM3.1 masks | 0.728 | 0.796 | 0.235 | 0.214 | 0.181 |

E-SAM2相对A的sparse/shape/texture AUC改善约2.0%/6.0%/4.6%。E-SAM3.1只改善约2.8%/0.8%，texture AUC反而恶化约13.3%。相对E-SAM2，SAM3.1的sparse AUC再降低0.8%，但shape/texture AUC升高5.5%/18.9%。虽然SAM3.1的texture final更低，早期轨迹冲突更大，因此完整AUC仍明显退化。

更高的稀疏点mask召回没有转化成更一致的shape/texture预测；两种mask都远低于25%的强成功阈值。

## 几何

| 条件 | Heldout台面核心中位 | Heldout台面核心P90 | Heldout台面 <=10cm | Anchor-holdout中位/P90 | 平面拟合/水平P90 | 全局heldout中位 |
|---|---:|---:|---:|---:|---:|---:|
| A current | 1.66cm | 12.48cm | 84.27% | 1.40 / 5.32cm | 1.12 / 2.40mm | 6.59cm |
| B1 in-frustum anchors | 3.61cm | 14.76cm | 80.65% | 1.16 / 11.68cm | 3.18 / 7.54mm | 7.49cm |
| B2 observed anchors | 2.36cm | 12.87cm | 81.45% | 1.37 / 7.07cm | 5.30 / 11.06mm | 6.43cm |
| C object views | 1.64cm | 12.09cm | 84.27% | 0.72 / 6.57cm | 0.55 / 0.85mm | 7.49cm |
| E SAM2 masks | 2.31cm | 15.30cm | 79.03% | 0.85 / 5.75cm | 1.00 / 3.64mm | 6.64cm |
| E SAM3.1 masks | 1.66cm | 13.51cm | 81.85% | 0.86 / 6.53cm | 2.18 / 5.82mm | 6.88cm |

SAM3.1相对SAM2改善heldout台面核心：中位/P90降低28.3%/11.7%，10cm内比例提高2.82个百分点。Heldout柜体正面中位/P90也从0.84/1.48cm改善到0.83/1.27cm。

但相对A baseline，SAM3.1的台面核心中位基本相同，P90恶化8.3%，10cm内比例下降2.42个百分点；台面平面拟合P90从1.12mm升到2.18mm，全局heldout中位从6.59cm升到6.88cm。Anchor-holdout的中位改善也伴随P90从5.32cm升到6.53cm。因此它优于本轮E-SAM2，但仍不是相对baseline的可靠几何成功。

## 结论

1. SAM2实例mask可以温和降低三个阶段的cross-chunk predictor disagreement，说明排除无关图像context有真实作用。
2. SAM3.1在严格point-only对照中的稀疏点召回高于SAM2，但计算代价显著增加，而且shape/texture disagreement不如E-SAM2。
3. SAM3.1修复了E-SAM2的一部分台面核心和柜体误差，却同时恶化texture AUC、台面平面和全局heldout中位；这是误差重新分配，不是稳定净收益。
4. 更低的latent冲突或更高的mask点召回都没有单调转化为更好的heldout几何，instance awareness不是充分条件。
5. 真实track过滤使共享tokens的AUC方向一致，但改善很小且几何退化；未训练token不能作为最终设计。
6. 按稀疏覆盖选图显著扰乱shape生成，不能作为最终设计。
7. 自适应chunk的几何margin目标与生成质量不相关；当前布局暂时更好。
8. 首要瓶颈仍是visibility/depth ownership。SAM mask只告诉模型射线属于哪个实例，不告诉它该实例位于射线上的哪个深度。
9. 更合理的下一版应使用`SAM mask + 三角化track depth + free/surface/occluded gating`，之后再训练共享object/part token adapter。

因此本轮Goal的判定是：**SAM3.1提供了更高的稀疏点mask召回并优于本轮E-SAM2的部分几何，但instance-aware条件的完整几何成功仍未成立。**

## 限制

- 只有seed 42；小于几个百分点的AUC或几何变化尚未经过多seed确认。
- SAM3.1和已部署SAM2的提示协议不同；严格point-only、single-mask SAM2对照已补充，但下游E-SAM2仍代表原部署配置。
- 稀疏点召回不是dense mask GT IoU；ETH3D没有该工作台的实例mask真值。
- ETH3D是无序多基线相机集合，本实验没有使用SAM3.1视频传播；跨视角身份仍由COLMAP tracks负责。
- 当前官方`start_session`会向multiplex `init_state`传入不支持的`offload_state_to_cpu`；适配层只过滤该关键字并注册相同session状态，不改变模型前向。
- 官方multiplex首对象空输出需要相同提示重试；这可能是当前commit的接口/状态初始化问题。
- Mask仅作用于每个chunk的主`cond_2D`，没有作用于scene-wide `cond_3D`投影。
- Anchor-bank holdout只是不进入token/mask提示，并不代表其图像像素从scene-wide DINO条件中完全消失。
- ETH3D test split没有公开dense GT，不能报告绝对surface precision。

## 产物

- `outputs/eth3d/lecture_room/fallback_instance_experiment/a0_analysis/`
- `outputs/eth3d/lecture_room/fallback_instance_experiment/final_analysis/`
- `outputs/eth3d/lecture_room/fallback_instance_experiment/sam_mask_comparison/`
- `outputs/eth3d/lecture_room/fallback_instance_experiment/sam2_masks/`
- `outputs/eth3d/lecture_room/fallback_instance_experiment/sam2_masks_point_only_single/`
- `outputs/eth3d/lecture_room/fallback_instance_experiment/sam31_masks_official/`
- `outputs/eth3d/lecture_room/fallback_instance_experiment/{a0_current,b_shared_anchors,b2_observed_anchors,c_object_views,e_sam2_masks,e_sam31_masks}/`
- `tools/evaluate_instance_chunk_layouts.py`
- `tools/generate_sam2_instance_masks.py`
- `tools/generate_sam31_instance_masks.py`

## 后续：独立 Object-Centered Chunks

后续实验已实现并运行 `SAM3.1 mask + COLMAP track depth + free/surface/occluded gating + 独立object branch`，并完成11-chunk全场景object-centered生成。结果见：

- `reports/ETH3D_LECTURE_ROOM_INDEPENDENT_OBJECT_CHUNKS_EXPERIMENT_zh.md`
- `outputs/eth3d/lecture_room/fallback_instance_experiment/independent_object_chunks/final_visualization/index.html`

后续结果进一步确认：SAM-only独立branch退化，只有加入track depth ownership后heldout工作台几何才明显恢复；强AUC标准仍未达到。
