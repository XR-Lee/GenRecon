# ETH3D `lecture_room` Strict Object-Centric 实验

## Goal

验证真正的single-instance object branch：chunk内只生成工作台/柜体，图像条件只使用SAM3.1判定为同一实例的feature，房间背景、黑板、地面和座椅不参与object feature，也不能出现在raw object mesh中。

正式 `outputs/eth3d/lecture_room/final` 未修改。

## Camera-mask索引修正

SAM masks以`fallback_instance_experiment/a0_current/cameras.json`的32个crops为canonical顺序。局部三chunk selector重新选择了物理相机，只有24/32 crops能映射到canonical集合。旧局部N1/N2/N2b错误地按数组下标套用mask；其中旧local view 12实际为`DSC_0916`，canonical mask 12来自`DSC_0925`。

因此旧局部masked N1/N2/N2b及其融合结果不再用于因果结论。完整11-chunk `full_scene_obj_center_n2b`的32/32 camera keys与canonical完全一致，不受该问题影响。

修正后，mask按图像basename和左右crop内参匹配。无法匹配的视角被禁用，不再静默套用其他图像的mask。三chunk主视图改为local 1/24/10，分别映射canonical 1/24/12。

## Strict实现

O3c同时执行：

1. 仅启用至少有3个训练tracks且能匹配canonical mask的scene crops，共16/32。
2. Scene RGB和主`cond_2D`的mask外像素全部清零。
3. DINO token层再次清零所有非实例patch token；只保留7,432/32,768个scene patch tokens和1,642/3,072个主视图patch tokens。
4. `cond_3D`在整个chunk范围只接受SAM实例patch；不再在ROI外恢复scene feature。
5. 使用COLMAP训练tracks区分surface/free/occluded；heldout point IDs不参与。
6. 台面和柜体support boxes以外的sparse occupancy强制删除。
7. Decode后再次删除越过support的三角面并压缩顶点。

O3c删除10,792个已生成但不具备object support所有权的occupancy cells。Decode mesh从1,909,036顶点/3,971,630面裁为1,742,546顶点/3,634,179面。最终所有顶点均位于object support内，世界坐标范围约为`4.26 × 1.25 × 1.06m`，raw PLY不再包含地面、黑板或座椅。

## 有效对照

- `N0 raw`：相同三chunk布局，无SAM。
- `N2c contextual`：canonical mask/depth映射正确，mask外保留15%图像上下文，ROI外仍使用scene features。
- `O3c strict object`：只使用同实例features，并执行hard object support。
- `Full contextual valid`：此前完整11-chunk object-centered结果，canonical 32/32匹配。
- `O1 big seeded`：单个4.5m object chunk；使用相同strict features并以训练track surface cells提供正occupancy种子。

## Heldout结果

| 条件 | 台面中位/P90 | 台面coverage | 柜面中位/P90 | 柜面coverage | 平面P90 |
|---|---:|---:|---:|---:|---:|
| A baseline | 2.36 / 11.33cm | 98.91% | 1.35 / 1.77cm | 100.00% | 5.94mm |
| N0 raw | 4.55 / 14.68cm | 100.00% | 3.04 / 4.51cm | 100.00% | 6.23mm |
| N2c contextual | 2.06 / 9.44cm | 98.47% | 0.99 / 1.74cm | 100.00% | 1.39mm |
| O3c strict object | 4.64 / 8.12cm | 82.88% | 1.75 / 3.56cm | 100.00% | 11.19mm |
| Full contextual valid | 1.59 / 7.33cm | 98.48% | 0.42 / 1.10cm | 98.55% | 1.04mm |
| O1 big seeded | 8.82 / 10.94cm | 9.35% | 3.45 / 8.81cm | 31.88% | 0.65mm* |

这里的平面P90统一使用`DSC_0903`渲染中工作台ROI的robust plane fit，与早期fidelity报告中的其他平面采样口径不可直接混用。`O1 big seeded`的平面P90只基于极低coverage命中，不能解释为平面质量优于其他条件。

O3c相对N2c：

- 台面P90从9.44cm降到8.12cm。
- 台面中位从2.06cm恶化到4.64cm。
- 台面coverage从98.47%降到82.88%。
- 柜面中位/P90从0.99/1.74cm恶化到1.75/3.56cm。
- 平面P90从1.39mm恶化到11.19mm。

因此严格feature隔离减少了一部分长尾错误，但显著损害典型位置、平面一致性、纹理和完整性。

## Cross-chunk disagreement

| 条件 | Sparse AUC | Shape AUC | Texture AUC |
|---|---:|---:|---:|
| N0 raw | 1.219 | 0.909 | 0.244 |
| N2c contextual | 1.039 | 0.825 | 0.217 |
| O3c strict object | 0.869 | 0.840 | 0.265 |

O3c相对N2c的sparse AUC下降约16.4%，但shape AUC恶化约1.8%，texture AUC恶化约22.0%。它让粗occupancy更一致，却没有得到更准确、更完整的表面。

## 4.5m单体chunk

单体4.5m strict run首先在不注入正表面cells时失败：模型生成的3,660个occupied cells全部位于object support外，删除后内部occupancy为零。该失败不是OOM，峰值进程显存约4.1GiB。

加入116个COLMAP训练track surface cells后能够解码，但只得到7,542顶点、13,314面：

- 台面coverage 9.35%。
- 柜面coverage 31.88%。
- Anchor-holdout coverage 20.97%。

这证明当前预训练模型不能仅靠把4.25m柜体缩放进一个4.5m chunk和SAM feature定位出完整对象。低物理分辨率与分布外尺度是实质问题。

## 运行资源

| Run | 时间 | 峰值进程显存 | 峰值进程树RAM | 状态 |
|---|---:|---:|---:|---|
| N2c contextual | 214.7s | 8,396MiB | 17.33GiB | 成功 |
| O3c strict object | 140.5s | 4,128MiB | 17.92GiB | 成功 |
| O1 4.5m无seed | 88.2s | 4,112MiB | 16.15GiB | 内部occupancy归零，按预期失败 |
| O1b 4.5m seeded | 152.1s | 4,114MiB | 18.05GiB | 成功但coverage不足 |

## 视觉判断

O3c raw mesh确实只包含柜体support，但存在：

- 台面遮挡孔上方的白色伪结构。
- 台面断裂和高度起伏。
- 柜门纹理发白、细节变软。
- 边缘孔洞和悬浮小片。
- 从部分侧后视角出现较大缺失。

根因是SAM提供modal可见表面，不提供遮挡后的amodal柜体；同时预训练GenRecon主要接收带场景上下文的图像，黑底孤立实例和大面积DINO token清零属于分布外输入。

后续feature-flow审计进一步确认：保留前景DINO特征的原图/strict输入cosine中位仍为`0.980`，不是前景语义整体崩坏；主要问题是chunks 0/1/2分别有`9.4%/33.3%/23.6%`输出occupancy没有任何strict `cond_3D`来源，而eligible来源中`84.8%/61.5%/88.8%`没有track depth。若拒绝全部unknown，则`84.0%/76.3%/89.9%`输出点会失去全部3D来源。详见`reports/ETH3D_LECTURE_ROOM_STRICT_OBJECT_FEATURE_FLOW_AUDIT_zh.md`。

## 判定

本实验完成了真正的object-only推理路径和raw mesh纯度验证，但质量不通过：

> 同实例feature隔离能消除背景几何并改善粗sparse一致性，但当前预训练GenRecon需要场景上下文才能保持台面平面、纹理和完整表面。SAM身份信息不能单独替代amodal几何和训练过的object prior。

当前最佳结果仍是canonical映射正确的contextual N2c或完整11-chunk contextual结果，而不是strict O3c或4.5m O1。

后续O4将完整RGB先送入DINO、再使用相同object patch/global token gate。它恢复台面coverage但将台面P90恶化到12.05cm、柜面中位恶化到3.09cm；进一步清零global/register tokens的O4p使台面core中位/P90恶化到28.98/48.76cm。两者均不采用。详见`reports/ETH3D_LECTURE_ROOM_CONTEXT_ENCODED_OBJECT_TOKENS_EXPERIMENT_zh.md`。

## 产物

- Strict O3c PLY：`outputs/eth3d/lecture_room/fallback_instance_experiment/independent_object_chunks/o3c_strict_sam31_canonical/mesh.ply`
- Corrected contextual N2c：`outputs/eth3d/lecture_room/fallback_instance_experiment/independent_object_chunks/n2c_contextual_canonical/mesh.ply`
- 4.5m单体结果：`outputs/eth3d/lecture_room/fallback_instance_experiment/independent_object_chunks/o1b_big_strict_surface_seed/mesh.ply`
- 可视化：`outputs/eth3d/lecture_room/fallback_instance_experiment/independent_object_chunks/strict_object_visualization/index.html`
- 有效summary：`outputs/eth3d/lecture_room/fallback_instance_experiment/independent_object_chunks/strict_object_canonical_analysis/summary.json`
- 索引修正说明：`outputs/eth3d/lecture_room/fallback_instance_experiment/independent_object_chunks/MASK_INDEX_CORRECTION.md`

正式结果保持不变：`final/mesh.ply` SHA-256为`b24fe5628305e120b945dd72503620ab6984016b31a70a1d26f3b8aba4afcd83`，`final/scene.glb`为`6fd4000afd4287ea947d53de6a4dda92ae483136693946efb78b19df67db4a7d`。
