# ETH3D `lecture_room` Strict Object Feature-Flow 审计

## 目的

解释O3c strict object-only为何在背景排他成功后，台面平面、coverage和纹理明显退化。审计对象包括：

- 32个local scene crops及其canonical SAM映射。
- SAM3.1像素mask、part masks和模型实际消费的32×32 DINO ownership mask。
- 原图、15% contextual输入和strict黑底输入的DINO特征变化。
- 三个chunk在输出occupancy坐标上的精确`cond_3D`候选来源。
- Track-depth surface与`allow_unknown=true`放行来源的比例。

本审计没有把候选来源解释为learned attribution。原推理未保存aggregator softmax权重，无法恢复每个来源的实际贡献大小。

## 复现校验

工具按原iPhone selector规则从ETH3D原图重新执行左右square crop、512/1024 LANCZOS resize、SAM bilinear mask和黑底乘法。重建的32张1024 strict输入与O3c保存的`scene/view_NNN.png`逐像素一致：

- 差异通道值：`0`。
- 最大绝对差：`0`。
- 最大平均绝对差：`0`。

DINO使用原推理相同的本地`DINOv3 ViT-L/16`权重。诊断tensor以FP16持久化，PCA只用于可视化。

## View gate

32个local crops中：

- 24个能按图像basename和crop intrinsics匹配canonical SAM mask。
- 16个满足至少3个训练tracks并进入object feature encoder。
- 8个local crops没有canonical匹配。
- 8个有canonical位置但提示tracks不足或mask为空，被禁用。

全部scene patch位置为32,768个，最终保留7,432个，即所有local crops的`22.68%`。只在16个active views内计算时，保留率为`45.36%`。Feature-view gate本身对当前输出点的candidate pair只额外删除约`0.1–1.0pp`，因此它不是主要损失来源；主要损失发生在SAM范围和depth gating。

## SAM到DINO patch

| Chunk | cond2D local/canonical | SAM像素覆盖 | 保留DINO patches | 混合边界patches | 无track-depth patches |
|---|---:|---:|---:|---:|---:|
| 0 | 1 / 1 | 51.56% | 560/1024 | 146，26.1% | 252，45.0% |
| 1 | 24 / 24 | 50.95% | 547/1024 | 159，29.1% | 145，26.5% |
| 2 | 10 / 12 | 48.25% | 535/1024 | 207，38.7% | 274，51.2% |

混合边界patch同时覆盖前景和黑底/遮挡孔。32×32 token gate因此不能保持像素级SAM边界，chunk 2受影响最明显。

## DINO变化

所有active scene views中，原图与strict黑底输入在保留前景patch上的DINO cosine：

- 均值：`0.9714`。
- 中位：`0.9800`。
- P10：`0.9394`。

三个主cond2D从15% contextual输入改为strict黑底后的前景特征：

| Chunk | Cosine中位 | Relative-L2中位 |
|---|---:|---:|
| 0 | 0.9956 | 0.0938 |
| 1 | 0.9923 | 0.1243 |
| 2 | 0.9906 | 0.1374 |

因此不是“DINO前景语义整体失效”。前景向量仍较接近，但chunk 2变化最大；同时54.6%的active-view patch tokens以及全部inactive-view tokens被置零，这仍是预训练模型未见过的输入分布。

## cond3D来源漏斗

以下统计使用O3c保存的shape occupancy坐标，并精确复现positive-depth、图像范围、SAM patch、active gate、track-depth surface band和unknown规则。

| Chunk | 输出points | SAM/in-frustum | Strict eligible/in-frustum | 每点eligible中位 | Zero-source points | Eligible中的unknown |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1,449 | 68.72% | 41.97% | 4 | 9.39% | 84.79% |
| 1 | 2,238 | 61.03% | 24.03% | 2 | 33.33% | 61.48% |
| 2 | 1,597 | 49.80% | 32.80% | 2 | 23.61% | 88.78% |

这说明strict分支生成的不是一个被可靠多视图深度约束的object field：

- Chunk 1有三分之一输出occupancy没有任何`cond_3D`来源。
- Chunks 0/2的eligible候选约85%/89%没有track depth，只因`allow_unknown=true`而沿整条SAM射线放行。
- 若改为只接受track-depth surface，三个chunk分别有83.99%、76.27%、89.92%的输出点失去全部`cond_3D`来源。

所以直接关闭unknown会导致更严重的coverage崩塌；保留unknown则继续存在射线深度歧义。

## Tabletop与cabinet

严格ROI是窄评测带，`support_shell`是hard support内但不在这两个窄带中的区域，不能全部解释为背景。

- Chunk 0：预期tabletop高度带内没有sparse occupancy中心；70.67%的点位于support shell。已有柜体点的65.72% eligible来源为unknown。
- Chunk 1：tabletop有84点，9.52% zero-source、84.07%来源为unknown；cabinet有781点，17.67% zero-source。
- Chunk 2：tabletop仅78点，其中70.51% zero-source；cabinet有727点，但86.84% eligible来源为unknown。
- Support shell在chunks 0/1/2中分别有96.91%/85.10%/93.36%的eligible来源为unknown。

这与视觉失败一致：台面高度偏移、右侧台面缺失、support边缘悬浮片和遮挡孔附近伪结构主要位于深度无约束或无3D来源区域。

## 判定

质量下降的主因不是SAM主体轮廓完全错误，也不是保留的DINO前景向量整体崩坏。更直接的原因是：

1. Strict token清零移除了模型训练时依赖的大量上下文。
2. SAM仍是modal silhouette，不提供遮挡后柜体的amodal几何。
3. Track depth覆盖不足；unknown候选被沿射线复制，仍然没有第一表面所有权。
4. 拒绝unknown又会让76%–90%的输出点没有任何3D来源。
5. Hard support box只限制生成范围，不提供台面高度、厚度或统一平面。
6. 32×32 DINO patch在SAM边界存在26%–39%的混合tokens。

当前结果支持继续使用N2c contextual，而不是当前O3c strict。下一步不应只调SAM阈值或直接关闭unknown；需要先构建稠密/正则化的amodal object surface envelope，或由dense MVS/depth提供第一表面。

## 后续O4验证

“完整RGB先编码DINO、之后只保留object/global tokens”的O4已实跑。它将台面coverage从82.88%恢复到97.25%，但完整台面P90从8.12cm恶化到12.05cm，柜面中位从1.75cm恶化到3.09cm，并产生更大的台面伪面和柜门孔洞。进一步清零全部global/register tokens的O4p发生结构崩塌：台面core中位/P90为28.98/48.76cm。这确认上下文编码只能在缺面与错误补面之间重新分配，而零训练删除global tokens也不可行；两者都不能替代第一表面约束。详见`reports/ETH3D_LECTURE_ROOM_CONTEXT_ENCODED_OBJECT_TOKENS_EXPERIMENT_zh.md`。

## 产物

- HTML：`outputs/eth3d/lecture_room/fallback_instance_experiment/independent_object_chunks/strict_object_feature_flow/index.html`
- 总览：`strict_object_feature_flow/overview.png`
- Tabletop/cabinet拆分：`strict_object_feature_flow/region_sources.png`
- 32视角gate：`strict_object_feature_flow/all_scene_view_gate.jpg`
- Active-view DINO：`strict_object_feature_flow/active_scene_views.jpg`
- 每chunk条件流：`strict_object_feature_flow/chunk_NNN/condition_flow.jpg`
- 每chunk来源图：`strict_object_feature_flow/chunk_NNN/source_eligibility.png`
- Summary：`strict_object_feature_flow/summary.json`
- FP16 DINO dump：`strict_object_feature_flow/dino_feature_diagnostics_fp16.pt`
