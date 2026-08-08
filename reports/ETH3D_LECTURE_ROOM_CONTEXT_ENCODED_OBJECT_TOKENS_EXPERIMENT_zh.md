# ETH3D `lecture_room` Context-Encoded Object Tokens 实验

## 目的

测试strict O3c退化是否主要来自“先将RGB变黑，再送入DINO”的分布偏移。O4只改变特征编码顺序：

1. Scene crops和三个主`cond_2D`以完整RGB进入DINO。
2. DINO编码后仍使用同一SAM 32×32 ownership mask清零非实例patch tokens。
3. Active view gate、canonical mask映射、track-depth、`allow_unknown`、hard support、chunk布局、seed42和world noise保持不变。

该条件准确名称是**context-encoded object/global tokens**。最终传递object patch tokens及每个active view原有的5个global/register tokens，这些保留token已经通过DINO self-attention吸收完整图像上下文，因此不是严格context-free实例特征。

为按字面验证“只保留object patch tokens”，另运行O4p：使用相同完整RGB编码，但同时清零全部scene 80个和cond2D 15个global/register tokens。

## 因果固定项

O3c与O4完全相同：

- 24/32 canonical mask matches。
- 16/32 active feature views。
- Scene patch保留`7,432/32,768`。
- cond2D patch保留`1,642/3,072`。
- Track-depth known patches为`4,761`。
- 相同ROI、support boxes、12cm surface band和`allow_unknown=true`。
- `chunk_transforms.json` SHA-256均为`a270555e7eda917d3c020131c280ebdcce22525249b99dd0dd02c5e3e15865a5`。

唯一预期条件差异是DINO看到的RGB上下文。

## 输出

- 推理时间：`151.2s`。
- 峰值进程显存：`4,128MiB`。
- 峰值进程树RAM：`16.84GiB`。
- Mesh：`1,319,078`顶点、`2,745,565`面。
- O3c mesh：`1,742,546`顶点、`3,634,179`面。
- O4全部顶点仍位于登记object support内，顶点有限且面索引合法。
- O4p：`142.7s`，峰值进程显存`4,128MiB`，`816,339`顶点、`1,670,021`面。

三个chunk的O3c/O4 sparse support IoU分别为`0.347/0.506/0.379`。O4/O4p IoU为`0.731/0.453/0.475`。编码上下文和95个global tokens均会大幅重写occupancy，而不是轻微调整纹理。

## Heldout结果

| 条件 | 台面中位/P90 | 台面coverage | 台面core中位/P90 | 柜面中位/P90 | 柜面coverage | 平面P90 |
|---|---:|---:|---:|---:|---:|---:|
| N2c contextual | 2.06/9.44cm | 98.47% | 1.38/3.69cm | 0.99/1.74cm | 100.00% | 1.39mm |
| O3c pre-mask | 4.64/8.12cm | 82.88% | 4.38/7.04cm | 1.75/3.56cm | 100.00% | 11.19mm |
| O4 context+global | 5.06/12.05cm | 97.25% | 4.20/7.43cm | 3.09/4.38cm | 98.77% | 10.63mm |
| O4p patches only | 11.61/38.38cm | 62.69% | 28.98/48.76cm | 17.74/30.18cm | 81.97% | 20.74mm |

O4相对O3c：

- 完整台面coverage提高`14.37pp`。
- 台面core coverage提高`12.43pp`。
- 台面core中位改善`4.1%`，但P90恶化`5.6%`。
- 完整台面中位/P90恶化`9.1%/48.5%`。
- 柜面中位/P90恶化`76.5%/23.3%`。
- Anchor holdout中位/P90恶化`62.8%/24.5%`。
- 平面P90只改善约`5.0%`，仍比N2c差约7.7倍。

因此coverage恢复并不是准确表面恢复。O4p则发生完整几何崩塌，说明当前预训练模型强依赖global/register tokens，不能在零训练条件下直接删除。

## Cross-chunk disagreement

| 条件 | Sparse AUC | Shape AUC | Texture AUC |
|---|---:|---:|---:|
| N2c contextual | 1.039 | 0.825 | 0.217 |
| O3c pre-mask | 0.869 | 0.840 | 0.265 |
| O4 context+global | 0.939 | 0.824 | 0.262 |
| O4p patches only | 0.847 | 0.748 | 0.273 |

O4相对O3c的shape/texture AUC改善`1.88%/1.25%`，sparse AUC恶化`8.12%`。O4p的shape AUC虽然下降到`0.748`，但台面core中位误差升至`28.98cm`、coverage降至`70.62%`。这是“chunks更一致地生成错误结构”的直接反例，不能把AUC下降解释为质量提高。

## O4来源审计

O4保持相同projection规则，但生成了不同occupancy：

| Chunk | O4 points | Zero-source | Eligible中unknown | 仅surface时zero-source |
|---|---:|---:|---:|---:|
| 0 | 1,018 | 14.34% | 79.76% | 77.90% |
| 1 | 2,074 | 32.79% | 58.17% | 73.48% |
| 2 | 1,535 | 27.04% | 85.56% | 87.62% |

O4在预期tabletop窄带内生成了更多occupancy，但这些点仍缺少可靠深度：

- Chunk 0 tabletop：214点，88.58%的eligible来源为unknown。
- Chunk 1 tabletop：303点，10.23% zero-source、68.08%来源为unknown。
- Chunk 2 tabletop：140点，61.43% zero-source。

这解释了“看起来更完整但误差更大”：上下文帮助生成先验填满support，却没有提供正确第一表面。

## 视觉检查

相对O3c，O4出现：

- 台面左中部更大范围的白色管状/层状伪面。
- 柜门大孔洞和错误first-hit surface。
- 柜面材质被拉成灰白色，真实木纹一致性下降。
- 台面右侧coverage增加，但高度和前缘仍不稳定。

稀疏COLMAP point coverage没有充分惩罚柜门大孔洞和错误附加表面，因此不能只依据`97.25%`台面coverage选择O4。

## 判定

O4/O4p验证了前一审计的判断：

> 先编码上下文再筛object/global tokens可以恢复coverage，但不能修复unknown射线、第一表面所有权或amodal几何；恢复的部分表面是更完整但更错误的生成式补全。进一步删除global tokens会让预训练模型直接失去结构稳定性。

O4和O4p均不替代O3c，也不替代N2c。当前最佳仍是N2c contextual。下一步应停止继续调RGB背景比例或token种类，优先构建稠密/正则化object surface envelope或dense depth first-hit；否则只是在“缺面”“错误补面”和“结构崩塌”之间移动。

## 产物

- O4 PLY：`outputs/eth3d/lecture_room/fallback_instance_experiment/independent_object_chunks/o4_context_encoded_object_tokens/mesh.ply`
- O4 fidelity：`o4_context_encoded_object_tokens/fidelity/`
- O4p PLY：`independent_object_chunks/o4p_context_encoded_patch_tokens/mesh.ply`
- 最终对照summary：`independent_object_chunks/context_encoded_patch_analysis/summary.json`
- 最终同视角HTML：`independent_object_chunks/context_encoded_patch_visualization/index.html`
- O4 source audit：`independent_object_chunks/o4_feature_flow/index.html`
- Feature-flow前置审计：`reports/ETH3D_LECTURE_ROOM_STRICT_OBJECT_FEATURE_FLOW_AUDIT_zh.md`

O4 PLY SHA-256：`3155bf91de085e4be63f98e4e4a6165fb63c72bf7234b15b8120d2ad10bb9a63`。O4p PLY SHA-256：`3a52579b70c4dc81c3337ad20d0d2fe94734cd278a549f5b4282b82f5d108a01`。
