# ETH3D `lecture_room` Cross-Chunk Instance Context 代理消融

## 问题

本实验检查：工作台跨越多个 chunks，而采样过程没有共享实例身份时，各 chunk 是否会在重叠区产生不同的结构/纹理预测。

工作台主体依次穿过 `chunk 003/004/005`。当前 MultiDiffusion 在每个 Euler step 中先让各 chunk 独立执行 model forward，再平均重叠 latent。最终 mesh 连续并不能证明各 chunk 原始预测一致，因为冲突可能已被平均隐藏。

## 实验设计

### A. 原始条件诊断

- 与最终 `high_precision` 相同的数据、seed 42、512配置、11 chunks和阈值。
- 在每个 Euler step 更新后、执行 overlap aggregation 前，匹配相邻 chunks 的同一世界体素。
- 记录 state RMSE、相对 RMSE、cosine 和由 step size 还原的 velocity disagreement RMSE。
- 诊断只读取 tensor，不改变聚合结果。

### B. 共享 `cond_2D` 代理

- 仅将 `chunk 003/004/005` 的主 `cond_2D` 统一为 scene crop `#13 DSC_0925 right`。
- scene-wide 32个 crops、`cond_3D`、相机、seed、chunk布局和聚合算法均不变。
- 这是“共享实例 context”的粗代理，不是真正训练过的 instance token。
- 选择 `#13` 是因为它在32个候选 crops 中对三个 chunks 的最小整体覆盖率最高。

该 crop 对完整占据的投影覆盖率为 `34.5%/80.9%/40.9%`，对台面 ROI 的覆盖率为 `28.0%/91.2%/44.0%`。因此即使图像相同，它给三个 chunks 的几何证据仍高度不对称。

## 原始条件下的重叠冲突

最后一个 Euler step 的 velocity disagreement RMSE：

| 阶段 | `003-004` | 16对中的排名 | `004-005` | 16对中的排名 |
|---|---:|---:|---:|---:|
| sparse structure | 1.079 | 1 | 0.899 | 3 |
| shape SLat | 0.856 | 2 | 0.234 | 约14 |
| texture SLat | 0.722 | 1 | 0.208 | 约13 |

`003-004` 在三个阶段都属于冲突最高的重叠边界；`004-005` 只在早期 occupancy 阶段较高，后续 shape/texture 已较一致。

这支持“局部 cross-chunk 条件不一致是真实存在的”，但不支持“整张工作台所有边界都同样失败”。

## 共享 `cond_2D` 的影响

| 阶段 | 边界 | 原始 final RMSE | 共享图 final RMSE | 变化 |
|---|---|---:|---:|---:|
| sparse structure | `003-004` | 1.079 | 1.126 | +4.4% |
| shape SLat | `003-004` | 0.856 | 3.217 | +275.8% |
| texture SLat | `003-004` | 0.722 | 2.384 | +230.3% |
| sparse structure | `004-005` | 0.899 | 0.911 | +1.3% |
| shape SLat | `004-005` | 0.234 | 0.331 | +41.3% |
| texture SLat | `004-005` | 0.208 | 0.297 | +42.8% |

作为远端对照，未直接修改的 `006-010` 在 final shape/texture 中仅变化约 `+7.0%/+1.5%`。`003-004` 的数倍增幅远大于该运行噪声尺度。

共享 raw image 没有产生共享实例语义，反而使 `003-004` 成为 shape/texture 中显著最差的边界。原因是同一 crop 在 chunk 004 中提供强证据，但对 chunk 003 和005只覆盖少量台面；相同 token 不等于相同、正确、视角不变的对象表示。

## 几何结果

所有数值均来自23个原始 COLMAP 相机的 PLY z-buffer；同一个 SfM 点可按真实观测在多个视角出现。共享版 fidelity 页面中的 GLB 列是固定的原始 baseline GLB，仅用于并排参考，本节比较只使用两组 PLY 指标。

| 指标 | 原始条件 | 共享 `cond_2D` | 变化 |
|---|---:|---:|---:|
| 全视角 SfM 深度跨视角中位数 | 5.22cm | 6.26cm | 变差约19.8% |
| 全视角 SfM `<=10cm` | 66.78% | 66.66% | 基本不变 |
| 台面观测中位深度误差 | 2.34cm | 2.27cm | 略改善 |
| 台面观测 P90 | 12.02cm | 12.83cm | 变差约6.7% |
| 台面 `<=10cm` | 86.55% | 83.93% | -2.62pp |
| 台面核心中位深度误差 | 1.12cm | 1.58cm | 变差约41.2% |
| 台面核心 `<=10cm` | 84.38% | 82.29% | -2.09pp |
| 柜体正面中位深度误差 | 1.31cm | 0.61cm | 改善约52.9% |
| 柜体正面 P90 | 1.77cm | 1.07cm | 改善约40.0% |

共享的近正面 crop 明显改善柜体正面，却没有一致改善掠射台面。它重新分配了误差，而不是形成一个全局一致的工作台实例。

`DSC_0903` 可见台面平面诊断：

| 指标 | 原始条件 | 共享 `cond_2D` |
|---|---:|---:|
| 拟合平面残差 P90 | 1.12mm | 1.64mm |
| 相对水平中位高度残差 P90 | 2.40mm | 4.09mm |
| 拟合倾角 | 0.246度 | 0.190度 |
| 台面中位高度 | -0.8763m | -0.8798m |

共享版倾角略小，但局部平整度变差，台面整体下移约3.5mm。

## 结论

1. `003-004` 确实存在持续、较强的跨 chunk 预测冲突；当前 MultiDiffusion 每一步都在主动消除它，最终没有裂缝不代表生成假设一致。
2. 冲突是局部的，`004-005` 后期 shape/texture 较稳定，不能把全部台面问题归因于 lack of instance awareness。
3. 简单共享一张 raw `cond_2D` 不是 instance-aware 建模，并会因视角覆盖不对称显著放大冲突。
4. 真正需要的是视角不变的 object token、跨视角关联后的实例 mask，或共享台面平面/厚度/前缘参数，而不是让所有 chunks 看同一张图。
5. 可见性和深度归属仍是首要问题；instance-aware context 更像是在观测歧义出现后，阻止各 chunks 独立选择不同补全模式的第二层约束。

因此强化后的结论是：

> 非 instance-aware cross-chunk sampling 在工作台的 `003-004` 边界上确实产生了可测量的局部冲突，但它不是唯一或全局主因。共享 raw image context 不能解决该问题；需要与 visibility-aware lifting 配合的对象级、视角不变表示。

## 产物

- `outputs/eth3d/lecture_room/cross_chunk_ablation/overlap_ablation.png`
- `outputs/eth3d/lecture_room/cross_chunk_ablation/records.csv`
- `outputs/eth3d/lecture_room/cross_chunk_ablation/pair_summary.csv`
- `outputs/eth3d/lecture_room/cross_chunk_ablation/summary.json`
- `outputs/eth3d/lecture_room/overlap_diagnostics/overlap_diagnostics.json`
- `outputs/eth3d/lecture_room/overlap_shared_cond2d/overlap_diagnostics.json`
- `outputs/eth3d/lecture_room/overlap_shared_cond2d/fidelity/index.html`

## 限制

- 共享 `cond_2D` 是诊断代理，不是训练过的 instance-aware 模型，因此负结果不能证明 object token 无效。
- 两次运行的 shape decoder mesh 面数有轻微差异；FlashAttention和稀疏GPU kernel不是字节级确定。主要结论采用同一次运行内的配对 overlap 排名，并用远端邻接对作对照。
- ETH3D test包没有公开 dense ground truth，本实验使用原始相机、稀疏 SfM 和自一致性指标，不是绝对表面精度评测。
