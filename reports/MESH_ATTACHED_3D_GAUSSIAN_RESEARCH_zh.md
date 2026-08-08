# 在GenRecon Mesh上携带3D Gaussian基元的可行性调研

完整H20实施规格见：`reports/GENRECON_H20_FULL_FACE_GAUSSIAN_IMPLEMENTATION_GUIDE_zh.md`。

## 结论

可行，而且已有多条公开研究路线直接覆盖这个设想。最接近的工作是：

- **SuGaR（CVPR 2024）**：把Gaussians绑定到mesh表面，并通过Gaussian rasterization联合优化mesh与Gaussians。
- **GaMeS（2024）**：每个Gaussian由所属三角面的三个顶点参数化，支持输入已有mesh。
- **Gaussian Frosting（ECCV 2024 Oral）**：在mesh内外建立可变厚度Gaussian壳层，用于表面细节、毛发和半体积效果。
- **2D Gaussian Splatting（SIGGRAPH 2024）**：把Gaussian压成有方向的二维盘面，以获得更一致的表面几何；它不是显式mesh绑定，但适合作为几何一致性对照。

对当前GenRecon工程，可以把最终目标明确设为：

> 正式PLY的每个有效三角面都携带至少一个稳定绑定的叶级surface Gaussian，即完整保留26,854,772个face-to-Gaussian绑定；正式mesh顶点和面不变。实时渲染可以选择LOD父节点，但资产中不删除叶级基元。

实测该mesh没有零面积面，三角形最大边中位仅`5.884mm`、P90为`6.151mm`，总表面积约`212.94m²`。因此一面一个Gaussian不是在粗三角形上稀疏采样，而是在一张面尺度较均匀的高密度表面上建立叶级辐射基元，表示上是合理的。

ROI或简化proxy仍有价值，但只是验证训练损失、坐标、可见性和渲染器的阶段性手段，不是最终规模上限。该Gaussian层应被定义为**外观/新视角渲染资产**，不能替代权威几何。它很可能改善纹理接缝、抗锯齿、细线、小物体轮廓和视角相关高光，但不能自动修复GenRecon已经错误的第一表面、厚度、孔洞或双层几何。

## 1. “Mesh携带Gaussian”的准确含义

对三角面顶点 $v_0,v_1,v_2$，一个绑定Gaussian可以写为：

$$
\mu = b_0v_0+b_1v_1+b_2v_2+d\,n,
\qquad b_i\geq 0,
\qquad \sum_i b_i=1,
$$

其中：

- $\mu$ 是Gaussian中心。
- $b_i$ 是面内重心坐标。
- $n$ 是三角面法向。
- $d$ 是可选法向偏移；严格surface-only时固定为零。
- Gaussian旋转由三角面的切向坐标系 $[t_1,t_2,n]$ 给出。
- 切向尺度控制覆盖范围，法向尺度保持很小。
- 透明度控制可见贡献。
- SH系数存储随观察方向变化的颜色。

如果mesh发生编辑，重心坐标、局部旋转和尺度会随三角面同步变化。这正是SuGaR和GaMeS所实现的核心关系。

仅把mesh采样成一批椭球并复制顶点颜色，只是换一种光栅化方式。它可能减少锯齿或产生平滑效果，但没有从多视角图像学习新的辐射信息，因此不能被当成真正的渲染质量提升。

## 2. 相关工作

### 2.1 SuGaR

论文：<https://arxiv.org/abs/2311.12775>
代码：<https://github.com/Anttwo/SuGaR>

完整流程为：

1. 先训练7,000 iterations的普通3DGS。
2. 增加surface-alignment正则。
3. 从对齐后的Gaussians提取Poisson mesh。
4. 把新Gaussians绑定到mesh表面。
5. 通过Gaussian splatting联合细化mesh和Gaussians。

SuGaR是“mesh作为控制结构，Gaussian作为高质量渲染表示”的直接证据。论文在Mip-NeRF360上的mesh rendering消融如下：

| Mesh/渲染方式 | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| 1M vertices + bound Gaussians | 24.51 | 0.768 | 0.295 |
| 1M vertices + optimized UV | 21.24 | 0.609 | 0.478 |
| 200K vertices + bound Gaussians | 24.24 | 0.757 | 0.300 |
| 200K vertices + optimized UV | 21.44 | 0.656 | 0.419 |

该对照还刻意只使用Gaussian的diffuse SH分量，因此收益不完全来自高阶视角相关颜色。它证明面上Gaussians可以成为比普通UV更有效的图像拟合表示。

限制：

- 论文默认先从3DGS得到自己的mesh，不是直接接受任意GenRecon mesh。
- 论文实验使用32GB V100。
- 其官方实现继承原始3DGS的非商业研究许可证。
- SuGaR通常允许mesh联合细化；若要保持GenRecon几何，需要冻结顶点或施加强约束。

### 2.2 GaMeS

论文：<https://arxiv.org/abs/2402.01459>
代码：<https://github.com/waczjoan/gaussian-mesh-splatting>

GaMeS与本项目的接口最接近：

- 可以输入已有三角mesh。
- 每个Gaussian的中心、旋转和尺度都由所属三角面参数化。
- 可以固定mesh，也可以优化mesh顶点。
- 官方代码的mesh顶点学习率默认是零。
- 默认每面2个Gaussians，README说明论文常用每面5或10个。
- COLMAP场景通过其`gs_multi_mesh`分支读取，mesh放在`sparse/0/*.obj`。

Gaussian数量是 $kN_f$，其中 $N_f$ 为面数。当前正式lecture_room PLY有26,854,772个面。每面一个Gaussian作为最终表示是可行的，但不能直接套GaMeS的单进程、全量常驻、普通Adam训练方式；官方推荐的每面5至10个则会达到约134M至269M，超出本项目必要范围。全量目标需要一面一个基础叶节点、分页优化、可见性裁剪和LOD渲染。

GaMeS官方代码同样继承原始3DGS的非商业研究许可证，并使用较旧的Python/PyTorch/CUDA 11.8环境。它适合验证论文路径，但不适合直接嵌入当前GenRecon环境。

### 2.3 Gaussian Frosting

论文：<https://arxiv.org/abs/2403.14554>
代码：<https://github.com/Anttwo/Frosting>

Frosting不把所有Gaussians严格压在表面，而是在mesh内外构建可变厚度壳层：

- 平面、墙面和桌面使用薄壳。
- 毛发、草、细线和复杂边界使用厚壳。
- mesh可用于Gaussian occlusion culling。
- 编辑mesh时，壳层中的Gaussians随之变形。

论文中真实场景通常使用约5M Gaussians，完整优化在32GB V100上约45至90分钟。该路线渲染质量高，但对当前A4000 16GB和lecture_room并不是首选：

- 工作台主要问题是第一表面和深度所有权，不是毛发类体积效果。
- 壳层允许法向厚度，可能掩盖或扩大现有错误几何。
- 5M规模和官方实现的内存要求偏高。

Frosting适合第二阶段，仅用于线缆、水龙头边缘、薄物体或反射边界等明确需要离面自由度的区域。

### 2.4 2D Gaussian Splatting

论文：<https://arxiv.org/abs/2403.17888>
代码：<https://github.com/hbb1/2d-gaussian-splatting>

2DGS使用有方向的平面Gaussian盘，并加入depth-distortion与normal-consistency约束。它不以现有mesh为控制笼，但比自由3D椭球更接近surface rendering，可作为重要对照：

- 如果2DGS明显优于mesh-bound Gaussians，说明GenRecon mesh本身是主要限制。
- 如果mesh-bound方法相近或更好，说明已有mesh提供了有效几何先验。

### 2.5 Mesh2Gaussian一类直接转换器

示例：<https://github.com/hwanhuh/mesh2gaussian>

这类工具从mesh表面采样各向异性Gaussians，可以快速验证格式、坐标、viewer和渲染吞吐，但不使用原始照片进行优化。它适合作为pipeline smoke test，不适合作为质量方案。

## 3. 对当前GenRecon场景可能改善什么

### 高概率改善

- UV图集接缝和纹理烘焙模糊。
- 三角边缘锯齿和远距离亚像素细节。
- 木材、金属和反光台面上的视角相关外观。
- 线缆、桌沿、小物体和薄结构的视觉连续性。
- 因GLB简化或纹理分块造成的局部外观损失。
- 新视角RGB的PSNR、SSIM和LPIPS。

### 不会自动改善

- 错误first-hit surface。
- 台面高度、厚度和平面统一性。
- 柜体被遮挡部分的真实amodal几何。
- 双层面、错误孔洞或缺失表面。
- COLMAP稀疏点到mesh的几何距离。
- 可用于测量的精确深度。

Gaussian具有透明度和方向相关颜色，能在图像上遮盖错误mesh。渲染“看起来更好”不代表几何更正确，因此必须并行保留mesh深度和Gaussian深度评估。

## 4. 与PBR mesh的关系

3DGS的SH颜色是一种训练视角条件下的baked radiance，不是标准PBR材质：

- 优点：可拟合高光、曝光差异和难以烘焙的细节。
- 缺点：通常不能在新灯光下正确重照明。

因此建议保留两种模式：

1. **PBR mode**：继续使用当前GLB，支持普通引擎、重照明和几何查询。
2. **Radiance mode**：使用mesh-bound Gaussian renderer，追求输入照明下的新视角照片质量。

不要默认同时把不透明PBR面和不透明Gaussians叠加，否则会出现双重着色、错误alpha和深度排序。第一版应让Gaussian层负责颜色，mesh只提供绑定、深度、遮挡和选择；或者使用明确设计的residual compositor。

## 5. 存储与GLB可行性

Khronos glTF仓库当前包含`KHR_gaussian_splatting`，状态标记为**Release Candidate**：

<https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_gaussian_splatting>

它在glTF point primitive上定义：

- `POSITION`
- `KHR_gaussian_splatting:ROTATION`
- `KHR_gaussian_splatting:SCALE`
- `KHR_gaussian_splatting:OPACITY`
- 0至3阶SH系数

一个GLB可以同时包含：

- 普通triangle primitive形式的PBR mesh。
- `KHR_gaussian_splatting` point primitive形式的Gaussian field。

但该扩展没有定义“某Gaussian绑定到哪个三角面”的编辑关系。静态渲染时可以直接烘焙Gaussian世界位置；若要保持mesh编辑能力，还需额外存储`face_id + barycentric + normal_offset + local_frame`，通常放入sidecar或自定义扩展。

当前建议资产布局：

```text
scene_hybrid/
  scene.glb                 # 原PBR mesh，保持权威几何
  appearance.spz            # 压缩Gaussian渲染资产
  appearance.ply            # 可选、便于调试的原始3DGS PLY
  attachment.npz            # face id、重心坐标、局部偏移
  manifest.json             # 坐标系、mesh hash、训练split和版本
```

Niantic SPZ实现宣称相对普通Gaussian PLY约缩小10倍：

<https://github.com/nianticlabs/spz>

单GLB的`KHR_gaussian_splatting`版本可以作为实验导出，但当前普通GLB viewer不会普遍支持该Release Candidate扩展。兼容性优先时，保留`GLB + SPZ`双文件更稳妥。

坐标转换是高风险点。Gaussian的position、quaternion、anisotropic scale和SH方向都必须使用与GLB相同的世界变换；旋转坐标系时，高阶SH还需要对应的Wigner-D旋转，不能只改XYZ。

## 6. 推荐实现路线

### 6.1 第一阶段：固定表面Gaussian验证

目标是先验证“Gaussian外观层能否改善heldout渲染”，不允许它改变几何；该阶段通过后保持同一参数化扩展到全部26,854,772个面。

1. 从正式PLY抽取工作台ROI或独立proxy，正式文件不变。
2. 验证版控制在约100K至800K个face-bound Gaussians。
3. 每个测试面固定一个Gaussian，保持与最终一面一基元相同的face ID语义。
4. 中心严格限制在三角面上，即 $d=0$。
5. 法向scale固定在很小值，优化两个切向scale。
6. mesh顶点冻结。
7. 先训练SH degree 0，再训练自适应SH对照。
8. 只使用现有16个input相机；7个heldout相机完全不参与训练、密度选择或剪枝规则。
9. 使用mesh depth prepass做可见性和后表面culling。

建议用`gsplat`实现训练和渲染：

- 代码为Apache-2.0。
- 支持当前PyTorch/CUDA栈的概率高于SuGaR/GaMeS旧环境。
- 内存通常低于原始3DGS rasterizer。
- mesh绑定参数化需要本项目自己实现，但核心公式不复杂。

如果优先追求最快论文复现，则在独立环境运行GaMeS；如果优先追求可维护和未来商用，则使用`gsplat`重写最小surface-bound模型。

### 6.2 第二阶段：受限法向残差

只有第一阶段heldout RGB通过后才允许：

- 法向偏移限制在约2至5mm，或由每个区域的几何置信度决定。
- 对offset、法向scale和opacity施加强正则。
- 保持Gaussian期望深度接近mesh z-buffer。
- 禁止在无训练视图支持区域densify。

这一步可以改善微小几何和轮廓，但已经不再严格等价于原mesh表面。

### 6.3 第三阶段：局部Frosting/residual Gaussians

仅对确定的高残差区域启用，例如：

- 水龙头和线缆。
- 桌沿亚像素结构。
- 反射高光边界。
- 薄物体和植被类区域。

不要先对整个房间使用自由壳层，否则容易重新引入当前strict/object实验已经暴露的射线深度二义性。

## 7. 最小可证伪实验

建议先在lecture_room工作台ROI运行以下对照，固定同一16/7相机split：

| 变体 | 作用 |
|---|---|
| A PBR mesh | 当前基线 |
| B direct mesh-to-Gaussian | 验证格式和纯光栅化收益 |
| C vanilla 3DGS | 渲染质量上限及自由floaters对照 |
| D fixed mesh-bound, SH0 | 验证表面Gaussian本身 |
| E fixed mesh-bound, SH3 | 验证视角相关外观收益 |
| F bounded normal offset | 仅在D/E通过后验证微小离面自由度 |

### 成功条件

- 7个heldout相机的PSNR、SSIM或LPIPS相对PBR mesh稳定改善。
- 有效mesh覆盖不能下降。
- Gaussian expected/median depth相对mesh深度不出现大面积漂移。
- 不能只改善16个训练视角。
- 不能通过扩大splat、模糊边界或填充错误深度获得表面上的RGB提升。
- 工作台和柜体的原mesh几何指标保持完全不变。

### 必须保留的诊断

- Mesh-only、Gaussian-only和hybrid同相机图。
- Gaussian alpha、expected depth、median depth和最大贡献Gaussian ID。
- 每个Gaussian所属face ID、训练可见视图数和最大投影面积。
- heldout RGB error overlay。
- mesh depth与Gaussian depth差异图。
- splat数量、显存、FPS和资产体积。

## 8. 当前场景的规模判断

正式lecture_room资产：

- PLY：12,732,795 vertices / 26,854,772 faces，约552MiB。
- GLB：8,251,478 vertices / 10,724,971 faces，约668MiB。
- 图像：23张，原始分辨率6211×4139。
- GPU：RTX A4000 16GB。

对正式PLY逐面扫描得到：

- 零面积面：`0`。
- 面积P10/中位/P90：`5.030/8.635/9.403 mm²`。
- 最大边P10/中位/P90：`4.743/5.884/6.151mm`。
- 最大边最大值：`16.946mm`。
- 总三角面积：`212.937m²`。

这些面已经相当均匀，一面一个Gaussian具有明确物理尺度。26.85M规模的主要约束是训练状态和每帧工作集，而不是最终静态参数本身：

| 表示/状态 | 估算字节/基元 | 26.85M总量 |
|---|---:|---:|
| 显式FP32、SH0 | 56B | 1.40GiB |
| 显式FP32、SH3 | 236B | 5.90GiB |
| 内部FP16 SH3近似 | 112B | 2.80GiB |
| 32B量化运行时 | 32B | 0.80GiB |
| 48B量化运行时 | 48B | 1.20GiB |
| FP32 Adam，51个appearance参数 | 816B | 20.41GiB |
| 混合精度Adam，51个appearance参数 | 612B | 15.31GiB |

最后两行还没有包含投影、排序、梯度和framebuffer，所以A4000不能用普通Adam把全量SH3一次性常驻训练。反过来，5.90GiB的显式推理参数或约0.8至1.2GiB的量化运行时参数本身可以容纳；配合frustum、mesh occlusion和LOD，渲染目标成立。

公开的大场景工作也验证了该工程路径：

- Hierarchical 3D Gaussians把场景分chunk训练、合并成层级，并在viewer中按VRAM budget选择粒度。
- CityGaussian使用分块训练和合并处理大场景。
- Octree-GS按视距动态选择多分辨率Gaussian anchors。

ROI smoke test仍建议先做，预计约30至90分钟。它验证正确性后，完整训练应改为分页/分块任务，不能沿用单次全量训练的1至3小时估计。

## 9. 全量一面一Gaussian目标架构

### 9.1 叶级一对一身份

- 叶级Gaussian数量严格等于有效face数量，即`26,854,772`。
- `gaussian_id == face_id`或通过无歧义置换表对应。
- 每个face恰好一个基础Gaussian，不因不可见、低opacity或LOD而从资产删除。
- 未被16个训练视角观察的face保留Gaussian，使用PBR面颜色初始化并标记为`untrained`，不生成伪SH细节。
- 正式mesh的顶点、面、顺序和哈希保持不变。

### 9.2 几何参数隐式派生

内部格式无需为每个Gaussian重复存储position、rotation和基础scale。对face中心 $c$，可直接从mesh派生：

$$
c=\frac{v_0+v_1+v_2}{3},
$$

$$
\Sigma_f=\alpha\left(\frac{1}{12}\sum_{i=0}^{2}(v_i-c)(v_i-c)^T\right)+\epsilon_n^2nn^T.
$$

其中 $\alpha$ 是覆盖校准或小范围可学习的切向scale，$\epsilon_n$ 是固定的小法向厚度。这样一面一Gaussian的几何成本主要由已有mesh承担，训练和存储集中在opacity与appearance。

导出`KHR_gaussian_splatting`时再把派生参数烘焙为显式position/quaternion/scale。

### 9.3 分页训练

- 按face centroid的Morton code或空间meshlet把全量叶节点切成约64K至256K faces/page。
- 先用mesh rasterizer为16个训练相机生成准确的first-hit face-ID buffer。
- 每个image tile只加载其可见pages以及splat footprint halo。
- optimizer状态保存在CPU内存或mmap分片中，GPU只保留当前active pages。
- 相机曝光/白平衡等少量参数保持全局共享，避免各page独立训练产生色差。
- 第一遍只优化SH0和opacity；高阶SH按有效观察方向数量自适应开启，避免为单视图face拟合欠约束SH3。

### 9.4 两种渲染模式

- **Exact/offline**：按相机frustum和mesh depth选择所有可见叶节点，分tile投影、排序并合成；不使用LOD替代近景face。
- **Realtime/LOD**：所有26.85M叶节点仍在资产中，但远处用层级聚合父Gaussian替代，近处逐步展开到一面一Gaussian。

LOD不是删除目标，而是运行时选择。Hierarchical 3D Gaussians、CityGaussian和Octree-GS已经证明分块和层级预算是大规模Gaussian场景的有效实现方式。

### 9.5 全量验收条件

- `leaf_gaussian_count == valid_face_count == 26,854,772`。
- face映射一对一、无遗漏、无重复绑定。
- 全部参数有限，scale和opacity合法。
- 正式mesh哈希不变。
- Exact模式在关键近景不调用父级替代。
- LOD模式报告每帧叶级/父级选择数、显存和FPS。
- 7个heldout相机同时报告RGB和mesh/Gaussian depth差异。
- 未观察face与已训练face在manifest中可区分。

## 10. 风险与决策

### 主要风险

1. **错误mesh成为硬偏置**：表面Gaussian只能在错误表面上拟合颜色。
2. **外观掩盖几何问题**：透明和SH可以让错误几何在少数视角看似正确。
3. **稀疏视角过拟合**：lecture_room只有16个训练相机。
4. **mesh规模过大**：必须实现分页optimizer、first-hit face可见性、层级LOD和流式导出；proxy只用于验证，不能代替最终叶级映射。
5. **格式生态仍在成熟**：`KHR_gaussian_splatting`尚不能假设所有GLB viewer支持。
6. **PBR语义丢失**：Gaussian radiance不能自然替代金属度、粗糙度和重照明。
7. **许可证**：SuGaR、Frosting和GaMeS官方实现继承原始3DGS的非商业研究限制；`gsplat`本身是Apache-2.0，但最终采用前仍需单独审查训练代码、数据和专利风险。

### 推荐决策

- **接受完整26,854,772面一面一叶级Gaussian作为最终goal**。
- **ROI pilot用于验证参数化和guardrail，不用于决定是否保留全量目标**。
- **全量实现采用固定mesh-bound、分页训练和层级渲染，而非全场自由Frosting**。
- **把Gaussian定义为appearance layer，不是几何修复**。
- **保留正式PLY/GLB及其哈希不变**。
- **先输出`GLB + SPZ + attachment sidecar`，再实验单GLB KHR扩展**。
- **heldout RGB与depth guardrail决定该外观层是否成为推荐渲染模式，而不是决定26.85M叶级资产能否生成**。

## 参考资料

- SuGaR: <https://arxiv.org/abs/2311.12775>
- SuGaR code: <https://github.com/Anttwo/SuGaR>
- GaMeS: <https://arxiv.org/abs/2402.01459>
- GaMeS code: <https://github.com/waczjoan/gaussian-mesh-splatting>
- Gaussian Frosting: <https://arxiv.org/abs/2403.14554>
- Gaussian Frosting code: <https://github.com/Anttwo/Frosting>
- 2D Gaussian Splatting: <https://arxiv.org/abs/2403.17888>
- gsplat: <https://github.com/nerfstudio-project/gsplat>
- Khronos `KHR_gaussian_splatting`: <https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_gaussian_splatting>
- Niantic SPZ: <https://github.com/nianticlabs/spz>
- PlayCanvas SuperSplat: <https://github.com/playcanvas/supersplat>
- Mesh2Gaussian converter: <https://github.com/hwanhuh/mesh2gaussian>
- Hierarchical 3D Gaussians: <https://github.com/graphdeco-inria/hierarchical-3d-gaussians>
- CityGaussian: <https://github.com/Linketic/CityGaussian>
- Octree-GS: <https://github.com/city-super/Octree-GS>
