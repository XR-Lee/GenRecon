# ETH3D `lecture_room` 干净腰高工作台 GenRecon 推理报告

## 结论

在当前可直接测试的数据中，ETH3D `lecture_room` 是最符合“半人高、台面较干净”要求的场景。主体是一条长木质讲台/实验演示工作台，台面连续，前半区域基本无遮挡；后沿仅有少量线缆、终端和水路设备。

从清理后的 SfM 点高度峰值估计，地面约为 `z=-1.82m`，主工作台面约为 `z=-0.89m`，台面离地约 `0.93m`。该数值是稀疏点平面峰值估计，不是语义标注尺寸。

ScanNet++ `c5439f4607` 是更工业化的备选，但其电子仪器、抽风管、线缆和周边遮挡更多；本次因此选用 `lecture_room` 作为主交付。

## 数据

- 场景：ETH3D multi-view test `lecture_room`
- 下载：<https://www.eth3d.net/data/lecture_room_dslr_undistorted.7z>
- 许可：CC BY-NC-SA 4.0
- 图像：23/23 张无畸变 DSLR JPG，`6211x4139`
- 相机：1 个 COLMAP `PINHOLE` 相机模型
- 稀疏点：6,239 个，平均重投影误差约 `0.936px`
- 高置信点：3,642 个（`error<=2px`、`track>=3`）
- chunker 清理后：1,886 个
- 坐标：`z-up`，米制尺度；相机约位于地面上方 `1.20m`

原始归档 SHA-256：

```text
0508d09efedb803b8600ce63484d5307f410f976216b1f6981a8ab933acdad2f
```

## 推理配置

- GenRecon 512
- 16 个物理相机，非方形图像展开为 32 个方形 crop
- seed `42`
- `occ_threshold=-1`
- `chunk_size_factor=1.04`
- `max_reproj_error=2px`
- `min_track_len=3`
- `min_points_per_chunk=30`
- 11 个 chunks，边长约 `2.17m`
- 联合解码分为 `[5,3,3]` 三组，输入膨胀体素数为 `[19105,16155,13552]`

未分组的首次运行在联合 shape 解码时发生 CUDA OOM，进程显存峰值约 `13.8GiB`。分组重跑保持输入、seed、分辨率和阈值不变，成功耗时 `473.2s`，进程显存峰值约 `7.0GiB`，进程树 RSS 峰值约 `18.26GB`。

## 最终产物

| 资产 | 顶点 | 三角面 | 大小 | SHA-256 |
|---|---:|---:|---:|---|
| PLY | 12,732,795 | 26,854,772 | 578,302,659 B | `b24fe5628305e120b945dd72503620ab6984016b31a70a1d26f3b8aba4afcd83` |
| GLB | 8,251,478 | 10,724,971 | 699,554,508 B | `6fd4000afd4287ea947d53de6a4dda92ae483136693946efb78b19df67db4a7d` |

PLY 保留最高几何细节。GLB 包含 11 个 meshes、11 个 materials 和 22 张内嵌 4096² PBR PNG 纹理，每 chunk 最多约 100 万面。

最终路径：

- `outputs/eth3d/lecture_room/final/mesh.ply`
- `outputs/eth3d/lecture_room/final/scene.glb`

## 同相机视角检查

使用全部 23 个原始 COLMAP 相机渲染，其中 16 个为输入视角、7 个为 heldout 视角。

| 指标 | PLY | GLB |
|---|---:|---:|
| 全视角覆盖率 | 75.2% | 75.0% |
| 曝光补偿 PSNR | 14.26dB | 14.53dB |
| SfM 深度覆盖率 | 78.1% | 77.8% |
| 跨视角中位深度误差 | 5.2cm | 5.5cm |
| SfM 深度误差 <=10cm | 66.8% | 66.4% |

PLY 与 GLB 的 mask IoU 为 `0.998`；共同覆盖像素中约 `98.0%` 的深度差不超过 2cm，说明 GLB 简化没有明显破坏主体几何。

工作台正面和近距离视角显示：台面主体、前沿和柜体连续，主要缺失集中在远端教室、窗边和少纹理黑板区域。GLB 的无环境光诊断渲染偏暗，实际 PBR 查看器会受环境光照影响。

## SfM 到 GLB 直接距离

| 点集 | 点数 | 中位距离 | P90 | <=10cm |
|---|---:|---:|---:|---:|
| chunker 清理后全部点 | 1,886 | 0.62cm | 28.75cm | 86.5% |
| 高置信点且位于 GLB AABB | 3,008 | 0.84cm | 28.26cm | 84.3% |

外围 `chunk 006` 和 `chunk 010` 的中位距离约 29–31cm，它们位于教室远端；中央工作台区域的支撑点明显更好。直接最近面距离不检查遮挡，因此应与上面的相机 z-buffer 指标一起解释。

## 校验

- PLY 文件长度与 header 声明完全一致。
- 全部 12,732,795 个顶点坐标有限。
- 全部 face 均为三角形，索引范围为 `0..12,732,794`。
- GLB magic/version/声明长度、JSON/BIN chunks、bufferViews 和 accessors 均有效。
- 22 个内嵌图像 bufferView 均具有 PNG signature。
- 23 个同位姿比较目录和 HTML 资源引用均可用。

## 局限

`lecture_room` 属于 ETH3D test split，公开包没有激光点云或官方 dense ground truth。因此当前可以验证 SfM 稀疏一致性、相机可见深度和 PLY/GLB 相互一致性，但不能报告绝对 surface precision、双向 Chamfer 或法向一致性。若任务必须同时满足“工业工作台”和“扫描真值”，应使用备选 ScanNet++ `c5439f4607`；若优先要求台面平整、干净和高分辨率成像，本次 `lecture_room` 更合适。

## 查看入口

- `outputs/eth3d/lecture_room/fidelity_comparison/index.html`
- `outputs/eth3d/lecture_room/fidelity_comparison/overview.jpg`
- `outputs/eth3d/lecture_room/fidelity_comparison/sfm_glb_geometry/README.md`
- `reports/generated/eth3d/lecture_room/final/input_views_contact.jpg`
- `reports/generated/eth3d/lecture_room/final/source_views_contact.jpg`
