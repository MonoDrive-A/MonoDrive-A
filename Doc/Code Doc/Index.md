# Code Doc Index

本目录用于存放 MonoDrive 的逐文件代码文档。每个源码文件都应有对应文档，并按照源码路径镜像组织。

## 目录规范

| 源码目录 | 文档目录 |
| --- | --- |
| `model/` | `doc/Code Doc/model/` |
| `visualization/` | `doc/Code Doc/visualization/` |
| `data/` | `doc/Code Doc/data/` |
| `config/` | `doc/Code Doc/config/` |

## 文档目录

| 源码文件 | 基本功能 | 摘要文档 | 完整文档 | 状态 |
| --- | --- | --- | --- | --- |
| `data/b2d_dataset.py` | 读取逐场景 B2D H5，并以 PyTorch Dataset 返回 8 帧前视图像、目标候选、Agent future 和局部 Map 标签。 | [data/b2d_dataset_abstract.md](data/b2d_dataset_abstract.md) | [data/b2d_dataset.md](data/b2d_dataset.md) | 已同步 |
| `data/b2d_preprocess.py` | 将 B2D 原始场景预处理为逐场景 H5，构造 5Hz 输入滑窗、目标候选、Agent future 和局部 Map 标签。 | [data/b2d_preprocess_abstract.md](data/b2d_preprocess_abstract.md) | [data/b2d_preprocess.md](data/b2d_preprocess.md) | 已同步 |
| `data/detection_class_stats.py` | 跨 H5 数据集统计 Agent、Map、Traffic Light 和 Stop Sign 的检测类别分布。 | [data/detection_class_stats_abstract.md](data/detection_class_stats_abstract.md) | [data/detection_class_stats.md](data/detection_class_stats.md) | 已同步 |
| `data/trajectory_vocab.py` | 从逐场景 H5 全局读取 ego future 轨迹，并用 FTS 采样规划轨迹词表，第 0 条强制为静止轨迹。 | [data/trajectory_vocab_abstract.md](data/trajectory_vocab_abstract.md) | [data/trajectory_vocab.md](data/trajectory_vocab.md) | 已同步 |
| `config/target_point_embedding.toml` | 保存目标点嵌入层配置，包括 18x16 ego 栅格、Symlog 向量变换、三层卷积下采样、线性投影为 2 个目标导航点 Token 和 FP32 精度。 | [config/target_point_embedding_abstract.md](config/target_point_embedding_abstract.md) | [config/target_point_embedding.md](config/target_point_embedding.md) | 已同步 |
| `config/trajectory_vocab.toml` | 保存模型侧轨迹词表加载、嵌入和解码配置，包括归一化词表字段、384 维特征和解码初始化。 | [config/trajectory_vocab_abstract.md](config/trajectory_vocab_abstract.md) | [config/trajectory_vocab.md](config/trajectory_vocab.md) | 已同步 |
| `config/vision_embedding.toml` | 保存骨干视觉嵌入层配置，包括 DINOv3 本地加载、输入图像约束、DINOv3 后 3D 卷积压缩、精度和输出 token 数。 | [config/vision_embedding_abstract.md](config/vision_embedding_abstract.md) | [config/vision_embedding.md](config/vision_embedding.md) | 已同步 |
| `model/rope_3d.py` | 提供通用 3D RoPE 旋转位置编码，只消费外部传入的三维位置坐标和轴通道划分。 | [model/rope_3d_abstract.md](model/rope_3d_abstract.md) | [model/rope_3d.md](model/rope_3d.md) | 已同步 |
| `model/residual_block.py` | 定义 2D / 3D RMSNorm、瓶颈残差块、时空解耦 3D 残差块和 ConvNeXt 风格 depthwise separable 残差块。 | [model/residual_block_abstract.md](model/residual_block_abstract.md) | [model/residual_block.md](model/residual_block.md) | 已同步 |
| `model/swiglu.py` | 提供函数式和 `nn.Module` 形式的通用 SwiGLU 激活。 | [model/swiglu_abstract.md](model/swiglu_abstract.md) | [model/swiglu.md](model/swiglu.md) | 已同步 |
| `model/target_point_embedding.py` | 将 ego 坐标系目标点构造成 Symlog 栅格向量场并编码为 2 个 384 维目标导航点 Token，同时强制保持 FP32。 | [model/target_point_embedding_abstract.md](model/target_point_embedding_abstract.md) | [model/target_point_embedding.md](model/target_point_embedding.md) | 已同步 |
| `model/vision_embedding.py` | 加载冻结 DINOv3-ViT-B，只取 Patch 序列，并在 DINOv3 后执行 3D 卷积时间压缩，输出 2304 个 384 维视觉 token。 | [model/vision_embedding_abstract.md](model/vision_embedding_abstract.md) | [model/vision_embedding.md](model/vision_embedding.md) | 已同步 |
| `model/trajectory_vocab/__init__.py` | 作为轨迹词表模型包入口，重新导出配置、加载、嵌入和解码公开接口。 | [model/trajectory_vocab/__init___abstract.md](model/trajectory_vocab/__init___abstract.md) | [model/trajectory_vocab/__init__.md](model/trajectory_vocab/__init__.md) | 已同步 |
| `model/trajectory_vocab/trajectory_vocab.py` | 从 TOML 配置加载归一化轨迹词表，生成 384 维轨迹查询，并解码轨迹 logit 与 Tanh 残差。 | [model/trajectory_vocab/trajectory_vocab_abstract.md](model/trajectory_vocab/trajectory_vocab_abstract.md) | [model/trajectory_vocab/trajectory_vocab.md](model/trajectory_vocab/trajectory_vocab.md) | 已同步 |
| `visualization/b2d_h5_viewer.py` | 复用 `B2DH5Dataset` 读取 B2D 训练样本并导出 PNG 诊断图，用于检查输入帧、BEV 轨迹、Agent future、Map 和交通元素标签。 | [visualization/b2d_h5_viewer_abstract.md](visualization/b2d_h5_viewer_abstract.md) | [visualization/b2d_h5_viewer.md](visualization/b2d_h5_viewer.md) | 已同步 |
| `visualization/trajectory_vocab_viewer.py` | 读取轨迹词表 NPZ，反求归一化轨迹到物理空间，计算 MSE，并导出逐条叠图与所有轨迹全局叠图。 | [visualization/trajectory_vocab_viewer_abstract.md](visualization/trajectory_vocab_viewer_abstract.md) | [visualization/trajectory_vocab_viewer.md](visualization/trajectory_vocab_viewer.md) | 已同步 |
| `visualization/target_point_embedding_viewer.py` | 根据命令行目标点坐标调用真实目标点嵌入层，并导出 18x16 Symlog 向量场、9x8 卷积特征和目标导航点 Token 统计图。 | [visualization/target_point_embedding_viewer_abstract.md](visualization/target_point_embedding_viewer_abstract.md) | [visualization/target_point_embedding_viewer.md](visualization/target_point_embedding_viewer.md) | 已同步 |
| `visualization/vision_embedding_viewer.py` | 读取 B2D H5 样本，直接调用骨干视觉嵌入层并以 FP32 导出输入帧、DINOv3 PCA 上采样图、latent PCA 上采样图和 token norm 诊断图。 | [visualization/vision_embedding_viewer_abstract.md](visualization/vision_embedding_viewer_abstract.md) | [visualization/vision_embedding_viewer.md](visualization/vision_embedding_viewer.md) | 已同步 |

## 登记规则

1. 新增源码文件时，必须在本页新增一行。
2. 删除源码文件时，必须同步删除或标记对应代码文档。
3. 移动源码文件时，必须同步移动代码文档，并更新本页链接。
4. 每个源码文件必须同时登记摘要文档和完整文档。
5. `基本功能` 用一句话说明文件职责，方便从 Index 快速定位。
6. 文档状态使用：`待补充实现细节`、`已同步`、`需要更新`。
