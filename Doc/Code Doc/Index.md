# Code Doc Index

本目录用于存放 MonoDrive 的逐文件代码文档。每个源码文件都应有对应文档，并按照源码路径镜像组织。

## 目录规范

| 源码目录 | 文档目录 |
| --- | --- |
| `model/` | `doc/Code Doc/model/` |
| `visualization/` | `doc/Code Doc/visualization/` |
| `data/` | `doc/Code Doc/data/` |
| `config/` | `doc/Code Doc/config/` |
| `train/` | `doc/Code Doc/train/` |

## 文档目录

| 源码文件 | 基本功能 | 摘要文档 | 完整文档 | 状态 |
| --- | --- | --- | --- | --- |
| `data/b2d_dataset.py` | 读取逐场景 B2D H5，并以 PyTorch Dataset 返回 8 帧前视图像、目标候选、Agent future 和局部 Map 标签。 | [data/b2d_dataset_abstract.md](data/b2d_dataset_abstract.md) | [data/b2d_dataset.md](data/b2d_dataset.md) | 已同步 |
| `data/b2d_preprocess.py` | 将 B2D 原始场景预处理为逐场景 H5，构造 5Hz 输入滑窗、目标候选、Agent future 和局部 Map 标签。 | [data/b2d_preprocess_abstract.md](data/b2d_preprocess_abstract.md) | [data/b2d_preprocess.md](data/b2d_preprocess.md) | 已同步 |
| `data/detection_class_stats.py` | 跨 H5 数据集统计 Agent、Map、Traffic Light 和 Stop Sign 的检测类别分布。 | [data/detection_class_stats_abstract.md](data/detection_class_stats_abstract.md) | [data/detection_class_stats.md](data/detection_class_stats.md) | 已同步 |
| `data/trajectory_vocab.py` | 从逐场景 H5 全局读取 ego future 轨迹，并用 FTS 采样规划轨迹词表，第 0 条强制为静止轨迹。 | [data/trajectory_vocab_abstract.md](data/trajectory_vocab_abstract.md) | [data/trajectory_vocab.md](data/trajectory_vocab.md) | 已同步 |
| `config/backbone.toml` | 保存统一序列 Transformer 主干配置，包括子配置路径、12 层主干、视觉 Token 专用 3D RoPE、模态独立 FFN 和精度策略。 | [config/backbone_abstract.md](config/backbone_abstract.md) | [config/backbone.md](config/backbone.md) | 已同步 |
| `config/detection_head.toml` | 保存检测查询初始化和检测解码头配置，包括 96 个检测查询、无类别硬分配、Agent 4-mode 120 度均匀初始化和 FP32 解码精度。 | [config/detection_head_abstract.md](config/detection_head_abstract.md) | [config/detection_head.md](config/detection_head.md) | 已同步 |
| `config/target_point_embedding.toml` | 保存目标点嵌入层配置，包括 18x16 ego 栅格、Symlog 向量变换、三层卷积下采样、线性投影为 2 个目标导航点 Token 和 FP32 精度。 | [config/target_point_embedding_abstract.md](config/target_point_embedding_abstract.md) | [config/target_point_embedding.md](config/target_point_embedding.md) | 已同步 |
| `config/training_data.toml` | 保存训练数据读取、样本校验、轨迹词表标签和 Agent / Map Hungarian matching cost 配置，不启用危险轨迹判断。 | [config/training_data_abstract.md](config/training_data_abstract.md) | [config/training_data.md](config/training_data.md) | 已同步 |
| `config/training.toml` | 保存训练主流程配置，包括运行设备、DataLoader、AdamW、学习率调度、loss 权重、梯度监测、checkpoint 和日志。 | [config/training_abstract.md](config/training_abstract.md) | [config/training.md](config/training.md) | 已同步 |
| `config/trajectory_vocab.toml` | 保存模型侧轨迹词表加载、嵌入和解码配置，包括归一化词表字段、384 维特征和解码初始化。 | [config/trajectory_vocab_abstract.md](config/trajectory_vocab_abstract.md) | [config/trajectory_vocab.md](config/trajectory_vocab.md) | 已同步 |
| `config/vision_embedding.toml` | 保存骨干视觉嵌入层配置，包括 DINOv3 本地加载、输入图像约束、DINOv3 后 3D 卷积压缩、精度和输出 token 数。 | [config/vision_embedding_abstract.md](config/vision_embedding_abstract.md) | [config/vision_embedding.md](config/vision_embedding.md) | 已同步 |
| `model/backbone.py` | 复用已有嵌入与解码模块，构造 2662 个 Token 的统一序列并执行 12 层 Transformer 主干，检测解码前加回初始检测查询和零初始化线性残差。 | [model/backbone_abstract.md](model/backbone_abstract.md) | [model/backbone.md](model/backbone.md) | 已同步 |
| `model/detection_head.py` | 生成无类别硬分配的 Agent/Map 检测查询 Token，并用 FP32 线性层解码 Agent 和 Map 检测输出。 | [model/detection_head_abstract.md](model/detection_head_abstract.md) | [model/detection_head.md](model/detection_head.md) | 已同步 |
| `model/rope_3d.py` | 提供通用 3D RoPE 旋转位置编码，只消费外部传入的三维位置坐标和轴通道划分。 | [model/rope_3d_abstract.md](model/rope_3d_abstract.md) | [model/rope_3d.md](model/rope_3d.md) | 已同步 |
| `model/residual_block.py` | 定义 2D / 3D RMSNorm、瓶颈残差块、时空解耦 3D 残差块和 ConvNeXt 风格 depthwise separable 残差块。 | [model/residual_block_abstract.md](model/residual_block_abstract.md) | [model/residual_block.md](model/residual_block.md) | 已同步 |
| `model/swiglu.py` | 提供函数式和 `nn.Module` 形式的通用 SwiGLU 激活。 | [model/swiglu_abstract.md](model/swiglu_abstract.md) | [model/swiglu.md](model/swiglu.md) | 已同步 |
| `model/target_point_embedding.py` | 将 ego 坐标系目标点构造成 Symlog 栅格向量场并编码为 2 个 384 维目标导航点 Token，同时强制保持 FP32。 | [model/target_point_embedding_abstract.md](model/target_point_embedding_abstract.md) | [model/target_point_embedding.md](model/target_point_embedding.md) | 已同步 |
| `model/vision_embedding.py` | 加载冻结 DINOv3-ViT-B，只取 Patch 序列，并在 DINOv3 后执行 3D 卷积时间压缩，输出 2304 个 384 维视觉 token。 | [model/vision_embedding_abstract.md](model/vision_embedding_abstract.md) | [model/vision_embedding.md](model/vision_embedding.md) | 已同步 |
| `model/trajectory_vocab/__init__.py` | 作为轨迹词表模型包入口，重新导出配置、加载、嵌入和解码公开接口。 | [model/trajectory_vocab/__init___abstract.md](model/trajectory_vocab/__init___abstract.md) | [model/trajectory_vocab/__init__.md](model/trajectory_vocab/__init__.md) | 已同步 |
| `model/trajectory_vocab/trajectory_vocab.py` | 从 TOML 配置加载归一化轨迹词表，生成 384 维轨迹查询，并解码轨迹 logit 与 Tanh 残差。 | [model/trajectory_vocab/trajectory_vocab_abstract.md](model/trajectory_vocab/trajectory_vocab_abstract.md) | [model/trajectory_vocab/trajectory_vocab.md](model/trajectory_vocab/trajectory_vocab.md) | 已同步 |
| `train/__init__.py` | 作为训练辅助包入口，懒加载并导出训练数据处理、配置、loss、梯度监测、checkpoint 和训练主入口接口。 | [train/__init___abstract.md](train/__init___abstract.md) | [train/__init__.md](train/__init__.md) | 已同步 |
| `train/checkpointing.py` | 保存和加载训练 checkpoint，记录 model、optimizer、scheduler、step、epoch、batch、metrics 和 RNG 状态。 | [train/checkpointing_abstract.md](train/checkpointing_abstract.md) | [train/checkpointing.md](train/checkpointing.md) | 已同步 |
| `train/data_processing.py` | 复用 H5 Dataset 读取训练样本，过滤无效数据，构造轨迹词表标签，并在物理空间执行 Agent / Map Hungarian matching。 | [train/data_processing_abstract.md](train/data_processing_abstract.md) | [train/data_processing.md](train/data_processing.md) | 已同步 |
| `train/gradient_monitor.py` | 监测可训练参数梯度范数，报告缺失、过小、过大和非有限梯度。 | [train/gradient_monitor_abstract.md](train/gradient_monitor_abstract.md) | [train/gradient_monitor.md](train/gradient_monitor.md) | 已同步 |
| `train/losses.py` | 汇总轨迹、Agent 和 Map 训练 loss，其中轨迹词表分数使用 soft CE，分类和 mode 使用 hard-label CE。 | [train/losses_abstract.md](train/losses_abstract.md) | [train/losses.md](train/losses.md) | 已同步 |
| `train/trainer.py` | 运行完整训练流程，包含模型、数据、loss、优化器、学习率调度、梯度监测、自动保存和断点恢复。 | [train/trainer_abstract.md](train/trainer_abstract.md) | [train/trainer.md](train/trainer.md) | 已同步 |
| `train/training_config.py` | 读取和校验 `config/training.toml`，生成训练主流程 dataclass 配置。 | [train/training_config_abstract.md](train/training_config_abstract.md) | [train/training_config.md](train/training_config.md) | 已同步 |
| `visualization/b2d_h5_viewer.py` | 复用 `B2DH5Dataset` 读取 B2D 训练样本并导出 PNG 诊断图，用于检查输入帧、BEV 轨迹、Agent future、Map 和交通元素标签。 | [visualization/b2d_h5_viewer_abstract.md](visualization/b2d_h5_viewer_abstract.md) | [visualization/b2d_h5_viewer.md](visualization/b2d_h5_viewer.md) | 已同步 |
| `visualization/backbone_feature_pca_viewer.py` | 直接调用统一主干并以 FP32 导出每层视觉 Token PCA 图，支持加载 checkpoint 权重，同时展示非 `none` 检测查询、轨迹词表概率和 top-k residual 修正。 | [visualization/backbone_feature_pca_viewer_abstract.md](visualization/backbone_feature_pca_viewer_abstract.md) | [visualization/backbone_feature_pca_viewer.md](visualization/backbone_feature_pca_viewer.md) | 已同步 |
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
