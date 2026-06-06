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
| `model/residual_block.py` | 定义 2D / 3D RMSNorm、瓶颈残差块、时空解耦 3D 残差块和 ConvNeXt 风格 depthwise separable 残差块。 | [model/residual_block_abstract.md](model/residual_block_abstract.md) | [model/residual_block.md](model/residual_block.md) | 已同步 |
| `visualization/b2d_h5_viewer.py` | 复用 `B2DH5Dataset` 读取 B2D 训练样本并导出 PNG 诊断图，用于检查输入帧、BEV 轨迹、Agent future、Map 和交通元素标签。 | [visualization/b2d_h5_viewer_abstract.md](visualization/b2d_h5_viewer_abstract.md) | [visualization/b2d_h5_viewer.md](visualization/b2d_h5_viewer.md) | 已同步 |
| `visualization/trajectory_vocab_viewer.py` | 读取轨迹词表 NPZ，反求归一化轨迹到物理空间，计算 MSE，并导出逐条叠图与所有轨迹全局叠图。 | [visualization/trajectory_vocab_viewer_abstract.md](visualization/trajectory_vocab_viewer_abstract.md) | [visualization/trajectory_vocab_viewer.md](visualization/trajectory_vocab_viewer.md) | 已同步 |

## 登记规则

1. 新增源码文件时，必须在本页新增一行。
2. 删除源码文件时，必须同步删除或标记对应代码文档。
3. 移动源码文件时，必须同步移动代码文档，并更新本页链接。
4. 每个源码文件必须同时登记摘要文档和完整文档。
5. `基本功能` 用一句话说明文件职责，方便从 Index 快速定位。
6. 文档状态使用：`待补充实现细节`、`已同步`、`需要更新`。
