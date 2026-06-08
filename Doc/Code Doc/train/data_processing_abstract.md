# train/data_processing.py 摘要

## 1. 文件基本功能

`train/data_processing.py` 提供训练阶段数据处理：复用 H5 Dataset 读取样本，过滤无效数据，构造轨迹词表标签，并在物理空间对 Agent / Map 执行 Hungarian matching。

## 2. 主要公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `TrainingDataConfig` | dataclass | 训练数据处理配置。 |
| `ValidatedTrainingDataset` | class | 过滤 NaN、Inf 和越界样本的 Dataset 包装器。 |
| `build_training_dataset` | function | 构建训练 Dataset。 |
| `training_collate` | function | 合并 batch。 |
| `build_training_batch_labels` | function | 构造全部训练标签。 |
| `build_trajectory_vocab_labels` | function | 构造轨迹词表 soft label 和残差目标。 |
| `build_agent_matching_targets` | function | 构造 Agent 匹配目标和 winner mode 逐点 future mask。 |
| `build_map_matching_targets` | function | 构造 Map 匹配目标，并对无方向类别沿用最小误差点序。 |

## 3. Shape 概览

| 名称 | Shape | 说明 |
| --- | --- | --- |
| 轨迹 soft label | `[B, V]` | 由词表轨迹和 GT 的物理空间 MSE 构造。 |
| 轨迹残差目标 | `[B, V, 6, 2]` | 只监督 winner 轨迹。 |
| Agent 分类目标 | `[B, 48]` | 未匹配 query 为 none。 |
| Agent 状态目标 | `[B, 48, 11]` | 匹配后写回监督空间。 |
| Agent future mask | `[B, 48, 4, 6]` | 只监督 winner mode 中有效 future 点。 |
| Map 分类目标 | `[B, 48]` | 未匹配 query 为 none。 |
| Map 点目标 | `[B, 48, 100, 2]` | 匹配后写回 Symlog 空间。 |

## 4. 使用规范

训练入口应先读取 `config/training_data.toml`，再构建 Dataset 和 labels。调用方传入模型输出时，应使用 `model.detection_head.DetectionDecoderOutput` 和 `model.trajectory_vocab.TrajectoryDecoderOutput`。

本模块不实现危险轨迹判断，避免稀疏 H5 未来 Agent 标签造成错误监督。

## 5. 最小示例

适合在训练入口中组合使用：加载配置，构造 `ValidatedTrainingDataset`，用 `training_collate` 生成 batch，再把模型输出和 batch 传入 `build_training_batch_labels`。

## 6. 维护注意事项

- 修改模型输出 shape 后必须同步本文件和代码文档。
- 修改配置字段后必须同步 `config/training_data.toml` 文档。
- Hungarian cost 新增项必须说明其物理空间单位和监督空间回写方式。
- 无方向 Map 类别的正反点序选择必须保持 matching 和 loss 目标一致。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-07 | 1os3_Codex | AI 完成：新增训练数据处理模块摘要。 |
| 2026-06-08 | 1os3_Codex | AI 完成：更新 Agent future 逐点 mask 和 Map 正反点序监督摘要。 |
