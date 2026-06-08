# train/trainer.py 摘要

## 1. 文件基本功能

`train/trainer.py` 运行完整训练流程，包括模型、数据、loss、优化器、梯度监测、checkpoint 和恢复，并在训练开始前打印预期单个 epoch step 数。断点恢复时会按保存的 batch index 切分首个恢复 epoch 的 sampler，避免重新读取已完成 batch。

## 2. 主要公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `TrainingSummary` | dataclass | 训练结果摘要。 |
| `WarmupCosineLRScheduler` | class | warmup + 末尾余弦退火调度器。 |
| `run_training` | function | 启动训练。 |

## 3. Shape 概览

| 名称 | Shape | 说明 |
| --- | --- | --- |
| `images` | `[B, 8, 3, 288, 512]` | 模型输入图像。 |
| `target_point` | `[B, 2]` | 目标点。 |
| `ego_motion` | `[B, 3]` | 自车运动状态。 |

## 4. 使用规范

从项目根目录运行训练入口，并传入项目内训练配置路径。`--max-steps` 只用于临时调试。

## 5. 最小示例

适合使用 `run_training("config/training.toml")` 或命令行 `python -m train.trainer --config config/training.toml`。

## 6. 维护注意事项

修改训练循环状态、checkpoint 字段或 LR 调度时，同步 checkpoint 文档和配置文档。恢复训练不得通过读取后 `continue` 丢弃历史 batch，应在 sampler 层跳过已完成样本，避免 H5 I/O 和 CPU 校验开销。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-08 | 1os3_Codex | AI 完成：记录断点恢复 sampler 级跳过已完成 batch。 |
| 2026-06-08 | 1os3_Codex | AI 完成：新增训练入口摘要。 |
| 2026-06-08 | 1os3_Codex | AI 完成：记录训练开始前打印 epoch step 数。 |
