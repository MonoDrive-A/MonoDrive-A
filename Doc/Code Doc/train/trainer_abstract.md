# train/trainer.py 摘要

## 1. 文件基本功能

`train/trainer.py` 运行完整训练流程，包括模型、数据、loss、优化器、梯度监测、checkpoint 和恢复。

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

修改训练循环状态、checkpoint 字段或 LR 调度时，同步 checkpoint 文档和配置文档。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-08 | 1os3_Codex | AI 完成：新增训练入口摘要。 |
