# config/training.toml 摘要

## 1. 文件基本功能

`config/training.toml` 保存训练主流程配置，覆盖运行设备、随机种子、DataLoader、AdamW、学习率调度、loss 权重、梯度监测、checkpoint 和日志。

## 2. 主要公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `[modules]` | TOML table | 引用主干和训练数据配置。 |
| `[optimization]` | TOML table | 优化器和学习率调度。 |
| `[loss_weights]` | TOML table | loss 权重。 |
| `[checkpoint]` | TOML table | 自动保存和断点恢复。 |

## 3. Shape 概览

本文件不直接处理张量；训练张量 shape 由模型和数据配置决定。

## 4. 使用规范

训练入口通过 `load_training_run_config` 读取本文件。配置路径和输出目录必须解析到项目目录内。

## 5. 最小示例

适合在项目根目录运行训练入口时使用默认路径 `config/training.toml`。

## 6. 维护注意事项

- 不在本文件重复已有模型结构默认值。
- 修改训练配置字段时，同步 `train/training_config.py` 和本文档。
- 轨迹词表概率 loss 使用 BCEWithLogits，不使用 softmax 或 CE。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-08 | 1os3_Codex | AI 完成：新增训练主配置摘要。 |
