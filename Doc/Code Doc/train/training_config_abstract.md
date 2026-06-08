# train/training_config.py 摘要

## 1. 文件基本功能

`train/training_config.py` 读取 `config/training.toml`，校验训练运行配置，并返回 `TrainingRunConfig`。

## 2. 主要公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `TrainingRunConfig` | dataclass | 训练主配置聚合对象。 |
| `LossWeights` | dataclass | loss 权重配置。 |
| `GradientMonitorConfig` | dataclass | 梯度监测配置。 |
| `load_training_run_config` | function | 读取训练配置。 |

## 3. Shape 概览

本文件不直接处理张量。

## 4. 使用规范

配置路径和输出目录必须是项目内相对路径。实现端只读取和校验配置，不填补配置默认值。

## 5. 最小示例

适合训练入口调用 `load_training_run_config("config/training.toml")`。

## 6. 维护注意事项

新增字段时同步 TOML、dataclass、完整文档和摘要文档。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-08 | 1os3_Codex | AI 完成：新增训练配置解析摘要。 |
