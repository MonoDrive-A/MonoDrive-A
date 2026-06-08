# train/__init__.py 摘要

## 1. 文件基本功能

`train/__init__.py` 作为训练辅助包入口，懒加载并导出训练数据处理、训练配置、loss、梯度监测、checkpoint 和训练主入口公开接口。

## 2. 主要公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `TrainingDataConfig` | dataclass | 训练数据配置。 |
| `TrainingRunConfig` | dataclass | 训练主配置。 |
| `ValidatedTrainingDataset` | class | 校验后的训练 Dataset。 |
| `MonoDriveTrainingLoss` | class | 训练 loss 汇总模块。 |
| `build_training_batch_labels` | function | 训练标签构造入口。 |
| `run_training` | function | 训练主入口。 |

## 3. Shape 概览

本文件不直接处理张量。

## 4. 使用规范

从 `train` 包导入公开训练接口即可，不需要直接依赖懒加载表。

## 5. 最小示例

适合在实验脚本中执行 `from train import run_training, build_training_dataset`。

## 6. 维护注意事项

新增公开接口时同步 `__all__`、`_LAZY_EXPORTS` 和文档。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-07 | 1os3_Codex | AI 完成：新增训练包入口摘要。 |
| 2026-06-08 | 1os3_Codex | AI 完成：扩展训练包入口摘要，覆盖新增训练主流程接口。 |
