# train/__init__.py

## 1. 文件职责

`train/__init__.py` 是训练辅助包入口，使用懒加载方式导出训练数据处理、训练配置、loss、梯度监测、checkpoint 和训练主入口公开接口。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `TrainingRunConfig` | dataclass | 训练主流程配置。 |
| `TrainingDataConfig` | dataclass | 训练数据处理配置。 |
| `ValidatedTrainingDataset` | class | 过滤无效样本的 Dataset。 |
| `MonoDriveTrainingLoss` | class | 训练 loss 汇总模块。 |
| `GradientMonitorResult` | NamedTuple | 梯度监测结果。 |
| `WarmupCosineLRScheduler` | class | 学习率调度器。 |
| `build_training_batch_labels` | function | 构造训练标签。 |
| `build_training_dataset` | function | 构建训练 Dataset。 |
| `load_training_run_config` | function | 读取训练主配置。 |
| `run_training` | function | 启动训练主流程。 |
| `training_collate` | function | 合并 batch。 |

## 3. 关键类和函数

本文件仅提供懒加载导出，不实现训练逻辑。

## 4. 输入输出与 Shape

| 名称 | Shape | 说明 |
| --- | --- | --- |
| 无 | 无 | 本文件不直接处理张量。 |

## 5. 关键实现逻辑

`__getattr__` 在访问公开名称时动态导入对应训练模块，降低包初始化时的依赖加载成本。新增训练模块接口通过 `_LAZY_EXPORTS` 显式登记。

## 6. 配置项

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| 无 | 无 | 本文件不读取配置。 |

## 7. 依赖关系

- 上游：`train.data_processing`、`train.training_config`、`train.losses`、`train.gradient_monitor`、`train.checkpointing`、`train.trainer`。
- 下游：训练入口和测试代码。

## 8. 注意事项

- 新增训练公开接口时，应同步 `__all__` 和 `_LAZY_EXPORTS`。

## 9. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-07 | 1os3_Codex | AI 完成：新增训练包入口文档。 |
| 2026-06-08 | 1os3_Codex | AI 完成：扩展训练包懒加载导出，覆盖训练主流程、loss、梯度监测、checkpoint 和配置接口。 |
