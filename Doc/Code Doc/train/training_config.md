# train/training_config.py

## 1. 文件职责

`train/training_config.py` 负责读取和校验 `config/training.toml`，并把训练主流程配置解析为不可变 dataclass。该文件只处理训练运行层配置，不读取 H5、不创建模型、不启动训练。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `RuntimeConfig` | dataclass | 设备和张量搬运配置。 |
| `RandomConfig` | dataclass | 随机种子和确定性配置。 |
| `DataLoaderConfig` | dataclass | DataLoader 配置。 |
| `OptimizationConfig` | dataclass | AdamW 和学习率调度配置。 |
| `LossWeights` | dataclass | 各项 loss 权重。 |
| `GradientMonitorConfig` | dataclass | 梯度监测配置。 |
| `CheckpointConfig` | dataclass | checkpoint 配置。 |
| `LoggingConfig` | dataclass | 日志配置。 |
| `TrainingRunConfig` | dataclass | 训练主配置聚合对象。 |
| `load_training_run_config` | function | 读取并校验训练主配置。 |

## 3. 关键类和函数

### `load_training_run_config`

- 功能：读取 TOML 并解析全部训练主流程配置。
- 输入：配置路径和可选项目根目录。
- 输出：`TrainingRunConfig`。
- 约束：所有路径配置必须是项目内相对路径或允许为空的恢复 checkpoint 路径。

### `OptimizationConfig`

- 功能：校验 AdamW、warmup、峰值学习率、末尾余弦退火和梯度裁剪字段。
- 关键约束：`warmup_steps + cosine_decay_steps <= total_steps`。

### `LossWeights`

- 功能：保存 loss 权重。
- 关键约束：权重不能为负，且不能全部为 0。

## 4. 输入输出与 Shape

| 名称 | Shape | 说明 |
| --- | --- | --- |
| TOML 配置 | 无 | 结构化配置文本。 |
| `TrainingRunConfig` | 无 | Python dataclass 聚合对象。 |

## 5. 关键实现逻辑

配置读取使用 `tomllib`。实现端不提供训练字段默认值，缺表、缺字段或类型错误会直接抛出异常。路径字段会拒绝绝对路径，并要求解析结果位于项目根目录内。

## 6. 配置项

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `config/training.toml` | 见配置文件 | 本文件读取的主配置。 |

## 7. 依赖关系

- 上游：`config/training.toml`。
- 下游：`train/trainer.py`、`train/losses.py`、`train/gradient_monitor.py`、`train/checkpointing.py`。
- 标准库：`dataclasses`、`pathlib`、`tomllib`、`typing`。

## 8. 注意事项

- 不要在实现文件中写入配置文件已有默认值。
- 新增训练配置字段时必须同步 dataclass、读取函数、配置文档和摘要文档。

## 9. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-08 | 1os3_Codex | AI 完成：新增训练主配置解析模块。 |
