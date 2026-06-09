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
| `DetectionClassWeightConfig` | dataclass | 检测分类 CE 的 none / non-none 类别权重配置。 |
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

### `DetectionClassWeightConfig`

- 功能：保存 Agent / Map 分类 CE 的 none / non-none 组权重策略与标准 Focal Loss 超参。
- 输入：`mode`、`focal_gamma` 与组权重。
- 输出：不可变 dataclass，供 `train/losses.py` 构造分类 loss。
- 关键约束：`mode` 仅支持 `auto`、`manual`、`disabled`；`focal_gamma` 不能为负；组权重不能为负；同一检测分支的 none 与 non-none 权重不能同时为 0。

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
| `detection_class_weights.mode` | `auto` | 检测分类 CE 策略；`auto` / `manual` 为标准 Focal Loss，`disabled` 为 CE。 |
| `detection_class_weights.focal_gamma` | `2.0` | 标准 Focal Loss 的 ``γ``。 |
| `detection_class_weights.focal_alpha` | `0.25` | 标准 Focal Loss 的 ``α``；`auto` 模式下 none 类 ``α_t = 1 - focal_alpha``。 |
| `detection_class_weights.*_non_none_weight` | 见配置文件 | `manual` 模式下 Agent / Map 前景类 ``α_t``。 |
| `detection_class_weights.*_none_weight` | 见配置文件 | `manual` 模式下 Agent / Map none 类 ``α_t``。 |

## 7. 依赖关系

- 上游：`config/training.toml`。
- 下游：`train/trainer.py`、`train/losses.py`、`train/gradient_monitor.py`、`train/checkpointing.py`。
- 标准库：`dataclasses`、`pathlib`、`tomllib`、`typing`。

## 8. 注意事项

- 不要在实现文件中写入配置文件已有默认值。
- 新增训练配置字段时必须同步 dataclass、读取函数、配置文档和摘要文档。
- `DetectionClassWeightConfig` 只描述权重策略与 `focal_gamma`；标准 Focal Loss 实现在 `train/losses.py` 中。

## 9. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-09 | 1os3_Composer | AI 完成：`DetectionClassWeightConfig` 新增 `focal_alpha`，默认 0.25。 |
| 2026-06-08 | 1os3_Codex | AI 完成：`auto` 模式改为分组归一化 Focal Loss，移除梯度预算相关配置字段。 |
| 2026-06-08 | 1os3_Codex | AI 完成：新增检测分类 none / non-none 类别权重配置解析。 |
| 2026-06-08 | 1os3_Codex | AI 完成：新增训练主配置解析模块。 |
