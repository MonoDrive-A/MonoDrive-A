# train/trainer.py

## 1. 文件职责

`train/trainer.py` 是 MonoDrive 训练主入口，负责读取配置、构建模型和数据集、执行前向与反向、计算 loss、监测梯度、更新优化器、记录日志、自动保存 checkpoint 和断点恢复。

该文件不实现模型结构、H5 读取细节、Hungarian matching 细节或单项 loss 公式；这些逻辑由上游模块提供。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `TrainingSummary` | dataclass | 训练完成后的摘要。 |
| `WarmupCosineLRScheduler` | class | 训练用学习率调度器。 |
| `run_training` | function | 按配置运行训练。 |

## 3. 关键类和函数

### `run_training`

- 功能：执行完整训练流程。
- 输入：训练配置路径和可选 `max_steps` 临时限制。
- 输出：`TrainingSummary`。
- Shape：训练 batch 中 `images [B, 8, 3, 288, 512]`、`target_point [B, 2]`、`ego_motion [B, 3]` 传入主干。

### `WarmupCosineLRScheduler`

- 功能：按 `initial_lr -> peak_lr -> min_lr` 执行 warmup 和末尾余弦退火。
- 输入：优化器和 `OptimizationConfig`。
- 输出：当前 step 学习率。

## 4. 输入输出与 Shape

| 名称 | Shape | 说明 |
| --- | --- | --- |
| `batch["images"]` | `[B, 8, 3, 288, 512]` | 前视 RGB 输入。 |
| `batch["target_point"]` | `[B, 2]` | ego 坐标系目标点，单位 meter。 |
| `batch["ego_motion"]` | `[B, 3]` | `[V_x, V_y, W]`。 |
| `MonoDriveBackboneOutput` | 见 `model/backbone.py` | 模型输出。 |
| `TrainingLossOutput.total_loss` | `[]` | 标量 FP32 loss。 |

## 5. 关键实现逻辑

训练入口先读取 `config/training.toml`，再读取主干和训练数据配置。Dataset 构建完成后，会打印样本数、batch size、`drop_last` 和预期每个 epoch 的 step 数。模型构建后会显式检查 DINOv3 是否冻结；优化器只接收 `requires_grad=True` 参数，因此冻结 DINOv3 不参与反向更新。

每步训练流程为：设置学习率、搬运 batch 到设备、模型前向、构造训练标签、计算 FP32 loss、反向、梯度监测、可选梯度裁剪、优化器更新、日志记录和按间隔保存 checkpoint。Loss 模块由 `loss_weights` 和 `detection_class_weights` 共同构造，其中后者控制 Agent / Map 分类 CE 的 none / non-none 类别权重策略。控制台与 JSONL metrics 会记录 Agent / Map 检测分类 CE 的 none / non-none 分项（`agent_ce_fg` / `agent_ce_bg`、`map_ce_fg` / `map_ce_bg`）。

断点恢复会加载模型、优化器、调度器、global step、epoch、batch index 和 RNG 状态。DataLoader 以 epoch seed 重建同一份样本顺序，并在构建首个恢复 epoch 的 sampler 时按 `batch_index * batch_size` 切掉已完成样本，避免重新读取和 collate 恢复点之前的 batch。

## 6. 配置项

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `config/training.toml` | 见配置文件 | 训练主流程配置。 |
| `config/backbone.toml` | 由训练配置引用 | 模型主干和精度配置。 |
| `config/training_data.toml` | 由训练配置引用 | H5 读取、标签构造和匹配配置。 |
| `detection_class_weights` | 见 `config/training.toml` | 传入 `MonoDriveTrainingLoss` 的检测分类 none / non-none 权重策略。 |

## 7. 依赖关系

- 上游：`model/backbone.py`、`train/data_processing.py`、`train/losses.py`、`train/training_config.py`、`train/gradient_monitor.py`、`train/checkpointing.py`。
- 下游：命令行训练、实验脚本。
- 第三方：`torch`、`numpy`。

## 8. 注意事项

- 模块级训练入口不额外开启全局 BF16 autocast，精度边界由模型子模块内部和配置控制。
- DINOv3 必须冻结；若发现 DINOv3 参数可训练，会直接抛出异常。
- 断点恢复首个 epoch 不应通过迭代 DataLoader 再 `continue` 跳过历史 batch，否则会重新读取 H5、执行校验和 collate，造成恢复阶段 CPU 与 I/O 压力。
- 检测分类自动类别权重在 loss 内按当前 batch 标签和 raw logits 的 CE 梯度范数计算，训练入口只负责传递配置。
- `--max-steps` 仅作为 smoke test 或临时调试覆盖，不改变 TOML 中的真实训练计划。

## 9. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-09 | 1os3_Composer | AI 完成：控制台训练日志新增 Agent / Map 检测分类 CE none / non-none 分项。 |
| 2026-06-08 | 1os3_Codex | AI 完成：同步检测分类自动权重改为 logits 梯度预算口径。 |
| 2026-06-08 | 1os3_Codex | AI 完成：训练入口向 loss 模块传入检测分类 none / non-none 类别权重配置。 |
| 2026-06-08 | 1os3_Codex | AI 完成：断点恢复时按 batch index 切分 sampler，避免读取并丢弃已完成 batch。 |
| 2026-06-08 | 1os3_Codex | AI 完成：新增训练主入口、学习率调度、自动保存和断点恢复。 |
| 2026-06-08 | 1os3_Codex | AI 完成：训练开始前打印数据集规模和预期单个 epoch step 数。 |
