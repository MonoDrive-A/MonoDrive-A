# train/losses.py

## 1. 文件职责

`train/losses.py` 汇总 MonoDrive 训练 loss，覆盖轨迹词表分数、轨迹残差、Agent 分类/状态/mode/future 和 Map 分类/点回归。Agent / Map 分类 CE 支持 none 与 non-none 组的类别权重，默认按 batch 标签分布自动调整。该文件不执行数据读取、Hungarian matching、优化器更新或 checkpoint 保存。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `TrainingLossOutput` | NamedTuple | 单个 batch 的总 loss 和分项 loss。 |
| `MonoDriveTrainingLoss` | class | 汇总全部训练 loss 的 `nn.Module`。 |

## 3. 关键类和函数

### `MonoDriveTrainingLoss`

- 功能：根据模型输出和训练标签计算加权总 loss。
- 输入：`MonoDriveBackboneOutput` 和 `TrainingBatchLabels`。
- 输出：`TrainingLossOutput`。
- Shape：总 loss 为标量；分项 loss 均为标量。
- 关键参数：`LossWeights` 和 `DetectionClassWeightConfig`，来自 `config/training.toml`。

## 4. 输入输出与 Shape

| 名称 | Shape | 说明 |
| --- | --- | --- |
| `trajectory_output.logits` | `[B, V]` | 轨迹词表 raw logits。 |
| `trajectory.soft_labels` | `[B, V]` | 和为 1 的轨迹词表 soft label。 |
| `trajectory_output.residuals` | `[B, V, 6, 2]` | Symlog 空间残差预测。 |
| `agent_class_logits` | `[B, 16, C_agent + 1]` | Agent 分类 raw logits。 |
| `agent_future_trajectories` | `[B, 16, 4, 6, 2]` | Agent future Symlog 空间预测。 |
| `agent.future_mask` | `[B, 16, 4, 6]` | winner mode 和有效未来点 mask。 |
| `map_points` | `[B, 32, 100, 2]` | Map 点 Symlog 空间预测。 |

## 5. 关键实现逻辑

轨迹词表分数使用 soft cross entropy，输入是模型 raw logits，目标是和为 1 的 soft label。实现对模型输出执行 `log_softmax`，再与 soft label 做逐类加权求和。

Agent / Map 分类使用 hard-label CE，并在 none / non-none 两组之间应用检测分类权重。`disabled` 模式等价于未加类别权重；`manual` 模式使用配置中的 `*_non_none_weight` 和 `*_none_weight`；`auto` 模式基于当前 batch 的 raw logits 估算未加权 CE 对 logits 的梯度范数，使 non-none 组获得 `auto_non_none_gradient_mass` 指定的组级梯度预算，none 组获得剩余预算，并用 `auto_min_weight` / `auto_max_weight` 限制极端值。若当前 batch 只有 none 或只有 non-none 目标，或某一组梯度范数为 0，则自动模式退回未加类别权重的 CE。

Agent mode 使用 PyTorch `cross_entropy`。该函数内部包含 `log_softmax`，调用方必须传入 raw logits。Agent mode 只在存在有效 future 点的匹配 query 上监督。

连续回归项使用 mask MSE。Agent future 的 mask 是 `[B, Q, M, K]`，只监督匹配 query 的 winner mode 和有效未来点；Map 点只监督匹配 query。

## 6. 配置项

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `loss_weights.trajectory_logit_soft_ce` | 见 `config/training.toml` | 轨迹词表分数 soft CE 权重。 |
| `loss_weights.trajectory_residual_mse` | 见 `config/training.toml` | winner 轨迹残差 MSE 权重。 |
| `loss_weights.agent_*` | 见 `config/training.toml` | Agent 分类、状态、mode、future 权重。 |
| `loss_weights.map_*` | 见 `config/training.toml` | Map 分类和点回归权重。 |
| `detection_class_weights.mode` | `auto` | Agent / Map 分类 CE 的 none / non-none 权重模式，支持 `auto`、`manual`、`disabled`。 |
| `detection_class_weights.agent_non_none_weight` | 见 `config/training.toml` | 手动模式下 Agent 前景类权重。 |
| `detection_class_weights.agent_none_weight` | 见 `config/training.toml` | 手动模式下 Agent none 类权重。 |
| `detection_class_weights.map_non_none_weight` | 见 `config/training.toml` | 手动模式下 Map 前景类权重。 |
| `detection_class_weights.map_none_weight` | 见 `config/training.toml` | 手动模式下 Map none 类权重。 |
| `detection_class_weights.auto_min_weight` | 见 `config/training.toml` | 自动权重下限。 |
| `detection_class_weights.auto_max_weight` | 见 `config/training.toml` | 自动权重上限。 |
| `detection_class_weights.auto_non_none_gradient_mass` | 见 `config/training.toml` | 自动模式下 non-none 组目标 logits 梯度预算比例。 |

## 7. 依赖关系

- 上游：`model/backbone.py`、`train/data_processing.py`、`train/training_config.py`。
- 下游：`train/trainer.py`。
- 第三方：`torch`。

## 8. 注意事项

- 所有 loss 计算都转为 FP32。
- 轨迹词表概率分数使用 soft CE，模型输出在 loss 内部做 `log_softmax`。
- Agent / Map 分类 none 类下标沿用检测标签约定，位于最后一个分类通道。
- 自动检测分类权重使用当前 batch 中已构造的分类目标和模型 raw logits，不读取或写入额外统计文件；权重计算使用 `logits.detach()`，权重本身不进入反向图。
- 修改 mask shape 或监督空间时，需要同步本文件和 `train/data_processing.py` 文档。
- Agent mode CE 依赖 `future_mask.any(dim=(2, 3))`，没有有效 future 点的匹配 Agent 不参与 mode 监督。

## 9. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-08 | 1os3_Codex | AI 完成：同步 Agent 16 / Map 32 loss 输入 shape。 |
| 2026-06-08 | 1os3_Codex | AI 完成：自动检测分类权重改为按当前 logits CE 梯度预算调整。 |
| 2026-06-08 | 1os3_Codex | AI 完成：Agent / Map 分类 CE 新增 none 与 non-none 类别权重，默认按 batch 自动调整。 |
| 2026-06-08 | 1os3_Codex | AI 完成：轨迹词表分数 loss 从 BCEWithLogits 改为 soft CE。 |
| 2026-06-08 | 1os3_Codex | AI 完成：新增训练 loss 汇总模块。 |
