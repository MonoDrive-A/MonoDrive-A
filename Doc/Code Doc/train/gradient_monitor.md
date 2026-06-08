# train/gradient_monitor.py

## 1. 文件职责

`train/gradient_monitor.py` 负责训练反向传播后的梯度范数监测，报告可训练参数中缺失梯度、过小梯度、过大梯度和非有限梯度。该文件不执行反向传播、梯度裁剪或参数更新。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `GradientParameterStat` | dataclass | 单个参数的梯度统计。 |
| `GradientMonitorResult` | NamedTuple | 一次梯度监测结果。 |
| `inspect_gradients` | function | 遍历可训练参数并统计梯度。 |

## 3. 关键类和函数

### `inspect_gradients`

- 功能：检查 `requires_grad=True` 的参数梯度。
- 输入：模型和 `GradientMonitorConfig`。
- 输出：`GradientMonitorResult`。
- Shape：不直接改变张量 shape；统计参数梯度的向量范数。

## 4. 输入输出与 Shape

| 名称 | Shape | 说明 |
| --- | --- | --- |
| `parameter.grad` | 与参数相同 | 被展开为向量计算 L2 范数。 |
| `GradientParameterStat.shape` | tuple | 原参数 shape。 |

## 5. 关键实现逻辑

监测函数只遍历可训练参数，因此冻结的 DINOv3 参数不会进入监测。对每个参数，若梯度缺失则按配置记录；若存在 NaN 或 Inf，则记录为非有限梯度；有限梯度会计算 L2 范数，并按阈值归入过小或过大列表。

## 6. 配置项

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `gradient_monitor.enabled` | 见 `config/training.toml` | 是否启用监测。 |
| `gradient_monitor.large_grad_norm` | 见 `config/training.toml` | 过大梯度阈值。 |
| `gradient_monitor.small_grad_norm` | 见 `config/training.toml` | 过小梯度阈值。 |
| `gradient_monitor.fail_on_nonfinite` | 见 `config/training.toml` | 非有限梯度是否中止训练。 |

## 7. 依赖关系

- 上游：训练循环反向传播后的模型。
- 下游：`train/trainer.py`。
- 第三方：`torch`。

## 8. 注意事项

- 该模块只监测，不裁剪梯度。
- 过小梯度包含 0 范数梯度，用于发现无学习信号或被 mask 掉的路径。
- 缺失梯度是否记录由配置控制。

## 9. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-08 | 1os3_Codex | AI 完成：新增梯度监测模块。 |
