# config/trajectory_vocab.toml

## 1. 文件职责

`config/trajectory_vocab.toml` 集中保存模型侧轨迹词表加载、嵌入层和解码层配置。该文件定义词表 `.npz` 路径、字段名、词表 shape、384 维轨迹特征、高频编码参数、SwiGLU 中间维度，以及解码层初始输出口径。

该文件不负责离线 FTS 词表构建，也不保存本机绝对路径、训练输出路径或实验临时覆盖项。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `[vocabulary]` | TOML table | 词表路径、字段名和 shape 约束。 |
| `[embedding]` | TOML table | 高频编码、SwiGLU 嵌入层和输出特征维度。 |
| `[decoder]` | TOML table | 解码层 logit 和残差初始输出口径。 |

## 3. 关键类和函数

### `[vocabulary]`

- 功能：指定模型侧加载的轨迹词表文件和 `.npz` 字段。
- 输入：无运行时输入，由 `model/trajectory_vocab/trajectory_vocab.py` 读取。
- 输出：`TrajectoryVocabModelConfig` 的词表相关字段。
- Shape：`[256, 6, 2]`。
- 关键参数：`path`、`normalized_key`、`num_trajectories`、`future_points`、`trajectory_dim`。

### `[embedding]`

- 功能：控制从归一化词表到轨迹查询特征的嵌入层。
- 输入：`.npz` 中 `trajectory_vocab_normalized`。
- 输出：`[256, 384]` 轨迹查询特征。
- Shape：高频编码展平维度为 `6 * 2 * 64 * 2 = 1536`。
- 关键参数：`hidden_dim`、`frequency_count`、`frequency_base=10.0`、`frequency_scale=2π`、`swiglu_hidden_dim`。

### `[decoder]`

- 功能：控制轨迹解码层的初始输出行为。
- 输入：`[B, 256, 384]` 轨迹 token 特征。
- 输出：logits `[B, 256]` 和残差 `[B, 256, 6, 2]`。
- Shape：单层线性层输出 `1 + 6 * 2 = 13` 个通道。
- 关键参数：`logit_init_value`、`residual_output_init_value`、`residual_activation`。

## 4. 输入输出与 Shape

| 名称 | Shape | 说明 |
| --- | --- | --- |
| `trajectory_vocab_normalized` | `[256, 6, 2]` | 模型嵌入层使用的已归一化轨迹词表。 |
| 高频编码 | `[256, 1536]` | 每个时间步按 `[phi_y(y), phi_x(x)]` 拼接，每个坐标 64 频，每个频率包含 sin/cos。 |
| 轨迹查询特征 | `[256, 384]` | 嵌入层输出，供 Transformer 轨迹查询使用。 |
| 解码输入 | `[B, 256, 384]` | Transformer 后的轨迹 token 特征。 |
| `logits` | `[B, 256]` | 轨迹词表概率的未激活 logit，初始输出为 1。 |
| `residuals` | `[B, 256, 6, 2]` | 经 Tanh 激活后的 Symlog 空间残差，初始输出为 0。 |

## 5. 关键实现逻辑

配置文件分为三个表。`[vocabulary]` 固定模型侧使用的 `.npz` 字段，嵌入层只读取 `normalized_key` 指向的已归一化字段。`[embedding]` 决定高频编码和两层线性嵌入结构；当前频带由 `frequency_scale / frequency_base ** (i / frequency_count)` 生成，对应 $2\pi / 10^{i/64}$。`[decoder]` 决定单层线性解码头的初始化：logit 通道权重为 0、bias 为 `logit_init_value`；残差通道权重为 0、bias 反解为 Tanh 前值，使 Tanh 后初始输出等于 `residual_output_init_value`。

相对路径按项目根目录解析。配置文件不得写入本机绝对路径，避免不同机器和运行目录之间出现不可复现实验。

## 6. 配置项

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `vocabulary.path` | `model/trajectory_vocab/trajectory_vocab_256.npz` | 项目内相对词表路径。 |
| `vocabulary.physical_key` | `trajectory_vocab_m` | 米制物理词表字段。 |
| `vocabulary.symlog_key` | `trajectory_vocab_symlog` | Symlog 词表字段。 |
| `vocabulary.normalized_key` | `trajectory_vocab_normalized` | 已归一化词表字段，模型嵌入层使用该字段。 |
| `vocabulary.scale_key` | `symlog_scale` | 共享 Symlog 缩放系数字段。 |
| `vocabulary.num_trajectories` | `256` | 词表轨迹数量。 |
| `vocabulary.future_points` | `6` | 每条轨迹未来点数，对应未来 3 秒 2Hz。 |
| `vocabulary.trajectory_dim` | `2` | 每个轨迹点的 ego XY 坐标维度。 |
| `embedding.hidden_dim` | `384` | 轨迹查询和解码输入特征维度。 |
| `embedding.frequency_count` | `64` | 每个归一化坐标的频率数量。 |
| `embedding.frequency_base` | `10.0` | 高频编码分母底数。 |
| `embedding.frequency_scale` | `6.283185307179586` | 高频编码角度前置系数，即 $2\pi$。 |
| `embedding.swiglu_hidden_dim` | `768` | SwiGLU 激活后的中间维度。 |
| `decoder.logit_init_value` | `1.0` | 解码层 logit 初始输出值。 |
| `decoder.residual_output_init_value` | `0.0` | Tanh 后残差初始输出值。 |
| `decoder.residual_activation` | `tanh` | 残差激活函数。 |

## 7. 依赖关系

- 上游：`model/trajectory_vocab/trajectory_vocab_256.npz`。
- 下游：`model/trajectory_vocab/trajectory_vocab.py`。
- 第三方依赖：Python 标准库 `tomllib` 读取 TOML，`numpy` 和 `torch` 由模型模块使用。

## 8. 注意事项

- 数值稳定性：`residual_output_init_value` 必须位于 `(-1, 1)`，否则无法反解 Tanh 前 bias。
- 性能：64 频高频编码在 FP32 中执行，输出维度随 `future_points`、`trajectory_dim` 和 `frequency_count` 线性增长。
- 兼容性：新增项目配置统一使用 TOML；不要在实现文件中重复写入本文件已有默认值。
- 路径：`vocabulary.path` 必须是项目内相对路径。

## 9. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-07 | 1os3_Codex | AI 完成：同步轨迹词表高频编码配置为 $2\pi / 10^{i/64}$ 形式。 |
| 2026-06-06 | 1os3_Codex | AI 完成：新增模型侧轨迹词表加载、嵌入和解码配置。 |
