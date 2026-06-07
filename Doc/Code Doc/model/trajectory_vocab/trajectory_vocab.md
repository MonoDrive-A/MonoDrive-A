# model/trajectory_vocab/trajectory_vocab.py

## 1. 文件职责

`model/trajectory_vocab/trajectory_vocab.py` 负责模型侧轨迹词表加载、轨迹查询嵌入和轨迹词表解码。该文件与 `trajectory_vocab_256.npz` 位于同一目录，从 TOML 配置读取所有可变参数，加载词表 `.npz`，使用 `.npz` 中的已归一化字段生成轨迹查询特征，并从 Transformer 输出的轨迹 token 解码轨迹词表 logit 和 Symlog 残差。

该文件不负责离线 FTS 词表构建、不生成 `.npz`、不计算 loss，也不在实现文件内重复定义配置文件中已有的默认值。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `TrajectoryVocabModelConfig` | dataclass | 模型侧轨迹词表配置对象，由 TOML 配置加载。 |
| `TrajectoryVocabData` | dataclass | 保存加载后的物理、Symlog、归一化词表和缩放系数。 |
| `TrajectoryDecoderOutput` | NamedTuple | 解码层输出，包含 logits 和 residuals。 |
| `TrajectoryVocabularyEmbedding` | class | 将已归一化词表编码为 `[256, 384]` 轨迹查询特征。 |
| `TrajectoryVocabularyDecoder` | class | 单层线性头解码 logits 和 Tanh 残差。 |
| `load_trajectory_vocab_config` | function | 读取 TOML 配置并构造 `TrajectoryVocabModelConfig`。 |
| `load_trajectory_vocabulary` | function | 从 `.npz` 加载并校验轨迹词表。 |

## 3. 关键类和函数

### `TrajectoryVocabModelConfig`

- 功能：承载配置文件中的词表、嵌入和解码参数。
- 输入：`config/trajectory_vocab.toml`。
- 输出：供加载、嵌入和解码模块使用的只读配置对象。
- Shape：`trajectory_shape` 为 `[256, 6, 2]`，`high_frequency_encoding_dim` 为 `1536`。
- 关键参数：`hidden_dim`、`frequency_count`、`swiglu_hidden_dim`、`logit_init_value`、`residual_output_init_value`。

### `TrajectoryVocabData`

- 功能：保存 `.npz` 中加载出的词表张量。
- 输入：`.npz` 字段 `trajectory_vocab_m`、`trajectory_vocab_symlog`、`trajectory_vocab_normalized`、`symlog_scale`。
- 输出：PyTorch FP32 张量。
- Shape：三个词表字段均为 `[256, 6, 2]`。
- 关键参数：由 `TrajectoryVocabModelConfig` 决定校验 shape。

### `TrajectoryVocabularyEmbedding`

- 功能：将归一化轨迹词表编码为轨迹查询 embedding。
- 输入：`TrajectoryVocabData.trajectory_vocab_normalized`。
- 输出：轨迹查询特征 `[256, 384]`。
- Shape：`[256, 6, 2] -> y/x 分别编码为 [256, 6, 128] -> [256, 6, 256] -> [256, 1536] -> [256, 384]`。
- 关键参数：`frequency_count`、`frequency_base`、`frequency_scale`、`swiglu_hidden_dim`、`hidden_dim`。

### `TrajectoryVocabularyDecoder`

- 功能：单层线性层从轨迹 token 特征输出轨迹词表 logit 和残差。
- 输入：`trajectory_features`，shape 为 `[B, 256, 384]`。
- 输出：`TrajectoryDecoderOutput`。
- Shape：`logits` 为 `[B, 256]`，`residuals` 为 `[B, 256, 6, 2]`。
- 关键参数：`logit_init_value=1.0`，`residual_output_init_value=0.0`，`residual_activation=tanh`。

### `load_trajectory_vocab_config`

- 功能：读取 TOML 配置，解析项目内相对路径，并校验字段类型。
- 输入：配置文件路径和可选项目根目录。
- 输出：`TrajectoryVocabModelConfig`。
- Shape：无张量输入输出。
- 关键参数：`config_path`、`project_root`。

### `load_trajectory_vocabulary`

- 功能：从 `.npz` 加载词表并校验字段、shape、NaN 和 Inf。
- 输入：`TrajectoryVocabModelConfig`。
- 输出：`TrajectoryVocabData`。
- Shape：加载三个 `[256, 6, 2]` 词表张量和一个标量 `symlog_scale`。
- 关键参数：`config.normalized_key` 指定模型嵌入层实际使用的归一化字段。

## 4. 输入输出与 Shape

| 名称 | Shape | 说明 |
| --- | --- | --- |
| `trajectory_vocab_m` | `[256, 6, 2]` | ego 坐标系米制词表，单位 meter。 |
| `trajectory_vocab_symlog` | `[256, 6, 2]` | Symlog 空间词表。 |
| `trajectory_vocab_normalized` | `[256, 6, 2]` | 已归一化词表，嵌入层使用该字段。 |
| `frequency_bands` | `[64]` | 高频编码频带，当前为 $2\pi / 10^{i/64}$。 |
| 高频编码特征 | `[256, 1536]` | 每个时间步按 `[phi_y(y), phi_x(x)]` 拼接，每坐标 64 个频率，每个频率 sin/cos 两项。 |
| `trajectory_queries` | `[256, 384]` | 轨迹查询特征。 |
| `trajectory_features` | `[B, 256, 384]` | Transformer 输出的轨迹 token 特征。 |
| `logits` | `[B, 256]` | 未激活轨迹词表 logit。 |
| `residuals` | `[B, 256, 6, 2]` | Tanh 后 Symlog 残差。 |

## 5. 关键实现逻辑

`load_trajectory_vocab_config` 使用 Python 标准库 `tomllib` 读取 TOML。配置中的词表路径必须是项目内相对路径，加载时解析到项目根目录内；若解析到项目目录外会直接报错，避免配置文件携带本机绝对路径。

`load_trajectory_vocabulary` 要求 `.npz` 同时包含物理词表、Symlog 词表、已归一化词表和 `symlog_scale`。三个词表字段必须与配置中的 `[V, K, D]` 一致，且不能包含 NaN 或 Inf。模型嵌入层只注册 `trajectory_vocab_normalized` 作为 buffer；物理和 Symlog 字段保留给后续 loss、推理或可视化使用。

`TrajectoryVocabularyEmbedding` 对已归一化轨迹执行高频编码。归一化轨迹最后一维按 ego XY 坐标解释，`x` 为前向、`y` 为左向。第 `i` 个频带为：

$$
\omega_i = \frac{2\pi}{10^{i/64}}
$$

代码中由 `frequency_scale / frequency_base ** (i / frequency_count)` 表示，当前配置为 `frequency_scale=2π`、`frequency_base=10`、`frequency_count=64`。单个坐标的编码顺序为 `[sin_0, cos_0, ..., sin_63, cos_63]`；每个时间步按 `[phi_y(y), phi_x(x)]` 拼接，再展平进入 `Linear -> SwiGLU -> Linear`，输出 384 维轨迹查询特征。SwiGLU 激活来自公共模块 `model/swiglu.py`，轨迹词表文件不再维护私有激活实现。

`TrajectoryVocabularyDecoder` 使用一个线性层输出 `1 + K * D` 个通道。第 0 个通道作为 logit，不做激活；剩余通道 reshape 为 `[B, V, K, D]` 并经过 Tanh 得到残差。初始化时线性层所有权重为 0，logit bias 为 1，使初始 logit 全部输出 1；残差 bias 设置为 `atanh(residual_output_init_value)`，当前配置下 Tanh 后初始残差为 0。

## 6. 配置项

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `vocabulary.path` | `model/trajectory_vocab/trajectory_vocab_256.npz` | 轨迹词表路径。 |
| `vocabulary.physical_key` | `trajectory_vocab_m` | 物理空间词表字段。 |
| `vocabulary.symlog_key` | `trajectory_vocab_symlog` | Symlog 词表字段。 |
| `vocabulary.normalized_key` | `trajectory_vocab_normalized` | 已归一化词表字段，嵌入层使用。 |
| `vocabulary.scale_key` | `symlog_scale` | 共享缩放系数字段。 |
| `vocabulary.num_trajectories` | `256` | 词表轨迹数量。 |
| `vocabulary.future_points` | `6` | 每条轨迹未来点数。 |
| `vocabulary.trajectory_dim` | `2` | 每个轨迹点坐标维度。 |
| `embedding.hidden_dim` | `384` | 轨迹查询和解码输入特征维度。 |
| `embedding.frequency_count` | `64` | 每坐标高频编码频率数量。 |
| `embedding.frequency_base` | `10.0` | 高频编码分母底数。 |
| `embedding.frequency_scale` | `6.283185307179586` | 高频编码角度前置系数，即 $2\pi$。 |
| `embedding.swiglu_hidden_dim` | `768` | SwiGLU 后中间特征维度。 |
| `decoder.logit_init_value` | `1.0` | logit 初始输出。 |
| `decoder.residual_output_init_value` | `0.0` | Tanh 后残差初始输出。 |
| `decoder.residual_activation` | `tanh` | 残差激活函数。 |

## 7. 依赖关系

- 上游：`config/trajectory_vocab.toml`、`model/trajectory_vocab/trajectory_vocab_256.npz`。
- 下游：Transformer 轨迹查询初始化、轨迹词表概率输出、Winner 残差回归监督。
- 项目内依赖：`model/swiglu.py`。
- 第三方依赖：`numpy`、`torch`。
- 标准库依赖：`dataclasses`、`math`、`pathlib`、`tomllib`、`typing`。

## 8. 注意事项

- 数值稳定性：高频编码和 `.npz` 加载均使用 FP32；残差输出经 Tanh 限制到 `[-1, 1]`。
- 初始化：logit 初始输出为 1，不代表已经 Softmax 后的概率为 1；所有词表项 logit 相同，Softmax 后为均匀分布。
- 配置：实现文件只读取配置，不重复定义配置文件已有默认值。
- 字段：嵌入层必须使用 `trajectory_vocab_normalized`，不能直接使用物理空间或 Symlog 字段替代。
- 维度：高频编码固定按 ego XY 二维轨迹执行，`trajectory_dim` 必须为 2。
- 路径：配置中的词表路径必须解析到项目目录内。

## 9. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-07 | 1os3_Codex | AI 完成：将轨迹词表高频编码改为 $2\pi / 10^{i/64}$ 频带，并按 `[phi_y(y), phi_x(x)]` 拼接。 |
| 2026-06-06 | 1os3_Codex | AI 完成：改为复用公共 `model/swiglu.py` 中的 `SwiGLU` 激活。 |
| 2026-06-06 | 1os3_Codex | AI 完成：将模型侧轨迹词表模块移动到 `model/trajectory_vocab/`，与词表 `.npz` 同目录。 |
| 2026-06-06 | 1os3_Codex | AI 完成：新增模型侧轨迹词表加载、归一化词表嵌入和单层线性解码模块。 |
