# train/data_processing.py

## 1. 文件职责

`train/data_processing.py` 负责训练阶段的数据读取包装、样本数值校验、轨迹词表标签构造，以及 Agent / Map 预测与 H5 标注之间的匈牙利匹配。该文件不重写 B2D H5 读取逻辑，而是直接复用 `data.b2d_dataset.B2DH5Dataset`。读取阶段遇到未通过数值校验的单个样本时，会向后寻找可替代有效样本，避免单个坏样本终止训练。

本文件不实现危险轨迹判断。H5 未来 Agent 标注稀疏时，训练数据处理不根据不完整未来框屏蔽规划词表候选。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `TrainingDataConfig` | dataclass | 训练数据处理配置对象。 |
| `ValidatedTrainingDataset` | class | 包装 H5 Dataset 并剔除无效样本。 |
| `TrajectoryVocabLabels` | NamedTuple | 轨迹词表 soft label 和 winner-only 残差标签。 |
| `AgentMatchingTargets` | NamedTuple | Agent 匹配和监督目标。 |
| `MapMatchingTargets` | NamedTuple | Map 匹配和监督目标。 |
| `TrainingBatchLabels` | NamedTuple | 单个 batch 的全部训练标签。 |
| `load_training_data_config` | function | 读取 `config/training_data.toml`。 |
| `build_training_dataset` | function | 构建经过校验的训练 Dataset。 |
| `training_collate` | function | 将样本列表合并为 batch。 |
| `build_training_batch_labels` | function | 从模型输出和 batch 构造全部训练标签。 |
| `build_trajectory_vocab_labels` | function | 构造轨迹词表标签。 |
| `build_agent_matching_targets` | function | 构造 Agent 匹配目标，并保留 winner mode 的逐点 future mask。 |
| `build_map_matching_targets` | function | 构造 Map 匹配目标，并对无方向类别沿用匹配时误差更小的点序。 |
| `symlog` / `inverse_symlog` | function | Symlog 正反变换。 |

## 3. 关键类和函数

### `ValidatedTrainingDataset`

- 功能：包装 `B2DH5Dataset`，在初始化或读取时剔除 NaN、Inf 和明显越界样本。
- 输入：底层 H5 Dataset 和 `TrainingDataConfig`。
- 输出：与 `B2DH5Dataset` 相同的样本字典。
- Shape：保持 H5 Dataset 的字段 shape，不改变样本内容。
- 读取行为：若当前索引样本失效，会在 `valid_indices` 内向后查找替代样本；只有找不到任何有效样本时才抛出异常并报告首个失败原因。

### `build_trajectory_vocab_labels`

- 功能：为规划词表构造 soft label、winner 索引和 winner-only 残差监督。
- 输入：`TrajectoryDecoderOutput`、`future_trajectory [B, 6, 2]`、轨迹词表数据和训练数据配置。
- 输出：`TrajectoryVocabLabels`。
- Shape：`soft_labels [B, V]`，`residual_targets [B, V, 6, 2]`。

### `build_agent_matching_targets`

- 功能：把 Agent 预测反变换到物理空间并执行 Hungarian matching。
- 输入：检测输出、batch、检测配置和训练数据配置。
- 输出：Agent 分类、状态、mode、future 目标和匹配索引。
- Shape：分类 `[B, 16]`，状态 `[B, 16, 11]`，future `[B, 16, 4, 6, 2]`，future mask `[B, 16, 4, 6]`。

### `build_map_matching_targets`

- 功能：把 Map 点预测反 Symlog 到物理空间并执行 Hungarian matching。
- 输入：检测输出、batch、检测配置和训练数据配置。
- 输出：Map 分类、点监督目标和匹配索引。
- Shape：分类 `[B, 32]`，点 `[B, 32, 100, 2]`。

## 4. 输入输出与 Shape

| 名称 | Shape | 说明 |
| --- | --- | --- |
| `images` | `[B, 8, 3, H, W]` | 训练输入图像，由 H5 Dataset 返回。 |
| `future_trajectory` | `[B, 6, 2]` | ego 坐标系未来规划 GT，单位 meter。 |
| `trajectory_output.logits` | `[B, V]` | 轨迹词表未激活 logit。 |
| `trajectory_output.residuals` | `[B, V, 6, 2]` | Symlog 空间残差。 |
| `trajectory_soft_labels` | `[B, V]` | 由物理空间 MSE 构造、和为 1 的 soft label。 |
| `agent_class_logits` | `[B, 16, C_agent + 1]` | Agent 分类预测。 |
| `agent_states` | `[B, 16, 11]` | Agent 监督空间状态预测。 |
| `agent_future_trajectories` | `[B, 16, 4, 6, 2]` | Agent future Symlog 空间预测。 |
| `agent_future_mask` | `[B, 16, 4, 6]` | 只在匹配 query 的 winner mode 和有效 future 点为真。 |
| `map_points` | `[B, 32, 100, 2]` | Map 点 Symlog 空间预测。 |

## 5. 关键实现逻辑

训练 Dataset 先调用 `B2DH5Dataset` 读取样本，再对图像、ego future、ego motion、目标点、Agent 和 Map 字段执行 finite 与范围校验。初始化扫描开启时，无效样本会从 `ValidatedTrainingDataset` 的索引中剔除，不写回 H5。读取阶段若仍遇到无效样本，则跳过该样本并返回后续有效样本；若整个索引集合都无效，异常信息会包含首个失败样本的具体校验原因。

轨迹词表标签构造中，词表物理轨迹与 GT 轨迹在 ego meter 空间计算 MSE。MSE 取倒数、归一化到最大 logit 后 softmax，得到和为 1 的 soft label。winner 取物理空间 MSE 最小的词表项，残差目标为 `(symlog(GT) - vocab_symlog[winner]) / symlog_scale`。模型 residual 反解到物理空间时使用 `inverse_symlog(vocab_symlog + residual * symlog_scale)`。

Agent 匹配中，模型输出先恢复为 FP32 物理空间：位置、速度、加速度和 future 使用反 Symlog；尺寸使用 `expm1`；yaw 使用 sin/cos 向量 cost。匹配完成后，监督目标再写回模型训练空间。Agent future mask 保留 H5 中 `[K]` 逐点有效性，只在匹配 query 的 winner mode 写入 `[K]` mask，避免 padding 未来点进入 loss。

Map 匹配中，预测点反 Symlog 到 ego meter 空间计算点误差。`lane_divider` 和 `road_edge` 按配置视为点序正反等价，cost 取正向和反向的较小值；`centerline` 保留方向。若无方向类别在匹配时反向点序误差更小，监督目标也会写入反向点序，保证 matching 和 loss 口径一致。

## 6. 配置项

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `config/training_data.toml` | 见配置文件 | 本文件读取的主配置。 |
| `config/detection_head.toml` | 由主配置引用 | 类别、query 数、状态字段顺序和 future 结构。 |
| `config/trajectory_vocab.toml` | 由主配置引用 | 词表路径、词表键名、词表规模和轨迹 shape。 |

## 7. 依赖关系

- 上游：`data.b2d_dataset.B2DH5Dataset`、`model.detection_head`、`model.trajectory_vocab`。
- 下游：训练入口和 loss 计算。
- 第三方：`torch`、`scipy.optimize.linear_sum_assignment`。

## 8. 注意事项

- 数值空间：Hungarian matching 和轨迹 MSE 均在 FP32 物理空间执行。
- 样本跳过：单个样本不通过校验不会终止训练；只有找不到任何可替代有效样本时才报错。
- 监督空间：匹配完成后的回归目标仍按模型输出空间构造，包括 Symlog、log1p 和 sin/cos。
- 稀疏 H5：本文件不进行危险轨迹判断，也不使用当前帧 Agent 外推未来碰撞。
- 路径口径：`dataset.h5_dir` 和 `dataset.h5_paths` 是只读数据源，允许使用项目外绝对路径；模块配置路径仍必须位于项目目录内。

## 9. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-08 | 1os3_Codex | AI 完成：同步 Agent 16 / Map 32 检测监督 shape。 |
| 2026-06-08 | 1os3_Codex | AI 完成：轨迹 residual 目标改为除以 `symlog_scale` 的归一化空间，反解时乘回 `symlog_scale`。 |
| 2026-06-07 | 1os3_Codex | AI 完成：新增训练数据处理模块，支持 H5 样本过滤、轨迹词表标签、Agent/Map Hungarian matching，并移除危险轨迹判断。 |
| 2026-06-08 | 1os3_Codex | AI 完成：将 Agent future mask 改为 winner mode 逐点 mask，并让 Map 无方向类别监督沿用匹配时误差更小的点序。 |
| 2026-06-08 | 1os3_Codex | AI 完成：放开 H5 只读数据源路径限制，允许 `h5_dir` 和 `h5_paths` 使用项目外绝对路径。 |
| 2026-06-08 | 1os3_Codex | AI 完成：样本校验增加失败原因，并在读取阶段跳过单个无效样本。 |
