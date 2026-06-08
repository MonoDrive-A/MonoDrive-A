# config/training_data.toml

## 1. 文件职责

`config/training_data.toml` 集中保存训练阶段数据读取、样本数值校验、轨迹词表标签构造和 Agent / Map 匈牙利匹配的配置。它只保存训练数据处理模块自己拥有的阈值和 cost 权重；词表规模、检测类别、query 数量、future 点数等既有配置继续由 `config/trajectory_vocab.toml` 和 `config/detection_head.toml` 管理。

本配置不启用危险轨迹判断。H5 未来 Agent 标注较稀疏时，不在训练数据处理阶段用不完整未来框去屏蔽轨迹词表候选。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `[dataset]` | TOML table | H5 输入、图像归一化、目标点采样和初始化扫描策略。 |
| `[modules]` | TOML table | 引用已有检测头和轨迹词表配置路径。 |
| `[validation]` | TOML table | 输入样本数值合理性阈值。 |
| `[trajectory_label]` | TOML table | 轨迹词表 soft label 构造参数。 |
| `[agent_matching]` | TOML table | Agent 匈牙利匹配 cost 权重。 |
| `[map_matching]` | TOML table | Map 匈牙利匹配 cost 权重和无方向类别。 |

## 3. 关键类和函数

本文件没有 Python 类或函数。它由 `train/data_processing.py` 中的 `load_training_data_config` 读取并解析为 `TrainingDataConfig`。

## 4. 输入输出与 Shape

| 名称 | Shape | 说明 |
| --- | --- | --- |
| H5 样本 | 见 `data/b2d_dataset.py` | 数据字段由已有 H5 Dataset 返回。 |
| 轨迹词表标签 | `[B, V]`、`[B, V, 6, 2]` | soft label 和 winner-only 残差标签。 |
| Agent 匹配目标 | `[B, 16]`、`[B, 16, 11]` | 分类、状态、mode 和 future 监督。 |
| Map 匹配目标 | `[B, 32]`、`[B, 32, 100, 2]` | 分类和局部地图点监督。 |

## 5. 关键实现逻辑

训练数据配置允许 H5 只读数据源位于项目目录外。`dataset.h5_dir` 可以是项目内相对路径，也可以是绝对路径；加载时展开当前目录下的 `*.h5`。`dataset.h5_paths` 同样允许混合项目内相对路径和绝对路径。检测头和轨迹词表配置路径仍必须解析到项目目录内。

样本校验阈值用于剔除 NaN、Inf 和明显越界数据。剔除发生在训练 Dataset 包装层，不修改 H5 文件。

轨迹词表 soft label 使用词表物理空间轨迹与 GT 轨迹之间的 MSE 构造，不做危险轨迹屏蔽。Agent 和 Map 匹配使用 SciPy Hungarian，并且 cost 在物理空间计算。

## 6. 配置项

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `dataset.h5_dir` | `data/preprocessed` | 逐场景 H5 目录，可为项目内相对路径或绝对路径。 |
| `dataset.normalize_images` | `true` | 图像是否归一化到 `[0, 1]`。 |
| `dataset.random_target_point` | `true` | 是否随机抽取有效目标点候选。 |
| `dataset.scan_on_init` | `true` | 初始化时是否扫描并剔除无效样本。 |
| `modules.detection_head_config_path` | `config/detection_head.toml` | 检测头配置引用。 |
| `modules.trajectory_vocab_config_path` | `config/trajectory_vocab.toml` | 轨迹词表配置引用。 |
| `validation.*` | 见配置文件 | 样本数值范围阈值。 |
| `trajectory_label.inverse_mse_eps` | `1e-6` | 轨迹 MSE 取倒数时的稳定项。 |
| `trajectory_label.inverse_mse_max_logit` | `10.0` | 倒数 MSE 归一化后的最大 logit。 |
| `agent_matching.*_cost_weight` | 见配置文件 | Agent 匹配 cost 权重。 |
| `map_matching.*_cost_weight` | 见配置文件 | Map 匹配 cost 权重。 |
| `map_matching.bidirectional_class_names` | `["lane_divider", "road_edge"]` | 点序正反等价的 Map 类别。 |

## 7. 依赖关系

- 上游：`data/b2d_dataset.py`、`config/detection_head.toml`、`config/trajectory_vocab.toml`。
- 下游：训练入口、loss 计算和训练可视化。
- 第三方：Agent / Map 匹配依赖 SciPy 的 `linear_sum_assignment`。

## 8. 注意事项

- 数值稳定性：轨迹 soft label 使用 `inverse_mse_eps` 防止除零。
- 坐标与单位：匹配 cost 使用 ego 坐标系物理空间，单位 meter。
- 数据路径：H5 是只读数据源，允许项目外绝对路径；训练输出路径不由本配置控制。
- 配置一致性：不得在实现文件重复写入 detection head、trajectory vocab 中已有默认值。
- 危险轨迹：本文件不提供危险轨迹判断配置，避免稀疏 H5 未来 Agent 标注导致错误屏蔽。

## 9. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-08 | 1os3_Codex | AI 完成：同步 Agent 16 / Map 32 匹配目标 shape。 |
| 2026-06-08 | 1os3_Codex | AI 完成：移除轨迹 soft label 最大值归一化配置，保持 soft label 为和为 1 的概率分布。 |
| 2026-06-07 | 1os3_Codex | AI 完成：新增训练数据处理配置，覆盖 H5 读取、样本校验、轨迹标签和 Agent / Map Hungarian 权重，并明确不启用危险轨迹判断。 |
| 2026-06-08 | 1os3_Codex | AI 完成：放开 H5 只读数据源路径说明，允许项目外绝对路径。 |
