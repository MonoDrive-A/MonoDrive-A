# config/detection_head.toml

## 1. 文件职责

`config/detection_head.toml` 集中保存模型侧检测查询初始化和检测解码头配置。它定义 48 个检测查询的组织方式、查询 Token 初始 anchor 特征、Agent / Map 输出字段、Agent 4 个 future mode 的 120 度均匀初始化、解码线性层初始 logit 和强制 FP32 精度。

该文件不保存本机绝对路径、训练输出路径或实验临时覆盖项。实现文件只读取并校验本文件，不在 Python 实现中重复写入本文件已有默认值。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `[query]` | TOML table | 检测查询数量、hidden dim 和序列顺序。 |
| `[query_embedding]` | TOML table | 写入检测查询初值的 anchor 特征顺序和未占用通道初值。 |
| `[agent]` | TOML table | Agent 类别、状态字段、future mode 和未来点配置。 |
| `[agent_query_initialization]` | TOML table | Agent 查询在前方 120 度区域内的空间 anchor 采样。 |
| `[agent_state_initialization]` | TOML table | Agent 状态输出的初始来源和物理初值。 |
| `[agent_decoder_initialization]` | TOML table | Agent 分类和 mode logit 初值。 |
| `[agent_mode_initialization]` | TOML table | 4 个 Agent future mode 的角度和未来距离初始化。 |
| `[map]` | TOML table | Map 类别、点数和点维度。 |
| `[map_query_initialization]` | TOML table | Map 查询在前方 120 度区域内的空间 anchor 采样。 |
| `[map_point_initialization]` | TOML table | Map 点输出的初始来源和数值空间。 |
| `[map_decoder_initialization]` | TOML table | Map 分类 logit 初值。 |
| `[precision]` | TOML table | 检测解码线性层强制运行精度。 |

## 3. 关键类和函数

本文件没有 Python 类或函数。它由 `model/detection_head.py` 中的 `load_detection_head_config` 读取，并解析为 `DetectionHeadConfig`。

### `[query]`

- 功能：声明检测查询总组织。
- 输入：无运行时输入。
- 输出：`DetectionHeadConfig` 的基础 query 字段。
- Shape：Agent 查询 16 个、Map 查询 32 个，总计 48 个，每个 384 维。

### `[query_embedding]`

- 功能：声明查询 Token 初始化时写入 hidden 前若干通道的 anchor 特征。
- 输入：由 Agent / Map 空间采样得到的 ego XY anchor。
- 输出：`[48, 384]` 检测查询初值。
- Shape：当前 9 个 anchor 特征写入前 9 个 hidden 通道，其余通道为 `unfilled_value`。

### `[agent]`

- 功能：声明 Agent 检测输出结构。
- 输入：Transformer 后的 Agent Token 特征。
- 输出：Agent 类别、状态、mode logits 和 4-mode future。
- Shape：类别为 3 个前景类加“无”类别；状态 11 维；future 为 `[16, 4, 6, 2]`。

### `[agent_mode_initialization]`

- 功能：声明 Agent 4 个 mode 的特殊初始化。
- 输入：mode 角度和每个未来点的距离。
- 输出：解码线性层 future bias。
- Shape：`mode_angles_deg` 长度为 4，`future_distances_m` 长度为 6，生成 `[4, 6, 2]` Symlog 空间位移模板。

### `[map]`

- 功能：声明 Map 检测输出结构。
- 输入：Transformer 后的 Map Token 特征。
- 输出：Map 类别和局部 Map 点。
- Shape：类别为 3 个前景类加“无”类别；点输出为 `[32, 100, 2]`。

## 4. 输入输出与 Shape

| 名称 | Shape | 说明 |
| --- | --- | --- |
| 检测查询 Token | `[48, 384]` | 前 16 个为 Agent，后 32 个为 Map。 |
| Agent anchor | `[16, 2]` | ego 坐标系空间 anchor，前方 120 度内均匀采样。 |
| Map anchor | `[32, 2]` | ego 坐标系空间 anchor，前方 120 度内均匀采样。 |
| Agent class logits | `[B, 16, 4]` | `car/bicycle/pedestrian/none` 未激活 logit。 |
| Agent states | `[B, 16, 11]` | `[x, y, length_log1p, width_log1p, height_log1p, sin_yaw, cos_yaw, vx, vy, ax, ay]`。 |
| Agent mode logits | `[B, 16, 4]` | 4 个 Agent future mode 的未激活 logit。 |
| Agent future | `[B, 16, 4, 6, 2]` | Symlog 空间 future 位移预测，不做反 Symlog。 |
| Map class logits | `[B, 32, 4]` | `lane_divider/road_edge/centerline/none` 未激活 logit。 |
| Map points | `[B, 32, 100, 2]` | Symlog 空间 ego XY Map 点预测，不做反 Symlog。 |

## 5. 关键实现逻辑

Agent 查询和 Map 查询都不按类别做硬性数量分配。查询初始化只按空间 anchor 均匀覆盖前方 120 度区域；类别预测完全由解码线性层输出的分类 logit 决定。

Agent / Map 空间 anchor 由 `angle_min_deg`、`angle_max_deg`、`radial_min_m`、`radial_max_m`、`radial_count`、`angle_count` 和 `spatial_order` 决定。当前 Agent 使用 `4 * 4 = 16` 个 anchor，Map 使用 `4 * 8 = 32` 个 anchor，正好对应各自查询数。

查询 Token 初值通过 `[query_embedding].anchor_feature_order` 写入前若干 hidden 通道。当前特征包括 `x_symlog`、`y_symlog`、半径归一化、角度归一化、角度 sin/cos、任务标记和查询进度；其余 hidden 通道使用 `unfilled_value`。

Agent 解码初始化中，`x/y` 状态从查询 anchor 的 Symlog 坐标读出，`sin_yaw/cos_yaw` 从查询角度读出，尺寸按 `log1p` 物理长宽高写入 bias，速度和加速度按 Symlog 写入 bias。4 个 mode 的 future bias 由配置角度和距离生成，每个未来点按 ego 坐标 `[x, y] = [d cos(theta), d sin(theta)]` 构造，再做 Symlog 变换。

Map 解码初始化中，Map 点输出从查询 anchor 的 Symlog 坐标读出，使每个 Map 查询初始点位于其空间 anchor 附近。Map 查询不绑定具体地图类别。

## 6. 配置项

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `query.hidden_dim` | `384` | 检测 Token 特征维度。 |
| `query.agent_query_count` | `16` | Agent 查询数量，不按类别硬分配。 |
| `query.map_query_count` | `32` | Map 查询数量，不按类别硬分配。 |
| `query.token_order` | `agent_then_map` | 检测查询序列顺序。 |
| `query_embedding.anchor_feature_order` | 见配置文件 | 查询初值写入的 anchor 特征。 |
| `query_embedding.unfilled_value` | `0.0` | 未占用 hidden 通道初值。 |
| `agent.class_names` | `["car", "bicycle", "pedestrian"]` | Agent 前景类别。 |
| `agent.none_class_name` | `none` | Agent “无”类别名。 |
| `agent.state_order` | 见配置文件 | Agent 状态输出字段顺序。 |
| `agent.future_mode_count` | `4` | Agent future mode 数。 |
| `agent.future_points` | `6` | 每个 mode 的未来点数，对应未来 3 秒 2Hz。 |
| `agent.trajectory_dim` | `2` | Agent future 位移维度。 |
| `agent_query_initialization.*` | 见配置文件 | Agent 空间 anchor 范围和采样数。 |
| `agent_state_initialization.*` | 见配置文件 | Agent 状态输出初始来源和物理初值。 |
| `agent_decoder_initialization.*` | 见配置文件 | Agent 分类和 mode logit 初值。 |
| `agent_mode_initialization.mode_angles_deg` | `[-60.0, -20.0, 20.0, 60.0]` | 4 个 mode 在 120 度区域内等间隔散布。 |
| `agent_mode_initialization.future_distances_m` | `[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]` | 每个未来点的初始距离。 |
| `agent_mode_initialization.future_transform` | `symlog` | Agent future 初始输出空间。 |
| `map.class_names` | `["lane_divider", "road_edge", "centerline"]` | Map 前景类别。 |
| `map.none_class_name` | `none` | Map “无”类别名。 |
| `map.point_count` | `100` | 每条 Map 元素点数。 |
| `map.point_dim` | `2` | Map 点坐标维度。 |
| `map_query_initialization.*` | 见配置文件 | Map 空间 anchor 范围和采样数。 |
| `map_point_initialization.*` | 见配置文件 | Map 点输出初始来源和数值空间。 |
| `map_decoder_initialization.*` | 见配置文件 | Map 分类 logit 初值。 |
| `precision.decoder_dtype` | `float32` | 检测解码线性层强制运行精度。 |

## 7. 依赖关系

- 下游读取端：`model/detection_head.py`。
- 上游设计：`Doc/Model.md` 的 Token 序列、检测任务和精度策略。
- 下游模块：Transformer 主干装配、检测 loss、Hungarian matching 和推理后处理。

## 8. 注意事项

- 查询不按类别硬分配；不要把 Agent 或 Map 查询固定绑定到某个类别。
- Agent 4 个 mode 的角度必须等间隔，且首尾对齐 Agent 查询角度范围。
- Agent future 和 Map 点输出保持 Symlog 空间，不在检测解码头内做反 Symlog。
- 解码线性层强制 FP32；修改精度配置必须同步更新模型设计文档和实现校验。
- 修改配置字段、输出 shape 或初始化口径时，必须同步更新模型文件 Code Doc 和 `Doc/Code Doc/Index.md`。

## 9. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-08 | 1os3_Codex | AI 完成：同步 Agent 16 个、Map 32 个检测查询和对应 anchor 网格配置。 |
| 2026-06-07 | 1os3_Codex | AI 完成：新增检测头配置，记录 96 个检测查询、无类别硬分配、Agent 4-mode 120 度均匀初始化和 FP32 解码精度。 |
