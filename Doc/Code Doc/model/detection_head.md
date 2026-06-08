# model/detection_head.py

## 1. 文件职责

`model/detection_head.py` 实现模型侧检测查询初始化和检测解码头。它读取 `config/detection_head.toml`，生成 Agent / Map 检测查询 Token 初值，并用 FP32 线性层从 Transformer 输出的检测 Token 特征解码 Agent 和 Map 检测结果。

该文件不负责 Transformer 主干、Hungarian matching、检测 loss、分类 Softmax、反 Symlog、可见性过滤或数据预处理。连续输出保持模型监督空间，由训练或推理流程按任务需要解释。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `DetectionHeadConfig` | dataclass | 检测头配置对象，所有结构默认值来自 `config/detection_head.toml`。 |
| `DetectionDecoderOutput` | NamedTuple | 检测解码输出，包含 Agent 和 Map 的分类与连续预测。 |
| `DetectionQueryEmbedding` | class | 生成 `[48, 384]` 检测查询 Token 初值。 |
| `DetectionHeadDecoder` | class | 用 FP32 线性层解码 Agent 和 Map 输出。 |
| `load_detection_head_config` | function | 读取并校验 TOML 配置。 |

## 3. 关键类和函数

### `DetectionHeadConfig`

- 功能：承载检测查询、Agent 输出、Map 输出、初始化和精度配置。
- 输入：`config/detection_head.toml`。
- 输出：不可变配置对象。
- Shape：推导总检测查询数 48、Agent 输出维度 67、Map 输出维度 204。
- 关键约束：
  - Agent / Map 查询不按类别硬分配。
  - 空间采样数 `radial_count * angle_count` 必须等于对应查询数量。
  - Agent mode 角度必须等间隔，并且首尾对齐 Agent 查询角度范围。
  - 解码线性层当前只支持 FP32。

### `DetectionQueryEmbedding`

- 功能：根据空间 anchor 初始化检测查询 Token。
- 输入：无运行时输入；初始化时读取配置。
- 输出：`[48, 384]` FP32 检测查询 Token。
- Shape：
  - Agent anchor：`[16, 2]`。
  - Map anchor：`[32, 2]`。
  - 查询 Token：`[48, 384]`。
- 关键参数：`anchor_feature_order`、Agent / Map 空间角度范围和半径范围。

### `DetectionHeadDecoder`

- 功能：从检测 Token 特征解码 Agent 和 Map 输出。
- 输入：`detection_features`，shape 为 `[B, 48, 384]`。
- 输出：`DetectionDecoderOutput`。
- Shape：
  - Agent class logits：`[B, 16, 4]`。
  - Agent states：`[B, 16, 11]`。
  - Agent mode logits：`[B, 16, 4]`。
  - Agent future：`[B, 16, 4, 6, 2]`。
  - Map class logits：`[B, 32, 4]`。
  - Map points：`[B, 32, 100, 2]`。
- 关键参数：`agent_output_linear` 和 `map_output_linear` 均强制 FP32。

### `load_detection_head_config`

- 功能：读取 `config/detection_head.toml` 并解析为 `DetectionHeadConfig`。
- 输入：配置路径和可选项目根目录。
- 输出：`DetectionHeadConfig`。
- 约束：配置路径必须解析到项目目录内，所有表和字段均为必填。

## 4. 输入输出与 Shape

| 名称 | Shape | 说明 |
| --- | --- | --- |
| `DetectionQueryEmbedding.forward()` | `[48, 384]` | FP32 检测查询 Token。 |
| `detection_features` | `[B, 48, 384]` | Transformer 后的检测 Token 特征。 |
| `agent_class_logits` | `[B, 16, 4]` | Agent 分类未激活 logit，最后一类为“无”。 |
| `agent_states` | `[B, 16, 11]` | Agent 状态，字段顺序来自配置。 |
| `agent_mode_logits` | `[B, 16, 4]` | Agent future mode 未激活 logit。 |
| `agent_future_trajectories` | `[B, 16, 4, 6, 2]` | Agent future Symlog 空间位移。 |
| `map_class_logits` | `[B, 32, 4]` | Map 分类未激活 logit，最后一类为“无”。 |
| `map_points` | `[B, 32, 100, 2]` | Map 点 Symlog 空间 ego XY 预测。 |

## 5. 关键实现逻辑

`load_detection_head_config` 使用 `tomllib` 读取 TOML，并要求配置文件位于项目目录内。实现端不提供结构默认值；缺少表、字段、字段类型错误或 shape 约束不满足都会直接抛出异常。

`DetectionQueryEmbedding` 在初始化时分别构造 Agent 和 Map 空间 anchor。当前配置中 Agent 使用半径 4 档、角度 4 档形成 16 个空间位置；Map 使用半径 4 档、角度 8 档形成 32 个空间位置；两者角度范围均为 `[-60°, 60°]`。查询不绑定类别，只在前若干 hidden 通道写入空间和任务标记特征，包括 Symlog 后的 `x/y`、半径归一化、角度归一化、角度 sin/cos、Agent / Map 标记和查询进度。

`DetectionHeadDecoder` 使用两个任务分支线性层。Agent 分支输出分类、状态、mode logits 和 future；Map 分支输出分类和 Map 点。前向时会校验输入为浮点 `[B, 48, 384]`，随后禁用 autocast，把输入转为 FP32，再执行线性层。解码输出不做 Softmax、不做 Tanh、不做 Sigmoid、不做反 Symlog。

Agent 解码初始化利用查询 anchor 产生初始检测先验：`x/y` 输出读取查询 Token 中的 `x_symlog/y_symlog` 通道；`sin_yaw/cos_yaw` 读取查询角度；尺寸 bias 写入 `log1p(l/w/h)`；速度和加速度 bias 写入 Symlog 空间初值。4 个 mode 的 future bias 按配置角度和未来距离构造，使 mode 初始方向等间隔覆盖前方 120 度。

Map 解码初始化把每个 Map 查询的 100 个点都接到查询 anchor 的 Symlog 坐标，使初始 Map 点位于对应空间 anchor。Map 查询同样不绑定具体地图类别。

`DetectionQueryEmbedding` 和 `DetectionHeadDecoder` 都在初始化和 `_apply` 后调用 FP32 恢复逻辑。即使外层启用 BF16 autocast 或父模型整体 `.to(dtype=torch.bfloat16)`，检测查询参数、解码线性层参数、buffer 和输出仍会恢复为 `torch.float32`。

## 6. 配置项

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `query.*` | 由 `config/detection_head.toml` 提供 | 检测查询数量、hidden dim 和顺序。 |
| `query_embedding.*` | 由 `config/detection_head.toml` 提供 | 查询 Token 初始化特征。 |
| `agent.*` | 由 `config/detection_head.toml` 提供 | Agent 类别、状态字段和 future shape。 |
| `agent_query_initialization.*` | 由 `config/detection_head.toml` 提供 | Agent 空间 anchor 初始化。 |
| `agent_state_initialization.*` | 由 `config/detection_head.toml` 提供 | Agent 状态初始输出。 |
| `agent_decoder_initialization.*` | 由 `config/detection_head.toml` 提供 | Agent 分类和 mode logit 初值。 |
| `agent_mode_initialization.*` | 由 `config/detection_head.toml` 提供 | Agent 4-mode future 初始化。 |
| `map.*` | 由 `config/detection_head.toml` 提供 | Map 类别和点输出 shape。 |
| `map_query_initialization.*` | 由 `config/detection_head.toml` 提供 | Map 空间 anchor 初始化。 |
| `map_point_initialization.*` | 由 `config/detection_head.toml` 提供 | Map 点初始输出。 |
| `map_decoder_initialization.*` | 由 `config/detection_head.toml` 提供 | Map 分类 logit 初值。 |
| `precision.decoder_dtype` | 由 `config/detection_head.toml` 提供 | 解码线性层强制运行精度。 |

## 7. 依赖关系

- 上游：`config/detection_head.toml`、Transformer 主干输出的检测 Token 特征。
- 下游：检测 loss、Hungarian matching、训练日志、推理后处理。
- 项目内依赖：无。
- 第三方依赖：`torch`。
- 标准库依赖：`contextlib`、`dataclasses`、`math`、`pathlib`、`tomllib`、`typing`。

## 8. 注意事项

- 查询初始化：Agent 和 Map 查询不按类别硬分配；类别由解码分类 logit 学习。
- 坐标系：所有空间 anchor、Agent future 和 Map 点均使用当前帧 ego 坐标系，`x` 前向、`y` 左向。
- 数值空间：Agent future 和 Map 点输出为 Symlog 空间；检测解码头不做反 Symlog。
- 精度：检测查询和检测解码线性层强制 FP32。
- 初始化：4 个 Agent mode 的角度必须等间隔，且首尾对齐查询角度范围，当前覆盖前方 120 度。
- 配置：不要在实现文件内重复配置 TOML 中已有的结构默认值。

## 9. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-08 | 1os3_Codex | AI 完成：同步 Agent 16 个、Map 32 个检测查询和输出 shape。 |
| 2026-06-07 | 1os3_Codex | AI 完成：新增检测查询初始化和检测解码头，实现无类别硬分配查询、Agent 4-mode 120 度均匀初始化、FP32 解码线性层和模型空间输出。 |
