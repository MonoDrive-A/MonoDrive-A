# visualization/target_point_embedding_viewer.py

## 1. 文件职责

`visualization/target_point_embedding_viewer.py` 负责可视化目标点嵌入策略。用户在命令行指定 ego 坐标系目标点 `[x, y]` 后，脚本加载 `config/target_point_embedding.toml`，直接实例化并调用 `model.target_point_embedding.TargetPointEmbedding`，导出目标点 `18x16` Symlog 栅格向量场、x/y 分量热力图、`9x8` 卷积输出 norm 和 2 个目标导航点 Token 统计。

该文件不复制目标点栅格构造、向量场构造、卷积或线性投影逻辑；中间结果从真实模型模块的 buffer、方法和层取得，用于避免训练实现与可视化实现不一致。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `TargetPointEmbeddingVisualizationData` | dataclass | 渲染 PNG 所需的目标点、栅格、向量场、卷积特征和 Token。 |
| `run_target_point_embedding` | function | 调用真实目标点嵌入层并收集中间可视化数据。 |
| `render_target_point_embedding` | function | 运行嵌入层并导出 PNG。 |
| `render_visualization` | function | 把可视化数据渲染为 PIL 图像。 |
| `main` | function | 命令行入口。 |

## 3. 关键类和函数

### `TargetPointEmbeddingVisualizationData`

- 功能：保存目标点嵌入诊断图需要的数据。
- 输入：由 `run_target_point_embedding` 构造。
- 输出：供 `render_visualization` 使用。
- Shape：
  - `target_point`: `[2]`。
  - `grid_xy`: `[18, 16, 2]`。
  - `meter_vector_field`: `[18, 16, 2]`。
  - `symlog_vector_field`: `[18, 16, 2]`。
  - `embedded_feature_norm`: `[9, 8]`。
  - `tokens`: `[2, 384]`。

### `run_target_point_embedding`

- 功能：加载配置、实例化 `TargetPointEmbedding`，并通过真实模块生成目标点 Token 和中间特征。
- 输入：目标点 `target_x/target_y`、配置路径、项目根目录和设备。
- 输出：`TargetPointEmbeddingVisualizationData`。
- 约束：目标点坐标必须为有限数；配置路径必须位于项目目录内。

### `render_target_point_embedding`

- 功能：运行目标点嵌入层并保存 PNG。
- 输入：目标点、配置路径、输出路径、项目根目录和设备。
- 输出：PNG 输出路径。
- 约束：输出路径必须位于项目目录内，避免写出项目目录。

### `render_visualization`

- 功能：绘制四类诊断内容：
  - `18x16` Symlog 目标点向量场和箭头；
  - Symlog 向量场 x 分量、y 分量和 norm 热力图；
  - `9x8` 卷积输出 norm 热力图；
  - 2 个目标导航点 Token 的 norm、mean、std 和前若干维柱状图。
- 输入：`TargetPointEmbeddingVisualizationData`。
- 输出：PIL `Image`。

## 4. 输入输出与 Shape

| 名称 | Shape / 类型 | 说明 |
| --- | --- | --- |
| 命令行 `--x/--y` | scalar float | ego 坐标系目标点，`x` 前向、`y` 左向，单位 meter。 |
| `target_points` | `[1, 2] float32` | 送入真实 `TargetPointEmbedding.forward` 的输入。 |
| `grid_xy` | `[18, 16, 2] float32` | 来自目标点嵌入层的 buffer。 |
| `meter_vector_field` | `[18, 16, 2] float32` | 通过真实模块 `_build_meter_vector_field` 生成，单位 meter。 |
| `vector_features` | `[1, 2, 18, 16] float32` | 通过真实模块 `_build_vector_features` 生成，已做 Symlog。 |
| `symlog_vector_field` | `[18, 16, 2] float32` | 渲染用 channel-last Symlog 向量场。 |
| `embedded_features` | `[1, 16, 9, 8] float32` | 使用真实 `conv1 -> conv2 -> downsample` 计算。 |
| `embedded_feature_norm` | `[9, 8] float32` | 卷积输出通道维 L2 norm。 |
| `tokens` | `[2, 384] float32` | 真实 `TargetPointEmbedding.forward` 输出。 |
| 输出 PNG | image file | 目标点嵌入策略诊断图。 |

## 5. 关键实现逻辑

命令行入口要求用户提供 `--x` 和 `--y`。脚本默认读取 `config/target_point_embedding.toml`，默认输出到 `visualization/outputs/target_point_embedding/`，并且会校验配置路径和输出路径都位于项目目录内。

`run_target_point_embedding` 先调用 `load_target_point_embedding_config` 读取配置，再实例化 `TargetPointEmbedding`。目标点 Token 通过 `module(target_points)` 获取；栅格来自 `module.grid_xy`；米制向量场通过 `module._build_meter_vector_field(target_points)` 获取；Symlog 卷积输入通过 `module._build_vector_features(target_points)` 获取；卷积下采样特征通过同一个模块实例的 `conv1`、`conv2` 和 `downsample` 获取。这样可视化图展示的是当前真实模型实现，而不是脚本内的复制逻辑。

渲染阶段只做图像表达：BEV 面板用颜色表示 Symlog 向量 norm，并在每个栅格内绘制 Symlog 向量短箭头；右侧热力图展示 Symlog 向量 x/y 分量、Symlog 向量 norm 和卷积输出 norm；底部面板展示配置、FP32 dtype 和输出 Token 统计。

## 6. 配置项

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `--x` | 无 | 必填，目标点 ego x 坐标，单位 meter，前向为正。 |
| `--y` | 无 | 必填，目标点 ego y 坐标，单位 meter，左向为正。 |
| `--config` | `config/target_point_embedding.toml` | 目标点嵌入层配置路径。 |
| `--output` | 可选 | 输出 PNG 路径；必须位于项目目录内。 |
| `--output-dir` | `visualization/outputs/target_point_embedding` | 未指定 `--output` 时的输出目录；必须位于项目目录内。 |
| `--device` | `cpu` | 运行设备，例如 `cpu` 或 `cuda`。 |

命令行默认值只影响可视化入口；模型结构默认值仍以 `config/target_point_embedding.toml` 为准。

## 7. 依赖关系

- 核心实现：`model/target_point_embedding.py`。
- 配置：`config/target_point_embedding.toml`。
- 输出目录：默认写入 `visualization/outputs/target_point_embedding/`，该目录位于项目内并被 `.gitignore` 忽略。
- 第三方依赖：`torch`、`numpy`、`PIL`。

## 8. 注意事项

- 坐标系：命令行坐标必须是当前帧 ego 坐标系，`x` 前向、`y` 左向，单位 meter。
- 数值空间：可视化主面板和热力图显示的是送入卷积的 Symlog 向量场，不是原始 meter 向量。
- 一致性：不要在脚本中复制目标点嵌入逻辑；需要中间特征时继续从 `TargetPointEmbedding` 实例取得。
- 路径：脚本会拒绝项目目录外的配置路径和输出路径。
- 精度：可视化数据来自强制 FP32 的目标点嵌入层，图中会显示参数、buffer 和输出 Token dtype。
- 维护：修改命令行参数、输出面板或中间特征口径时，必须同步更新摘要文档和 `Doc/Code Doc/Index.md`。

## 9. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-07 | 1os3_Codex | AI 完成：将可视化主图和热力图改为展示真实送入卷积的 Symlog 向量场。 |
| 2026-06-07 | 1os3_Codex | AI 完成：新增目标点嵌入策略可视化工具文档，支持命令行坐标并直接调用真实目标点嵌入实现。 |
