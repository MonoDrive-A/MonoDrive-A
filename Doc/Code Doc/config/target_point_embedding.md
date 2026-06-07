# config/target_point_embedding.toml

## 1. 文件职责

`config/target_point_embedding.toml` 集中保存目标点嵌入层配置，包括目标点坐标维度、栅格覆盖范围、送入卷积前的 Symlog 向量变换、三层卷积结构、下采样输出尺寸、线性投影输出 Token 数和强制 FP32 精度。

该文件不保存本机绝对路径、训练输出路径、缓存路径或实验临时覆盖项。实现文件只读取并校验本文件中的字段，不在 Python 实现中重复写入这些结构默认值。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `[target_point]` | TOML table | 目标点输入坐标维度。 |
| `[grid]` | TOML table | 目标点向量场栅格尺寸、覆盖范围和向量方向。 |
| `[normalization]` | TOML table | 米制向量场送入卷积前的数值空间变换。 |
| `[convolution]` | TOML table | 三层卷积的通道数、kernel、stride、padding 和下采样后尺寸。 |
| `[output]` | TOML table | 线性投影后的目标导航点 Token 数和隐藏维度。 |
| `[precision]` | TOML table | 目标点嵌入层强制运行精度。 |

## 3. 关键类和函数

本文件没有 Python 类或函数。它由 `model/target_point_embedding.py` 中的 `load_target_point_embedding_config` 读取，并解析为 `TargetPointEmbeddingConfig`。

### `[target_point]`

- 功能：声明输入目标点坐标维度。
- 输入：`target_points` 的最后一维。
- 输出：`TargetPointEmbeddingConfig.coordinate_dim`。
- Shape：`[B, 2]`。

### `[grid]`

- 功能：声明目标点向量场的空间范围和方向。
- 输入：ego 坐标系米制目标点 `[x, y]`。
- 输出：`[B, 18, 16, 2]` 向量场。
- Shape：栅格覆盖 `x ∈ [-4m, 32m]`、`y ∈ [-32m, 32m]`。

### `[normalization]`

- 功能：声明目标点米制向量场送入卷积前的变换。
- 输入：`[B, 18, 16, 2]` 米制向量场。
- 输出：`[B, 18, 16, 2]` Symlog 空间向量场。
- Shape：shape 不变，逐坐标计算 $Sign(x) \times \ln(|x|+1)$。

### `[convolution]`

- 功能：声明 1 层 `1x1`、1 层 `3x3` 和 1 层 `2x2` 下采样卷积。
- 输入：`[B, 2, 18, 16]`。
- 输出：`[B, 16, 9, 8]`。
- Shape：下采样后展平维度为 `16 * 9 * 8 = 1152`。

### `[output]`

- 功能：声明展平后线性层投影口径。
- 输入：`[B, 1152]`。
- 输出：`[B, 2, 384]`。
- Shape：线性层直接输出 `2 * 384 = 768` 个通道，再 reshape 为目标导航点 Token。

## 4. 输入输出与 Shape

| 名称 | Shape | 说明 |
| --- | --- | --- |
| `target_points` | `[B, 2]` | ego 坐标系米制目标点，坐标为 `[x, y]`。 |
| `grid_xy` | `[18, 16, 2]` | 每个栅格中心的 ego XY 坐标，`x` 前向、`y` 左向。 |
| 米制目标点向量场 | `[B, 18, 16, 2]` | 当前配置为 `grid_xy - target_point`，单位 meter。 |
| Symlog 向量场 | `[B, 18, 16, 2]` | 对米制向量场逐坐标做 Symlog 变换。 |
| 卷积输入 | `[B, 2, 18, 16]` | Symlog 向量场转为 channel-first。 |
| 下采样特征 | `[B, 16, 9, 8]` | 三层卷积后的目标点局部特征。 |
| 展平特征 | `[B, 1152]` | 按 `channel_height_width` 展平。 |
| 目标导航点 Token | `[B, 2, 384]` | 线性层投影后的输出。 |

## 5. 关键实现逻辑

配置读取端要求所有表和字段显式存在。`[grid]` 中的范围用于生成栅格中心坐标；当前 `vector_order = "grid_minus_target"` 表示目标点到每个栅格中心的相对偏移按 `grid_xy - target_point` 计算。`[normalization].vector_transform = "symlog"` 表示该米制向量场必须逐坐标变换为 Symlog 空间后再送入卷积。

卷积结构完全由 `[convolution]` 控制。读取配置时会根据 kernel、stride 和 padding 推导三层卷积后的空间尺寸，并与 `output_height/output_width` 校验，避免配置与实现 shape 口径不一致。

`[precision].dtype` 当前只允许 `float32`。模型模块会在初始化、设备或 dtype 迁移后把所有浮点参数、buffer 和已有梯度恢复为 FP32，并在前向中禁用 autocast。

## 6. 配置项

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `target_point.coordinate_dim` | `2` | ego XY 目标点坐标维度。 |
| `grid.height` | `18` | 栅格 x 方向数量。 |
| `grid.width` | `16` | 栅格 y 方向数量。 |
| `grid.x_min_m` | `-4.0` | 后向覆盖边界，单位 meter。 |
| `grid.x_max_m` | `32.0` | 前向覆盖边界，单位 meter。 |
| `grid.y_min_m` | `-32.0` | 右向覆盖边界，单位 meter。 |
| `grid.y_max_m` | `32.0` | 左向覆盖边界，单位 meter。 |
| `grid.vector_order` | `grid_minus_target` | 目标点向量场方向。 |
| `normalization.vector_transform` | `symlog` | 米制向量场送入卷积前的逐坐标变换。 |
| `convolution.feature_channels` | `16` | 三层卷积中间通道数。 |
| `convolution.conv1_kernel_size` | `[1, 1]` | 第一层卷积核。 |
| `convolution.conv1_stride` | `[1, 1]` | 第一层卷积步长。 |
| `convolution.conv1_padding` | `[0, 0]` | 第一层卷积 padding。 |
| `convolution.conv2_kernel_size` | `[3, 3]` | 第二层卷积核。 |
| `convolution.conv2_stride` | `[1, 1]` | 第二层卷积步长。 |
| `convolution.conv2_padding` | `[1, 1]` | 第二层卷积 padding。 |
| `convolution.downsample_kernel_size` | `[2, 2]` | 下采样卷积核。 |
| `convolution.downsample_stride` | `[2, 2]` | 下采样卷积步长。 |
| `convolution.downsample_padding` | `[0, 0]` | 下采样卷积 padding。 |
| `convolution.output_height` | `9` | 下采样后的栅格高度。 |
| `convolution.output_width` | `8` | 下采样后的栅格宽度。 |
| `output.goal_token_count` | `2` | 输出目标导航点 Token 数。 |
| `output.hidden_dim` | `384` | 每个目标导航点 Token 的特征维度。 |
| `output.flatten_order` | `channel_height_width` | 卷积输出展平顺序。 |
| `precision.dtype` | `float32` | 目标点嵌入层强制运行精度。 |

## 7. 依赖关系

- 读取端：`model/target_point_embedding.py`。
- 上游：Dataset 或训练流程提供的 `labels/target_point`。
- 下游：Transformer 主干中的目标导航点 Token。
- 相关设计：`Doc/Model.md` 的目标点编码流程和精度策略。

## 8. 注意事项

- 坐标系：目标点和栅格均使用当前帧 ego 坐标系，`x` 前向、`y` 左向，单位 meter。
- 数值空间：卷积输入不是原始 meter，而是 Symlog 空间向量。
- 精度：本配置强制 `float32`；如果后续扩展其他精度，必须先更新模型设计文档和实现校验。
- Shape：修改卷积 kernel、stride 或 padding 时，必须同步调整 `output_height/output_width`，配置加载会校验推导结果。
- 兼容性：不要在实现文件中重复写入本文件已有默认值。

## 9. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-07 | 1os3_Codex | AI 完成：将目标点向量 Symlog 公式说明从 `Log` 修正为自然对数 `ln`。 |
| 2026-06-07 | 1os3_Codex | AI 完成：新增 `normalization.vector_transform = "symlog"`，明确目标点米制向量场送入卷积前必须做 Symlog。 |
| 2026-06-07 | 1os3_Codex | AI 完成：新增目标点嵌入层配置文档，记录栅格、卷积、线性投影和 FP32 精度配置。 |
