# model/target_point_embedding.py

## 1. 文件职责

`model/target_point_embedding.py` 负责实现目标点嵌入层：读取 `config/target_point_embedding.toml`，把 ego 坐标系下的米制目标点 `[x, y]` 转为 `18x16` 栅格米制向量场，再逐坐标做 Symlog 变换，经三层卷积下采样到 `[9, 8]`，展平后通过线性层投影为 2 个 384 维目标导航点 Token。

该文件不负责目标候选采样、H5 读取、Transformer 主干、目标点 loss 或轨迹解码。目标点候选池和随机抽样由数据或训练流程完成。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `TargetPointEmbeddingConfig` | dataclass | 目标点嵌入配置对象，所有结构默认值来自 `config/target_point_embedding.toml`。 |
| `TargetPointEmbedding` | class | 将 `[B, 2]` 目标点编码为 `[B, 2, 384]` 目标导航点 Token。 |
| `load_target_point_embedding_config` | function | 读取并校验 TOML 配置。 |

## 3. 关键类和函数

### `TargetPointEmbeddingConfig`

- 功能：封装目标点坐标维度、栅格范围、向量 Symlog 变换、卷积结构、输出 Token 维度和强制精度。
- 输入：由 `load_target_point_embedding_config` 从 TOML 解析得到。
- 输出：不可变配置对象。
- Shape：配置推导 `18x16 -> 9x8`，展平维度为 `16 * 9 * 8 = 1152`。
- 关键约束：
  - `coordinate_dim` 必须为 2。
  - `vector_order` 支持 `grid_minus_target` / `target_minus_grid`。
  - `vector_transform` 当前只支持 `symlog`。
  - `flatten_order` 当前支持 `channel_height_width`。
  - `dtype` 当前只支持 `float32`。

### `TargetPointEmbedding`

- 功能：生成目标导航点 Token。
- 输入：`target_points`，shape 为 `[B, 2]`，ego 坐标系米制 `[x, y]`。
- 输出：`[B, goal_token_count, hidden_dim]`，当前配置为 `[B, 2, 384]`。
- Shape：
  - 栅格中心：`[18, 16, 2]`。
  - 米制向量场：`[B, 18, 16, 2]`。
  - Symlog 向量场：`[B, 18, 16, 2] -> [B, 2, 18, 16]`。
  - 卷积输出：`[B, 16, 9, 8]`。
  - 线性投影：`[B, 1152] -> [B, 768] -> [B, 2, 384]`。
- 关键参数：全部来自 `TargetPointEmbeddingConfig`。

### `load_target_point_embedding_config`

- 功能：读取 `config/target_point_embedding.toml` 并解析为 `TargetPointEmbeddingConfig`。
- 输入：配置路径和可选项目根目录。
- 输出：`TargetPointEmbeddingConfig`。
- 约束：配置路径必须解析到项目目录内，所有表和字段均为必填。

## 4. 输入输出与 Shape

| 名称 | Shape | 说明 |
| --- | --- | --- |
| `target_points` | `[B, 2]` | ego 坐标系目标点，单位 meter，坐标为 `[x, y]`。 |
| `grid_xy` | `[18, 16, 2] float32` | 栅格中心坐标 buffer，`x` 前向、`y` 左向。 |
| `meter_vector_field` | `[B, 18, 16, 2] float32` | 每个栅格中心与目标点之间的米制向量，单位 meter。 |
| `normalized_vector_field` | `[B, 18, 16, 2] float32` | 逐坐标 Symlog 后的向量场。 |
| `vector_features` | `[B, 2, 18, 16] float32` | 送入卷积的 Symlog channel-first 特征。 |
| `embedded_features` | `[B, 16, 9, 8] float32` | 三层卷积下采样后的特征。 |
| `flattened_features` | `[B, 1152] float32` | 卷积输出展平结果。 |
| `projected_tokens` | `[B, 768] float32` | 线性层输出，表示 2 个 384 维 Token。 |
| 返回值 | `[B, 2, 384] float32` | 目标导航点 Token。 |

## 5. 关键实现逻辑

配置加载时会读取 `[target_point]`、`[grid]`、`[normalization]`、`[convolution]`、`[output]` 和 `[precision]` 六个表。实现端不提供结构默认值；缺少字段或字段类型错误会直接抛出异常。卷积输出空间尺寸通过标准 2D 卷积公式从配置推导，并与配置中的 `output_height/output_width` 对齐。

初始化时根据栅格范围构造 `grid_xy` buffer。x 方向对应车辆后向到前向，当前范围为 `[-4m, 32m]`；y 方向对应右向到左向，当前范围为 `[-32m, 32m]`。栅格坐标取每个 cell 中心。

前向时先校验 `target_points` 为浮点 `[B, 2]`。随后禁用 autocast，把目标点转为 FP32，并按配置的 `vector_order` 生成米制向量场。米制向量场逐坐标执行 `Symlog(x)=Sign(x) * Log(|x|+1)`，再从 `[B, H, W, 2]` 转为 `[B, 2, H, W]` 后依次进入 `1x1`、`3x3`、`2x2` 卷积。最后按 `channel_height_width` 展平，用单个线性层投影到 `goal_token_count * hidden_dim`，并 reshape 为目标导航点 Token。

模块初始化和 `_apply` 后都会调用 FP32 恢复逻辑，将所有浮点参数、buffer 和已有梯度恢复为 `torch.float32`。因此即使外层启用 BF16 autocast 或调用父模型整体 `.to(dtype=torch.bfloat16)`，目标点嵌入层内部仍保持 FP32 参数、buffer 和输出。

## 6. 配置项

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `target_point.*` | 由 `config/target_point_embedding.toml` 提供 | 输入目标点维度。 |
| `grid.*` | 由 `config/target_point_embedding.toml` 提供 | 栅格尺寸、米制覆盖范围和向量方向。 |
| `normalization.vector_transform` | 由 `config/target_point_embedding.toml` 提供 | 米制向量场送入卷积前的逐坐标 Symlog 变换。 |
| `convolution.*` | 由 `config/target_point_embedding.toml` 提供 | 三层卷积结构和下采样输出尺寸。 |
| `output.*` | 由 `config/target_point_embedding.toml` 提供 | 线性投影输出 Token 数、hidden dim 和展平顺序。 |
| `precision.dtype` | 由 `config/target_point_embedding.toml` 提供 | 强制运行精度，当前只支持 `float32`。 |

## 7. 依赖关系

- 上游：Dataset 或训练流程抽取的 `labels/target_point`。
- 下游：Transformer 主干的 Goal Token 序列。
- 配置：`config/target_point_embedding.toml`。
- 第三方依赖：`torch`。

## 8. 注意事项

- 坐标系：输入目标点必须已经转换到当前帧 ego 坐标系，`x` 前向、`y` 左向，单位 meter。
- 数值空间：目标点到栅格中心的米制向量必须先做 Symlog 变换，卷积不能直接消费原始 meter。
- 精度：本模块整体强制 FP32，不应由外层 autocast 或父模型 dtype 改写。
- Shape：当前输出固定由配置决定为 2 个 384 维目标导航点 Token；修改输出维度需同步 Transformer 序列组织。
- 配置：不要在实现文件内重复配置 TOML 中已经出现的结构默认值。
- 兼容性：如果未来加入激活、归一化或更多卷积层，需要先更新配置文件和模型设计文档。

## 9. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-07 | 1os3_Codex | AI 完成：修正目标点嵌入层，使米制向量场送入卷积前先做 Symlog，并补充配置校验。 |
| 2026-06-07 | 1os3_Codex | AI 完成：新增目标点嵌入层，实现配置加载、栅格向量场、三层卷积下采样、线性投影为 2 个目标导航点 Token，并强制 FP32。 |
