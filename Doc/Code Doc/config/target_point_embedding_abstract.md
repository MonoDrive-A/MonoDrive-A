# config/target_point_embedding.toml 摘要

## 1. 文件基本功能

`config/target_point_embedding.toml` 保存目标点嵌入层的可提交配置：输入目标点维度、`18x16` ego 栅格覆盖范围、米制向量场的 Symlog 变换、三层卷积下采样结构、展平后线性投影为 2 个目标导航点 Token 的输出口径，以及强制 FP32 精度。

## 2. 主要公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `[target_point]` | TOML table | 输入目标点坐标维度。 |
| `[grid]` | TOML table | 栅格尺寸、米制覆盖范围和向量方向。 |
| `[normalization]` | TOML table | 米制向量场送入卷积前的 Symlog 变换。 |
| `[convolution]` | TOML table | 1x1、3x3、2x2 卷积和输出空间尺寸。 |
| `[output]` | TOML table | 目标导航点 Token 数、hidden dim 和展平顺序。 |
| `[precision]` | TOML table | 强制运行精度，当前为 `float32`。 |

## 3. 输入输出 Shape 概览

| 阶段 | Shape | 说明 |
| --- | --- | --- |
| 目标点输入 | `[B, 2]` | ego 坐标系 `[x, y]`，单位 meter。 |
| 米制栅格向量场 | `[B, 18, 16, 2]` | 当前为 `grid_xy - target_point`，单位 meter。 |
| 卷积输入 | `[B, 2, 18, 16]` | Symlog 后的 channel-first 向量场。 |
| 下采样输出 | `[B, 16, 9, 8]` | 三层卷积后的特征。 |
| 目标导航点 Token | `[B, 2, 384]` | 展平后经线性层投影得到。 |

## 4. 公开接口使用规范

| 接口 | 使用规范 |
| --- | --- |
| `[target_point]` | `coordinate_dim` 必须为 `2`，与 `labels/target_point` 的 ego XY 坐标一致。 |
| `[grid]` | 范围字段使用 meter；修改范围时需确认与目标点采样距离和规划覆盖范围一致。 |
| `[normalization]` | 当前 `vector_transform` 必须为 `symlog`，与模型整体数值空间一致。 |
| `[convolution]` | kernel、stride、padding 必须推导出 `output_height=9`、`output_width=8` 或同步修改输出配置。 |
| `[output]` | `goal_token_count=2`、`hidden_dim=384` 必须与 Transformer 序列组织一致。 |
| `[precision]` | 当前只允许 `dtype="float32"`。 |

## 5. 最小使用示例

本文件不直接运行。最小使用路径是调用 `model/target_point_embedding.py` 中的 `load_target_point_embedding_config` 读取本配置，再实例化 `TargetPointEmbedding`。

## 6. 维护注意事项

- 实现文件只能读取和校验本配置，不要重复写入配置文件中已有默认值。
- 修改目标点 shape、栅格范围、卷积结构、输出 Token 数或精度时，必须同步更新完整文档、模型文档和 `Doc/Code Doc/Index.md`。
- 本配置描述当前真实模型结构，不记录未落地的实验选项。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-07 | 1os3_Codex | AI 完成：补充目标点向量场 Symlog 变换配置说明。 |
| 2026-06-07 | 1os3_Codex | AI 完成：新增目标点嵌入层配置摘要文档。 |
