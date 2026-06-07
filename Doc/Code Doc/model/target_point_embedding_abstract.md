# model/target_point_embedding.py 摘要

## 1. 文件基本功能

`model/target_point_embedding.py` 实现目标点嵌入层。它读取 `config/target_point_embedding.toml`，把 ego 坐标系 `[B, 2]` 目标点构造成 `18x16` 米制栅格向量场，逐坐标做 Symlog 变换后，经 `1x1`、`3x3`、`2x2` 三层卷积下采样到 `[9, 8]`，展平后用线性层投影为 2 个 384 维目标导航点 Token。

## 2. 主要公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `TargetPointEmbeddingConfig` | dataclass | 目标点嵌入配置对象，默认值来自 TOML。 |
| `TargetPointEmbedding` | class | 目标点到目标导航点 Token 的 FP32 嵌入层。 |
| `load_target_point_embedding_config` | function | 读取并校验 TOML 配置。 |

## 3. 输入输出 Shape 概览

| 接口 | 输入 | 输出 |
| --- | --- | --- |
| `load_target_point_embedding_config` | TOML 路径 | `TargetPointEmbeddingConfig` |
| `TargetPointEmbedding.forward` | `target_points: [B, 2]` | `[B, 2, 384]` |
| `_build_meter_vector_field` | `[B, 2]` | `[B, 18, 16, 2]` |
| `_build_vector_features` | `[B, 2]` | `[B, 2, 18, 16]` Symlog 特征 |
| `_flatten_embedded_features` | `[B, 16, 9, 8]` | `[B, 1152]` |

## 4. 公开接口使用规范

| 接口 | 使用规范 |
| --- | --- |
| `TargetPointEmbeddingConfig` | 只应由配置加载函数构造；所有结构字段来自 `config/target_point_embedding.toml`。 |
| `TargetPointEmbedding` | 构造时传入已校验配置；模块会在初始化和 `_apply` 后恢复 FP32 参数和 buffer。 |
| `TargetPointEmbedding.forward` | 输入必须是浮点 `[B, 2]`，坐标为当前帧 ego 系 `[x, y]`，单位 meter。 |
| `load_target_point_embedding_config` | 配置路径必须在项目目录内；缺失表或字段会抛出异常。 |

## 5. 最小使用示例

不提供完整运行示例，因为该模块通常由主模型装配并接收 Dataset 已抽取的 `target_point`。维护时可用 smoke test 构造 `[B, 2]` FP32 / BF16 输入，检查输出 shape 为 `[B, 2, 384]` 且 dtype 为 FP32。

## 6. 维护注意事项

- 目标点和栅格坐标均为 ego 坐标系，`x` 前向、`y` 左向，单位 meter。
- 目标点米制向量场必须先做 Symlog，再送入卷积。
- 展平后必须经过线性层投影到 `goal_token_count * hidden_dim`，再 reshape 为目标导航点 Token。
- 模块整体强制 FP32；修改精度策略必须同步更新 `Doc/Model.md`。
- 修改配置字段、shape、公开接口或精度行为时，必须同步更新完整文档和 `Doc/Code Doc/Index.md`。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-07 | 1os3_Codex | AI 完成：补充目标点向量场 Symlog 变换说明。 |
| 2026-06-07 | 1os3_Codex | AI 完成：新增目标点嵌入层摘要文档。 |
