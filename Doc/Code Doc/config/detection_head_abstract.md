# config/detection_head.toml 摘要

## 1. 文件基本功能

`config/detection_head.toml` 保存模型侧检测查询初始化和检测解码头配置。它定义 Agent 16 个查询、Map 32 个查询、384 维 hidden、空间 anchor 初始化、Agent 4 个 future mode 在 120 度范围内均匀散布、Agent / Map 输出字段，以及检测解码线性层的 FP32 精度。

## 2. 主要公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `[query]` | TOML table | 检测查询数量、hidden dim 和顺序。 |
| `[query_embedding]` | TOML table | 查询 Token 初始 anchor 特征。 |
| `[agent]` | TOML table | Agent 类别、状态字段和 future 配置。 |
| `[agent_query_initialization]` | TOML table | Agent 空间 anchor 初始化。 |
| `[agent_mode_initialization]` | TOML table | Agent 4 个 mode 的 120 度均匀初始化。 |
| `[map]` | TOML table | Map 类别和点输出配置。 |
| `[map_query_initialization]` | TOML table | Map 空间 anchor 初始化。 |
| `[precision]` | TOML table | 解码线性层精度。 |

## 3. 输入输出 Shape 概览

| 阶段 | Shape | 说明 |
| --- | --- | --- |
| 检测查询 Token | `[48, 384]` | Agent 16 个，Map 32 个。 |
| Agent class logits | `[B, 16, 4]` | 3 个前景类加“无”类别。 |
| Agent states | `[B, 16, 11]` | Agent 平面状态、尺寸、朝向、速度和加速度。 |
| Agent future | `[B, 16, 4, 6, 2]` | 4-mode Symlog 空间 future 位移。 |
| Map class logits | `[B, 32, 4]` | 3 个前景类加“无”类别。 |
| Map points | `[B, 32, 100, 2]` | Symlog 空间 Map 点。 |

## 4. 公开接口使用规范

| 接口 | 使用规范 |
| --- | --- |
| `[query]` | 查询数量必须与空间采样数乘积一致；查询不按类别硬分配。 |
| `[query_embedding]` | `hidden_dim` 必须不小于 anchor 特征数量。 |
| `[agent]` | Agent 类别不包含 `motorcycle`，无类别由 `none_class_name` 表示。 |
| `[agent_mode_initialization]` | `mode_angles_deg` 必须等间隔，且首尾对齐 Agent 查询角度范围。 |
| `[map]` | Map 类别不包含 CrossWalk，局部元素点数为 100。 |
| `[precision]` | 当前只允许 `decoder_dtype="float32"`。 |

## 5. 最小使用示例

本文件不直接运行。最小使用路径是调用 `model/detection_head.py` 中的 `load_detection_head_config` 读取本配置，再实例化 `DetectionQueryEmbedding` 和 `DetectionHeadDecoder`。

## 6. 维护注意事项

- 实现文件只能读取和校验本配置，不要重复写入配置文件中已有默认值。
- 修改查询数量、类别、输出字段、mode 初始化或精度时，必须同步更新完整配置文档、模型文件文档和 `Doc/Code Doc/Index.md`。
- 检测解码头不做反 Symlog，相关后处理应放在 loss 或推理流程中。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-08 | 1os3_Codex | AI 完成：同步 Agent 16 / Map 32 检测查询摘要。 |
| 2026-06-07 | 1os3_Codex | AI 完成：新增检测头配置摘要文档。 |
