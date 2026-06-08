# model/detection_head.py 摘要

## 1. 文件基本功能

`model/detection_head.py` 实现模型侧检测查询初始化和检测解码头。它读取 `config/detection_head.toml`，生成 Agent 16 个、Map 32 个共 48 个检测查询 Token，并用 FP32 线性层从 `[B, 48, 384]` 检测 Token 特征解码 Agent 和 Map 输出。

## 2. 主要公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `DetectionHeadConfig` | dataclass | 检测头配置对象，默认值来自 TOML。 |
| `DetectionDecoderOutput` | NamedTuple | Agent 和 Map 检测解码输出。 |
| `DetectionQueryEmbedding` | class | 生成 FP32 检测查询 Token。 |
| `DetectionHeadDecoder` | class | FP32 检测解码线性层。 |
| `load_detection_head_config` | function | 读取并校验 TOML 配置。 |

## 3. 输入输出 Shape 概览

| 接口 | 输入 | 输出 |
| --- | --- | --- |
| `load_detection_head_config` | TOML 路径 | `DetectionHeadConfig` |
| `DetectionQueryEmbedding.forward` | 无 | `[48, 384]` |
| `DetectionHeadDecoder.forward` | `[B, 48, 384]` | `DetectionDecoderOutput` |
| `agent_class_logits` | - | `[B, 16, 4]` |
| `agent_states` | - | `[B, 16, 11]` |
| `agent_future_trajectories` | - | `[B, 16, 4, 6, 2]` |
| `map_points` | - | `[B, 32, 100, 2]` |

## 4. 公开接口使用规范

| 接口 | 使用规范 |
| --- | --- |
| `DetectionHeadConfig` | 只应由配置加载函数构造；所有结构字段来自 `config/detection_head.toml`。 |
| `DetectionQueryEmbedding` | 构造时传入已校验配置；查询初值为空间 anchor，不按类别硬分配。 |
| `DetectionHeadDecoder` | 输入必须是浮点 `[B, 48, 384]`；内部禁用 autocast 并输出 FP32。 |
| `load_detection_head_config` | 配置路径必须在项目目录内；缺失表或字段会抛出异常。 |

## 5. 最小使用示例

不提供完整运行示例，因为该模块通常由主模型装配。维护时可用 smoke test 加载配置，实例化 `DetectionQueryEmbedding` 和 `DetectionHeadDecoder`，检查输出 shape 和 dtype。

## 6. 维护注意事项

- Agent 和 Map 查询不按类别硬分配。
- Agent 4 个 future mode 的初始化角度必须等间隔，且首尾对齐查询角度范围以覆盖前方 120 度。
- 解码输出不做反 Symlog；Agent future 和 Map 点保持模型空间。
- 检测查询和解码线性层强制 FP32；修改精度策略必须同步更新 `Doc/Model.md`。
- 修改配置字段、shape、公开接口或初始化口径时，必须同步更新完整文档和 `Doc/Code Doc/Index.md`。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-08 | 1os3_Codex | AI 完成：同步 Agent 16 / Map 32 检测查询摘要。 |
| 2026-06-07 | 1os3_Codex | AI 完成：新增检测头模型文件摘要文档。 |
