# config/training_data.toml 摘要

## 1. 文件基本功能

`config/training_data.toml` 保存训练阶段数据处理配置，包括 H5 数据源、样本校验范围、轨迹词表 soft label 参数和 Agent / Map Hungarian matching cost 权重。

## 2. 主要公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `[dataset]` | TOML table | H5 读取和 Dataset 行为。 |
| `[modules]` | TOML table | 引用已有模型子配置。 |
| `[validation]` | TOML table | 样本数值校验阈值。 |
| `[trajectory_label]` | TOML table | 轨迹词表标签构造参数。 |
| `[agent_matching]` | TOML table | Agent 匹配 cost 权重。 |
| `[map_matching]` | TOML table | Map 匹配 cost 权重。 |

## 3. Shape 概览

| 名称 | Shape | 说明 |
| --- | --- | --- |
| 轨迹 soft label | `[B, V]` | 由词表轨迹与 GT 的物理空间 MSE 构造。 |
| Agent 目标 | `[B, 16]`、`[B, 16, 11]` | Hungarian 匹配后的分类和状态目标。 |
| Map 目标 | `[B, 32]`、`[B, 32, 100, 2]` | Hungarian 匹配后的分类和点目标。 |

## 4. 使用规范

本配置只由 `train/data_processing.py` 读取。词表规模、类别列表、query 数量和 future 点数不得复制到本文件，必须继续从已有配置读取。

H5 数据源是只读输入，`dataset.h5_dir` 和 `dataset.h5_paths` 允许使用项目外绝对路径；模型子配置路径仍由实现限制在项目目录内。

本配置不包含危险轨迹判断开关；训练数据处理阶段不根据稀疏 H5 未来 Agent 标签屏蔽候选轨迹。

## 5. 最小示例

适合通过训练入口间接使用，不建议单独手写解析逻辑。直接调用 `load_training_data_config("config/training_data.toml")` 即可得到校验后的配置对象。

## 6. 维护注意事项

- 修改 cost 权重后同步检查训练 loss 的量纲。
- 修改样本校验范围后用 H5 smoke test 确认有效样本数量没有异常下降。
- 新增配置项时同步更新完整文档和 `Doc/Code Doc/Index.md`。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-08 | 1os3_Codex | AI 完成：同步 Agent 16 / Map 32 匹配目标 shape 摘要。 |
| 2026-06-07 | 1os3_Codex | AI 完成：新增训练数据处理配置摘要。 |
| 2026-06-08 | 1os3_Codex | AI 完成：记录 H5 数据源可使用项目外绝对路径。 |
