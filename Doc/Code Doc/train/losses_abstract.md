# train/losses.py 摘要

## 1. 文件基本功能

`train/losses.py` 汇总规划、Agent 和 Map 训练 loss，并输出加权总 loss。

## 2. 主要公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `TrainingLossOutput` | NamedTuple | 总 loss 和分项 loss。 |
| `MonoDriveTrainingLoss` | class | 训练 loss 汇总模块。 |

## 3. Shape 概览

| 名称 | Shape | 说明 |
| --- | --- | --- |
| 轨迹 logits | `[B, V]` | 使用 BCEWithLogits。 |
| Agent future mask | `[B, 48, 4, 6]` | 逐 mode / 逐点有效 mask。 |
| Map 点 | `[B, 48, 100, 2]` | 匹配 query 的 Symlog 空间监督。 |

## 4. 使用规范

传入 `MonoDriveBackboneOutput` 和 `TrainingBatchLabels`。轨迹词表分数不使用 Softmax；分类和 mode CE 传入 raw logits，mode 只监督存在有效 future 点的匹配 query。

## 5. 最小示例

适合训练入口构造 `MonoDriveTrainingLoss(config.loss_weights)` 后在每步调用。

## 6. 维护注意事项

修改 loss 权重字段、mask shape 或监督空间时，同步训练配置和文档。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-08 | 1os3_Codex | AI 完成：新增训练 loss 摘要。 |
