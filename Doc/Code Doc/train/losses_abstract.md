# train/losses.py 摘要

## 1. 文件基本功能

`train/losses.py` 汇总规划、Agent 和 Map 训练 loss，并输出加权总 loss。Agent / Map 分类 CE 支持 none 与 non-none 组类别权重，默认自动按当前 logits 的 CE 梯度预算调整。

## 2. 主要公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `TrainingLossOutput` | NamedTuple | 总 loss 和分项 loss。 |
| `MonoDriveTrainingLoss` | class | 训练 loss 汇总模块。 |

## 3. Shape 概览

| 名称 | Shape | 说明 |
| --- | --- | --- |
| 轨迹 logits | `[B, V]` | 使用 soft cross entropy。 |
| Agent future mask | `[B, 16, 4, 6]` | 逐 mode / 逐点有效 mask。 |
| Map 点 | `[B, 32, 100, 2]` | 匹配 query 的 Symlog 空间监督。 |

## 4. 使用规范

传入 `MonoDriveBackboneOutput` 和 `TrainingBatchLabels`。Agent / Map 分类 CE 在 `auto` / `manual` 下使用标准 Focal Loss（Agent / Map 各自独立 ``γ`` 与 ``α``）；`disabled` 为 batch mean CE。

## 5. 最小示例

适合训练入口构造 `MonoDriveTrainingLoss(config.loss_weights, config.detection_class_weights)` 后在每步调用。

## 6. 维护注意事项

修改 loss 权重字段、检测分类权重模式、mask shape 或监督空间时，同步训练配置和文档。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-10 | 1os3_Composer | AI 完成：同步 Agent / Map 独立 Focal 超参摘要。 |
| 2026-06-09 | 1os3_Composer | AI 完成：同步标准 Focal Loss 与 `focal_gamma` 配置摘要。 |
| 2026-06-09 | 1os3_Composer | AI 完成：同步检测分类 CE none / non-none 分项日志摘要。 |
| 2026-06-08 | 1os3_Codex | AI 完成：同步 Agent 16 / Map 32 loss shape 摘要。 |
| 2026-06-08 | 1os3_Codex | AI 完成：同步自动检测分类权重的 logits 梯度预算口径。 |
| 2026-06-08 | 1os3_Codex | AI 完成：同步检测分类 none / non-none 类别权重摘要。 |
| 2026-06-08 | 1os3_Codex | AI 完成：同步轨迹词表 soft CE loss 摘要。 |
| 2026-06-08 | 1os3_Codex | AI 完成：新增训练 loss 摘要。 |
