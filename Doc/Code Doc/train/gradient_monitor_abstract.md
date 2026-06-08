# train/gradient_monitor.py 摘要

## 1. 文件基本功能

`train/gradient_monitor.py` 检查可训练参数的梯度范数，报告过小、过大、缺失和非有限梯度。

## 2. 主要公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `GradientParameterStat` | dataclass | 单参数梯度记录。 |
| `GradientMonitorResult` | NamedTuple | 梯度监测结果。 |
| `inspect_gradients` | function | 执行梯度检查。 |

## 3. Shape 概览

本文件不改变张量 shape，只读取参数梯度并计算 L2 范数。

## 4. 使用规范

在 `loss.backward()` 后、`optimizer.step()` 前调用。冻结参数不会被检查。

## 5. 最小示例

适合训练循环用 `inspect_gradients(model, config.gradient_monitor)` 获取告警计数。

## 6. 维护注意事项

新增梯度统计字段时，同步训练日志和文档。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-08 | 1os3_Codex | AI 完成：新增梯度监测摘要。 |
