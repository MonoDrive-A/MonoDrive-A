# config/training.toml 摘要

## 1. 文件基本功能

`config/training.toml` 保存训练主流程配置，覆盖运行设备、随机种子、DataLoader、AdamW、学习率调度、loss 权重、检测分类类别权重、梯度监测、checkpoint 和日志。

## 2. 主要公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `[modules]` | TOML table | 引用主干和训练数据配置。 |
| `[optimization]` | TOML table | 优化器和学习率调度。 |
| `[loss_weights]` | TOML table | loss 权重。 |
| `[detection_class_weights]` | TOML table | Agent / Map 分类 CE 的 none / non-none 组权重策略。 |
| `[checkpoint]` | TOML table | 自动保存和断点恢复。 |

## 3. Shape 概览

本文件不直接处理张量；训练张量 shape 由模型和数据配置决定。

## 4. 使用规范

训练入口通过 `load_training_run_config` 读取本文件。配置路径和输出目录必须解析到项目目录内。检测分类类别权重默认 `mode = "auto"`，按 logits 梯度预算动态调整，也可切换为 `manual` 或 `disabled`。

## 5. 最小示例

适合在项目根目录运行训练入口时使用默认路径 `config/training.toml`。

## 6. 维护注意事项

- 不在本文件重复已有模型结构默认值。
- 修改训练配置字段时，同步 `train/training_config.py` 和本文档。
- 轨迹词表概率 loss 使用 soft cross entropy，标签为和为 1 的 inverse-MSE soft label。
- 检测分类自动类别权重按当前 batch 的 none / non-none logits 梯度预算动态计算，默认 non-none 预算比例为 `0.25`。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-08 | 1os3_Codex | AI 完成：同步自动检测分类 logits 梯度预算配置摘要。 |
| 2026-06-08 | 1os3_Codex | AI 完成：同步检测分类 none / non-none 类别权重配置摘要。 |
| 2026-06-08 | 1os3_Codex | AI 完成：同步轨迹词表 soft CE loss 口径。 |
| 2026-06-08 | 1os3_Codex | AI 完成：新增训练主配置摘要。 |
