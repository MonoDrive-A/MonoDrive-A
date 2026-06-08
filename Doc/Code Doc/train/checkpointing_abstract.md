# train/checkpointing.py 摘要

## 1. 文件基本功能

`train/checkpointing.py` 保存和加载训练 checkpoint，并捕获/恢复随机状态。

## 2. 主要公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `save_checkpoint` | function | 保存 checkpoint。 |
| `find_resume_checkpoint` | function | 查找恢复路径。 |
| `load_checkpoint` | function | 加载 checkpoint。 |
| `capture_rng_state` / `restore_rng_state` | function | RNG 状态保存和恢复。 |

## 3. Shape 概览

本文件不直接处理训练张量 shape，只保存 PyTorch state_dict。

## 4. 使用规范

输出目录必须由训练配置解析为项目内路径。保存时传入模型、优化器、调度器状态和训练 step。

## 5. 最小示例

适合训练入口按保存间隔调用 `save_checkpoint(...)`，启动时调用 `find_resume_checkpoint(...)`。

## 6. 维护注意事项

修改 checkpoint payload 字段时同步 schema、恢复逻辑和文档。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-08 | 1os3_Codex | AI 完成：新增 checkpoint 摘要。 |
