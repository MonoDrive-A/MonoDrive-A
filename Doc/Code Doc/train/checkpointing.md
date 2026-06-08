# train/checkpointing.py

## 1. 文件职责

`train/checkpointing.py` 负责训练 checkpoint 的保存、查找、加载和随机状态捕获/恢复。该文件不创建模型、不执行训练 step，也不决定保存时机。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `CheckpointLoadResult` | NamedTuple | checkpoint 加载结果。 |
| `capture_rng_state` | function | 捕获 Python、NumPy、PyTorch 和可选 CUDA RNG 状态。 |
| `restore_rng_state` | function | 恢复 RNG 状态。 |
| `save_checkpoint` | function | 保存 checkpoint 并更新 latest 文件。 |
| `find_resume_checkpoint` | function | 按配置查找恢复路径。 |
| `load_checkpoint` | function | 加载 checkpoint payload。 |

## 3. 关键类和函数

### `save_checkpoint`

- 功能：保存 model、optimizer、scheduler、step、epoch、batch、metrics 和 RNG 状态。
- 输入：`CheckpointConfig`、模型、优化器、调度器状态和训练状态。
- 输出：写入的 checkpoint 路径。
- 约束：输出目录由配置解析保证位于项目目录内。

### `find_resume_checkpoint`

- 功能：优先使用显式恢复路径，其次按配置查找 latest 或最新 step checkpoint。
- 输出：可恢复路径或 `None`。

## 4. 输入输出与 Shape

| 名称 | Shape | 说明 |
| --- | --- | --- |
| `model_state` | dict | PyTorch state_dict，包含模型参数和 buffer。 |
| `optimizer_state` | dict | PyTorch optimizer state_dict。 |
| `rng_state` | dict | 随机状态，不是张量训练输入。 |

## 5. 关键实现逻辑

保存时会写入 `step_{global_step:08d}.pt`，并额外更新配置中的 latest 文件。写入采用临时文件后替换目标文件，避免半写入 checkpoint。`keep_last` 大于 0 时会清理旧的 step checkpoint；latest 文件不参与清理。

加载时使用 `torch.load(..., map_location=device)`，使恢复的张量落在训练设备上。RNG 状态由训练入口决定是否恢复。

## 6. 配置项

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `checkpoint.output_dir` | 见 `config/training.toml` | checkpoint 保存目录。 |
| `checkpoint.save_interval_steps` | 见 `config/training.toml` | 自动保存间隔。 |
| `checkpoint.keep_last` | 见 `config/training.toml` | 保留最近 step checkpoint 数；0 表示不清理。 |
| `checkpoint.resume_from_latest` | 见 `config/training.toml` | 是否自动恢复 latest。 |
| `checkpoint.resume_checkpoint_path` | 见 `config/training.toml` | 显式恢复路径，空字符串表示不指定。 |

## 7. 依赖关系

- 上游：`train/training_config.py`、训练循环状态。
- 下游：`train/trainer.py`。
- 第三方：`torch`、`numpy`。

## 8. 注意事项

- checkpoint、日志和训练输出不应提交到 Git。
- 当前 checkpoint 保存 step、epoch 和 batch 位置；DataLoader 迭代器内部状态不单独序列化，训练入口通过固定 epoch seed 和 batch 跳过尽量复现恢复位置。

## 9. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-08 | 1os3_Codex | AI 完成：新增 checkpoint 保存与恢复模块。 |
