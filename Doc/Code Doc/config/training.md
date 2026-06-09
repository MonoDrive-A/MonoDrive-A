# config/training.toml

## 1. 文件职责

`config/training.toml` 集中保存训练主流程配置，包括模型与数据配置引用、运行设备、随机种子、DataLoader、AdamW 优化器、学习率调度、loss 权重、检测分类 none / non-none 类别权重、梯度监测、checkpoint 和日志输出。

本文件不保存模型结构默认值。主干、视觉嵌入、检测头、轨迹词表、目标点嵌入和训练数据处理的结构性配置继续由各自 TOML 文件维护。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `[modules]` | TOML table | 引用已有主干和训练数据配置。 |
| `[runtime]` | TOML table | 训练设备和张量搬运策略。 |
| `[random]` | TOML table | 随机种子和确定性算法开关。 |
| `[dataloader]` | TOML table | Batch、shuffle 和 worker 配置。 |
| `[optimization]` | TOML table | AdamW、warmup、余弦退火和梯度裁剪配置。 |
| `[loss_weights]` | TOML table | 各项 loss 权重。 |
| `[detection_class_weights]` | TOML table | Agent / Map 分类 CE 的 none / non-none 权重策略。 |
| `[gradient_monitor]` | TOML table | 梯度过大、过小和非有限值监测阈值。 |
| `[checkpoint]` | TOML table | 自动保存和断点恢复配置。 |
| `[logging]` | TOML table | 训练指标日志配置。 |

## 3. 关键类和函数

本文件没有 Python 类或函数。它由 `train/training_config.py` 的 `load_training_run_config` 读取并解析为 `TrainingRunConfig`。

## 4. 输入输出与 Shape

| 名称 | Shape | 说明 |
| --- | --- | --- |
| H5 batch | 见 `data/b2d_dataset.py` | 由训练入口按 DataLoader 配置读取。 |
| 模型输出 | 见 `model/backbone.py` | 训练入口调用 `MonoDriveBackbone`。 |
| loss 标量 | `[]` | 由 `train/losses.py` 汇总。 |

## 5. 关键实现逻辑

训练入口先读取本配置，再根据 `[modules]` 中的相对路径读取 `config/backbone.toml` 和 `config/training_data.toml`。所有路径都必须解析到项目目录内；checkpoint 和日志输出目录也必须位于项目目录内。

优化器配置使用 AdamW。学习率调度由 `initial_lr` 开始，经 `warmup_steps` 线性升至 `peak_lr`，中间保持峰值学习率，最后 `cosine_decay_steps` 使用余弦退火降至 `min_lr`。

轨迹词表概率监督使用 `trajectory_logit_soft_ce` 权重，对模型 raw logits 使用 soft cross entropy。标签由 `train/data_processing.py` 在物理空间按 inverse-MSE 构造，并保持为和为 1 的概率分布。

Agent / Map 分类 CE 可通过 `[detection_class_weights]` 控制 none 与 non-none 组的相对权重。默认 `mode = "auto"` 时，`train/losses.py` 使用标准 Focal Loss（``γ = focal_gamma``，``α_t`` 由组权重映射）；`mode = "manual"` 使用相同公式但由用户手动指定 ``α_t``；`mode = "disabled"` 时保持未加类别权重的 CE。

## 6. 配置项

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `modules.backbone_config_path` | `config/backbone.toml` | 主干配置引用。 |
| `modules.training_data_config_path` | `config/training_data.toml` | 数据处理配置引用。 |
| `runtime.device` | `auto` | 自动选择 CUDA 或 CPU。 |
| `random.seed` | `20260608` | 训练随机种子。 |
| `dataloader.batch_size` | `1` | 单卡 batch size。 |
| `optimization.optimizer` | `adamw` | 当前仅支持 AdamW。 |
| `optimization.initial_lr` | `1e-5` | warmup 起始学习率。 |
| `optimization.peak_lr` | `1e-4` | warmup 后峰值学习率。 |
| `optimization.warmup_steps` | `5000` | 线性 warmup step 数。 |
| `optimization.cosine_decay_steps` | `5000` | 末尾余弦退火 step 数。 |
| `loss_weights.*` | `1.0` | 各项 loss 权重。 |
| `detection_class_weights.mode` | `auto` | 检测分类类别权重模式，支持 `auto` / `manual`（标准 Focal Loss）、`disabled`。 |
| `detection_class_weights.focal_gamma` | `2.0` | 标准 Focal Loss 的 ``γ``。 |
| `detection_class_weights.focal_alpha` | `0.25` | 标准 Focal Loss 的 ``α``；`auto` 模式下 none 类 ``α_t = 1 - focal_alpha``。 |
| `detection_class_weights.agent_non_none_weight` | `0.25` | `manual` 模式下 Agent 前景类 ``α_t``。 |
| `detection_class_weights.agent_none_weight` | `0.75` | `manual` 模式下 Agent none 类 ``α_t``。 |
| `detection_class_weights.map_non_none_weight` | `0.25` | `manual` 模式下 Map 前景类 ``α_t``。 |
| `detection_class_weights.map_none_weight` | `0.75` | `manual` 模式下 Map none 类 ``α_t``。 |
| `gradient_monitor.*` | 见配置文件 | 梯度范数监测阈值和报告数量。 |
| `checkpoint.output_dir` | `checkpoints/training` | checkpoint 保存目录。 |
| `logging.output_dir` | `logs/training` | 指标日志目录。 |

## 7. 依赖关系

- 上游：已有模型和数据配置文件。
- 下游：`train/training_config.py`、`train/trainer.py`。

## 8. 注意事项

- 输出目录必须位于项目目录内，且不应提交 checkpoint、日志或训练中间产物。
- 本文件不重复配置 DINOv3、3D Conv、Transformer、检测头或轨迹词表的结构默认值。
- 轨迹词表分数使用 soft CE；Agent / Map 分类和 Agent mode 使用 hard-label CE。
- 检测分类 `auto` / `manual` 模式使用标准 Focal Loss，不会读取或写入额外统计文件。

## 9. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-09 | 1os3_Composer | AI 完成：检测分类改为标准 Focal Loss，新增 `focal_gamma` 配置。 |
| 2026-06-09 | 1os3_Composer | AI 完成：同步 `auto` 检测分类全类分组 Focal Loss 与组间权重配置说明。 |
| 2026-06-08 | 1os3_Codex | AI 完成：自动检测分类权重改为按当前 logits CE 梯度预算调整，默认 non-none 预算为 0.25。 |
| 2026-06-08 | 1os3_Codex | AI 完成：新增检测分类 none / non-none 类别权重配置，默认自动调整。 |
| 2026-06-08 | 1os3_Codex | AI 完成：轨迹词表概率 loss 从 BCE 改为 soft CE，并同步 loss 权重字段名。 |
| 2026-06-08 | 1os3_Codex | AI 完成：新增训练主流程配置。 |
