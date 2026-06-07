# config/vision_embedding.toml

## 1. 文件职责

`config/vision_embedding.toml` 负责保存骨干视觉嵌入层的可提交配置，包括本地 DINOv3 模型路径、输入图像约束、DINOv3 后接 3D 卷积压缩结构、精度策略和输出 token 约束。

该配置文件不保存本机私有绝对路径、训练输出路径、缓存路径或大文件内容。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `[dinov3]` | TOML table | DINOv3 本地模型加载与冻结配置。 |
| `[input]` | TOML table | 输入帧、通道、分辨率、值域和归一化配置。 |
| `[compression]` | TOML table | DINOv3 后接 3D 卷积压缩结构。 |
| `[precision]` | TOML table | DINOv3 和卷积压缩运行精度。 |
| `[output]` | TOML table | 输出 token 顺序和 token 数校验。 |

## 3. 关键类和函数

本文件没有 Python 类或函数。它由 `model/vision_embedding.py` 中的 `load_vision_embedding_config` 读取，并解析为 `VisionEmbeddingConfig`。

## 4. 输入输出与 Shape

| 配置组 | Shape / 语义 | 说明 |
| --- | --- | --- |
| `[input]` | `[T, C, H, W]` | 单样本输入图像 shape；当前语义为 8 帧 RGB 288x512。 |
| `[compression]` | `[B, 768, 8, 18, 32] -> [B, 384, 4, 18, 32]` | DINOv3 Patch 网格经过 3D 卷积压缩后的 shape 变化。 |
| `[output]` | `[B, 2304, 384]` | 输出视觉 token shape。 |

## 5. 关键实现逻辑

配置读取端会把 `model_path` 解析为项目根目录内路径，并拒绝绝对路径或逃逸出项目目录的相对路径。输入尺寸用于校验 Dataset 输出，也用于结合 DINOv3 `patch_size` 推导 Patch 网格。

`[precision]` 中的精度字段用于选择 DINOv3 前向和卷积压缩前向的 autocast dtype。配置默认反映模型设计精度；本地可视化工具会在运行时把精度覆盖为 FP32，但不修改本配置文件。

## 6. 配置项

| 配置项 | 默认值来源 | 说明 |
| --- | --- | --- |
| `dinov3.model_path` | 本文件 | DINOv3 本地模型目录，项目内相对路径。 |
| `dinov3.local_files_only` | 本文件 | 是否禁止 Transformers 访问网络。 |
| `dinov3.trust_remote_code` | 本文件 | 是否允许加载模型目录声明的 remote code。 |
| `dinov3.freeze_dinov3` | 本文件 | 是否冻结 DINOv3。 |
| `input.input_frames` | 本文件 | 历史输入帧数。 |
| `input.input_channels` | 本文件 | 输入通道数。 |
| `input.input_height` | 本文件 | 输入图像高度，单位 pixel。 |
| `input.input_width` | 本文件 | 输入图像宽度，单位 pixel。 |
| `input.input_value_range` | 本文件 | 输入值域，当前实现支持 `zero_one`。 |
| `input.image_mean` | 本文件 | DINOv3 mean/std 归一化均值。 |
| `input.image_std` | 本文件 | DINOv3 mean/std 归一化标准差。 |
| `compression.residual_block_count` | 本文件 | DINOv3 后接 3D 残差块数量。 |
| `compression.temporal_compression_after_block` | 本文件 | 时间压缩卷积插入位置。 |
| `compression.temporal_compression_kernel` | 本文件 | 时间压缩卷积核 `[T, H, W]`。 |
| `compression.temporal_compression_stride` | 本文件 | 时间压缩卷积步长 `[T, H, W]`。 |
| `compression.output_hidden_dim` | 本文件 | 输出视觉 token 维度。 |
| `precision.dinov3_dtype` | 本文件 | DINOv3 前向精度。 |
| `precision.conv_dtype` | 本文件 | 3D 卷积压缩前向精度。 |
| `output.token_order` | 本文件 | token 展平顺序。 |
| `output.expected_token_count` | 本文件 | 输出视觉 token 数校验。 |

## 7. 依赖关系

- 读取端：`model/vision_embedding.py`。
- 使用端：训练主干、`visualization/vision_embedding_viewer.py`。
- 相关设计：`Doc/Model.md` 的视觉编码器和精度策略。

## 8. 注意事项

- 本文件中的输入尺寸必须与 Dataset 输出和下游 token 数一致。
- DINOv3 前处理只允许归一化，不应通过配置开启 resize。
- 修改精度字段时不需要修改实现文件；实现文件只读取和校验字段。
- 若新增配置字段，必须同步更新 `VisionEmbeddingConfig`、本配置文档、摘要文档和 `doc/Code Doc/Index.md`。

## 9. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-07 | 1os3_Codex | AI 完成：新增骨干视觉嵌入层配置文档。 |
