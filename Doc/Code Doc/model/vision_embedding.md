# model/vision_embedding.py

## 1. 文件职责

`model/vision_embedding.py` 负责实现骨干视觉嵌入层：加载本地 DINOv3-ViT-B，冻结 DINOv3 参数，从 DINOv3 输出中只截取 Patch 序列，并在 DINOv3 后接 3D 卷积压缩模块，把 8 帧视觉特征压缩为 4 帧 latent，最后输出 384 维视觉 Token。

该文件不负责图像 resize、crop、padding、H5 数据读取、Transformer 主干、3D RoPE 位置编码或训练 loss。DINOv3 前处理只做张量 mean/std 归一化。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `VisionEmbeddingConfig` | dataclass | 视觉嵌入配置对象，所有默认值来自 `config/vision_embedding.toml`。 |
| `VisionEmbeddingOutput` | NamedTuple | 视觉嵌入输出，包含 token、压缩后特征图和网格 shape。 |
| `BackboneVisionEmbedding` | class | DINOv3 后接 3D 卷积压缩的骨干视觉嵌入层。 |
| `load_vision_embedding_config` | function | 读取并校验 TOML 配置。 |
| `override_vision_embedding_precision` | function | 返回只替换 DINOv3 和卷积精度字段的新配置。 |

## 3. 关键类和函数

### `VisionEmbeddingConfig`

- 功能：封装本地 DINOv3 路径、输入 shape、归一化参数、卷积压缩结构、精度和输出 token 约束。
- 输入：由 `load_vision_embedding_config` 从 TOML 解析得到。
- 输出：不可变配置对象。
- Shape：`input_shape` 为 `[T, C, H, W]`。
- 关键约束：
  - `input_value_range` 当前仅支持 `zero_one`。
  - `dinov3_dtype` 和 `conv_dtype` 支持 `float32` / `bfloat16`。
  - `token_order` 当前仅支持 `time_height_width`。

### `VisionEmbeddingOutput`

- 功能：返回主干视觉嵌入结果和下游位置编码需要的网格信息。
- 输入：由 `BackboneVisionEmbedding.forward` 构造。
- 输出：
  - `tokens`: Transformer 视觉 token。
  - `dinov3_feature_map`: DINOv3 Patch 网格恢复后的原始特征图，用于诊断和可视化。
  - `feature_map`: 卷积压缩后的 5D 特征图。
  - `patch_grid_shape`: DINO patch 网格 `[H_patch, W_patch]`。
  - `latent_grid_shape`: 压缩后 latent 网格 `[T_latent, H_patch, W_patch]`。

### `BackboneVisionEmbedding`

- 功能：执行 DINOv3 Patch 抽取、DINOv3 后 3D 卷积压缩和 token 展平。
- 输入：`images`，shape 为 `[B, 8, 3, 288, 512]`，值域为 `[0, 1]`。
- 输出：`VisionEmbeddingOutput`。
- Shape：
  - DINOv3 Patch 序列：`[B, 8, 576, 768]`。
  - 卷积输入特征图：`[B, 768, 8, 18, 32]`。
  - 压缩后特征图：`[B, 384, 4, 18, 32]`。
  - 输出 Token：`[B, 2304, 384]`。
- 关键参数：全部来自 `VisionEmbeddingConfig`。

### `load_vision_embedding_config`

- 功能：读取 `config/vision_embedding.toml` 并解析为 `VisionEmbeddingConfig`。
- 输入：配置路径和可选项目根目录。
- 输出：`VisionEmbeddingConfig`。
- 约束：`model_path` 必须是项目目录内相对路径。

### `override_vision_embedding_precision`

- 功能：为可视化或本地调试返回只改精度的新配置。
- 输入：已有配置、DINOv3 dtype、卷积 dtype。
- 输出：新的 `VisionEmbeddingConfig`。
- 约束：只允许 `float32` / `bfloat16`。

## 4. 输入输出与 Shape

| 名称 | Shape | 说明 |
| --- | --- | --- |
| `images` | `[B, T, C, H, W]` | 输入 RGB 图像，当前配置为 8 帧、3 通道、288x512，值域 `[0, 1]`。 |
| `flat_images` | `[B*T, C, H, W]` | 送入 DINOv3 的展平图像批次，不做 resize。 |
| `last_hidden_state` | `[B*T, N, 768]` | DINOv3 输出序列，可能包含 CLS / register token。 |
| `patch_tokens` | `[B*T, 576, 768]` | 从序列末尾截取的 Patch token。 |
| `dinov3_feature_map` | `[B, 768, 8, 18, 32]` | DINOv3 Patch 网格恢复后的时空特征。 |
| `compressed_feature_map` | `[B, 384, 4, 18, 32]` | 4 层 3D 卷积和 1x1x1 投影后的特征。 |
| `tokens` | `[B, 2304, 384]` | 按 `time_height_width` 展平的视觉 token。 |

## 5. 关键实现逻辑

前向流程先校验输入 shape 和 dtype，然后只执行 DINOv3 mean/std 归一化。实现不会调用 Hugging Face image processor 的 resize 或 crop，也不会在代码内隐式改变图像空间尺寸。

DINOv3 前向根据配置精度进入 autocast；冻结模式下额外使用 `torch.no_grad()`，并在 `train()` 被调用后保持 DINOv3 为 eval。DINOv3 输出中只保留序列末尾的 Patch token，Patch 数由输入分辨率和 DINOv3 `patch_size` 推导。

卷积压缩严格接在 DINOv3 后：先把 Patch token 恢复为 `[B, C, T, H, W]`，经过配置数量的 `SpatioTemporalResidualBlock3d`。时间压缩卷积插在配置指定的残差块之后，默认结构对应第 3 层和第 4 层之间的 `[2, 1, 1]` 卷积。最后用 `1x1x1` 卷积降到输出隐藏维度，并展平成视觉 token。

## 6. 配置项

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `dinov3.*` | 由 `config/vision_embedding.toml` 提供 | DINOv3 本地路径、本地加载、remote code 和冻结配置。 |
| `input.*` | 由 `config/vision_embedding.toml` 提供 | 输入帧数、通道数、高宽、值域和 mean/std。 |
| `compression.*` | 由 `config/vision_embedding.toml` 提供 | DINOv3 后接卷积块数量、时间压缩位置、kernel/stride 和输出维度。 |
| `precision.*` | 由 `config/vision_embedding.toml` 提供 | DINOv3 和卷积压缩的运行精度。 |
| `output.*` | 由 `config/vision_embedding.toml` 提供 | token 展平顺序和期望 token 数。 |

## 7. 依赖关系

- 上游：`data/b2d_dataset.py` 返回的 `[0, 1]` RGB 图像张量。
- 复用模块：`model/residual_block.py` 中的 `SpatioTemporalResidualBlock3d`。
- 下游：Transformer 主干视觉 token 输入、3D RoPE 位置编码坐标构造。
- 第三方依赖：`torch`、`transformers`。

## 8. 注意事项

- DINOv3 前处理不允许 resize；如果输入尺寸改变，应先修改配置并确认 token 数与下游一致。
- 可视化脚本使用 FP32 覆盖精度，但仍调用本文件实现，避免诊断路径和训练路径分叉。
- DINOv3 Patch token 通过“截取序列末尾 `patch_count` 个 token”获得，避免依赖不同 DINOv3 版本中 CLS / register token 的具体数量。
- `bfloat16` 在当前本机资源上可能非常慢；实际运行精度由配置或调用方显式覆盖。
- 修改公开接口、shape、精度配置或卷积结构时，必须同步更新本文件文档、摘要文档和 `doc/Code Doc/Index.md`。

## 9. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-07 | 1os3_Codex | AI 完成：新增骨干视觉嵌入层，实现冻结 DINOv3 Patch 抽取、DINOv3 后 3D 卷积压缩、可配置精度和 2304 个视觉 token 输出。 |
| 2026-06-07 | 1os3_Codex | AI 完成：在 `VisionEmbeddingOutput` 中新增 `dinov3_feature_map`，供可视化展示 DINOv3 Patch 特征图。 |
