# visualization/vision_embedding_viewer.py

## 1. 文件职责

`visualization/vision_embedding_viewer.py` 负责对骨干视觉嵌入层做 FP32 诊断可视化。它从 B2D H5 读取一个样本，加载 `config/vision_embedding.toml`，把精度字段临时覆盖为 FP32，然后直接实例化并调用 `model.vision_embedding.BackboneVisionEmbedding`，最后导出输入帧、DINOv3 Patch 特征 PCA RGB 上采样图、压缩后 latent 特征 PCA RGB 上采样图和 token norm 直方图。

该文件不复制视觉嵌入逻辑，也不实现独立的 DINOv3 / 3D 卷积路径。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `VisionEmbeddingVisualizationData` | dataclass | 渲染 PNG 所需的样本、shape、精度和 PCA 可视化图。 |
| `render_vision_embedding_sample` | function | 运行嵌入层并导出 PNG。 |
| `run_vision_embedding_sample` | function | 读取样本、调用嵌入层并返回包含 DINOv3 和压缩后特征统计的可视化数据。 |
| `render_visualization` | function | 把可视化数据渲染为 PIL 图像。 |
| `main` | function | 命令行入口。 |

## 3. 关键类和函数

### `VisionEmbeddingVisualizationData`

- 功能：保存视觉嵌入诊断图需要的元数据和统计张量。
- 输入：由 `run_vision_embedding_sample` 构造。
- 输出：供 `render_visualization` 使用。
- Shape：
  - `images`: `[8, H, W, 3]`。
  - `dinov3_pca_images`: `[8, 288, 512, 3]`。
  - `latent_pca_images`: `[4, 288, 512, 3]`。
  - `token_norms`: `[4, 18, 32]`。

### `render_vision_embedding_sample`

- 功能：对单个 H5 样本运行视觉嵌入层并保存诊断 PNG。
- 输入：H5 路径、样本索引、视觉嵌入配置路径、输出路径和运行设备。
- 输出：PNG 输出路径。

### `run_vision_embedding_sample`

- 功能：读取 H5 样本，加载配置，覆盖精度为 FP32，直接调用 `BackboneVisionEmbedding`。
- 输入：H5 路径、样本索引、配置路径和设备。
- 输出：`VisionEmbeddingVisualizationData`。

### `render_visualization`

- 功能：绘制当前帧、历史帧缩略图、DINOv3 PCA RGB、latent PCA RGB、token norm 直方图和 shape 元数据。
- 输入：`VisionEmbeddingVisualizationData`。
- 输出：PIL `Image`。

## 4. 输入输出与 Shape

| 名称 | Shape / 类型 | 说明 |
| --- | --- | --- |
| `sample["images"]` | `[8, 3, 288, 512] float32` | 由 `B2DH5Dataset` 返回，值域 `[0, 1]`。 |
| `embedding_output.tokens` | `[1, 2304, 384]` | 通过真实视觉嵌入层得到。 |
| `embedding_output.dinov3_feature_map` | `[1, 768, 8, 18, 32]` | DINOv3 Patch 网格恢复后的特征图。 |
| `embedding_output.feature_map` | `[1, 384, 4, 18, 32]` | 卷积压缩后的特征图。 |
| `dinov3_pca_images` | `[8, 288, 512, 3]` | 对 DINOv3 特征通道 PCA 到 RGB 后，从 18x32 上采样到原图分辨率。 |
| `latent_pca_images` | `[4, 288, 512, 3]` | 对压缩后 latent 特征通道 PCA 到 RGB 后，从 18x32 上采样到原图分辨率。 |
| `token_norms` | `[4, 18, 32]` | 输出 token 的 L2 norm。 |
| 输出 PNG | image file | 视觉诊断图。 |

## 5. 关键实现逻辑

脚本通过 `B2DH5Dataset` 读取真实 H5 样本，并保持 Dataset 的图像归一化到 `[0, 1]`。加载视觉嵌入配置后，只使用 `override_vision_embedding_precision` 把 `dinov3_dtype` 和 `conv_dtype` 覆盖为 `float32`，其余结构配置完全沿用配置文件。

嵌入输出来自 `BackboneVisionEmbedding`，因此可视化中的 DINOv3 Patch 截取、DINOv3 后接卷积、时间压缩和 token 展平与训练实现保持一致。渲染阶段对 `dinov3_feature_map` 和压缩后 `feature_map` 分别在通道维做 PCA，取前 3 个主成分归一化为 RGB，并从 18x32 上采样到原始 288x512 分辨率，不参与模型逻辑。

## 6. 配置项

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `--h5` | 无 | 必填，预处理 H5 文件。 |
| `--sample-index` | 命令行默认 | 样本索引。 |
| `--config` | 命令行默认 | 视觉嵌入配置路径。 |
| `--output` | 可选 | 单张 PNG 输出路径。 |
| `--output-dir` | 命令行默认 | 未指定 `--output` 时的输出目录。 |
| `--device` | 命令行默认 | 运行设备。 |

命令行默认值只影响可视化入口；模型结构默认值仍以 `config/vision_embedding.toml` 为准。

## 7. 依赖关系

- 上游：`data/b2d_dataset.py`。
- 核心实现：`model/vision_embedding.py`。
- 输出目录：默认写入 `visualization/outputs/vision_embedding/`，该目录位于项目内并被 `.gitignore` 忽略。
- 第三方依赖：`torch`、`numpy`、`PIL`、`transformers`。

## 8. 注意事项

- 本脚本固定以 FP32 调用视觉嵌入层，避免本机 BF16 运行过慢。
- 诊断图中的 PCA RGB 只用于观察特征空间结构，不代表训练 loss 或注意力权重。
- 运行脚本会加载 DINOv3 大模型，CPU 上仍可能耗时较长。
- 修改视觉嵌入实现后无需同步复制可视化逻辑；脚本会自动调用最新实现。

## 9. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-07 | 1os3_Codex | AI 完成：新增调用真实视觉嵌入层的 FP32 诊断可视化工具文档。 |
| 2026-06-07 | 1os3_Codex | AI 完成：同步新增 DINOv3 Patch 特征能量热力图面板。 |
| 2026-06-07 | 1os3_Codex | AI 完成：将 DINOv3 和 latent 可视化从 RMS 热力图改为 PCA RGB，并上采样到原始图像分辨率。 |
