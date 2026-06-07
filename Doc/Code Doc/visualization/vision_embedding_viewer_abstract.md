# visualization/vision_embedding_viewer.py 摘要

## 1. 文件基本功能

`visualization/vision_embedding_viewer.py` 对骨干视觉嵌入层做 FP32 诊断可视化。它读取 B2D H5 样本，加载视觉嵌入配置，临时把精度覆盖为 FP32，并直接调用 `BackboneVisionEmbedding` 生成视觉 token、DINOv3 Patch 特征图和压缩后特征图，再把两类特征分别 PCA 到 RGB 并上采样到原始图像分辨率。

## 2. 主要公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `VisionEmbeddingVisualizationData` | dataclass | PNG 渲染所需的样本和 PCA 可视化图。 |
| `render_vision_embedding_sample` | function | 运行嵌入层并保存 PNG。 |
| `run_vision_embedding_sample` | function | 调用真实嵌入层并返回可视化数据。 |
| `render_visualization` | function | 渲染 PIL 图像。 |
| `main` | function | 命令行入口。 |

## 3. 输入输出 Shape 概览

| 接口 | 输入 Shape | 输出 Shape |
| --- | --- | --- |
| `run_vision_embedding_sample` | H5 样本图像 `[8, 3, 288, 512]` | `dinov3_pca_images: [8, 288, 512, 3]`，`latent_pca_images: [4, 288, 512, 3]`，`token_norms: [4, 18, 32]` |
| `render_visualization` | `VisionEmbeddingVisualizationData` | PIL `Image` |
| `render_vision_embedding_sample` | H5 路径和样本索引 | PNG 文件 |

## 4. 公开接口使用规范

| 接口 | 使用规范 |
| --- | --- |
| `render_vision_embedding_sample` | 传入预处理 H5、样本索引、配置文件和输出路径。 |
| `run_vision_embedding_sample` | 固定把视觉嵌入精度覆盖为 FP32，其余配置不改。 |
| `render_visualization` | 只消费已经由真实嵌入层生成并经 PCA 上采样后的可视化图，不执行模型逻辑。 |
| `main` | 命令行运行，默认输出到项目内 `visualization/outputs/vision_embedding/`。 |

## 5. 最小使用示例

不在摘要中提供可复制命令，因为具体 H5 路径依赖本机数据。命令行参数见完整文档；核心要求是必须提供 `--h5`。

## 6. 维护注意事项

- 可视化必须继续调用 `BackboneVisionEmbedding`，不要复制 DINOv3 或卷积压缩逻辑。
- 脚本固定 FP32 运行，避免本机 BF16 过慢。
- 修改输出统计或命令行参数时，同步更新完整文档和 `doc/Code Doc/Index.md`。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-07 | 1os3_Codex | AI 完成：新增视觉嵌入 FP32 可视化摘要文档。 |
| 2026-06-07 | 1os3_Codex | AI 完成：同步 DINOv3 Patch 特征图可视化输出。 |
| 2026-06-07 | 1os3_Codex | AI 完成：将特征可视化改为 PCA RGB，并上采样到原始图像分辨率。 |
