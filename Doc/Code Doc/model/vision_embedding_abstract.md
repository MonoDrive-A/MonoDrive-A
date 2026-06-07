# model/vision_embedding.py 摘要

## 1. 文件基本功能

`model/vision_embedding.py` 实现骨干视觉嵌入层：冻结本地 DINOv3-ViT-B，只使用 Patch 序列，并在 DINOv3 后接 3D 卷积压缩模块，输出 2304 个 384 维视觉 token。

DINOv3 前处理只执行 mean/std 归一化，不做 resize、crop 或 padding。

## 2. 主要公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `VisionEmbeddingConfig` | dataclass | 视觉嵌入配置对象。 |
| `VisionEmbeddingOutput` | NamedTuple | 视觉嵌入输出。 |
| `BackboneVisionEmbedding` | class | DINOv3 后接 3D 卷积压缩的视觉嵌入层。 |
| `load_vision_embedding_config` | function | 读取 TOML 配置。 |
| `override_vision_embedding_precision` | function | 只替换精度字段，用于可视化或本地调试。 |

## 3. 输入输出 Shape 概览

| 接口 | 输入 Shape | 输出 Shape |
| --- | --- | --- |
| `BackboneVisionEmbedding.forward` | `[B, 8, 3, 288, 512]` | `tokens: [B, 2304, 384]`，`dinov3_feature_map: [B, 768, 8, 18, 32]`，`feature_map: [B, 384, 4, 18, 32]` |
| `load_vision_embedding_config` | TOML 路径 | `VisionEmbeddingConfig` |
| `override_vision_embedding_precision` | `VisionEmbeddingConfig` 和 dtype 名称 | 新的 `VisionEmbeddingConfig` |

## 4. 公开接口使用规范

| 接口 | 使用规范 |
| --- | --- |
| `VisionEmbeddingConfig` | 通过 `load_vision_embedding_config` 构造，默认值只以 `config/vision_embedding.toml` 为准。 |
| `VisionEmbeddingOutput` | 下游使用 `tokens` 进入 Transformer，使用 `dinov3_feature_map` 做 DINOv3 诊断，使用 `latent_grid_shape` 构造视觉 RoPE 位置。 |
| `BackboneVisionEmbedding` | 输入必须为浮点 `[B, T, C, H, W]`，当前配置要求值域 `[0, 1]`。 |
| `load_vision_embedding_config` | `model_path` 必须为项目内相对路径。 |
| `override_vision_embedding_precision` | dtype 仅支持 `float32` 和 `bfloat16`。 |

## 5. 最小使用示例

不在摘要中提供可直接复制的完整 DINOv3 示例，因为该模块会加载本地大模型权重，运行耗时依赖本机设备。建议使用 `visualization/vision_embedding_viewer.py` 对 H5 样本做 FP32 诊断，该脚本会直接调用 `BackboneVisionEmbedding`。

## 6. 维护注意事项

- 不要在实现文件中重复写入配置默认值；默认值以 `config/vision_embedding.toml` 为准。
- 不要在 DINOv3 前处理路径加入 resize、crop 或 padding。
- 修改卷积结构、精度字段、输出 token 数或公开接口时，必须同步更新完整文档和 `doc/Code Doc/Index.md`。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-07 | 1os3_Codex | AI 完成：新增骨干视觉嵌入层摘要文档。 |
| 2026-06-07 | 1os3_Codex | AI 完成：同步 `dinov3_feature_map` 诊断输出字段。 |
