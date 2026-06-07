# config/vision_embedding.toml 摘要

## 1. 文件基本功能

`config/vision_embedding.toml` 保存骨干视觉嵌入层配置，包括本地 DINOv3 路径、输入图像约束、DINOv3 后接 3D 卷积压缩结构、精度和输出 token 数。

## 2. 主要公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `[dinov3]` | TOML table | DINOv3 加载和冻结配置。 |
| `[input]` | TOML table | 输入帧数、图像尺寸、值域和归一化参数。 |
| `[compression]` | TOML table | 3D 卷积压缩结构。 |
| `[precision]` | TOML table | DINOv3 和卷积压缩 dtype。 |
| `[output]` | TOML table | token 展平顺序和数量校验。 |

## 3. 输入输出 Shape 概览

| 配置组 | Shape |
| --- | --- |
| `[input]` | `[8, 3, 288, 512]` |
| `[compression]` | `[B, 768, 8, 18, 32] -> [B, 384, 4, 18, 32]` |
| `[output]` | `[B, 2304, 384]` |

## 4. 公开接口使用规范

| 配置组 | 使用规范 |
| --- | --- |
| `[dinov3]` | `model_path` 必须为项目内相对路径。 |
| `[input]` | 输入值域当前为 `[0, 1]`；DINOv3 前处理只做 mean/std 归一化。 |
| `[compression]` | 卷积结构接在 DINOv3 Patch 输出之后。 |
| `[precision]` | 支持 `float32` 和 `bfloat16`；可视化脚本运行时覆盖为 FP32。 |
| `[output]` | `expected_token_count` 用于防止视觉 token 数与下游不一致。 |

## 5. 最小使用示例

不提供单独命令示例。该配置通常由 `model/vision_embedding.py` 的 `load_vision_embedding_config` 读取，或由 `visualization/vision_embedding_viewer.py` 间接使用。

## 6. 维护注意事项

- 不要在实现文件中复制本文件中的默认值。
- 不要新增任何 resize 相关配置。
- 修改字段时必须同步更新配置读取、完整文档和 `doc/Code Doc/Index.md`。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-07 | 1os3_Codex | AI 完成：新增骨干视觉嵌入配置摘要文档。 |
