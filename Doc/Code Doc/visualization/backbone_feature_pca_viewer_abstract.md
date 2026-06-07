# visualization/backbone_feature_pca_viewer.py 摘要

## 1. 文件基本功能

`visualization/backbone_feature_pca_viewer.py` 对统一序列 Transformer 主干做 FP32 诊断可视化。它读取 B2D H5 样本，临时把主干、注意力和视觉嵌入精度覆盖为 FP32，直接调用 `MonoDriveBackbone`，收集 12 层输出后的视觉 Token，并把每层视觉特征 PCA 到 RGB 后导出 PNG。

## 2. 主要公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `BackboneFeaturePCAVisualizationData` | dataclass | PNG 渲染所需的数据。 |
| `run_backbone_feature_pca_sample` | function | 调用真实主干并返回 PCA 数据。 |
| `render_backbone_feature_pca_sample` | function | 运行主干并保存 PNG。 |
| `render_visualization` | function | 渲染 PIL 图像。 |
| `main` | function | 命令行入口。 |

## 3. 输入输出 Shape 概览

| 接口 | 输入 Shape | 输出 Shape |
| --- | --- | --- |
| `run_backbone_feature_pca_sample` | H5 样本图像 `[8, 3, 288, 512]`，目标点 `[2]`，自车运动 `[3]` | `layer_pca_images: [12, 4, 288, 512, 3]`，`layer_token_norms: [12, 4, 18, 32]` |
| `render_visualization` | `BackboneFeaturePCAVisualizationData` | PIL `Image` |
| `render_backbone_feature_pca_sample` | H5 路径和样本索引 | PNG 文件 |

## 4. 公开接口使用规范

| 接口 | 使用规范 |
| --- | --- |
| `run_backbone_feature_pca_sample` | 固定覆盖精度为 FP32，其余结构配置来自 `config/backbone.toml`。 |
| `render_backbone_feature_pca_sample` | 输出路径必须位于项目目录内。 |
| `render_visualization` | 只消费真实主干生成的数据，不执行模型逻辑。 |
| `main` | 命令行运行，默认输出到项目内 `visualization/outputs/backbone_feature_pca/`。 |

## 5. 最小使用示例

不在摘要中提供可复制命令，因为具体 H5 路径依赖本机数据。命令行参数见完整文档；核心要求是必须提供 `--h5`。

## 6. 维护注意事项

- 可视化必须继续调用 `MonoDriveBackbone`，不要复制主干或 RoPE 逻辑。
- 脚本固定 FP32 运行，避免本机 BF16 过慢。
- 修改输出统计或命令行参数时，同步更新完整文档和 `doc/Code Doc/Index.md`。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-07 | 1os3_Codex | AI 完成：新增统一主干每层视觉特征 PCA 可视化摘要文档。 |
