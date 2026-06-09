# visualization/backbone_feature_pca_viewer.py 摘要

## 1. 文件基本功能

`visualization/backbone_feature_pca_viewer.py` 对统一序列 Transformer 主干做 FP32 诊断可视化。它读取 B2D H5 样本，临时把主干、注意力和视觉嵌入精度覆盖为 FP32，可选从 checkpoint 加载真实训练权重，直接调用 `MonoDriveBackbone`，收集 16 层输出后的视觉 Token，并把每层视觉特征 PCA 到 RGB 后导出 PNG。它还会把同一次前向得到的轨迹、Agent 查询和 Map 查询反变换到 ego 米制 BEV 中展示；检测 query 先按包含 `none` 的完整类别 softmax argmax 定类，最高类别为 `none` 时不绘制；轨迹诊断栏显示词表概率统计和 top-k residual 修正。

## 2. 主要公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `ModelOutputVisualizationData` | dataclass | 模型输出 BEV 面板和轨迹概率/residual 诊断栏所需的数据。 |
| `BackboneFeaturePCAVisualizationData` | dataclass | PNG 渲染所需的数据。 |
| `run_backbone_feature_pca_sample` | function | 调用真实主干并返回 PCA 数据。 |
| `render_backbone_feature_pca_sample` | function | 运行主干并保存 PNG。 |
| `render_visualization` | function | 渲染 PIL 图像。 |
| `main` | function | 命令行入口。 |

## 3. 输入输出 Shape 概览

| 接口 | 输入 Shape | 输出 Shape |
| --- | --- | --- |
| `run_backbone_feature_pca_sample` | H5 样本图像 `[8, 3, 288, 512]`，目标点 `[2]`，自车运动 `[3]` | `layer_pca_images: [16, 4, 288, 512, 3]`，`layer_token_norms: [16, 4, 18, 32]`，模型输出 BEV 数据，轨迹词表概率 `[256]`，top-k residual/correction 数据，并过滤 `none` |
| `render_visualization` | `BackboneFeaturePCAVisualizationData` | PIL `Image` |
| `render_backbone_feature_pca_sample` | H5 路径和样本索引 | PNG 文件 |

## 4. 公开接口使用规范

| 接口 | 使用规范 |
| --- | --- |
| `run_backbone_feature_pca_sample` | 固定覆盖精度为 FP32，其余结构配置来自 `config/backbone.toml`。 |
| `render_backbone_feature_pca_sample` | 输出路径必须位于项目目录内。 |
| `render_visualization` | 只消费真实主干生成的数据，不执行模型逻辑。 |
| `main` | 命令行运行，默认输出到项目内 `visualization/outputs/backbone_feature_pca/`，可用 `--checkpoint` 加载真实模型权重，也可用 `--agent-top-k`、`--map-top-k` 和 `--agent-confidence-threshold` / `--map-confidence-threshold` 控制检测绘制数量与置信度过滤。 |

## 5. 最小使用示例

不在摘要中提供可复制命令，因为具体 H5 路径依赖本机数据。命令行参数见完整文档；核心要求是必须提供 `--h5`。

## 6. 维护注意事项

- 可视化必须继续调用 `MonoDriveBackbone`，不要复制主干或 RoPE 逻辑。
- 脚本固定 FP32 运行，避免本机 BF16 过慢。
- 模型输出 BEV 面板默认最多检查 16 个 Agent 和 32 个 Map，只绘制最高概率类别不是 `none` 且 argmax 概率不低于阈值的 query，不替代正式推理后处理。
- 每个检测 query 在包含 `none` 的完整类别 softmax 上取最高概率类别；`none` query 会被过滤，非 `none` 标签格式为 `class:argmax_prob none:none_prob`。
- 轨迹诊断栏显示 top-k 词表概率、概率质量、归一化熵，以及 top-k raw residual 和米制 correction；它只用于诊断，不替代训练 loss。
- `--checkpoint` 严格加载模型 state dict，不允许结构不匹配时静默继续。
- 修改输出统计或命令行参数时，同步更新完整文档和 `doc/Code Doc/Index.md`。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-09 | 1os3_Composer | AI 完成：已绘制 Agent 标签同时显示 argmax 类别概率和 `none` 概率。 |
| 2026-06-09 | 1os3_Composer | AI 完成：记录 Agent/Map argmax 置信度阈值过滤参数。 |
| 2026-06-08 | 1os3_Codex | AI 完成：同步 16 层主干和 Agent 16 / Map 32 默认展示数量摘要。 |
| 2026-06-08 | 1os3_Codex | AI 完成：记录轨迹词表概率与 top-k residual 修正诊断栏。 |
| 2026-06-08 | 1os3_Codex | AI 完成：记录检测 BEV 面板过滤最高概率类别为 `none` 的 query。 |
| 2026-06-08 | 1os3_Codex | AI 完成：记录检测 query 按完整类别 softmax argmax 显示最高概率类别。 |
| 2026-06-08 | 1os3_Codex | AI 完成：同步可选 checkpoint 加载和轨迹 residual 反解口径。 |
| 2026-06-07 | 1os3_Codex | AI 完成：记录模型输出 BEV 默认显示完整 48 个 Agent 和 48 个 Map。 |
| 2026-06-07 | 1os3_Codex | AI 完成：新增模型输出 BEV 面板摘要说明。 |
| 2026-06-07 | 1os3_Codex | AI 完成：新增统一主干每层视觉特征 PCA 可视化摘要文档。 |
