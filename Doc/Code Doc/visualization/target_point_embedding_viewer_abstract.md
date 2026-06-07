# visualization/target_point_embedding_viewer.py 摘要

## 1. 文件基本功能

`visualization/target_point_embedding_viewer.py` 可视化目标点嵌入策略。用户通过命令行指定 ego 坐标系目标点 `[x, y]`，脚本直接调用 `model.target_point_embedding.TargetPointEmbedding`，生成目标点 `18x16` Symlog 栅格向量场、分量热力图、`9x8` 卷积输出 norm 和 2 个目标导航点 Token 统计 PNG。

## 2. 主要公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `TargetPointEmbeddingVisualizationData` | dataclass | PNG 渲染所需的目标点嵌入中间数据。 |
| `run_target_point_embedding` | function | 调用真实嵌入层并返回可视化数据。 |
| `render_target_point_embedding` | function | 运行嵌入层并保存 PNG。 |
| `render_visualization` | function | 渲染 PIL 图像。 |
| `main` | function | 命令行入口。 |

## 3. 输入输出 Shape 概览

| 接口 | 输入 Shape | 输出 Shape |
| --- | --- | --- |
| `run_target_point_embedding` | `target_points: [1, 2]` | `grid_xy: [18, 16, 2]`，`symlog_vector_field: [18, 16, 2]`，`embedded_feature_norm: [9, 8]`，`tokens: [2, 384]` |
| `render_visualization` | `TargetPointEmbeddingVisualizationData` | PIL `Image` |
| `render_target_point_embedding` | 目标点坐标、配置路径和输出路径 | PNG 文件 |

## 4. 公开接口使用规范

| 接口 | 使用规范 |
| --- | --- |
| `run_target_point_embedding` | 传入有限的 `target_x/target_y`，配置路径必须在项目目录内。 |
| `render_target_point_embedding` | 输出路径必须在项目目录内；默认输出到 `visualization/outputs/target_point_embedding/`。 |
| `render_visualization` | 只消费真实嵌入层产生的数据，不重新实现模型逻辑。 |
| `main` | 命令行必须提供 `--x` 和 `--y`。 |

## 5. 最小使用示例

在项目根目录运行：

`.\.venv\Scripts\python.exe visualization\target_point_embedding_viewer.py --x 24 --y 0`

默认输出到 `visualization/outputs/target_point_embedding/`。

## 6. 维护注意事项

- 可视化必须继续调用 `TargetPointEmbedding`，不要复制目标点栅格、向量场、卷积或线性投影逻辑。
- 主图和热力图必须展示真实送入卷积的 Symlog 向量场。
- 坐标为 ego 系 `[x, y]`，单位 meter，`x` 前向、`y` 左向。
- 脚本拒绝写入项目目录外路径。
- 修改命令行参数、输出面板或中间特征时，同步更新完整文档和 `Doc/Code Doc/Index.md`。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-07 | 1os3_Codex | AI 完成：补充 Symlog 向量场可视化说明。 |
| 2026-06-07 | 1os3_Codex | AI 完成：新增目标点嵌入策略可视化摘要文档。 |
