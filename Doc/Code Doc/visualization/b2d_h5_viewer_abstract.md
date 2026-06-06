# visualization/b2d_h5_viewer.py 摘要

## 1. 文件基本功能

将 B2D 预处理 H5 的训练样本导出为 PNG 诊断图，用于检查 8 帧历史图像、未来轨迹、目标候选、Agent 10D 标签、Agent future、局部 Map 和交通元素标签是否正确。样本读取复用 `data.b2d_dataset.B2DH5Dataset`，降低可视化端与训练读取端不一致的风险。

## 2. 主要公开接口

| 名称 | 类型 | 功能 |
| --- | --- | --- |
| `BevViewConfig` | dataclass | 配置 BEV 坐标范围和绘制参数。 |
| `H5SampleData` | dataclass | 封装单样本可视化数据。 |
| `load_h5_sample` | function | 通过 `B2DH5Dataset` 读取单个样本。 |
| `render_sample` | function | 渲染样本为 `PIL.Image.Image`。 |
| `render_h5_sample` | function | 读取 H5 并导出 PNG。 |

## 3. 输入输出 Shape 概览

| 数据 | Shape | 说明 |
| --- | --- | --- |
| `images` | `[8, H, W, 3]` | 8 帧历史输入图像。 |
| `future_trajectory` | `[6, 2]` | 未来轨迹。 |
| `target_point` | `[2]` | 默认目标点，为候选池首个有效点或最远点兜底。 |
| `target_point_index` | scalar | 当前目标点在候选池中的索引。 |
| `target_points` | `[32, 2]` | 未来 24-30m 可达目标候选点。 |
| `target_valid` | `[32]` | 目标候选 padding mask。 |
| `agent_boxes` | `[194, 10]` | `[x, y, l, w, h, yaw, v_x, v_y, a_x, a_y]`。 |
| `agent_future_trajectory` | `[194, 6, 2]` | Agent future 位移，绘制时加回当前 Agent 中心。 |
| `map_points` | `[60, 100, 2]` | 局部 Map 元素。 |
| 输出 PNG | `[960, 1360, 3]` | 单样本诊断图。 |

## 4. 公开接口使用规范

| 接口 | 使用规范 |
| --- | --- |
| `load_h5_sample` | 输入应是 `b2d_h5_v5` 风格 H5；内部必须通过 `B2DH5Dataset` 读取样本，Agent 标签必须是 10D，并包含目标候选、Agent future 和 Map 字段。 |
| `render_h5_sample` | `agent_limit` 必须为正数；输出目录会自动创建。 |
| `BevViewConfig` | `x_min < x_max`、`y_min < y_max`，坐标单位为 meter。 |

## 5. 最小使用示例

```powershell
.\.venv\Scripts\python.exe -m visualization.b2d_h5_viewer `
  --h5 data/preprocessed/ControlLoss_Town11_Route402_Weather12.h5 `
  --sample-index 0 `
  --output visualization/outputs/sample_000000.png
```

## 6. 维护注意事项

- H5 schema 变化时必须同步更新 `load_h5_sample` 和文档。
- `load_h5_sample` 应保持通过 `B2DH5Dataset` 读取训练样本，不要恢复为可视化端手写 H5 字段切片。
- 可视化输出属于调试产物，不提交到 Git。
- BEV 只验证 ego 坐标标签，不代表相机平面投影效果。
- 可视化只显示 H5 中已经过滤和重采样的 Agent/Map，不做二次几何处理。
- Agent future 是当前 Agent 原点下的位移标签；BEV 绘制时加回 `agent_boxes[..., :2]`。
- metadata 会显示目标候选数量、目标距离范围、搜索范围与轨迹平滑配置，检查旧 H5 时应先确认这些字段。
- 未来轨迹第一个标签点是 `t+0.5s`，可视化折线会从 ego 原点连接到该点。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-05 | 1os3_Codex | AI 完成：同步 H5 v5 Agent future 位移可视化口径。 |
| 2026-06-03 | 1os3_Codex | AI 完成：同步可视化复用 `B2DH5Dataset`、随机目标点开关和目标索引显示。 |
| 2026-06-03 | 1os3_Codex | AI 完成：同步 H5 v4 Agent future 和 Map 可视化说明。 |
| 2026-06-02 | 1os3_Codex | AI 完成：同步目标候选池和平滑配置可视化说明。 |
| 2026-06-02 | 1os3_Codex | AI 完成：同步 BEV 未来轨迹从 ego 原点连线的可视化约定。 |
| 2026-06-02 | 1os3_Codex | AI 完成：新增 B2D H5 可视化工具摘要。 |
