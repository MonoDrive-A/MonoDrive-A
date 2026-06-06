# data/detection_class_stats.py 摘要

## 1. 文件基本功能

统计 B2D 预处理 H5 数据集的检测类别分布，支持输入整个 H5 目录并递归扫描所有场景文件，输出 Agent、Map、Traffic Light 和 Stop Sign 的全局类别计数与比例。

## 2. 主要公开接口

| 名称 | 类型 | 功能 |
| --- | --- | --- |
| `DetectionClassStatsConfig` | dataclass | 定义 H5 输入、JSON 输出、分块大小和 Traffic 无效样本统计开关。 |
| `DetectionClassStats` | dataclass | 保存跨 H5 聚合统计结果。 |
| `compute_detection_class_stats` | function | 计算检测类别分布。 |
| `save_detection_class_stats` | function | 保存 JSON 统计结果。 |
| `format_detection_class_stats` | function | 输出命令行表格文本。 |

## 3. 输入输出 Shape 概览

| 字段 | Shape | 说明 |
| --- | --- | --- |
| `labels/agent_classes` | `[S, 194]` | Agent 类别，使用 `agent_valid` 过滤。 |
| `labels/map_classes` | `[S, 60]` | Map 类别，使用 `map_valid` 过滤。 |
| `labels/traffic_light_state` | `[S]` | Traffic Light 状态，默认只统计有效标签。 |
| `labels/stop_sign_state` | `[S]` | Stop Sign 状态，默认只统计有效标签。 |
| JSON 输出 | object | 全局统计、逐场景统计和来源 H5 列表。 |

## 4. 公开接口使用规范

| 接口 | 使用规范 |
| --- | --- |
| `DetectionClassStatsConfig(h5_paths)` | 可传入 H5 文件、目录或路径列表；目录会递归扫描 `*.h5`。 |
| `batch_size` | 必须为正数，控制按样本维读取的分块大小。 |
| `include_invalid_traffic` | 默认为 `False`；若需要全样本 `none` 比例，显式开启。 |
| `compute_detection_class_stats` | 输入 H5 必须包含 H5 v4/v5 的检测标签字段。 |

## 5. 最小使用示例

```powershell
.\.venv\Scripts\python.exe -m data.detection_class_stats `
  --h5 data/preprocessed/MonoDrive `
  --output-json visualization/outputs/detection_class_stats.json
```

## 6. 维护注意事项

- 修改 H5 检测字段名或类别映射时必须同步更新本文件和完整 Code Doc。
- Agent 与 Map 必须使用 valid mask 过滤 padding 查询。
- Traffic Light 与 Stop Sign 默认只统计有效标签；不要把无效样本静默混入有效检测分布。
- JSON 输出应放在忽略目录，避免提交分析产物。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-05 | 1os3_Codex | AI 完成：新增跨 H5 检测类别分布统计工具摘要。 |
