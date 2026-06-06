# data/detection_class_stats.py

## 1. 文件职责

`data/detection_class_stats.py` 负责统计一个或多个 B2D 预处理 H5 的检测类别分布。该工具面向整个预处理数据集目录，而不是单个场景；目录输入会递归扫描 `*.h5`，按所有场景累加 Agent、Map、Traffic Light 和 Stop Sign 的类别计数，并可选保存 JSON 结果。

该文件不负责 H5 生成、不读取图像，也不执行训练 Dataset 的随机目标点抽样。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `DetectionClassStatsConfig` | dataclass | 检测类别统计配置，包含 H5 输入、JSON 输出、分块大小和 Traffic 无效样本统计开关。 |
| `DetectionClassStats` | dataclass | 跨 H5 数据集的聚合统计结果。 |
| `compute_detection_class_stats` | function | 读取一个或多个 H5 并返回检测类别分布。 |
| `save_detection_class_stats` | function | 将统计结果保存为 JSON。 |
| `format_detection_class_stats` | function | 将统计结果格式化为命令行文本表格。 |
| `main` | function | 命令行入口。 |

## 3. 关键类和函数

### `DetectionClassStatsConfig`

- 功能：定义统计任务输入和读取策略。
- 输入：H5 文件、H5 目录或路径列表。
- 输出：供统计函数使用的不可变配置对象。
- 关键参数：
  - `h5_paths`：可传入单个 H5、目录或多个路径；目录会递归扫描 `*.h5`。
  - `output_json`：可选 JSON 输出路径。
  - `batch_size=4096`：按样本维分块读取，避免一次性读取整个数据集。
  - `include_invalid_traffic=False`：默认只统计有效 Traffic Light / Stop Sign 标签；开启后把无效样本也计入默认 `none` 状态。

### `compute_detection_class_stats`

- 功能：跨场景聚合检测类别分布。
- 输入：`DetectionClassStatsConfig`。
- 输出：`DetectionClassStats`。
- 统计字段：
  - Agent：`labels/agent_classes`，使用 `labels/agent_valid` 过滤 padding。
  - Map：`labels/map_classes`，使用 `labels/map_valid` 过滤 padding。
  - Traffic Light：`labels/traffic_light_state`，默认使用 `labels/traffic_light_valid` 过滤。
  - Stop Sign：`labels/stop_sign_state`，默认使用 `labels/stop_sign_valid` 过滤。

### `format_detection_class_stats`

- 功能：生成可直接打印的表格文本。
- 输入：`DetectionClassStats`。
- 输出：字符串，包含全局场景数、样本数、有效标签数量和四类检测任务的分布表。

### `save_detection_class_stats`

- 功能：保存完整统计结果。
- 输入：`DetectionClassStats` 和输出路径。
- 输出：JSON 文件路径。
- JSON 内容：包含全局汇总、各类别 count/ratio、来源 H5 列表和逐场景统计。

## 4. 输入输出与 Shape

| H5 字段 | Shape | 统计方式 |
| --- | --- | --- |
| `labels/agent_classes` | `[S, 194]` | 只统计 `agent_valid=True` 的类别。 |
| `labels/agent_valid` | `[S, 194]` | Agent padding mask。 |
| `labels/map_classes` | `[S, 60]` | 只统计 `map_valid=True` 的类别。 |
| `labels/map_valid` | `[S, 60]` | Map padding mask。 |
| `labels/traffic_light_state` | `[S]` | 默认只统计 `traffic_light_valid=True` 的状态。 |
| `labels/traffic_light_valid` | `[S]` | Traffic Light 有效 mask。 |
| `labels/stop_sign_state` | `[S]` | 默认只统计 `stop_sign_valid=True` 的状态。 |
| `labels/stop_sign_valid` | `[S]` | Stop Sign 有效 mask。 |

类别映射：

| 任务 | 类别 |
| --- | --- |
| Agent | `car`、`bicycle`、`motorcycle`、`pedestrian` |
| Map | `lane_divider`、`road_edge`、`crosswalk`、`centerline` |
| Traffic Light | `red`、`green`、`yellow`、`none` |
| Stop Sign | `none`、`present` |

未知类别会输出为 `unknown_{id}`，避免静默丢弃异常标签。

## 5. 关键实现逻辑

输入路径解析接受文件和目录。目录使用递归 `rglob("*.h5")`，因此可直接传入整个预处理数据集根目录。所有路径去重后排序，保证统计结果稳定。

统计时按样本维切片读取 H5 label dataset。Agent 和 Map 是多查询检测任务，必须先用对应 valid mask 过滤 padding 查询，否则 padding 类别 `-1` 会污染分布。Traffic Light 和 Stop Sign 是每样本唯一标签，默认只统计有效标签；当需要衡量“无交通元素”比例时，使用 `--include-invalid-traffic` 将无效样本也计入默认 `none`。

JSON 输出保留逐场景统计，用于定位类别分布异常来自哪个 H5；命令行表格只显示全局汇总，适合快速查看整个数据集。

## 6. 命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--h5` | 必填 | H5 文件或目录；可重复传入，目录会递归扫描 `*.h5`。 |
| `--output-json` | `None` | 可选 JSON 输出路径。 |
| `--batch-size` | `4096` | 按样本维分块读取大小。 |
| `--include-invalid-traffic` | `False` | 是否将无效 Traffic Light / Stop Sign 样本计入默认 `none` 状态。 |

## 7. 使用示例

统计整个预处理数据集：

```powershell
.\.venv\Scripts\python.exe -m data.detection_class_stats `
  --h5 data/preprocessed/MonoDrive `
  --output-json visualization/outputs/detection_class_stats.json
```

同时统计多个数据目录：

```powershell
.\.venv\Scripts\python.exe -m data.detection_class_stats `
  --h5 data/preprocessed/train `
  --h5 data/preprocessed/val `
  --include-invalid-traffic
```

## 8. 依赖关系

- 上游：`data/b2d_preprocess.py` 生成的 H5。
- 下游：数据质量检查、类别重采样策略、训练日志分析。
- 第三方依赖：`h5py`、`numpy`。

## 9. 注意事项

- 该工具直接读取 H5，不经过 `B2DH5Dataset`，因此不会触发图像加载或目标点随机抽样。
- Agent 与 Map 分布默认只统计有效检测标签，不统计 padding。
- Traffic Light 和 Stop Sign 默认只统计有效标签；若需要全样本状态比例，应显式使用 `--include-invalid-traffic`。
- JSON 输出属于分析产物，建议放在已忽略目录，例如 `visualization/outputs/`。

## 10. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-05 | 1os3_Codex | AI 完成：新增跨 H5 数据集检测类别分布统计工具说明。 |
