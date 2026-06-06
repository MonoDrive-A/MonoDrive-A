# data/b2d_preprocess.py 摘要

## 1. 文件基本功能

将 B2D 原始场景预处理为 MonoDrive 训练用逐场景 H5，包含二级目录兼容、10Hz 到 5Hz 输入下采样、未来 6 点轨迹标签、未来 24-30m 可达目标候选、自车运动状态、可见范围内 Agent future、局部 Map 标签、可指定目录的 town/scene 两级 Map 紧凑缓存、详细日志和 Agent ID 索引。

## 2. 主要公开接口

| 名称 | 类型 | 功能 |
| --- | --- | --- |
| `B2DPreprocessConfig` | dataclass | 定义帧率、滑窗、图像尺寸、输出目录、Map 缓存目录和压缩参数。 |
| `discover_b2d_scenes` | function | 递归发现合法 B2D 场景。 |
| `build_sample_windows` | function | 构造 8 帧历史输入和 6 点未来标签索引。 |
| `B2DScenePreprocessor` | class | 构造预处理数组并写入 H5。 |
| `preprocess_b2d_dataset` | function | 批量预处理所有场景。 |

## 3. 输入输出 Shape 概览

| 数据 | Shape | 说明 |
| --- | --- | --- |
| `frames/rgb_front` | `[F, 288, 512, 3]` | 5Hz 去重前视 RGB。 |
| `samples/input_frame_indices` | `[S, 8]` | 每个样本的历史图像索引。 |
| `labels/future_trajectory` | `[S, 6, 2]` | 未来 3 秒、2Hz、ego 坐标系轨迹。 |
| `labels/target_point` | `[S, 2]` | 默认目标点，为候选池首个有效点或最远点兜底。 |
| `labels/target_points` | `[S, 32, 2]` | 未来 24-30m 可达目标候选点。 |
| `labels/target_valid` | `[S, 32]` | 目标候选 padding mask。 |
| `labels/ego_motion` | `[S, 3]` | `[Vx, Vy, W]`。 |
| `labels/agent_boxes` | `[S, 194, 10]` | padded Agent 平面运动标签。 |
| `labels/agent_future_trajectory` | `[S, 194, 6, 2]` | Agent 未来 3 秒 2Hz 位移，以当前 Agent 中心为原点，坐标轴沿当前 ego 坐标系。 |
| `labels/agent_future_valid` | `[S, 194, 6]` | Agent future 有效 mask。 |
| `labels/map_points` | `[S, 60, 100, 2]` | 局部 Map 元素。 |
| `labels/map_valid` | `[S, 60]` | Map padding mask。 |

## 4. 公开接口使用规范

| 接口 | 使用规范 |
| --- | --- |
| `B2DPreprocessConfig` | `raw_fps` 必须整除 `model_fps` 和 `trajectory_fps`；`future_seconds * trajectory_fps` 必须为整数。 |
| `map_cache_dir` | 默认为 `output_dir/map_cache`；可显式指向已有缓存目录，使多次预处理或不同输出目录直接复用同一批 Map 缓存。 |
| `target_min_distance/target_max_distance` | 默认 24-30m；修改时必须同步检查模型目标点栅格覆盖范围。 |
| `target_search_seconds` | 默认 `None`，表示读取当前帧之后全部未来点；若限制时长，应记录实验口径。 |
| `smooth_future_trajectory` | 默认关闭，通常不建议开启；若为消融实验显式开启，必须记录实验口径并可视化复核。 |
| `detection_forward_range/detection_lateral_range` | 默认前向 32m、左右各 32m；Agent 与 Map 都在预处理端裁剪。 |
| `map_point_count` | 默认 100；每条局部 Map polyline 都在预处理端重采样。 |
| `discover_b2d_scenes` | 传入原始数据根目录即可；函数会递归兼容 `Scene/Scene` 二级结构。 |
| `build_sample_windows` | 输入必须是整数帧号；缺少历史或未来帧的样本会被跳过。 |
| `B2DScenePreprocessor.preprocess_scene` | 需要 `h5py`；默认不覆盖已有 H5，覆盖需显式传入 `overwrite=True`。 |

## 5. 最小使用示例

```python
from data.b2d_preprocess import B2DPreprocessConfig, preprocess_b2d_dataset

config = B2DPreprocessConfig(raw_dataset_root="datasets", output_dir="data/preprocessed")
output_paths = preprocess_b2d_dataset(config, overwrite=True)
```

## 6. 维护注意事项

- 修改 H5 schema 时必须同步更新 `data/b2d_dataset.py` 和两份 Code Doc。
- 修改帧率、轨迹点数或坐标系时必须同步更新 `doc/Model.md`。
- 默认 target 来自未来真实 ego 轨迹候选池；不要回退为直接读取 B2D command 或 target 字段。
- B2D `theta` 到 world yaw 的换算使用 $yaw = \theta - \pi/2$；轨迹、Agent、速度、加速度和 Map 必须复用同一 ego 坐标变换。
- 未来轨迹默认不平滑；即使 1 次平滑也可能导致轨迹几何失真，危险轨迹碰撞判定仍必须使用未来每帧实际 Agent 标签。
- Agent 可见性过滤使用 `CAM_FRONT.world2cam/intrinsic` 投影 3D 角点；缺少有效角点、缺少相机内外参或投影失败时不应默认视为可见。
- Agent 准入要求同一 Agent ID 在 8 帧历史输入窗口内至少 2 帧满足单帧可见性条件；当前帧中心仍必须在检测范围内。
- Agent 范围裁剪、Agent future 构造、Map 可见性过滤和 Map 重采样必须在预处理端完成。
- Agent future 存储未来位移而不是绝对轨迹点；原点为当前 Agent 中心，坐标轴沿当前 ego 坐标系，不旋转到 Agent 自身坐标系。
- HD Map 匹配必须先扫描 `hd_map_root` 建立地图索引，再按场景 Town 精确名和 `Town\d+` 基础名匹配；多场景预处理时，每个场景应解析到各自 Town 的 `*_HD_map.npz`。
- HD Map 首次处理后会在 `map_cache_dir` 生成 town-level 通用缓存和 scene-level 裁剪缓存；自定义缓存目录也属于预处理产物，不应提交到 Git。
- Map 缓存 canonical 文件名不带 hash，只使用场景名和地图名；旧版 `{prefix}_{hash}.npz` 缓存可按前缀兼容读取。
- CLI 默认 `INFO` 日志会输出场景进度、缓存命中、H5 写入和耗时；需要排查缓存和样本构造时使用 `--log-level DEBUG`。
- 所有速度和加速度标签必须来自轨迹差分，不允许回退到数据集标注字段。
- 预处理输出属于大数据文件，不应提交到 Git。

## 7. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-05 | 1os3_Codex | AI 完成：同步 Map 缓存无 hash canonical 命名和旧 hash 前缀兼容读取。 |
| 2026-06-05 | 1os3_Codex | AI 完成：同步预处理日志和可指定 Map 缓存目录的口径。 |
| 2026-06-05 | 1os3_Codex | AI 完成：同步 H5 v5 Agent future 位移语义。 |
| 2026-06-05 | 1os3_Codex | AI 完成：同步 town/scene 两级 HD Map 缓存口径。 |
| 2026-06-04 | 1os3_Codex | AI 完成：同步 Agent 历史窗口至少 2 帧可见的准入口径。 |
| 2026-06-04 | 1os3_Codex | AI 完成：同步 B2D `theta` 换算修正和 HD Map 扫描索引匹配口径。 |
| 2026-06-04 | 1os3_Codex | AI 完成：同步 `Town10HD` 地图匹配和多场景 Map 缓存生成口径。 |
| 2026-06-03 | 1os3_Codex | AI 完成：同步 Agent/Map 前视投影内外参来源和投影失败过滤行为。 |
| 2026-06-03 | 1os3_Codex | AI 完成：同步未来轨迹平滑默认关闭和通常不建议开启的口径。 |
| 2026-06-03 | 1os3_Codex | AI 完成：同步场景级 HD Map 缓存与 Agent ID 索引优化。 |
| 2026-06-03 | 1os3_Codex | AI 完成：同步 H5 v4 Agent future、局部 Map 和预处理端过滤职责。 |
| 2026-06-02 | 1os3_Codex | AI 完成：同步未来 24-30m 可达目标候选池字段。 |
| 2026-06-02 | 1os3_Codex | AI 完成：同步未来轨迹轻量平滑配置。 |
| 2026-06-02 | 1os3_Codex | AI 完成：同步 Agent 10D 平面运动标签与差分速度/加速度口径。 |
| 2026-06-02 | 1os3_Codex | AI 完成：新增 B2D 预处理器摘要文档。 |
