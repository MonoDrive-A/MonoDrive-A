# visualization/b2d_h5_viewer.py

## 1. 文件职责

`visualization/b2d_h5_viewer.py` 负责将 B2D 预处理 H5 中的单个或多个样本导出为 PNG 诊断图，帮助开发者检查输入帧、滑窗索引、未来轨迹、目标候选、Agent 标签和交通元素标签是否符合预期。样本读取直接复用 `data.b2d_dataset.B2DH5Dataset`，避免可视化端和训练读取端解析口径不一致。

该文件不负责修改 H5，不负责训练，也不依赖 GUI。默认输出目录为 `visualization/outputs/`，该目录不提交到 Git。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `BevViewConfig` | dataclass | 配置 BEV 面板的 ego 坐标范围、尺寸、网格和速度箭头时长。 |
| `H5SampleData` | dataclass | 封装单个 H5 样本的图像、轨迹、Agent 和元数据。 |
| `render_h5_sample` | function | 从 H5 读取指定样本并导出 PNG。 |
| `load_h5_sample` | function | 通过 `B2DH5Dataset` 读取并校验单个 H5 样本。 |
| `render_sample` | function | 将 `H5SampleData` 渲染为 `PIL.Image.Image`。 |
| `main` | function | 命令行入口。 |

## 3. 关键类和函数

### `BevViewConfig`

- 功能：定义 BEV 可视化范围。
- 输入：`x_min/x_max/y_min/y_max`、面板宽高、网格步长和速度箭头时间。
- 输出：不可变配置对象。
- Shape：BEV 面板默认 `[640, 640]` 像素。
- 坐标系：ego 坐标系，`x` 前向、`y` 左向，单位 meter。

### `load_h5_sample`

- 功能：读取 H5 中一个样本并展开 8 帧输入图像。
- 输入：H5 路径、样本索引和是否随机抽样目标点。
- 输出：`H5SampleData`。
- 读取路径：调用 `B2DH5Dataset(normalize_images=False, image_dtype=torch.uint8)` 得到训练端同源样本，再将图像从 `[8, 3, H, W]` 转回 `[8, H, W, 3]` 供 PIL 绘制。
- 异常：当 `agent_boxes` 最后一维不是 10 时抛出 `ValueError`，提示重新预处理。

### `render_sample`

- 功能：把样本渲染为一张 PNG 诊断图。
- 输入：`H5SampleData`、`BevViewConfig`、`agent_limit`。
- 输出：`PIL.Image.Image`。
- 画面区域：
  - 左上：当前 `rgb_front`。
  - 左中：8 帧 5Hz 历史输入缩略图。
  - 左下：样本元数据。
  - 右上：ego BEV，包含未来轨迹、目标候选、交通元素和 Agent。
  - 右下：颜色图例与 Agent 10D 字段说明。

### `render_h5_sample`

- 功能：端到端读取 H5 样本并写出 PNG。
- 输入：H5 路径、样本索引、输出路径。
- 输出：输出 PNG 路径。

## 4. 输入输出与 Shape

| 名称 | Shape | 说明 |
| --- | --- | --- |
| `frames/rgb_front` | `[F, H, W, 3]` | H5 中的前视 RGB。 |
| `images` | `[8, H, W, 3]` | 单样本历史输入图像。 |
| `future_trajectory` | `[6, 2]` | ego 坐标系未来 3 秒 2Hz 轨迹。 |
| `target_point` | `[2]` | ego 坐标系默认目标点，为候选池首个有效点或最远点兜底。 |
| `target_point_index` | scalar | 当前 `target_point` 在候选池中的索引。 |
| `target_points` | `[32, 2]` | ego 坐标系未来 24-30m 可达目标候选点。 |
| `target_valid` | `[32]` | 目标候选 padding mask。 |
| `ego_motion` | `[3]` | `[v_x, v_y, w]`。 |
| `agent_boxes` | `[194, 10]` | `[x, y, l, w, h, yaw, v_x, v_y, a_x, a_y]`。 |
| `agent_classes` | `[194]` | `car/bicycle/motorcycle/pedestrian` 类别 id。 |
| `agent_valid` | `[194]` | Agent padding mask。 |
| `agent_future_trajectory` | `[194, 6, 2]` | 有效 Agent 的未来 3 秒 2Hz 位移，原点为当前 Agent 中心，坐标轴沿当前 ego 坐标系。 |
| `agent_future_valid` | `[194, 6]` | Agent 未来轨迹逐点 mask。 |
| `map_points` | `[60, 100, 2]` | 局部 Map 元素。 |
| `map_classes` | `[60]` | Map 类别 id。 |
| `map_valid` | `[60]` | Map padding mask。 |

BEV 像素映射为：

$$
p_x = \frac{y-y_{min}}{y_{max}-y_{min}}(W-1)
$$

$$
p_y = \frac{x_{max}-x}{x_{max}-x_{min}}(H-1)
$$

其中 ego `x` 越大越靠图像上方，ego `y` 越大越靠图像右侧。

## 5. 关键实现逻辑

命令行入口解析 H5、样本索引、输出路径和 BEV 范围。若导出多个样本，按 `sample_index + offset * stride` 生成多个 PNG。

`load_h5_sample` 不再直接从 `frames/` 和 `labels/` 手动切片，而是实例化 `B2DH5Dataset` 读取样本，确保图像归一化开关、目标候选抽样、字段 dtype 和训练读取端保持一致。可视化默认关闭随机目标点抽样，使同一个样本反复出图稳定；传入 `random_target_point=True` 或命令行 `--random-target-point` 时，会复用 Dataset 的随机目标点选择逻辑。该函数仍会单独读取 H5 attrs，用于显示 schema、目标距离范围、过滤范围和平滑配置。

该函数强制要求 `agent_boxes` 为 10D，并读取 H5 v5 的 `target_points/target_valid`、`agent_future_*` 与 `map_*` 字段，以避免旧 H5 schema 被误判为新格式。

BEV 面板中，ego 车辆位于 `(0, 0)`，蓝色折线表示 ego 未来轨迹。`future_trajectory` 的第一个标签点是 `t+0.5s`，可视化时会从 ego 原点连到第一个未来点，避免诊断图看起来像轨迹与车辆断开。绿色空心点表示有效目标候选，绿色十字表示 H5 中的默认 `target_point`。灰/青/蓝/粉色细线表示 H5 中已经裁剪和重采样后的 Map 元素。Agent 框只来自 H5 中已通过可见性与范围过滤的 `agent_valid`，橙/绿/紫/红色折线表示对应 Agent future；由于 H5 v5 中 Agent future 保存的是位移，可视化会先加回当前 Agent 中心再绘制到 BEV 绝对 ego 坐标。metadata 会显示候选数量、目标距离范围、过滤范围、Agent future 数量和 Map 数量。

## 6. 配置项

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--h5` | 无 | 必填，预处理 H5 文件。 |
| `--sample-index` | `0` | 起始样本索引。 |
| `--count` | `1` | 导出样本数量。 |
| `--stride` | `1` | 多样本导出时的样本间隔。 |
| `--output` | `None` | 单样本输出 PNG 路径。 |
| `--output-dir` | `visualization/outputs` | 默认输出目录。 |
| `--bev-x-min` | `-10` | BEV 前向最小距离。 |
| `--bev-x-max` | `90` | BEV 前向最大距离。 |
| `--bev-y-min` | `-40` | BEV 左向最小距离。 |
| `--bev-y-max` | `40` | BEV 左向最大距离。 |
| `--agent-limit` | `80` | 最多绘制的有效 Agent 数。 |
| `--random-target-point` | `False` | 是否启用 Dataset 的随机目标点抽样；默认关闭以保持可复现。 |

## 7. 依赖关系

- 上游：`data/b2d_preprocess.py` 生成的 H5、`data/b2d_dataset.py` 的 `B2DH5Dataset`。
- 下游：人工数据检查、数据预处理调试。
- 第三方依赖：`h5py`、`numpy`、`Pillow`、`torch`。

## 8. 注意事项

- 该工具只导出 PNG，不打开 GUI 窗口。
- `visualization/outputs/` 已加入 `.gitignore`，诊断图不应提交。
- 若 H5 仍是旧的 9D Agent schema，或缺少 H5 v5 的目标候选、Agent future 与 Map 字段，应先重新运行预处理。
- H5 v5 的 Agent future 是位移标签；BEV 绘制时会加回 `agent_boxes[..., :2]`，不要在 Dataset 读取端提前加回。
- BEV 仅使用 Dataset 返回的已预处理 ego 坐标标签，不做相机投影。
- 可视化程序不做 Agent 或 Map 的二次过滤；图中内容应直接反映 H5 预处理结果。
- 若修改 `B2DH5Dataset.__getitem__` 的字段或 dtype，应同步检查本文件的样本转换逻辑。

## 9. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-05 | 1os3_Codex | AI 完成：同步 H5 v5 Agent future 位移可视化，绘制时加回当前 Agent 中心。 |
| 2026-06-03 | 1os3_Codex | AI 完成：可视化样本读取改为复用 `B2DH5Dataset`，并新增随机目标点抽样开关和目标候选索引显示。 |
| 2026-06-03 | 1os3_Codex | AI 完成：显示 H5 v4 Agent future 和局部 Map，并声明可视化端不做二次过滤。 |
| 2026-06-02 | 1os3_Codex | AI 完成：显示 H5 目标候选池、候选距离范围和未来轨迹平滑配置。 |
| 2026-06-02 | 1os3_Codex | AI 完成：修正 BEV 未来轨迹折线，使其从 ego 原点连接到第一个未来标签点。 |
| 2026-06-02 | 1os3_Codex | AI 完成：新增 B2D H5 样本 PNG 可视化工具说明。 |
