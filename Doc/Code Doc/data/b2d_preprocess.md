# data/b2d_preprocess.py

## 1. 文件职责

`data/b2d_preprocess.py` 负责将 B2D 原始逐帧场景转换为 MonoDrive 训练用的逐场景 H5 文件。该文件处理场景发现、10Hz 到 5Hz 的输入下采样、滑窗样本索引、未来轨迹、未来可达目标候选与自车状态标签构造、Agent 索引、town/scene 两级 HD Map 紧凑缓存、前视 RGB 图像缩放、预处理日志，以及 H5 写入。

该文件不负责训练时的批量读取；训练读取由 `data/b2d_dataset.py` 负责。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `ScenePaths` | dataclass | 记录一个真实 B2D 场景目录及其 `anno/`、`camera/rgb_front/` 路径。 |
| `SampleWindow` | dataclass | 记录一个样本的当前帧、8 帧历史输入帧和未来 6 个监督帧。 |
| `B2DPreprocessConfig` | dataclass | 预处理配置，集中定义帧率、图像尺寸、滑窗、Map 缓存目录和 H5 压缩参数。 |
| `discover_b2d_scenes` | function | 递归发现包含 `anno/` 和 `camera/<camera_name>/` 的场景，兼容二级目录。 |
| `build_sample_windows` | function | 根据原始 10Hz 帧号构造 5Hz 输入滑窗和 2Hz 未来轨迹帧。 |
| `preprocess_b2d_dataset` | function | 批量预处理根目录下的所有场景。 |
| `B2DScenePreprocessor` | class | 单场景预处理器，提供数组构造和 H5 写入。 |
| `main` | function | 命令行入口，用于批量生成 H5。 |

## 3. 关键类和函数

### `B2DPreprocessConfig`

- 功能：定义预处理的唯一默认参数来源。
- 输入：原始数据根目录、输出目录、Map 缓存目录、相机名、图像尺寸、帧率、滑窗、最大 Agent 数和 H5 压缩设置。
- 输出：供预处理器使用的配置对象。
- Shape：图像尺寸使用 `[H, W]`，默认 `[288, 512]`。
- 关键参数：
  - `raw_fps=10`：B2D 原始帧率。
  - `model_fps=5`：模型输入帧率。
  - `input_frame_count=8`：模型输入最近 8 帧。
  - `trajectory_fps=2`、`future_seconds=3.0`：未来轨迹标签为 3 秒 2Hz，共 6 点。
  - `window_stride=1`：滑窗步长为 1 个 5Hz 模型帧，即 0.2 秒。
  - `target_min_distance=24.0`、`target_max_distance=30.0`：目标候选点默认来自未来 24-30m 可达轨迹点。
  - `max_target_points=32`：每个样本最多保存 32 个目标候选。
  - `target_search_seconds=None`：默认搜索当前帧之后全部未来帧；若配置为正数，则限制目标候选搜索时长。
  - `smooth_future_trajectory=False`：默认不对未来轨迹做平滑；通常不建议开启。

### `discover_b2d_scenes`

- 功能：从数据根目录递归寻找真实场景目录。
- 输入：`raw_root` 和 `camera_name`。
- 输出：`list[ScenePaths]`。
- 兼容性：既支持 `datasets/Scene/anno`，也支持 `datasets/Scene/Scene/anno`。

### `build_sample_windows`

- 功能：基于整数帧号生成样本窗口。
- 输入：原始帧号列表和 `B2DPreprocessConfig`。
- 输出：`list[SampleWindow]`。
- Shape：
  - `input_frame_ids`: `[8]`。
  - `future_frame_ids`: `[6]`。
- 采样规则：
  - 历史输入：`current - 14, current - 12, ..., current`，对应 10Hz 原始帧中每 2 帧取 1 帧。
  - 未来监督：`current + 5, current + 10, ..., current + 30`，对应未来 3 秒 2Hz。

### `B2DScenePreprocessor.build_scene_arrays`

- 功能：读取 annotation 并构造除图像像素外的全部 H5 数组。
- 输入：`ScenePaths`。
- 输出：包含 sample、label、frame 索引和图像路径的字典。
- Shape：
  - `future_trajectory`: `[S, 6, 2]`。
  - `ego_motion`: `[S, 3]`，内容为 `[Vx, Vy, W]`。
  - `target_point`: `[S, 2]`。
  - `target_points`: `[S, 32, 2]`。
  - `target_valid`: `[S, 32]`。
  - `agent_boxes`: `[S, 194, 10]`。
  - `agent_future_trajectory`: `[S, 194, 6, 2]`，以当前 Agent 为原点的未来位移。
  - `agent_future_valid`: `[S, 194, 6]`。
  - `map_points`: `[S, 60, 100, 2]`。
  - `map_classes`: `[S, 60]`。
  - `map_valid`: `[S, 60]`。

### `B2DScenePreprocessor.preprocess_scene`

- 功能：将单场景写入 H5。
- 输入：`ScenePaths`、输出路径、是否覆盖。
- 输出：H5 路径。
- H5 结构：

| 路径 | Shape / 类型 | 说明 |
| --- | --- | --- |
| `frames/rgb_front` | `[F, H, W, 3] uint8` | 去重后的 5Hz 前视 RGB 输入帧。 |
| `frames/frame_ids` | `[F] int32` | 图像帧号。 |
| `samples/input_frame_indices` | `[S, 8] int32` | 每个样本输入帧在 `frames/rgb_front` 中的索引。 |
| `samples/input_frame_ids` | `[S, 8] int32` | 每个样本输入帧原始帧号。 |
| `samples/future_frame_ids` | `[S, 6] int32` | 未来轨迹监督帧号。 |
| `labels/future_trajectory` | `[S, 6, 2] float32` | ego 坐标系未来轨迹，单位 meter，默认不平滑。 |
| `labels/ego_motion` | `[S, 3] float32` | 当前自车 `[Vx, Vy, W]`，由轨迹差分构造。 |
| `labels/target_point` | `[S, 2] float32` | ego 坐标系默认目标点，为候选池首个有效点或最远点兜底。 |
| `labels/target_points` | `[S, 32, 2] float32` | ego 坐标系未来可达目标候选点，默认保存 24-30m 范围内的未来轨迹点。 |
| `labels/target_valid` | `[S, 32] bool` | 目标候选 padding mask。 |
| `labels/commands` | `[S, 3] int16` | `[command_near, command_far, next_command]`。 |
| `labels/control` | `[S, 3] float32` | `[throttle, steer, brake]`。 |
| `labels/agent_boxes` | `[S, 194, 10] float32` | padded Agent 平面运动标签。 |
| `labels/agent_classes` | `[S, 194] int16` | Agent 类别，padding 为 `-1`。 |
| `labels/agent_valid` | `[S, 194] bool` | Agent padding mask。 |
| `labels/agent_future_trajectory` | `[S, 194, 6, 2] float32` | 每个有效 Agent 的未来 3 秒 2Hz 位移，以当前 Agent 中心为原点，坐标轴沿当前 ego 坐标系。 |
| `labels/agent_future_valid` | `[S, 194, 6] bool` | Agent 未来轨迹逐点有效 mask。 |
| `labels/map_points` | `[S, 60, 100, 2] float32` | 预处理端裁剪并重采样后的局部 Map 元素。 |
| `labels/map_classes` | `[S, 60] int16` | Map 类别，padding 为 `-1`。 |
| `labels/map_valid` | `[S, 60] bool` | Map padding mask。 |
| `labels/traffic_light_*` | 多组 | 影响 ego 的最近红绿灯标签。 |
| `labels/stop_sign_*` | 多组 | 影响 ego 的最近 Stop 标志标签。 |

## 4. 输入输出与 Shape

| 名称 | Shape | 说明 |
| --- | --- | --- |
| 原始 RGB | `[900, 1600, 3]` | 示例 B2D 前视 RGB 原图。 |
| H5 RGB | `[F, 288, 512, 3]` | 预处理后去重储存的输入帧。 |
| 模型输入窗口 | `[S, 8]` | 每个样本的 8 帧 5Hz 图像索引。 |
| 未来轨迹 | `[S, 6, 2]` | 未来 3 秒、2Hz、ego 坐标系米制轨迹，默认不平滑。 |
| 目标候选 | `[S, 32, 2]` | 当前帧之后未来真实 ego 轨迹中的 24-30m 可达目标点，若无候选则保存最远未来点。 |
| 目标候选 mask | `[S, 32]` | 目标候选有效标记。 |
| 自车运动 | `[S, 3]` | `[Vx, Vy, W]`，`Vx/Vy` 单位 m/s，`W` 单位 rad/s。 |
| Agent 运动标签 | `[S, 194, 10]` | `[x, y, l, w, h, yaw, v_x, v_y, a_x, a_y]`。 |
| Agent 未来轨迹 | `[S, 194, 6, 2]` | 有效 Agent 的未来 3 秒 2Hz 位移，以当前 Agent 中心为原点，坐标轴沿当前 ego 坐标系。 |
| Map 元素 | `[S, 60, 100, 2]` | 局部可见 Map polyline，当前 ego 坐标系，每条 100 点。 |

未来点数由公式确定：

$$
K = future\_seconds \times trajectory\_fps = 3 \times 2 = 6
$$

## 5. 关键实现逻辑

场景发现从 `raw_root` 递归搜索 `anno` 目录，再检查同级目录是否存在 `camera/<camera_name>/`。因此即使样例数据被放成 `Scene/Scene/anno`，也能定位到内部真实场景目录。

滑窗构造以原始 10Hz 帧号为基础。模型输入帧率为 5Hz，因此历史帧每 2 个原始帧取 1 帧；轨迹标签为 2Hz，因此未来每 5 个原始帧取 1 点。默认首个有效当前帧需要之前已有 14 个 10Hz 帧，最后一个有效当前帧需要之后还有 30 个 10Hz 帧。

未来轨迹和目标候选都转换到当前 ego 坐标系，约定 `x` 为前向、`y` 为左向、单位 meter。B2D annotation 中 `theta` 与 CARLA/world yaw 的关系按 $yaw = \theta - \pi/2$ 处理，该关系用样例帧 `CAM_FRONT.world2cam` 的前向轴反推校验。轨迹、目标候选、Agent 中心、Agent yaw、速度、加速度和 Map 点必须复用同一 ego 变换，避免只平移 XY 而没有同步旋转朝向。目标候选不直接使用 `x_command_near/y_command_near`、`x_command_far/y_command_far` 或 `x_target/y_target`，而是读取当前帧之后的真实 ego 轨迹点，默认搜索到场景结束，选择距离当前 ego 原点 24-30m 的全部点；若没有点落入该范围，则使用搜索范围内最远未来点作为兜底目标。

未来轨迹默认不做平滑，建议保留原始差分轨迹。实测即使 1 次三点核平滑也可能造成轨迹几何失真，尤其会削弱急弯、急停、避让和路口局部行为；除非用于明确消融实验并经过可视化复核，否则不建议开启。若显式开启，平滑使用三点核：

$$
p'_k = 0.25p_{k-1} + 0.5p_k + 0.25p_{k+1}
$$

当前 ego 原点作为第一个未来点的前置锚点，最后一个 3 秒端点保持不变。该平滑不改变 2Hz 点数、未来帧索引或碰撞判定口径，但可能改变轨迹形状和目标候选分布；开启后必须记录实验口径。

自车运动状态不直接信任标注速度和加速度，而是根据相邻原始帧位姿差分得到 `[Vx, Vy, W]`，与 `doc/Model.md` 中速度和加速度字段由轨迹差分计算的约束一致。

Agent 标签当前保存动态目标的 padded 平面运动状态，并映射到 `car/bicycle/motorcycle/pedestrian` 四类。Agent 位置不保存 `z`，速度与加速度分别保存为 ego 坐标系下的 `[v_x, v_y]` 和 `[a_x, a_y]`，全部通过同一 `id` 的相邻帧轨迹差分得到，不读取数据集提供的 `speed` 或 `acceleration` 字段。

Agent 只在预处理阶段保留当前帧中心位于前向 32m、左右各 32m 范围内，且在历史输入窗口内稳定前视可见的目标。单帧可见性使用 annotation 中 `sensors.CAM_FRONT.world2cam` 作为 world 到前视相机的外参，使用 `sensors.CAM_FRONT.intrinsic` 作为针孔内参，并将 CARLA 相机坐标 `[camera_x, camera_y, camera_z]` 转为 `[right, down, forward] = [camera_y, -camera_z, camera_x]` 后投影到原始前视图像平面。单帧内 Agent 必须有 8 个有效 `world_cord` 角点，至少 2 个角点落在图像内且 `forward > 0.1m`；若前视 8-bit 深度图存在，还要求这些角点在 2x2 邻域内有非 255 的有效表面支撑。最终准入按同一 Agent ID 在 8 帧历史输入窗口 `[t-14, ..., t]` 内计数，默认至少 2 帧满足上述单帧可见性条件才写入当前样本标签。深度图不做米制遮挡比较，历史 depth 图使用小型 LRU 缓存复用，避免滑窗预处理反复读取同一帧。训练读取端只消费已经过滤好的 `agent_*` 张量。每个有效 Agent 还会按未来监督帧 `[t+5, ..., t+30]` 构造 `agent_future_trajectory/agent_future_valid`；其中 future 存储的是以当前 Agent 中心为原点的未来位移，坐标轴仍沿当前 ego 坐标系，不旋转到 Agent 自身 yaw 坐标系。

Map 从 `datasets/hd_map/*_HD_map.npz` 在预处理端加载。预处理器会先扫描 `hd_map_root` 下所有 `*_HD_map.npz` 并建立 Town 索引，再从场景名和路径中提取 `Town\d+` 或 `Town\d+HD`；匹配时先用完整 Town 名精确匹配，例如 `Town10HD` 对应 `Town10HD_HD_map.npz`，再退回到 `Town\d+` 基础名兜底，以兼容文件名带 `HD` 后缀或额外前缀的地图。Map 缓存分为两级，均位于 `map_cache_dir` 且只保存 `class_ids/offsets/points`；未显式指定时 `map_cache_dir` 默认为 `output_dir/map_cache/`。town-level canonical 缓存形如 `town_{map_stem}.npz`，保存同一 Town 地图稀疏化后的通用元素；多个场景或多个输出目录指向同一 `map_cache_dir` 时，可直接读取已有通用缓存。scene-level canonical 缓存形如 `{scene}_{map_stem}.npz`，保存当前场景 bbox 粗裁剪后的元素。缓存匹配只使用场景名和地图名这些前半段有效信息，不再把路径、mtime、bbox 或 hash 写入新缓存文件名；若目录中存在旧版 `{prefix}_{hash}.npz`，则在 canonical 文件不存在时按前缀兼容读取最新可用旧缓存。处理流程优先命中 scene-level 缓存；若 miss，则读取或构建 town-level 缓存，再从中筛出与场景 bbox 相交的元素并写入 scene-level 缓存。局部 Map 元素随后按当前 ego 坐标裁剪到前向 32m、左右各 32m 范围内，按 `lane_divider/road_edge/crosswalk/centerline` 四类写入 H5，每条 polyline 统一重采样为 100 个 ego 坐标系 XY 点，并使用同一套 `CAM_FRONT.world2cam/intrinsic` 检查是否能投影到前视图像中；缺少相机内外参或投影失败时不默认视为可见，避免 Dataset 在训练时解析大体积 HD map 或二次过滤。

## 6. 配置项

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `raw_dataset_root` | `datasets` | 原始 B2D 数据根目录。 |
| `output_dir` | `data/preprocessed` | 逐场景 H5 输出目录，该目录被 Git 忽略。 |
| `hd_map_root` | `datasets/hd_map` | B2D HD Map 根目录。 |
| `map_cache_dir` | `None` | HD Map 紧凑缓存目录；`None` 表示使用 `output_dir/map_cache`，也可指定已有缓存目录直接复用。 |
| `camera_name` | `rgb_front` | 使用前视单目 RGB。 |
| `camera_sensor_name` | `CAM_FRONT` | annotation `sensors` 中的前视相机键。 |
| `image_size` | `(288, 512)` | 输出图像 `[H, W]`。 |
| `raw_fps` | `10` | 原始数据帧率。 |
| `model_fps` | `5` | 模型输入帧率。 |
| `input_frame_count` | `8` | 每个样本历史图像数。 |
| `trajectory_fps` | `2` | 未来规划轨迹标签频率。 |
| `future_seconds` | `3.0` | 未来规划监督时长。 |
| `window_stride` | `1` | 滑窗步长，单位为 5Hz 模型帧。 |
| `max_agents` | `194` | Agent 检测查询数量上限。 |
| `max_map_elements` | `60` | Map 查询数量上限。 |
| `map_point_count` | `100` | 每条 Map 元素重采样点数。 |
| `detection_forward_range` | `32.0` | Agent 与 Map 前向保留范围，单位 meter。 |
| `detection_lateral_range` | `32.0` | Agent 与 Map 左右保留范围，单位 meter。 |
| `min_visible_agent_vertices` | `2` | Agent 3D 框最少可见顶点数。 |
| `min_visible_agent_history_frames` | `2` | 同一 Agent 在历史输入窗口内至少满足单帧可见性条件的帧数。 |
| `map_min_visible_points` | `2` | Map 元素最少前视投影点数。 |
| `hd_map_min_point_spacing` | `0.5` | HD Map 载入后保留点的最小间距，单位 meter。 |
| `target_min_distance` | `24.0` | 目标候选最小距离，单位 meter。 |
| `target_max_distance` | `30.0` | 目标候选最大距离，单位 meter。 |
| `max_target_points` | `32` | 每个样本最多保存的目标候选数量。 |
| `target_search_seconds` | `None` | 目标候选搜索时长；`None` 表示搜索当前帧之后全部未来帧。 |
| `smooth_future_trajectory` | `False` | 是否对未来轨迹做轻量平滑；默认关闭，通常不建议开启。 |
| `trajectory_smoothing_iterations` | `1` | 显式开启平滑时的迭代次数；即使 1 次也可能导致轨迹失真。 |
| `compression` | `gzip` | H5 压缩算法。 |
| `compression_level` | `4` | gzip 压缩等级。 |

## 7. 依赖关系

- 上游：B2D 原始场景目录、`Pillow`、`numpy`、`h5py`。
- 下游：`data/b2d_dataset.py`、训练循环、模型输入管线。

## 8. 注意事项

- 数值稳定性：速度、加速度和角速度差分使用实际帧号间隔计算 `dt`，缺少相邻帧时返回零运动状态或零加速度，不回退到数据集速度/加速度标注。
- 性能：图像按去重后的 5Hz 帧储存，样本只保存图像索引，避免每个滑窗重复写入 8 张图；Agent 先建立逐帧 ID 索引，HD Map 使用 town-level 通用缓存和 scene-level 裁剪缓存，降低同 Town 多场景预处理时的重复解包、重复稀疏化和整城 Map 常驻内存。
- 日志：CLI 默认以 `INFO` 输出场景发现、样本数量、Map 缓存命中、H5 写入和耗时；`--log-level DEBUG` 会额外输出 Map bbox、样本构造进度和 RGB 写入进度。
- 兼容性：H5 写入需要 `h5py`；导入模块本身不强制导入 `h5py`，方便仅运行索引和编译检查。
- 使用端成本：Agent 可见性过滤、范围过滤、Agent future 构造和 Map 裁剪重采样都应保留在预处理端，Dataset 不做二次几何处理。
- 数据管理：`data/preprocessed/`、`data/preprocessed/map_cache/` 和 `*.h5` 已加入 `.gitignore`；若使用自定义 `map_cache_dir`，应确保该目录同样不提交。
- 缓存命名：新写入的 Map 缓存文件名不带 hash；旧版带 hash 缓存只作为前缀兼容读取，不再作为新写入格式。
- 标注口径：未来轨迹平滑默认关闭且通常不建议开启；若显式开启、修改目标候选距离或限制 `target_search_seconds`，应在实验记录中说明，并同步检查目标点栅格覆盖范围。

## 9. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-05 | 1os3_Codex | AI 完成：移除新 Map 缓存文件名 hash，并按前缀兼容读取旧 hash 缓存。 |
| 2026-06-05 | 1os3_Codex | AI 完成：新增预处理详细日志和可指定 `map_cache_dir` 的 HD Map 缓存复用机制。 |
| 2026-06-05 | 1os3_Codex | AI 完成：升级 H5 v5，将 Agent future 从当前 ego 绝对坐标改为当前 Agent 原点、ego 坐标轴下的未来位移。 |
| 2026-06-05 | 1os3_Codex | AI 完成：新增 town-level HD Map 通用缓存，并保留 scene-level 裁剪缓存以复用同 Town 多场景地图处理结果。 |
| 2026-06-04 | 1os3_Codex | AI 完成：将 Agent 可见性准入改为历史输入窗口内至少 2 帧满足单帧可见性条件，并新增历史 depth 图缓存。 |
| 2026-06-04 | 1os3_Codex | AI 完成：修正 B2D `theta` 到 world yaw 的换算，并将 HD Map 匹配改为先扫描地图索引再按 Town 精确/基础名匹配。 |
| 2026-06-04 | 1os3_Codex | AI 完成：修复 `Town10HD` 类场景的 HD Map 匹配，支持多场景从各自 Town 地图生成缓存。 |
| 2026-06-03 | 1os3_Codex | AI 完成：明确 Agent/Map 可见性使用 `CAM_FRONT.world2cam/intrinsic`，并收紧投影失败时的过滤行为。 |
| 2026-06-03 | 1os3_Codex | AI 完成：将未来轨迹平滑改为默认关闭，并记录不建议开启的标注口径。 |
| 2026-06-03 | 1os3_Codex | AI 完成：优化预处理端内存与计算效率，新增场景级 HD Map 紧凑缓存，并将 Agent 查找改为逐帧 ID 索引。 |
| 2026-06-03 | 1os3_Codex | AI 完成：升级 H5 v4，新增 Agent 可见/范围过滤、Agent 未来轨迹和局部 Map 预处理字段。 |
| 2026-06-02 | 1os3_Codex | AI 完成：将目标点默认来源改为未来 24-30m 可达候选池，新增 `target_points/target_valid` H5 字段说明。 |
| 2026-06-02 | 1os3_Codex | AI 完成：新增未来轨迹轻量平滑配置说明。 |
| 2026-06-02 | 1os3_Codex | AI 完成：将 Agent 标签改为 `[x, y, l, w, h, yaw, v_x, v_y, a_x, a_y]`，速度和加速度全部由轨迹差分构造。 |
| 2026-06-02 | 1os3_Codex | AI 完成：新增 B2D 场景发现、滑窗构造、标签构造和逐场景 H5 写入说明。 |
