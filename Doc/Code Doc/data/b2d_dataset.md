# data/b2d_dataset.py

## 1. 文件职责

`data/b2d_dataset.py` 负责读取 `data/b2d_preprocess.py` 生成的逐场景 H5，并以 PyTorch `Dataset` 形式返回训练样本。该文件处理多 H5 文件索引、worker 内 H5 句柄懒加载、H5 dataset 对象缓存、图像张量布局转换、目标候选随机抽样、标签张量类型转换和资源关闭。

该文件不负责原始 B2D annotation 解析或 H5 生成。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `B2DH5Dataset` | class | 读取一个 H5 文件、一个 H5 目录或 H5 文件列表，并返回训练样本字典。 |
| `SUPPORTED_SCHEMA_VERSIONS` | constant | 当前读取端支持的 H5 schema 集合，防止旧语义 H5 被静默用于训练。 |

## 3. 关键类和函数

### `B2DH5Dataset`

- 功能：训练时读取预处理后的 B2D H5。
- 输入：`h5_paths`、`normalize_images`、`image_dtype`、`random_target_point`。
- 输出：PyTorch 样本字典。
- Shape：
  - `images`: `[8, 3, 288, 512]`。
  - `future_trajectory`: `[6, 2]`。
  - `ego_motion`: `[3]`。
  - `agent_boxes`: `[194, 10]`。
- 关键参数：
  - `normalize_images=True` 时将图像从 uint8 转为浮点并除以 255。
  - `image_dtype=torch.float32` 默认返回 FP32 图像，后续训练可在模型或 dataloader 中切换精度。
  - `random_target_point=True` 时从 `target_valid` 标记的目标候选中随机抽取一个作为训练 `target_point`。

### `B2DH5Dataset.close`

- 功能：关闭当前进程缓存的 H5 文件句柄。
- 输入：无。
- 输出：无。
- 使用场景：长时间调试或显式释放文件锁时调用。

## 4. 输入输出与 Shape

| 返回字段 | Shape / 类型 | 说明 |
| --- | --- | --- |
| `images` | `[8, 3, H, W] float/uint8` | 最近 8 帧前视 RGB，默认已归一化到 `[0, 1]`。 |
| `ego_motion` | `[3] float32` | 当前自车 `[Vx, Vy, W]`。 |
| `future_trajectory` | `[6, 2] float32` | 未来 3 秒 2Hz ego 坐标系轨迹。 |
| `target_point` | `[2] float32` | ego 坐标系训练目标点；默认从有效 `target_points` 中随机抽取。 |
| `target_points` | `[32, 2] float32` | 未来 24-30m 可达目标候选点；若无范围内候选则包含最远未来点。 |
| `target_valid` | `[32] bool` | 目标候选 padding mask。 |
| `target_point_index` | scalar int64 | 本次返回的 `target_point` 在候选池中的索引。 |
| `commands` | `[3] int64` | `[command_near, command_far, next_command]`。 |
| `control` | `[3] float32` | `[throttle, steer, brake]`。 |
| `current_pose` | `[3] float32` | 当前 world 坐标 `[x, y, theta]`。 |
| `agent_boxes` | `[194, 10] float32` | padded Agent 平面运动标签：`[x, y, l, w, h, yaw, v_x, v_y, a_x, a_y]`。 |
| `agent_classes` | `[194] int64` | Agent 类别，padding 为 `-1`。 |
| `agent_valid` | `[194] bool` | Agent 有效 mask。 |
| `agent_future_trajectory` | `[194, 6, 2] float32` | 有效 Agent 的未来 3 秒 2Hz 位移，以当前 Agent 中心为原点，坐标轴沿当前 ego 坐标系。 |
| `agent_future_valid` | `[194, 6] bool` | Agent 未来轨迹逐点有效 mask。 |
| `map_points` | `[60, 100, 2] float32` | 已由预处理端裁剪和重采样的局部 Map 元素。 |
| `map_classes` | `[60] int64` | Map 类别，padding 为 `-1`。 |
| `map_valid` | `[60] bool` | Map 有效 mask。 |
| `traffic_light_state` | scalar int64 | 影响 ego 的最近红绿灯状态。 |
| `traffic_light_xy` | `[2] float32` | 红绿灯 ego 坐标。 |
| `traffic_light_valid` | scalar bool | 红绿灯标签是否有效。 |
| `stop_sign_state` | scalar int64 | Stop 标志状态。 |
| `stop_sign_xy` | `[2] float32` | Stop 标志 ego 坐标。 |
| `stop_sign_valid` | scalar bool | Stop 标志标签是否有效。 |
| `input_frame_ids` | `[8] int64` | 原始输入帧号。 |
| `future_frame_ids` | `[6] int64` | 未来监督帧号。 |
| `current_frame_id` | int | 当前样本原始帧号。 |
| `scene_name` | str | 场景名。 |
| `h5_path` | str | 样本来源 H5。 |

## 5. 关键实现逻辑

初始化时只打开每个 H5 读取样本数和场景名，不长期持有文件句柄。`__getitem__` 第一次访问某个场景时，在当前进程内懒加载对应 H5 句柄，并缓存常用 H5 dataset 对象，避免每个样本反复用字符串路径查找；缓存的是 h5py dataset 句柄，不会把整份数组读入内存。这使 PyTorch 多 worker 读取时每个 worker 拥有自己的文件句柄和 dataset 缓存。

初始化阶段会检查 `schema_version`，当前只接受 `b2d_h5_v5`。H5 v5 将 `agent_future_trajectory` 定义为当前 Agent 原点、当前 ego 坐标轴下的未来位移；旧 H5 v4 中该字段曾表示当前 ego 坐标系下的未来绝对点，因此不能混用。

图像在 H5 中按 `[T, H, W, 3]` 读取，返回前转换为 `[T, 3, H, W]`，以匹配 PyTorch 模型习惯。标签按训练常用 dtype 转为 `torch.Tensor`，字符串元数据保持 Python 类型。

目标点读取时保留完整候选池 `target_points/target_valid`，并默认在 `__getitem__` 中随机抽取一个有效候选作为 `target_point`。若 `random_target_point=False`，则稳定返回第一个有效候选，便于复现实验或调试可视化。

Agent 可见性过滤、范围裁剪、Agent future 构造和 Map 裁剪重采样都在 H5 预处理阶段完成。本文件只做 H5 切片、dtype 转换和目标点随机抽样，不执行投影、深度判断或 HD map 解析。`agent_future_trajectory` 直接返回 H5 中的位移标签，读取端不把它加回 Agent 当前中心。

全局样本索引通过累积长度映射到 `(scene_index, local_index)`，因此一个 Dataset 可以同时读取多个逐场景 H5。

## 6. 配置项

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `h5_paths` | 无 | 必填；可以是单文件、目录或文件列表。 |
| `normalize_images` | `True` | 是否将图像归一化到 `[0, 1]`。 |
| `image_dtype` | `torch.float32` | 返回图像 dtype。 |
| `random_target_point` | `True` | 是否在每次读取时随机抽取一个有效目标候选。 |

## 7. 依赖关系

- 上游：`data/b2d_preprocess.py` 生成的 H5。
- 下游：训练 dataloader、模型输入管线。
- 第三方依赖：`h5py`、`numpy`、`torch`。

## 8. 注意事项

- H5 读取需要 `h5py`；缺少依赖时会给出明确安装提示。
- Windows 下同一个 H5 文件正在写入时不能并行读取，验证应先等待写入完成。
- `normalize_images=False` 时仍会按 `image_dtype` 转换图像；若需要保留 `uint8`，应传入 `image_dtype=torch.uint8`。
- 修改 H5 schema 时必须同步修改本文件字段读取逻辑。
- 当前读取器只支持 `b2d_h5_v5`；旧 H5 必须重新预处理，避免 Agent future 绝对点和位移标签混用。
- Agent 速度和加速度由预处理阶段轨迹差分得到，读取端不应再读取或拼接数据集原始速度/加速度字段。
- Agent future 和 Map 标签必须直接读取 H5；读取端不应再做可见性过滤或 Map 重采样。Agent future 是当前 Agent 原点、ego 坐标轴下的未来位移。
- `target_point` 是读取端从 H5 目标候选池中抽取得到的训练条件；需要确定性评估时应设置 `random_target_point=False`。

## 9. 维护记录

| 日期 | 修改人 | 变更 |
| --- | --- | --- |
| 2026-06-05 | 1os3_Codex | AI 完成：同步 H5 v5 Agent future 位移读取语义。 |
| 2026-06-03 | 1os3_Codex | AI 完成：优化读取端 H5 访问路径，新增每场景 dataset 对象缓存并在关闭/序列化时清理。 |
| 2026-06-03 | 1os3_Codex | AI 完成：同步 H5 v4 的 Agent future 与 Map 读取字段，并声明读取端不做二次几何处理。 |
| 2026-06-02 | 1os3_Codex | AI 完成：同步目标候选池读取与训练随机抽取 `target_point` 语义。 |
| 2026-06-02 | 1os3_Codex | AI 完成：同步 Agent 10D 平面运动标签说明。 |
| 2026-06-02 | 1os3_Codex | AI 完成：新增 B2D H5 PyTorch Dataset 说明。 |
