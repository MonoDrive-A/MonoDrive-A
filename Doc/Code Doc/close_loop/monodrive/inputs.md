# close_loop/monodrive/inputs.py

## 1. 文件职责

负责 Carla 闭环中模型输入的构造：前视图像下采样与缓存、ego 状态缓存、anchor 帧 `ego_motion` 与 `target_point` 计算，以及 ego-local / 世界系 xy 变换。字段语义与 `data/b2d_dataset.py` 对齐。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `PAST_FRAMES` | int | 历史帧数 8。 |
| `TRAJECTORY_DT` | float | 2Hz 轨迹点间隔 0.5 s，供控制差分。 |
| `FrameBuffer` | class | RGB 帧 ring buffer。 |
| `EgoBuffer` | class | ego 快照 ring buffer。 |
| `build_ego_motion` | function | 构造 anchor ego-local 运动向量。 |
| `build_target_point` | function | 构造 anchor ego-local 目标点。 |
| `build_goal_dxy` | function | `build_target_point` 别名。 |
| `ego_local_to_world` | function | 批量 xy 变换到世界系。 |

## 3. 关键类和函数

### `FrameBuffer`

- 输入：Carla BGRA uint8 `(900, 1600, 4)`。
- 内部：RGB → resize 288×512 → `/255`。
- 输出：`stack()` → `[8, 3, 288, 512]` FP32。

### `EgoBuffer`

- 从 `carla.Vehicle` 读取世界系位姿、速度、加速度。
- `yaw_rate` 由相邻帧 yaw 差分 / `dt` 得到（默认 `dt=0.125`）。

### `build_target_point`

- 以 buffer 最后一帧为 anchor，将世界系 goal 变换为 ego-local `(x, y)` 物理量。

## 4. 配置与常量

| 常量 | 值 | 说明 |
| --- | --- | --- |
| `PAST_FRAMES` | 8 | 与 `config/vision_embedding.toml` 一致。 |
| `TRAJECTORY_DT` | 0.5 | 与 B2D 2Hz 未来轨迹标签一致。 |
| `FINAL_HW` | (288, 512) | 来自 `model.image_geometry.MODEL_HW`。 |

## 5. 依赖关系

- `model.image_geometry.resize_frame_chw`
- `numpy`, `torch`

## 6. 维护记录

| 日期 | 修改人 | 说明 |
| --- | --- | --- |
| 2026-06-09 | FuZiR_Cursor | 迁移至 MonoDrive 8 帧输入契约。 |
