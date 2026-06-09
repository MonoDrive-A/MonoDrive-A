# close_loop/monodrive/inputs.py 摘要

## 1. 文件基本功能

将 Carla 仿真状态转换为 `MonoDriveBackbone` 所需张量：8 帧图像 ring buffer、ego 状态 ring buffer，以及 anchor 帧 `ego_motion` / `target_point` 构造与坐标变换。

## 2. 主要公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `PAST_FRAMES` | constant | 历史帧数，值为 8。 |
| `TRAJECTORY_DT` | constant | 未来轨迹点时间间隔 (s)，值为 0.5。 |
| `FrameBuffer` | class | 缓存并 stack 最近 8 帧 `[0,1]` 图像。 |
| `EgoBuffer` | class | 缓存最近 8 帧 ego 世界系状态。 |
| `build_ego_motion` | function | 输出 `(3,)` 物理 `[Vx, Vy, W]`。 |
| `build_target_point` | function | 输出 `(2,)` ego-local 目标点。 |
| `ego_local_to_world` | function | 轨迹点 ego-local → 世界系。 |

## 3. 输入输出 Shape 概览

| 接口 | 输入 | 输出 Shape |
| --- | --- | --- |
| `FrameBuffer.stack` | 8 帧 push | `[8, 3, 288, 512]` |
| `build_ego_motion` | 满 `EgoBuffer` | `(3,)` |
| `build_target_point` | 满 buffer + goal 世界坐标 | `(2,)` |

## 4. 公开接口使用规范

- 图像 buffer 只存 `[0, 1]` FP32；DINO mean/std 由模型内部处理。
- `ego_motion` / `target_point` 必须为物理米制量，Symlog 在模型内完成。
- 摄像头分辨率必须为 1600×900（见 `model.image_geometry.SOURCE_HW`）。

## 5. 维护记录

| 日期 | 修改人 | 说明 |
| --- | --- | --- |
| 2026-06-09 | FuZiR_Cursor | 自 JEPA 16 帧输入迁移为 MonoDrive 8 帧与 `target_point` 命名。 |
