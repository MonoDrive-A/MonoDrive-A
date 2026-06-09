# close_loop/monodrive/agent.py 摘要

## 1. 文件基本功能

`MonoDriveAgent`：在 Carla 同步仿真中缓存 8 帧输入，调用 `MonoDriveBackbone` 推理，解码 winner 轨迹，并通过 PID / pure-pursuit 输出 `carla.VehicleControl`。支持 committed 轨迹持有与 BehaviorAgent 冷启动 fallback。

## 2. 主要公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `MonoDriveAgent` | class | 闭环推理 + 控制 agent。 |
| `InferenceResult` | dataclass | 单步控制与诊断信息。 |
| `CommittedTrajectory` | dataclass | 世界系持有轨迹。 |

## 3. Shape / 控制概览

| 项目 | 说明 |
| --- | --- |
| 模型输入 | `images [1,8,3,288,512]`, `target_points [1,2]`, `ego_motion [1,3]` |
| winner | `argmax(softmax(logits))`，可选迟滞 / 强制索引 |
| 纵向差分 | 轨迹点间隔 `TRAJECTORY_DT=0.5s` |
| 仿真 tick | `dt=0.125s`（8 FPS），用于 yaw_rate 与 PID |

## 4. 使用规范

- checkpoint 须为 `train.checkpointing` 保存格式（含 `model_state`）。
- buffer 未满时自动 fallback，不调用模型。
- `--flip-y` 会同步翻转 motion / target / 输出轨迹 y 分量。

## 5. 维护记录

| 日期 | 修改人 | 说明 |
| --- | --- | --- |
| 2026-06-09 | FuZiR_Cursor | 自 JEPAAgent 迁移至 MonoDriveBackbone。 |
