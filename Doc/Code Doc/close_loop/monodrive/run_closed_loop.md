# close_loop/monodrive/run_closed_loop.py

## 1. 文件职责

Carla Town10HD（或可配置地图）闭环评测脚本：world 同步设置、ego/NPC spawn、前视相机 1600×900、`DenseRoute` 目标点、`MonoDriveAgent` 控制循环、MP4 与 debug 绘制。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `parse_args` | function | CLI 参数（含 `--backbone-config`, `--viz-top-k`）。 |
| `main` | function | 完整闭环流程。 |

## 3. 主流程

1. `parse_args` / 连接 `carla.Client`
2. 加载地图、TrafficManager、同步 world
3. spawn ego + RGB 相机 + NPC
4. `DenseRoute.compute` 与 `MonoDriveAgent` 构造
5. 每 tick：取图 → push buffer → `run_step` → `apply_control`
6. finally 清理 actor 与 world settings

## 4. 关键 CLI 默认值

| 参数 | 默认 |
| --- | --- |
| `fixed_dt` | 0.125 s (8 FPS) |
| `n-ticks` | 960 |
| `replan-every` | 4 |
| `goal-min-dist-m` | 16.0 |
| `viz-top-k` | 8 |

## 5. 依赖关系

- 延迟导入：`carla`, `MonoDriveAgent`, `DenseRoute`, `visualizer`
- `numpy`, `argparse`

## 6. 维护记录

| 日期 | 修改人 | 说明 |
| --- | --- | --- |
| 2026-06-09 | FuZiR_Cursor | 模块路径改为 `close_loop.monodrive.run_closed_loop`。 |
