# close_loop/monodrive/agent.py

## 1. 文件职责

实现 Carla 闭环 `MonoDriveAgent`：构造模型输入、加载主干权重、解码 256 路轨迹词表、选择 winner、将 ego-local 轨迹转为世界系控制参考，并输出 throttle / brake / steer。保留 committed 跟踪、legacy 每 tick 推理、诊断 dump 与多种纵向模式。

## 2. 公开接口

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `MonoDriveAgent` | class | 主 agent。 |
| `InferenceResult` | dataclass | `run_step` 返回值；`used_model` 表示是否走模型。 |
| `CommittedTrajectory` | dataclass | 世界系 winner 路径及速度剖面。 |
| `detect_reverse_intent` | function | 由 ego-local x 净位移判断倒车。 |
| `sample_traj_diff_kinematics` | function | 折线差分速度/加速度。 |

## 3. 关键流程

### 模型加载

- `load_backbone_config(backbone_config_path, project_root)`
- `load_checkpoint` → `model.load_state_dict(payload["model_state"])`

### 推理 `_run_inference`

1. `FrameBuffer.stack()` → `[1, 8, 3, H, W]`
2. `build_ego_motion` / `build_target_point`
3. `MonoDriveBackbone.forward`
4. `decode_trajectories(..., top_k=viz_top_k)`
5. `_select_winner_idx`（迟滞 / force override）

### 控制

- **committed（默认）**：每 `replan_every` tick 推理；中间在世界系路径 pure-pursuit。
- **legacy**：每 tick 推理 + ego-local 预瞄。

## 4. 配置项（构造参数）

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `backbone_config_path` | `config/backbone.toml` | 主干配置 |
| `viz_top_k` | 8 | 可视化候选数 |
| `replan_every` | 4 | committed 重规划间隔 |
| `goal_min_dist_m` | 24.0 | 目标点最小直线距离 (m) |
| `goal_max_dist_m` | 30.0 | 目标点最大直线距离 (m) |
| `winner_hysteresis` | 0.15 | winner 切换迟滞 |

## 5. 依赖关系

- `model.backbone`, `train.checkpointing`
- `close_loop.monodrive.inputs`, `model_inference`, `planner_goal`
- `agents.navigation`（BehaviorAgent、PID 控制器）
- `carla`（运行时）

## 6. 维护记录

| 日期 | 修改人 | 说明 |
| --- | --- | --- |
| 2026-06-09 | FuZiR_Cursor | JEPA 8-mode 头替换为 MonoDrive 256 词表解码。 |
