"""MonoDriveAgent：把 ``MonoDriveBackbone`` 接入 Carla 闭环。

设计要点
========

- **完整主干**：``MonoDriveBackbone(images, target_points, ego_motion)`` 输出 256 路
  轨迹词表 logit 与 Tanh 残差；词表 + 残差反 Symlog 得物理轨迹。
- **winner**：``winner_idx = softmax(logits).argmax(-1)``，可选迟滞与强制索引。
- **控制**：
    * 横向：在 winner 折线或 committed 世界系路径上 pure-pursuit + PIDLateralController。
    * 纵向：对 winner ``(x,y)`` 折线按 ``TRAJECTORY_DT=0.5s`` 差分得 v/a，再经 PID / 前馈。
- **冷启动**：前 ``PAST_FRAMES - 1`` 个 tick buffer 没填满，跑 BehaviorAgent fallback。
"""

from __future__ import annotations

import contextlib
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

# 把项目根 (model 包) 与 close_loop/ (agents.navigation 包) 加入 sys.path
_THIS = Path(__file__).resolve()
_CLOSE_LOOP_ROOT = _THIS.parents[1]
_PROJECT_ROOT = _CLOSE_LOOP_ROOT.parent
for _p in (str(_PROJECT_ROOT), str(_CLOSE_LOOP_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import carla   # noqa: E402
from agents.navigation.behavior_agent import BehaviorAgent  # noqa: E402
from agents.navigation.controller import (  # noqa: E402
    PIDLateralController, PIDLongitudinalController,
)
from model.backbone import MonoDriveBackbone, load_backbone_config  # noqa: E402
from train.checkpointing import load_checkpoint  # noqa: E402

from .inputs import (  # noqa: E402
    PAST_FRAMES,
    TRAJECTORY_DT,
    EgoBuffer,
    FrameBuffer,
    build_ego_motion,
    build_target_point,
    ego_local_to_world,
)
from .model_inference import decode_trajectories, decode_winner_trajectory  # noqa: E402
from .planner_goal import DenseRoute, GOAL_MAX_DIST_M, GOAL_MIN_DIST_M  # noqa: E402

logger = logging.getLogger("monodrive_agent")


# ─────────────────────────────────────────────────────────────
# Committed trajectory（持有的世界系轨迹），用于跨多个 tick 跟踪
# ─────────────────────────────────────────────────────────────
@dataclass
class CommittedTrajectory:
    """模型 winner 轨迹的世界系副本，持有 ``ttl_ticks`` 个 tick 后过期重 plan。

    SOTA 实践：re-plan 频率（1–4 Hz）< 控制频率（≥ 10 Hz）。控制器在 committed
    路径上找最近点 + 前方 L 米的预瞄点（pure-pursuit-on-path）。这消除两个抖动源：

    1. 每 tick 重新跑模型导致 winner mode flicker
    2. 在「最新的 ego-local 第 k 步」找预瞄点导致目标点随 ego 漂移
    """

    xy_world: np.ndarray            # (T, 2) 世界系
    vel_phys: np.ndarray            # (T,) m/s, 切向标量（模型学的是 ≥0 的大小）
    accel_phys: np.ndarray          # (T,) m/s², 切向标量
    winner_idx: int                 # 生成时的 winner 索引（仅诊断）
    is_reverse: bool = False        # 该 winner 几何上是否向后行驶（由 ego-local x 净位移判断）
    ttl_ticks: int = 4              # 最多沿用多少 tick
    age_ticks: int = 0
    cum_arc: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        T = self.xy_world.shape[0]
        if T >= 2:
            diffs = np.diff(self.xy_world, axis=0)
            seg = np.linalg.norm(diffs, axis=1)
            self.cum_arc = np.concatenate([[0.0], np.cumsum(seg)])
        else:
            self.cum_arc = np.zeros(T, dtype=np.float64)

    def is_expired(self) -> bool:
        return self.age_ticks >= self.ttl_ticks

    def step(self) -> None:
        self.age_ticks += 1

    # ─────────────────────────────────────────────────────────────
    def closest_index(self, ego_xy: np.ndarray) -> int:
        d = self.xy_world - ego_xy
        d2 = np.einsum("ij,ij->i", d, d)
        return int(np.argmin(d2))

    def lookahead_world(
        self, ego_xy: np.ndarray, lookahead_dist: float
    ) -> Tuple[np.ndarray, int, int]:
        """从 ``ego_xy`` 在 committed 轨迹上的最近点出发，沿弧长前进 ``L`` 米。

        返回 ``(lookahead_world_xy, closest_idx, lookahead_idx)``。
        """
        i_close = self.closest_index(ego_xy)
        target_arc = float(self.cum_arc[i_close]) + float(lookahead_dist)
        i_la = int(np.searchsorted(self.cum_arc, target_arc, side="left"))
        i_la = min(max(i_la, i_close), self.xy_world.shape[0] - 1)
        return self.xy_world[i_la].copy(), i_close, i_la

    def sample_speed(self, idx: int) -> Tuple[float, float]:
        """读路径上某点的 ``(vel, accel)``。"""
        idx = int(np.clip(idx, 0, self.vel_phys.shape[0] - 1))
        return float(self.vel_phys[idx]), float(self.accel_phys[idx])


class _FakeWaypoint:
    """最小化 ``carla.Waypoint`` 替身：仅暴露 ``.transform.location`` 与 ``.transform.get_right_vector()``。

    用于把模型输出的 ego-local 点包装成 ``PIDLateralController`` 能消费的对象，
    避免每帧创建真实 ``carla.Waypoint``（且模型轨迹也可能不落在 lane 中心）。
    """

    class _FakeTransform:
        def __init__(self, location: carla.Location) -> None:
            self.location = location

        def get_right_vector(self) -> carla.Vector3D:
            # 仅在 PIDLateralController 的 offset != 0 时用到；这里默认 offset=0，永不调用。
            return carla.Vector3D(0.0, 0.0, 0.0)

    def __init__(self, location: carla.Location) -> None:
        self.transform = _FakeWaypoint._FakeTransform(location)


# ─────────────────────────────────────────────────────────────
# 推理结果
# ─────────────────────────────────────────────────────────────
def detect_reverse_intent(
    winner_xy_local: np.ndarray, threshold_m: float = 0.5
) -> bool:
    """从 winner 在 **ego-local** 系的 x 净位移判定是否倒车意图。

    模型学的切向速度 ``vel`` 与加速度 ``accel`` 都是 **非负标量**（数据集里 ``v_local``
    点乘瞬时 ``yaw_local`` 方向得到，正常驾驶里这个量为正），因此**不能**用 ``vel<0``
    判断倒车。轨迹本身的 ``x_local`` 净位移才是可靠信号：

    - ``Δx_local = winner_xy_local[-1, 0] - winner_xy_local[0, 0] < -threshold_m``
      → 轨迹整体向后 → 需要挂倒挡。
    - 介于 ``±threshold_m`` 之间 → 视为停车，不触发倒车。
    """
    if winner_xy_local.shape[0] < 2:
        return False
    dx = float(winner_xy_local[-1, 0] - winner_xy_local[0, 0])
    return dx < -float(threshold_m)


def sample_traj_diff_kinematics(
    xy: np.ndarray,
    idx: int,
    dt: float,
) -> Tuple[float, float]:
    """从轨迹折线逐帧差分估计 ``(speed_m_s, accel_m_s2)``。

    模型输出 ``(x, y)`` 序列与仿真同频（默认 ``dt=0.125`` s / 8 FPS），相邻点位移
    / ``dt`` 为段速度，段速度差分 / ``dt`` 为段加速度。比直接读 head 的 ``vel/accel``
    标量更贴合「几何轨迹即控制参考」的语义。

    Args:
        xy: ``(T, 2)`` ego-local 或 world 系折线。
        idx: 读取点（legacy 常用 0；committed 用路径最近点索引）。
        dt: 帧间隔 (s)。
    """
    T = int(xy.shape[0])
    if T < 2:
        return 0.0, 0.0
    idx = int(np.clip(idx, 0, T - 1))
    dt = max(float(dt), 1e-6)

    def _seg_speed(i: int) -> float:
        if i < 0 or i >= T - 1:
            return 0.0
        return float(np.linalg.norm(xy[i + 1] - xy[i]) / dt)

    v_now = _seg_speed(idx) if idx < T - 1 else _seg_speed(T - 2)

    if idx < T - 2:
        v_next = _seg_speed(idx + 1)
        accel = (v_next - v_now) / dt
    elif idx > 0:
        v_prev = _seg_speed(idx - 1)
        accel = (v_now - v_prev) / dt
    else:
        accel = 0.0
    return v_now, float(accel)


def compute_traj_kinematics_profile(
    xy: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """对整条 ``(T, 2)`` 折线逐点差分，返回 ``(vel, accel)`` 序列。"""
    T = int(xy.shape[0])
    vel = np.zeros(T, dtype=np.float64)
    accel = np.zeros(T, dtype=np.float64)
    for i in range(T):
        v_i, a_i = sample_traj_diff_kinematics(xy, i, dt)
        vel[i] = v_i
        accel[i] = a_i
    return vel, accel


@dataclass
class _InferBundle:
    """单次模型推理的原始结果（legacy / committed 共用）。"""

    winner_idx: int
    winner_traj_phys: np.ndarray    # (K, 2)
    top_trajs_phys: np.ndarray      # (top_k, K, 2)
    probs_np: np.ndarray            # (V,)
    all_world: np.ndarray           # (top_k, K, 2)
    yaw_ref: float
    p_ref: np.ndarray               # (2,)
    z_ref: float
    goal_dist_m: float = 0.0        # anchor 帧 ego-local goal 距离 (m)
    goal_refreshed: bool = False    # 本次推理是否刷新了 held goal


@dataclass
class InferenceResult:
    """单次 ``MonoDriveAgent.run_step`` 的产出。"""

    control: carla.VehicleControl
    used_model: bool                           # True = MonoDrive 推理；False = BehaviorAgent fallback
    winner_idx: int                            # 0..255；fallback 时 -1
    # top-k 候选轨迹（物理量、ego-local 系）：(top_k, 6, 2)
    all_trajs_local: Optional[np.ndarray] = None
    probs: Optional[np.ndarray] = None         # (V,) 全词表 softmax
    # 在世界坐标系下的所有候选轨迹：(n_traj, traj_horizon, 2)；含起点 ego 当前位置
    all_trajs_world: Optional[np.ndarray] = None
    target_speed_kmh: float = 0.0
    accel_cmd: float = 0.0
    look_ahead_world: Optional[Tuple[float, float, float]] = None  # (x, y, z) for viz
    replanned: bool = False                    # True = 本 tick 跑了模型推理；False = 沿用 committed
    committed_age: int = 0                     # committed 轨迹已被沿用了多少 tick
    committed_xy_world: Optional[np.ndarray] = None  # (T, 2) 当前持有的世界系轨迹，用于可视化
    goal_dist_m: float = 0.0                   # anchor 帧 ego-local goal 距离 (m)
    goal_refreshed: bool = False               # 本 tick 推理是否刷新了 held goal


# ─────────────────────────────────────────────────────────────
# MonoDriveAgent
# ─────────────────────────────────────────────────────────────
class MonoDriveAgent:
    """Carla 0.10.0 闭环 MonoDrive 推理 agent。"""

    LONG_MODES = ("speed", "accel", "ff_pid", "traj_diff")
    PRECISION_MODES = ("bf16", "fp16", "fp32")

    def __init__(
        self,
        vehicle: carla.Vehicle,
        route: DenseRoute,
        checkpoint: Optional[str] = None,
        backbone_config_path: str = "config/backbone.toml",
        viz_top_k: int = 8,
        camera_width: int = 800,
        camera_height: int = 450,
        device: str = "cuda",
        dt: float = 0.125,
        look_ahead_step: int = 2,                # 旧参数，仅在 use_pure_pursuit=False 时生效
        look_ahead_min_dist: float = 4.0,        # pure-pursuit 最小预瞄距离 (m)
        look_ahead_time: float = 0.8,            # pure-pursuit 时间预瞄 (s)
        use_pure_pursuit: bool = True,
        max_throttle: float = 0.75,
        max_brake: float = 0.5,
        max_steering: float = 0.8,
        long_mode: str = "traj_diff",
        accel_throttle_scale: float = 3.0,
        accel_brake_scale: float = 5.0,
        ff_pid_correction_weight: float = 0.3,
        fallback_target_speed_kmh: float = 20.0,
        precision: str = "bf16",
        flip_y: bool = False,
        allow_cpu_fallback: bool = False,
        replan_every: int = 4,
        use_committed_tracking: bool = True,
        allow_reverse: bool = False,
        reverse_dx_threshold: float = 0.5,
        force_winner_idx: Optional[int] = None,
        goal_min_dist_m: float = GOAL_MIN_DIST_M,
        goal_max_dist_m: float = GOAL_MAX_DIST_M,
        goal_hold_ticks: int = 1,
        winner_hysteresis: float = 0.15,
        diagnostic_dir: Optional[str] = None,
        diagnostic_every: int = 1,
    ) -> None:
        """
        Args:
            vehicle: 自车
            route: 已经 ``compute()`` 过的 ``DenseRoute``
            checkpoint: 训练 checkpoint 路径（含 ``model_state``）；None 时随机初始化
            backbone_config_path: 主干 TOML 配置路径（相对项目根）
            viz_top_k: 可视化 / 诊断导出的 top-k 候选数
            camera_width: Carla RGB 相机宽度（像素），默认 800
            camera_height: Carla RGB 相机高度（像素），默认 450
            device: 推理设备 "cuda" / "cpu"
            dt: Carla tick 周期（秒），8 FPS = 0.125s
            look_ahead_step: 用 winner 物理轨迹的第几步做横向 PID 目标（默认 step=2，约 0.25s 后）
            long_mode: 纵向控制模式，四选一::

                - ``"speed"``  : 纯 ``target_speed`` PID。winner[0].vel → target_speed_kmh
                                 喂 ``PIDLongitudinalController``，``[-1,1]`` 直接当
                                 throttle / brake。**默认**，最稳。
                - ``"accel"``  : 纯加速度前馈。winner[0].accel >= 0 → throttle = accel /
                                 ``accel_throttle_scale``，<0 → brake = |accel| /
                                 ``accel_brake_scale``。不走 PID。
                - ``"ff_pid"`` : 加速度前馈 + 速度 PID 小幅度纠偏。
                                 cmd = accel_ff + ``ff_pid_correction_weight`` × pid_out。
                - ``"traj_diff"``: 在 winner 折线上对相邻点差分得段速度（``TRAJECTORY_DT=0.5s``），
                                 再差分得加速度。纵向按 ``ff_pid`` 逻辑执行。
            accel_throttle_scale: ``"accel"`` / ``"ff_pid"`` / ``"traj_diff"`` 模式下，
                                  （>=accel_throttle_scale m/s² 即满油门）。
            accel_brake_scale:    同上，brake 归一化分母（满刹车对应 ``accel_brake_scale`` m/s² 减速）。
            ff_pid_correction_weight: ``"ff_pid"`` / ``"traj_diff"`` 模式下速度 PID 叠加权重。
            fallback_target_speed_kmh: BehaviorAgent fallback 阶段的 target speed
            precision: ``"bf16"`` (默认) / ``"fp16"`` / ``"fp32"``。CUDA 上启用 autocast
                       做混合精度前向；权重始终保留 FP32。``"fp32"`` 关闭 autocast。
                       CPU 模式下永远走 FP32。
            replan_every: 仅在 ``use_committed_tracking=True`` 时生效。每 N tick 推理一次，
                          中间复用世界系 ``CommittedTrajectory`` 做 pure-pursuit-on-path。
                          ``N=4`` ≈ 2 Hz re-plan（默认）。
            use_committed_tracking: ``True``（默认）= committed + 世界系 pure-pursuit；
                          ``False`` = **最初**跟踪方式：每 tick 推理，在**当前 tick**
                          的 ego-local winner 上取预瞄点，纵向用 winner 第 0 步 vel/accel。
                          CLI 对应 ``--legacy-tracking``。
            allow_reverse: 允许倒车。根据 winner 在 **ego-local** 的 x 净位移判断；
                          ``Δx_local < -reverse_dx_threshold`` 时挂 ``VehicleControl.reverse``。
                          模型学的 ``vel`` 是非负标量，**不**用它判反向。
            reverse_dx_threshold: 倒车判定阈值（米），winner 末点 vs 起点的 ego-local x 差
                                  小于 ``-该值`` 才挂倒挡（默认 0.5 m，过滤抖动）。
            force_winner_idx: 若不为 ``None``，强制把第 N 条词表轨迹作为 winner，
                              范围 ``[0, 255]``。
            goal_min_dist_m: 目标点最小直线距离 (m)，默认 24，与 B2D 训练一致。
            goal_max_dist_m: 目标点最大直线距离 (m)，默认 30。
            goal_hold_ticks: goal 在世界系中保持不变的 tick 数（默认 1 = 每 tick 重选）。
                             训练 anchor 的 ``||target_point||`` 应在 24–30 m；
                             hold > 1 时 ego 逼近目标后可能落入 OOD。
            winner_hysteresis: re-plan 时新 argmax 相对上一 winner 的 prob 领先不足该值则
                               保持上一 winner，减轻 mode 0 等 flicker（0 = 关闭）。
            diagnostic_dir: 若指定，则每次 re-plan 把输入与 top-k 轨迹 dump 成 PNG + NPZ。
            diagnostic_every: 每隔 N 次 re-plan 才 dump 一次（默认 1 = 每次都 dump）。
        """
        if long_mode not in self.LONG_MODES:
            raise ValueError(f"long_mode={long_mode!r} 不在 {self.LONG_MODES}")
        if precision not in self.PRECISION_MODES:
            raise ValueError(f"precision={precision!r} 不在 {self.PRECISION_MODES}")

        # ─── CUDA 硬校验：避免 --device cuda 被悄悄回退到 CPU 跑成 8s/step ───
        requested_cuda = device.startswith("cuda")
        if requested_cuda and not torch.cuda.is_available():
            msg = (
                "请求 device='cuda' 但 torch.cuda.is_available()=False。\n"
                "  - PyTorch 是否安装了 CUDA 版本？\n"
                "    python -c \"import torch; print(torch.version.cuda, torch.cuda.is_available())\"\n"
                "  - 驱动 / CUDA runtime 是否匹配（nvidia-smi）？\n"
                "  - 若要在 CPU 上跑（每步 ~8 秒，仅调试），请显式传 --device cpu，"
                "或在 MonoDriveAgent 构造时设 allow_cpu_fallback=True。"
            )
            if not allow_cpu_fallback:
                raise RuntimeError(msg)
            logger.warning(msg)

        self.vehicle = vehicle
        self.route = route
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        if requested_cuda and self.device.type != "cuda":
            logger.warning("device 被回退为 %s（请求 %s）", self.device, device)

        self.precision = precision if self.device.type == "cuda" else "fp32"
        self._amp_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[self.precision]
        self._use_amp = self.device.type == "cuda" and self.precision != "fp32"

        if self.device.type == "cuda":
            try:
                idx = self.device.index if self.device.index is not None else torch.cuda.current_device()
                name = torch.cuda.get_device_name(idx)
                mem_total = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
                logger.info(
                    "CUDA OK: cuda:%d (%s), total mem=%.1f GiB, precision=%s, autocast=%s",
                    idx, name, mem_total, self.precision, self._use_amp,
                )
            except Exception:    # noqa: BLE001
                logger.exception("查询 CUDA 设备信息失败（不致命）")

        self.dt = float(dt)
        self.look_ahead_step = int(look_ahead_step)
        self.look_ahead_min_dist = float(look_ahead_min_dist)
        self.look_ahead_time = float(look_ahead_time)
        self.use_pure_pursuit = bool(use_pure_pursuit)
        self.max_throttle = float(max_throttle)
        self.max_brake = float(max_brake)
        self.max_steering = float(max_steering)
        self.long_mode = long_mode
        self.accel_throttle_scale = float(accel_throttle_scale)
        self.accel_brake_scale = float(accel_brake_scale)
        self.ff_pid_correction_weight = float(ff_pid_correction_weight)
        self.flip_y = bool(flip_y)
        self.replan_every = max(1, int(replan_every))
        self.use_committed_tracking = bool(use_committed_tracking)
        self.allow_reverse = bool(allow_reverse)
        self.reverse_dx_threshold = float(reverse_dx_threshold)
        self._force_winner_idx: Optional[int] = (
            int(force_winner_idx) if force_winner_idx is not None else None
        )
        self.goal_min_dist_m = float(goal_min_dist_m)
        self.goal_max_dist_m = float(goal_max_dist_m)
        if self.goal_min_dist_m > self.goal_max_dist_m:
            raise ValueError(
                f"goal_min_dist_m 必须 <= goal_max_dist_m，"
                f"实际为 {self.goal_min_dist_m} > {self.goal_max_dist_m}"
            )
        self.goal_hold_ticks = max(1, int(goal_hold_ticks))
        self.winner_hysteresis = max(0.0, float(winner_hysteresis))
        self.diagnostic_dir = Path(diagnostic_dir).expanduser().resolve() if diagnostic_dir else None
        self.diagnostic_every = max(1, int(diagnostic_every))
        self.viz_top_k = max(1, int(viz_top_k))
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        if self.camera_width <= 0 or self.camera_height <= 0:
            raise ValueError(
                f"camera 分辨率必须为正数，实际为 {self.camera_width}x{self.camera_height}"
            )
        self.backbone_config_path = str(backbone_config_path)
        self._current_tick: int = 0
        self._infer_counter: int = 0
        if self.diagnostic_dir is not None:
            self.diagnostic_dir.mkdir(parents=True, exist_ok=True)
            logger.info("诊断 dump 启用 → %s（每 %d 次推理一张）",
                        self.diagnostic_dir, self.diagnostic_every)

        # Goal 持有：训练 clip 内 goal 固定，闭环不应每 0.5s re-plan 就换 goal 并回填历史帧。
        self._held_goal_xy: Optional[np.ndarray] = None
        self._last_goal_idx: int = 0
        self._goal_age_ticks: int = 0
        self._prev_winner_idx: int = -1

        # SOTA 风格「持有轨迹」：上一次跑模型生成的世界系 winner 轨迹。
        # 在 replan_every-1 个 tick 内只读不写，控制器在它上面做 pure-pursuit。
        self._committed: Optional[CommittedTrajectory] = None
        # 上一次 inference 产出的 viz / 诊断信息，持有期内复用以保持日志连贯。
        self._last_infer_aux: dict = {}

        self.model = self._load_model(checkpoint)

        # buffers
        self.frame_buf = FrameBuffer(
            maxlen=PAST_FRAMES,
            source_hw=(self.camera_height, self.camera_width),
        )
        self.ego_buf = EgoBuffer(maxlen=PAST_FRAMES)
        self.ego_buf.set_dt(self.dt)

        # PID 控制器
        # PID 系数沿用 BehaviorAgent 默认值（参考 local_planner.py）。
        # 仅在 long_mode != "accel" 时实际起作用；为简单起见两种模式下都构造。
        self._lon_pid = PIDLongitudinalController(
            self.vehicle, K_P=1.0, K_I=0.05, K_D=0.0, dt=self.dt
        )
        self._lat_pid = PIDLateralController(
            self.vehicle, offset=0.0, K_P=1.95, K_I=0.05, K_D=0.2, dt=self.dt
        )
        self._past_steering = 0.0

        # Fallback BehaviorAgent
        self._fallback_agent = BehaviorAgent(self.vehicle, behavior="normal")
        self._fallback_agent.set_destination(route.end_location())
        self._fallback_target_speed = float(fallback_target_speed_kmh)

        # 监控：上一次路径索引（保证 nearest_index 单调推进）
        self._last_route_idx = 0

    # ─────────────────────────────────────────────────────────────
    def _load_model(self, checkpoint: Optional[str]) -> MonoDriveBackbone:
        config = load_backbone_config(self.backbone_config_path, project_root=_PROJECT_ROOT)
        model = MonoDriveBackbone(config).to(self.device).eval()
        if checkpoint:
            path = Path(checkpoint).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            loaded = load_checkpoint(path, self.device)
            payload = loaded.payload
            model.load_state_dict(payload["model_state"])
            step = payload.get("global_step", payload.get("step", "?"))
            logger.info("已加载权重 %s (global_step=%s)", path, step)
        else:
            logger.warning("未指定 checkpoint，使用随机初始化（仅打通流水线）")
        logger.info(
            "MonoDriveAgent device=%s, precision=%s, autocast=%s",
            self.device, self.precision, self._use_amp,
        )
        return model

    def _amp_ctx(self) -> contextlib.AbstractContextManager:
        """根据 self.precision 返回 autocast context manager 或 nullcontext。"""
        if not self._use_amp:
            return contextlib.nullcontext()
        return torch.amp.autocast(device_type="cuda", dtype=self._amp_dtype)

    # ─────────────────────────────────────────────────────────────
    def set_force_winner_idx(self, idx: Optional[int]) -> None:
        """运行时设置/清除强制 winner 索引。传 ``None`` 恢复 ``probs.argmax``。

        若 committed 跟踪正在持有一条旧 winner，本次设置会在下次 re-plan 时生效。
        """
        if idx is None:
            self._force_winner_idx = None
            logger.info("clear force-winner override → 使用 probs.argmax")
            return
        idx = int(idx)
        self._force_winner_idx = idx
        logger.info("force-winner override → winner_idx=%d", idx)
        # 让 committed 立即过期，下个 tick 就重 plan 应用新 winner
        if self._committed is not None:
            self._committed.age_ticks = self._committed.ttl_ticks

    def get_force_winner_idx(self) -> Optional[int]:
        return self._force_winner_idx

    def push_camera_bgra(self, bgra: np.ndarray) -> None:
        """Carla ``sensor.camera.rgb`` 的 raw_data 走这里入帧 buffer。"""
        self.frame_buf.push_bgra_uint8(bgra)

    def push_ego_snapshot(self) -> None:
        """从 ``self.vehicle`` 抓一帧 ego 状态进 ego buffer。"""
        self.ego_buf.push_from_vehicle(self.vehicle)

    def set_current_tick(self, tick: int) -> None:
        """主循环每 tick 调用一次，仅用于诊断 dump 的文件名 / 日志关联。"""
        self._current_tick = int(tick)

    # ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def run_step(self) -> InferenceResult:
        """运行单步。

        - ``use_committed_tracking=False``（``--legacy-tracking``）：每 tick 推理 + ego-local 预瞄。
        - ``use_committed_tracking=True``（默认）：committed 持有 + 世界系 pure-pursuit-on-path。
        """
        if not (self.frame_buf.is_full() and self.ego_buf.is_full()):
            ctrl = self._run_fallback()
            return InferenceResult(control=ctrl, used_model=False, winner_idx=-1)

        self._goal_age_ticks += 1

        ego_loc = self.vehicle.get_location()
        self._last_route_idx = self.route.nearest_index(ego_loc, search_from=self._last_route_idx)

        if not self.use_committed_tracking:
            return self._run_step_legacy(ego_loc)

        return self._run_step_committed(ego_loc)

    def _run_step_legacy(self, ego_loc: carla.Location) -> InferenceResult:
        """最初跟踪：每 tick 推理；横向在 ego-local winner 上取预瞄；纵向用 winner[0]。"""
        bundle = self._run_inference(ego_loc)
        winner_idx = bundle.winner_idx
        winner_xy = bundle.winner_traj_phys
        winner_vel, winner_accel = sample_traj_diff_kinematics(
            winner_xy, idx=0, dt=TRAJECTORY_DT
        )
        is_reverse = (
            self.allow_reverse
            and detect_reverse_intent(winner_xy, self.reverse_dx_threshold)
        )

        ctrl, look_world = self._compute_control_legacy(
            winner_xy_local=winner_xy,
            target_vel=winner_vel,
            accel_cmd=winner_accel,
            is_reverse=is_reverse,
            yaw_ref=bundle.yaw_ref,
            p_ref=bundle.p_ref,
            z_ref=bundle.z_ref,
        )

        return InferenceResult(
            control=ctrl,
            used_model=True,
            winner_idx=winner_idx,
            all_trajs_local=bundle.top_trajs_phys,
            probs=bundle.probs_np,
            all_trajs_world=bundle.all_world,
            target_speed_kmh=self._target_speed_kmh(winner_vel),
            accel_cmd=winner_accel,
            look_ahead_world=look_world,
            replanned=True,
            committed_age=0,
            committed_xy_world=ego_local_to_world(winner_xy, bundle.yaw_ref, bundle.p_ref),
            goal_dist_m=bundle.goal_dist_m,
            goal_refreshed=bundle.goal_refreshed,
        )

    def _run_step_committed(self, ego_loc: carla.Location) -> InferenceResult:
        """committed 持有 + 世界系 pure-pursuit-on-path。"""
        need_replan = (self._committed is None) or self._committed.is_expired()
        if need_replan:
            self._replan(ego_loc)

        assert self._committed is not None
        committed = self._committed

        ego_world_xy = np.array([ego_loc.x, ego_loc.y], dtype=np.float64)
        z_ref = float(ego_loc.z)

        ctrl, look_world, closest_idx = self._compute_control_committed(
            committed=committed,
            ego_world_xy=ego_world_xy,
            z_ref=z_ref,
            is_reverse=bool(committed.is_reverse),
        )

        if self.long_mode == "traj_diff":
            target_vel, accel_cmd = sample_traj_diff_kinematics(
                committed.xy_world, closest_idx, TRAJECTORY_DT
            )
        else:
            target_vel, accel_cmd = committed.sample_speed(closest_idx)
        # 用 committed 持有时刻就计算好的 is_reverse；hold 期内不再变
        _ = committed.is_reverse
        committed.step()

        aux = self._last_infer_aux
        return InferenceResult(
            control=ctrl,
            used_model=True,
            winner_idx=committed.winner_idx,
            all_trajs_local=aux.get("all_trajs_local"),
            probs=aux.get("probs"),
            all_trajs_world=aux.get("all_trajs_world"),
            target_speed_kmh=self._target_speed_kmh(target_vel),
            accel_cmd=accel_cmd,
            look_ahead_world=look_world,
            replanned=need_replan,
            committed_age=int(committed.age_ticks),
            committed_xy_world=committed.xy_world,
            goal_dist_m=float(aux.get("goal_dist_m", 0.0)),
            goal_refreshed=bool(aux.get("goal_refreshed", False)),
        )

    def _resolve_goal_world_xy(self, p_ref: np.ndarray) -> Tuple[np.ndarray, int, bool]:
        """按 ``goal_hold_ticks`` 持有 goal，避免 re-plan 频繁跳变污染历史 ``goal_dx/dy``。"""
        refreshed = (
            self._held_goal_xy is None
            or self._goal_age_ticks >= self.goal_hold_ticks
        )
        if refreshed:
            _, goal_idx, goal_xy = self.route.goal_at_training_aligned(
                self._last_route_idx,
                p_ref_world=p_ref,
                min_dist_m=self.goal_min_dist_m,
                max_dist_m=self.goal_max_dist_m,
            )
            self._held_goal_xy = np.asarray(goal_xy, dtype=np.float64)
            self._last_goal_idx = int(goal_idx)
            self._goal_age_ticks = 0
        return self._held_goal_xy, self._last_goal_idx, refreshed

    def _select_winner_idx(self, probs: torch.Tensor) -> int:
        """``probs.argmax`` + 可选迟滞，减轻 mode flicker。"""
        if probs.ndim == 1:
            probs = probs.unsqueeze(0)
        n_traj_runtime = int(probs.shape[-1])
        if self._force_winner_idx is not None:
            forced = self._force_winner_idx
            if not 0 <= forced < n_traj_runtime:
                logger.warning(
                    "force_winner_idx=%d 超出 [0, %d)，本次回退到 probs.argmax",
                    forced, n_traj_runtime,
                )
                raw = int(probs.argmax(dim=-1).item())
            else:
                self._prev_winner_idx = forced
                return forced
        else:
            raw = int(probs.argmax(dim=-1).item())

        if self.winner_hysteresis <= 0.0 or self._prev_winner_idx < 0:
            self._prev_winner_idx = raw
            return raw

        prev = self._prev_winner_idx
        winner = raw
        if winner != prev and 0 <= prev < n_traj_runtime:
            delta = float(probs[0, winner].item()) - float(probs[0, prev].item())
            if delta < self.winner_hysteresis:
                logger.debug(
                    "winner hysteresis: keep %d over argmax %d (Δp=%.3f < %.3f)",
                    prev, winner, delta, self.winner_hysteresis,
                )
                winner = prev
        self._prev_winner_idx = winner
        return winner

    # ─────────────────────────────────────────────────────────────
    def _run_inference(self, ego_loc: carla.Location) -> _InferBundle:
        """跑一次模型推理，返回轨迹与参考位姿（不写 committed）。"""
        latest = self.ego_buf.latest()
        p_ref = np.array([latest.x, latest.y], dtype=np.float64)
        goal_world_xy, goal_idx, goal_refreshed = self._resolve_goal_world_xy(p_ref)
        if logger.isEnabledFor(logging.DEBUG):
            gdist = float(np.linalg.norm(goal_world_xy - p_ref))
            logger.debug(
                "goal@training_aligned: route_idx=%d goal_idx=%d ||d||=%.2f m (band=[%.1f, %.1f]) refreshed=%s",
                self._last_route_idx, goal_idx, gdist,
                self.goal_min_dist_m, self.goal_max_dist_m, goal_refreshed,
            )

        ego_motion = build_ego_motion(self.ego_buf)
        target_point = build_target_point(self.ego_buf, goal_world_xy)
        goal_dist_m = float(torch.linalg.norm(target_point).item())
        if self.flip_y:
            ego_motion = ego_motion.clone()
            ego_motion[1] *= -1.0
            ego_motion[2] *= -1.0
            target_point = target_point.clone()
            target_point[1] *= -1.0

        frames_past = self.frame_buf.stack()
        ego_motion_t = ego_motion.unsqueeze(0).to(self.device, non_blocking=True)
        target_point_t = target_point.unsqueeze(0).to(self.device, non_blocking=True)
        frames_t = frames_past.unsqueeze(0).to(self.device, non_blocking=True)

        with self._amp_ctx():
            backbone_output = self.model(
                images=frames_t,
                target_points=target_point_t,
                ego_motion=ego_motion_t,
            )

        decode = decode_trajectories(backbone_output, self.model, top_k=self.viz_top_k)
        probs_t = torch.from_numpy(decode.probs).to(self.device)
        winner_idx = self._select_winner_idx(probs_t)
        if winner_idx != decode.winner_idx:
            winner_traj_phys = decode_winner_trajectory(backbone_output, self.model, winner_idx)
        else:
            winner_traj_phys = decode.winner_traj_phys

        top_trajs_phys = decode.top_trajs_phys.copy()
        if self.flip_y:
            winner_traj_phys = winner_traj_phys.copy()
            winner_traj_phys[..., 1] *= -1.0
            top_trajs_phys = top_trajs_phys.copy()
            top_trajs_phys[..., 1] *= -1.0

        yaw_ref = float(latest.yaw)
        z_ref = float(ego_loc.z)

        n_top, t_horizon, _ = top_trajs_phys.shape
        all_world = np.zeros((n_top, t_horizon, 2), dtype=np.float64)
        for k in range(n_top):
            all_world[k] = ego_local_to_world(top_trajs_phys[k, :, :2], yaw_ref, p_ref)

        if self.diagnostic_dir is not None and self._infer_counter % self.diagnostic_every == 0:
            try:
                from .diagnostic import dump_openloop_snapshot, dump_replan_snapshot

                v_kmh = math.hypot(latest.vx, latest.vy) * 3.6
                dump_replan_snapshot(
                    out_dir=self.diagnostic_dir,
                    tick=self._current_tick,
                    frames_past=frames_past,
                    ego_motion=ego_motion,
                    target_point=target_point,
                    trajs_phys=top_trajs_phys,
                    probs=decode.top_probs,
                    winner_idx=winner_idx,
                    goal_local_xy=target_point.cpu().numpy().astype(np.float64),
                    v_kmh=v_kmh,
                    goal_d_m=goal_dist_m,
                    goal_refreshed=goal_refreshed,
                    extra_text=(
                        f"flip_y={self.flip_y} | precision={self.precision}"
                        f" | viz_top_k={self.viz_top_k}"
                    ),
                )
                dump_openloop_snapshot(
                    out_dir=self.diagnostic_dir,
                    tick=self._current_tick,
                    frames_past=frames_past,
                    ego_motion=ego_motion,
                    target_point=target_point,
                    v_kmh=v_kmh,
                    goal_d_m=goal_dist_m,
                    goal_refreshed=goal_refreshed,
                )
            except Exception:        # noqa: BLE001
                logger.exception("诊断 dump 失败（不致命）")
        self._infer_counter += 1

        return _InferBundle(
            winner_idx=winner_idx,
            winner_traj_phys=winner_traj_phys,
            top_trajs_phys=top_trajs_phys,
            probs_np=decode.probs,
            all_world=all_world,
            yaw_ref=yaw_ref,
            p_ref=p_ref,
            z_ref=z_ref,
            goal_dist_m=goal_dist_m,
            goal_refreshed=goal_refreshed,
        )

    def _replan(self, ego_loc: carla.Location) -> None:
        """跑一次模型推理，生成新的 ``CommittedTrajectory`` 并写入 ``self._committed``。"""
        bundle = self._run_inference(ego_loc)

        latest = self.ego_buf.latest()
        v_kmh = math.hypot(latest.vx, latest.vy) * 3.6
        probs_np = bundle.probs_np
        p_top3 = probs_np.argsort()[-3:][::-1]
        prob_top3 = [(int(i), float(probs_np[i])) for i in p_top3]
        logger.info(
            "replan: win=%d | prob[win]=%.3f"
            " | probs top3=%s"
            " | goal_d=%.1fm (refreshed=%s) | v=%.1f km/h",
            bundle.winner_idx,
            float(probs_np[bundle.winner_idx]),
            prob_top3,
            bundle.goal_dist_m,
            bundle.goal_refreshed,
            v_kmh,
        )

        winner_xy_local = bundle.winner_traj_phys
        winner_world_xy = ego_local_to_world(winner_xy_local, bundle.yaw_ref, bundle.p_ref)

        is_reverse = (
            self.allow_reverse
            and detect_reverse_intent(winner_xy_local, self.reverse_dx_threshold)
        )

        ego_xy = np.array([ego_loc.x, ego_loc.y], dtype=np.float64)
        winner_vel, winner_accel = compute_traj_kinematics_profile(
            winner_xy_local, TRAJECTORY_DT
        )
        if np.linalg.norm(winner_world_xy[0] - ego_xy) > 0.1:
            winner_world_xy = np.concatenate([ego_xy[None, :], winner_world_xy], axis=0)
            winner_vel = np.concatenate([[winner_vel[0]], winner_vel])
            winner_accel = np.concatenate([[winner_accel[0]], winner_accel])

        self._committed = CommittedTrajectory(
            xy_world=winner_world_xy,
            vel_phys=winner_vel,
            accel_phys=winner_accel,
            winner_idx=bundle.winner_idx,
            is_reverse=is_reverse,
            ttl_ticks=self.replan_every,
        )
        self._last_infer_aux = {
            "all_trajs_local": bundle.top_trajs_phys,
            "probs": bundle.probs_np,
            "all_trajs_world": bundle.all_world,
            "goal_dist_m": bundle.goal_dist_m,
            "goal_refreshed": bundle.goal_refreshed,
        }

    # ─────────────────────────────────────────────────────────────
    # Legacy 跟踪（每 tick ego-local winner + 预瞄）
    # ─────────────────────────────────────────────────────────────
    def _compute_control_legacy(
        self,
        winner_xy_local: np.ndarray,
        target_vel: float,
        accel_cmd: float,
        is_reverse: bool,
        yaw_ref: float,
        p_ref: np.ndarray,
        z_ref: float,
    ) -> Tuple[carla.VehicleControl, Tuple[float, float, float]]:
        """最初方式：ego-local winner 轨迹 → 控制；纵向固定读 winner[0]。"""
        steer, look_loc = self._compute_steering_legacy(
            winner_xy_local, yaw_ref, p_ref, z_ref
        )
        throttle, brake = self._compute_longitudinal(target_vel, accel_cmd, is_reverse)

        ctrl = carla.VehicleControl()
        ctrl.throttle = float(throttle)
        ctrl.brake = float(brake)
        ctrl.steer = float(steer)
        ctrl.reverse = bool(is_reverse)
        ctrl.hand_brake = False
        ctrl.manual_gear_shift = False
        return ctrl, (float(look_loc.x), float(look_loc.y), float(look_loc.z))

    def _pick_lookahead_local(self, winner_xy_local: np.ndarray) -> np.ndarray:
        """在 ego-local winner 上选预瞄点（距原点 >= L 的第一个点，或固定 step）。"""
        T = winner_xy_local.shape[0]
        if T == 0:
            return np.zeros(2, dtype=np.float64)
        if not self.use_pure_pursuit:
            k = max(0, min(self.look_ahead_step, T - 1))
            return winner_xy_local[k]

        latest = self.ego_buf.latest()
        v_ego = math.hypot(latest.vx, latest.vy)
        L = max(self.look_ahead_min_dist, self.look_ahead_time * v_ego)
        dists = np.linalg.norm(winner_xy_local, axis=1)
        over = np.flatnonzero(dists >= L)
        if over.size == 0:
            return winner_xy_local[-1]
        return winner_xy_local[int(over[0])]

    def _compute_steering_legacy(
        self,
        winner_xy_local: np.ndarray,
        yaw_ref: float,
        p_ref: np.ndarray,
        z_ref: float,
    ) -> Tuple[float, carla.Location]:
        """横向：ego-local 预瞄点 → 转世界系 → PIDLateralController。"""
        look_xy_local = self._pick_lookahead_local(winner_xy_local)
        look_xy_world = ego_local_to_world(look_xy_local, yaw_ref, p_ref)
        look_loc = carla.Location(
            x=float(look_xy_world[0]), y=float(look_xy_world[1]), z=z_ref
        )
        fake_wp = _FakeWaypoint(look_loc)
        steer_pid = float(self._lat_pid.run_step(fake_wp))

        if steer_pid > self._past_steering + 0.1:
            steer_pid = self._past_steering + 0.1
        elif steer_pid < self._past_steering - 0.1:
            steer_pid = self._past_steering - 0.1
        steer = float(np.clip(steer_pid, -self.max_steering, self.max_steering))
        self._past_steering = steer
        return steer, look_loc

    # ─────────────────────────────────────────────────────────────
    def _compute_control_committed(
        self,
        committed: CommittedTrajectory,
        ego_world_xy: np.ndarray,
        z_ref: float,
        is_reverse: bool = False,
    ) -> Tuple[carla.VehicleControl, Tuple[float, float, float], int]:
        """Pure-pursuit-on-path：在 committed 世界系轨迹上找预瞄点 → ``carla.VehicleControl``。

        返回 ``(control, look_world_xyz, closest_idx)``。``closest_idx`` 用来读 vel/accel。
        """
        # 自适应预瞄距离：L = max(min_dist, time * v_ego)
        latest = self.ego_buf.latest()
        v_ego = math.hypot(latest.vx, latest.vy)
        L = max(self.look_ahead_min_dist, self.look_ahead_time * v_ego)

        look_xy_world, closest_idx, _ = committed.lookahead_world(ego_world_xy, L)
        look_loc = carla.Location(
            x=float(look_xy_world[0]), y=float(look_xy_world[1]), z=z_ref
        )

        # 横向 PID（沿用 PIDLateralController）
        fake_wp = _FakeWaypoint(look_loc)
        steer_pid = float(self._lat_pid.run_step(fake_wp))   # [-1, 1]
        if steer_pid > self._past_steering + 0.1:
            steer_pid = self._past_steering + 0.1
        elif steer_pid < self._past_steering - 0.1:
            steer_pid = self._past_steering - 0.1
        steer = float(np.clip(steer_pid, -self.max_steering, self.max_steering))
        self._past_steering = steer

        # 纵向：读 committed 路径上 closest 处的 vel / accel；倒车标志由 committed 持有
        if self.long_mode == "traj_diff":
            target_vel, accel_cmd = sample_traj_diff_kinematics(
                committed.xy_world, closest_idx, TRAJECTORY_DT
            )
        else:
            target_vel, accel_cmd = committed.sample_speed(closest_idx)
        throttle, brake = self._compute_longitudinal(target_vel, accel_cmd, is_reverse)

        ctrl = carla.VehicleControl()
        ctrl.throttle = float(throttle)
        ctrl.brake = float(brake)
        ctrl.steer = float(steer)
        ctrl.reverse = bool(is_reverse)
        ctrl.hand_brake = False
        ctrl.manual_gear_shift = False
        return ctrl, (float(look_loc.x), float(look_loc.y), float(look_loc.z)), closest_idx

    # ─────────────────────────────────────────────────────────────
    def _target_speed_kmh(self, target_vel: float) -> float:
        """模型 ``vel`` 是非负切向标量，目标速度始终用其绝对值。"""
        return abs(float(target_vel)) * 3.6

    def _compute_longitudinal(
        self, target_vel: float, accel_cmd: float, is_reverse: bool
    ) -> Tuple[float, float]:
        """``(|vel|, accel, is_reverse)`` → ``(throttle, brake)``。

        ``vel`` 是非负标量；``is_reverse`` 完全由 winner 几何决定（见
        ``detect_reverse_intent``）。倒车时 PID 仍用速度大小，因为 Carla
        ``get_velocity()`` 模与是否倒车无关。
        """
        if self.long_mode == "accel":
            return self._lon_accel_only(accel_cmd)
        if self.long_mode in ("ff_pid", "traj_diff"):
            return self._lon_ff_pid(target_vel, accel_cmd)
        return self._lon_speed_pid(target_vel)

    def _lon_speed_pid(self, target_vel: float) -> Tuple[float, float]:
        target_kmh = abs(float(target_vel)) * 3.6
        cmd = float(self._lon_pid.run_step(target_kmh))
        return self._split_throttle_brake(cmd)

    def _lon_accel_only(self, accel_cmd: float) -> Tuple[float, float]:
        a = float(accel_cmd)
        if a >= 0.0:
            throttle = float(min(a / max(self.accel_throttle_scale, 1e-6), self.max_throttle))
            return throttle, 0.0
        brake = float(min(-a / max(self.accel_brake_scale, 1e-6), self.max_brake))
        return 0.0, brake

    def _lon_ff_pid(self, target_vel: float, accel_cmd: float) -> Tuple[float, float]:
        a = float(accel_cmd)
        if a >= 0.0:
            accel_ff = float(min(a / max(self.accel_throttle_scale, 1e-6), 1.0))
        else:
            accel_ff = float(max(a / max(self.accel_brake_scale, 1e-6), -1.0))
        target_kmh = abs(float(target_vel)) * 3.6
        pid_out = float(self._lon_pid.run_step(target_kmh))
        cmd = accel_ff + self.ff_pid_correction_weight * pid_out
        return self._split_throttle_brake(cmd)

    def _split_throttle_brake(self, cmd: float) -> Tuple[float, float]:
        if cmd >= 0.0:
            return float(min(cmd, self.max_throttle)), 0.0
        return 0.0, float(min(-cmd, self.max_brake))

    # ─────────────────────────────────────────────────────────────
    def _run_fallback(self) -> carla.VehicleControl:
        """前 ``PAST_FRAMES - 1`` 个 tick：buffer 未满 → BehaviorAgent 控制。

        BehaviorAgent 自身只接受 sync world.tick() 后调一次 ``run_step()``。
        我们也帮它做 ``_update_information``（BehaviorAgent 自带）。
        """
        try:
            ctrl = self._fallback_agent.run_step()
            return ctrl
        except Exception:   # noqa: BLE001
            logger.exception("BehaviorAgent fallback 异常，输出空控制")
            return carla.VehicleControl()

    # ─────────────────────────────────────────────────────────────
    def is_done(self, threshold: float = 3.0) -> bool:
        """是否抵达终点。"""
        ego_loc = self.vehicle.get_location()
        idx = self.route.nearest_index(ego_loc, search_from=self._last_route_idx)
        return self.route.is_done(idx, threshold=threshold)
