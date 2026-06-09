#!/usr/bin/env python3
"""Carla 0.10.0 Town10HD 闭环跑 MonoDrive 推理 agent。

用法（推荐）::

    # 服务器端先离屏启动：
    #   ./CarlaUE4.sh -RenderOffScreen -nosound -fps=8 -windowed -ResX=1 -ResY=1
    # 再运行本脚本：
    python -m close_loop.monodrive.run_closed_loop \\
        --host 127.0.0.1 --port 2000 \\
        --checkpoint ./checkpoints/step_000100.pt \\
        --start-idx 0 --end-idx 50 \\
        --n-traffic 50 \\
        --mp4 ./viz_out/monodrive_town10hd.mp4 \\
        --n-ticks 1200

未指定 ``--end-idx`` 且未指定 ``--random-target`` 时，会随机挑一个 spawn point 当终点。
"""

from __future__ import annotations

import argparse
import logging
import math
import queue
import random
import signal
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

# 由 ``_load_carla()`` 在 ``main()`` 内注入；帮助函数运行前必须已加载。
carla: Any = None

# sys.path 注入：项目根（model 包）+ close_loop/（agents.navigation 包）
_THIS = Path(__file__).resolve()
_CLOSE_LOOP_ROOT = _THIS.parents[1]
_PROJECT_ROOT = _CLOSE_LOOP_ROOT.parent
for _p in (str(_PROJECT_ROOT), str(_CLOSE_LOOP_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger("run_closed_loop")


def _load_carla() -> Any:
    """延迟导入 ``carla`` 并写入模块全局，供下方 Carla 帮助函数使用。"""
    global carla
    import carla as carla_module

    carla = carla_module
    return carla


# ─────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────
def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Carla 0.10.0 Town10HD + MonoDrive 闭环推理",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--tm-port", type=int, default=8000, help="TrafficManager 端口")
    p.add_argument("--checkpoint", default=None, help="训练 checkpoint 路径（含 model_state；缺省 = 随机初始化）")
    p.add_argument(
        "--backbone-config", default="config/backbone.toml",
        help="MonoDriveBackbone 配置 TOML（相对项目根）",
    )
    p.add_argument("--viz-top-k", type=int, default=8, help="可视化 / 诊断导出的 top-k 轨迹数")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument(
        "--precision", default="bf16", choices=["bf16", "fp16", "fp32"],
        help="模型前向精度（仅 cuda 生效，权重始终 FP32）。bf16 最稳定且 ~3x 提速",
    )
    p.add_argument(
        "--town", default="Town10HD",
        help="地图短名或关键字（如 Town10HD、Mine01）；会按 get_available_maps() 解析",
    )
    p.add_argument(
        "--list-maps", action="store_true",
        help="打印服务端可用地图列表后退出",
    )

    p.add_argument("--start-idx", type=int, default=None,
                   help="起点 spawn_points 索引；不传则默认随机起点（受 --seed 控制）")
    p.add_argument("--end-idx", type=int, default=None,
                   help="终点 spawn_points 索引；不传则默认随机终点（受 --seed 控制）")
    p.add_argument(
        "--random-start", action="store_true",
        help="（保留兼容；未指定 --start-idx 时默认就已是随机起点）",
    )
    p.add_argument(
        "--random-target", action="store_true",
        help="（保留兼容；未指定 --end-idx 时默认就已是随机终点）",
    )
    p.add_argument(
        "--no-random-start", action="store_true",
        help="未指定 --start-idx 时改为固定 spawn_points[0]，关闭随机起点",
    )
    p.add_argument(
        "--no-random-target", action="store_true",
        help="未指定 --end-idx 时改为固定 spawn_points[1]，关闭随机终点",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="随机数种子。不传则每次运行用系统时间生成不同 seed；传固定值（如 0）则起点/终点可复现",
    )

    p.add_argument(
        "--all-lights-green", action="store_true",
        help="将地图上所有红绿灯强制设为绿灯并冻结周期（主循环内定期刷新以防被仿真改回）",
    )
    p.add_argument(
        "--allow-reverse", action="store_true",
        help=(
            "允许倒车。判定依据是 winner 在 ego-local 系的 x 净位移（轨迹几何）。"
        ),
    )
    p.add_argument(
        "--reverse-dx-threshold", type=float, default=0.5,
        help=(
            "倒车判定阈值（米）：winner 末点 vs 起点的 ego-local x 差 < -该值 才挂倒挡，"
            "用于过滤抖动 / 静止时的方向噪声（默认 0.5 m）"
        ),
    )
    p.add_argument(
        "--force-winner-idx", type=int, default=None,
        help=(
            "强制把第 N 条词表轨迹当作 winner（绕开 probs.argmax），范围 [0, 255]。"
            "用于调试 / 可视化不同 mode 的跟踪效果。"
        ),
    )
    p.add_argument(
        "--goal-min-dist-m", type=float, default=16.0,
        help=(
            "模型 target_point 选点：相对 EgoBuffer 最新帧的直线欧氏距离阈值 (m)，"
            "规则与 data/b2d_preprocess.py 一致（默认 16）"
        ),
    )
    p.add_argument(
        "--goal-hold-ticks", type=int, default=1,
        help=(
            "goal 在世界系中保持不变的 tick 数（默认 1 = 每 tick 重选）。"
            "训练数据里 anchor 帧 ||goal_local|| 恒 >= 16m，hold > 1 时 ego 逐步逼近 goal、"
            "anchor goal_d 跌破 16m 落入 OOD（prob 平摊、mode 0 抬头）。仅调试时设大。"
        ),
    )
    p.add_argument(
        "--winner-hysteresis", type=float, default=0.15,
        help=(
            "re-plan 时新 argmax 相对上一 winner 的 prob 领先不足该值则保持上一 winner，"
            "减轻切到 mode 0 的 flicker（0 = 关闭）。"
        ),
    )
    p.add_argument(
        "--diagnostic-dir", type=str, default=None,
        help=(
            "诊断 dump 目录。开启后，每次模型推理把 frames / ego_motion / target_point / "
            "top-k 轨迹落盘为 PNG（+ NPZ）。"
        ),
    )
    p.add_argument(
        "--diagnostic-every", type=int, default=1,
        help="每隔 N 次推理 dump 一张（默认 1）。",
    )

    p.add_argument("--n-traffic", type=int, default=50,
                   help="额外 NPC 车辆数（spawn_points 充足时尽量满足）")
    p.add_argument("--sampling-resolution", type=float, default=0.5,
                   help="GlobalRoutePlanner 密集采样间距 (m)")

    p.add_argument("--n-ticks", type=int, default=8 * 120,
                   help="主循环最大 tick 数（默认 ~2 分钟 @ 8 FPS）")
    p.add_argument("--mp4", default=None,
                   help="MP4 输出路径；不传则不录制视频。")

    p.add_argument("--camera-width", type=int, default=800,
                   help="前视 RGB 相机宽度（像素）；降低可减轻 Carla 渲染与录像开销")
    p.add_argument("--camera-height", type=int, default=450,
                   help="前视 RGB 相机高度（像素）")
    p.add_argument(
        "--camera-full-res", action="store_true",
        help="使用 B2D 训练采集分辨率 1600×900（默认 800×450）",
    )

    # ── 纵向控制方案 ──
    p.add_argument(
        "--long-mode", choices=["speed", "accel", "ff_pid", "traj_diff"], default="traj_diff",
        help=(
            "纵向控制：traj_diff=对 winner (x,y) 逐帧差分得 v/a（默认，与 xy-only 头对齐）；"
            "speed/accel/ff_pid=用差分得到的 v/a 经 PID/前馈执行"
        ),
    )
    p.add_argument("--accel-throttle-scale", type=float, default=3.0,
                   help="accel/ff_pid/traj_diff 模式下 accel(m/s^2) → 满油门的分母")
    p.add_argument("--accel-brake-scale", type=float, default=5.0,
                   help="accel/ff_pid/traj_diff 模式下 |decel|(m/s^2) → 满刹的分母")
    p.add_argument("--ff-pid-correction-weight", type=float, default=0.3,
                   help="ff_pid/traj_diff 模式下，速度 PID 输出叠加权重 (0~1)")

    # ── 横向控制（pure-pursuit） ──
    p.add_argument(
        "--no-pure-pursuit", action="store_true",
        help="禁用 pure-pursuit 自适应预瞄，回退到 --look-ahead-step 固定步索引",
    )
    p.add_argument("--look-ahead-step", type=int, default=2,
                   help="禁用 pure-pursuit 时使用的固定预瞄步索引 (0..31)")
    p.add_argument("--look-ahead-min-dist", type=float, default=4.0,
                   help="pure-pursuit 最小预瞄距离 (m)")
    p.add_argument("--look-ahead-time", type=float, default=0.8,
                   help="pure-pursuit 时间预瞄 (s)，等效距离 = time × v_ego")

    # ── 坐标系翻转（训练数据 vs Carla y 轴方向） ──
    p.add_argument(
        "--flip-y", action="store_true",
        help="翻转 y 轴（含 heading/vy/yaw_rate/goal_dy 与模型输出 traj_y）；"
        "训练数据为右手系（y 向左）、Carla 为左手系（y 向右）时需要打开",
    )

    # ── 轨迹跟踪模式 ──
    p.add_argument(
        "--legacy-tracking", action="store_true",
        help=(
            "使用最初的轨迹跟踪：每 tick 跑模型，在**当前 tick** ego-local winner 上取预瞄点，"
            "legacy 模式：每 tick 推理 + ego-local 预瞄，纵向从 xy 差分 v/a。"
            "与 --replan-every 互斥（legacy 下忽略 replan-every）。"
        ),
    )
    p.add_argument(
        "--replan-every", type=int, default=4,
        help=(
            "仅 committed 模式（默认，未开 --legacy-tracking）：每 N tick 推理一次，"
            "中间复用世界系 winner 做 pure-pursuit-on-path。N=4 ≈ 2 Hz（默认）；N=1 每 tick 推理但仍走 committed 控制。"
        ),
    )

    # ── CPU fallback（默认不允许，避免 8s/step 跑成 CPU） ──
    p.add_argument(
        "--allow-cpu-fallback", action="store_true",
        help="允许 --device cuda 在 CUDA 不可用时回退到 CPU（默认严格报错）",
    )

    p.add_argument(
        "--world-debug", action="store_true",
        help=(
            "把 winner 轨迹 / 航点 / goal 直接用 world.debug.draw_* 渲染到 Carla 世界中。"
            "⚠️ 默认关闭：这些线段会被摄像头拍到，**模型会看到并躲避**它们。仅在不录制 MP4、"
            "且只在 spectator 窗口看时打开。MP4 录像的 2D 叠加可视化是另外一套，**不影响模型**。"
        ),
    )
    p.add_argument(
        "--no-viz", action="store_true",
        help="禁用所有可视化（既不画 world.debug 也不叠加 MP4 overlay）",
    )
    p.add_argument(
        "--no-mp4-overlay", action="store_true",
        help="MP4 仅写原始相机帧，不做 2D 轨迹叠加（默认会叠加，模型仍看不到）",
    )
    p.add_argument(
        "--draw-candidates", action="store_true",
        help="同时绘制除 winner 之外的 top-k 候选轨迹（默认只画 winner，减少 debug RPC）",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Carla 帮助函数
# ─────────────────────────────────────────────────────────────
def configure_world_sync(world: carla.World, fixed_dt: float = 0.125) -> carla.WorldSettings:
    """切换到严格同步模式 + 离线渲染兼容。返回旧 settings 以便恢复。"""
    old = world.get_settings()
    new = carla.WorldSettings(
        synchronous_mode=True,
        fixed_delta_seconds=fixed_dt,
        no_rendering_mode=False,        # 保持渲染，配合 -RenderOffScreen 即可
        substepping=True,
        max_substep_delta_time=0.01,
        max_substeps=10,
    )
    world.apply_settings(new)
    return old


def restore_world_settings(world: carla.World, old: carla.WorldSettings) -> None:
    try:
        world.apply_settings(old)
    except Exception:   # noqa: BLE001
        logger.exception("restore_world_settings 失败")


def map_short_name(map_id: str) -> str:
    """从 ``get_available_maps()`` / ``get_map().name`` 的路径里取短名。"""
    return map_id.rstrip("/").split("/")[-1]


def list_available_maps(client: carla.Client) -> List[str]:
    try:
        return list(client.get_available_maps())
    except Exception as exc:  # noqa: BLE001
        logger.error("无法获取可用地图列表: %s", exc)
        return []


def resolve_map_name(client: carla.Client, town_query: str) -> str:
    """把用户输入的 ``--town`` 解析成 ``load_world`` 可接受的名称。

    Carla 0.10 (UE5) 的 ``get_available_maps()`` 常返回完整 Unreal 路径，例如
    ``/Game/Carla/Maps/Town10HD/Town10HD``；直接 ``load_world('Town10HD')`` 有时会
    触发 ``RuntimeError: std::exception``。这里优先用服务端列表里的**精确字符串**。
    """
    available = list_available_maps(client)
    if not available:
        logger.warning("get_available_maps() 为空，回退使用原始 --town=%r", town_query)
        return town_query

    q = town_query.lower()
    by_short = {map_short_name(m).lower(): m for m in available}

    if q in by_short:
        return by_short[q]

    # 子串匹配：Town10HD 可匹配 Town10HD_Opt 等
    candidates = [
        m for m in available
        if q in map_short_name(m).lower() or q in m.lower()
    ]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        # 多个命中时优先短名完全等于 query 的，其次名字最短的（避免 _Opt 歧义时取第一个）
        exact = [m for m in candidates if map_short_name(m).lower() == q]
        if exact:
            return exact[0]
        candidates.sort(key=lambda m: (len(map_short_name(m)), map_short_name(m)))
        logger.info(
            "--town=%r 匹配到多张地图，选用 %s（短名 %s）",
            town_query, candidates[0], map_short_name(candidates[0]),
        )
        return candidates[0]

    lines = "\n  ".join(f"{map_short_name(m)}  <=  {m}" for m in available)
    raise ValueError(
        f"服务端没有与 --town={town_query!r} 匹配的地图。\n"
        f"可用地图（短名 <= 完整路径）:\n  {lines}\n"
        f"请改用 --town <短名> 或运行: python -m close_loop.monodrive.run_closed_loop --list-maps"
    )


def map_matches_query(cur_map_name: str, town_query: str) -> bool:
    short = map_short_name(cur_map_name).lower()
    q = town_query.lower()
    return q == short or q in short or q in cur_map_name.lower()


def _try_load_world(client: carla.Client, map_name: str) -> carla.World:
    """依次尝试完整路径、短名加载。"""
    short = map_short_name(map_name)
    for candidate in (map_name, short):
        if not candidate:
            continue
        try:
            logger.info("client.load_world(%r) ...", candidate)
            return client.load_world(candidate)
        except RuntimeError as exc:
            logger.warning("load_world(%r) 失败: %s", candidate, exc)
    raise RuntimeError(
        f"无法加载地图 {map_name!r}（短名 {short!r}）。"
        f"请运行 python -m close_loop.monodrive.run_closed_loop --list-maps 查看可用地图。"
    )


def get_or_load_world(client: carla.Client, town: str) -> carla.World:
    world = client.get_world()
    cur_map = world.get_map().name
    if map_matches_query(cur_map, town):
        logger.info("当前地图已满足 --town=%s（map name = %s）", town, cur_map)
        return world

    map_to_load = resolve_map_name(client, town)
    logger.info(
        "切换地图: 当前=%s → 目标=%s（短名 %s）",
        cur_map, map_to_load, map_short_name(map_to_load),
    )
    world = _try_load_world(client, map_to_load)
    # UE5 换图后给一帧时间稳定（异步模式下 wait_for_tick）
    try:
        settings = world.get_settings()
        if settings.synchronous_mode:
            world.tick()
        else:
            world.wait_for_tick(timeout=10.0)
    except Exception:  # noqa: BLE001
        pass
    logger.info("地图加载完成: %s", world.get_map().name)
    return world


def pick_spawn_point(
    spawn_points: List[carla.Transform],
    idx: Optional[int],
    use_random: bool,
    seed_rng: random.Random,
    fallback_label: str,
) -> int:
    """根据 (idx, use_random) 返回 spawn_points 上的索引。

    - 指定 ``idx`` → 固定该索引；
    - 未指定 ``idx`` 且 ``use_random`` → ``seed_rng`` 随机；
    - 未指定 ``idx`` 且非随机 → 固定 ``spawn_points[0]``。
    """
    if idx is not None:
        if not 0 <= idx < len(spawn_points):
            raise ValueError(
                f"{fallback_label} idx={idx} 越界，spawn_points 共 {len(spawn_points)} 个"
            )
        logger.info("%s 固定 spawn_points[%d]", fallback_label, idx)
        return int(idx)
    if use_random:
        chosen = seed_rng.randrange(0, len(spawn_points))
        logger.info("%s 随机 spawn_points[%d]", fallback_label, chosen)
        return chosen
    logger.info("%s 默认 spawn_points[0]（未开随机且未指定 idx）", fallback_label)
    return 0


def set_all_traffic_lights_green(world: carla.World, freeze: bool = True) -> int:
    """将当前世界中所有 ``traffic.traffic_light`` 设为绿灯。

    Carla 路口灯组会按周期把非当前相位灯置红，因此建议 ``freeze=True``，
    并在主循环中定期重复调用本函数（见 ``--all-lights-green``）。
    """
    try:
        tls = world.get_actors().filter("traffic.traffic_light")
    except Exception:  # noqa: BLE001
        logger.exception("枚举红绿灯失败")
        return 0

    n_ok = 0
    for tl in tls:
        try:
            tl.set_state(carla.TrafficLightState.Green)
            # 拉长绿灯相位，降低被周期逻辑抢回的概率
            if hasattr(tl, "set_green_time"):
                tl.set_green_time(1e6)
            if hasattr(tl, "set_red_time"):
                tl.set_red_time(0.01)
            if hasattr(tl, "set_yellow_time"):
                tl.set_yellow_time(0.01)
            if freeze and hasattr(tl, "freeze"):
                tl.freeze(True)
            n_ok += 1
        except Exception:  # noqa: BLE001
            logger.debug("设置红绿灯 %s 为绿灯失败", getattr(tl, "id", "?"), exc_info=True)
    return n_ok


def spawn_traffic(
    world: carla.World,
    tm: carla.TrafficManager,
    n_target: int,
    skip_idx: List[int],
    seed_rng: random.Random,
) -> List[carla.Vehicle]:
    """spawn N 辆随机 NPC 车辆并交给 TrafficManager 自动驾驶。"""
    bp_lib = world.get_blueprint_library()
    vehicle_bps = [
        bp for bp in bp_lib.filter("vehicle.*")
        if bp.get_attribute("number_of_wheels") is not None
        and int(bp.get_attribute("number_of_wheels").as_int()) == 4
    ]
    if not vehicle_bps:
        logger.warning("找不到任何 4 轮 vehicle blueprint，跳过交通流")
        return []

    spawn_points = world.get_map().get_spawn_points()
    candidate_idx = [i for i in range(len(spawn_points)) if i not in set(skip_idx)]
    seed_rng.shuffle(candidate_idx)

    spawned: List[carla.Vehicle] = []
    for i in candidate_idx:
        if len(spawned) >= n_target:
            break
        bp = seed_rng.choice(vehicle_bps)
        # 随机颜色 / 司机
        if bp.has_attribute("color"):
            colors = bp.get_attribute("color").recommended_values
            if colors:
                bp.set_attribute("color", seed_rng.choice(colors))
        if bp.has_attribute("driver_id"):
            ids = bp.get_attribute("driver_id").recommended_values
            if ids:
                bp.set_attribute("driver_id", seed_rng.choice(ids))
        bp.set_attribute("role_name", "autopilot")
        try:
            actor = world.try_spawn_actor(bp, spawn_points[i])
        except Exception:    # noqa: BLE001
            actor = None
        if actor is None:
            continue
        actor.set_autopilot(True, tm.get_port())
        # 给点变化避免太整齐
        tm.vehicle_percentage_speed_difference(actor, seed_rng.uniform(-10.0, 20.0))
        tm.ignore_lights_percentage(actor, 0.0)
        spawned.append(actor)
    logger.info("生成 %d / %d 辆 NPC 车辆", len(spawned), n_target)
    return spawned


def spawn_ego(
    world: carla.World,
    transform: carla.Transform,
    blueprint_id: str = "vehicle.mini.cooper",
) -> carla.Vehicle:
    bp_lib = world.get_blueprint_library()
    candidates = bp_lib.filter(blueprint_id)
    if not candidates:
        raise RuntimeError(f"找不到 blueprint: {blueprint_id}")
    bp = candidates[0]
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", "hero")
    actor = world.try_spawn_actor(bp, transform)
    if actor is None:
        # spawn 失败：尝试 spawn_actor（会抛异常，但更明确）
        actor = world.spawn_actor(bp, transform)
    return actor


def attach_front_camera(
    world: carla.World,
    parent: carla.Vehicle,
    width: int = 800,
    height: int = 450,
    fov: float = 120.0,
) -> carla.Sensor:
    """挂前视 RGB 摄像头。默认 800×450；模型侧仍会 resize 到 288×512。"""
    bp_lib = world.get_blueprint_library()
    bp = bp_lib.find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(width))
    bp.set_attribute("image_size_y", str(height))
    bp.set_attribute("fov", str(fov))
    bp.set_attribute("sensor_tick", "0.0")     # 同步模式下随 world.tick() 触发
    # 数据集里相机位姿大致在车头前方约 1.5m、离地 1.6m
    tf = carla.Transform(carla.Location(x=1.5, y=0.0, z=1.6))
    sensor = world.spawn_actor(bp, tf, attach_to=parent)
    return sensor


# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────
def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    _load_carla()

    from .agent import MonoDriveAgent  # noqa: E402
    from .planner_goal import DenseRoute  # noqa: E402
    from .visualizer import (  # noqa: E402
        BackgroundDebugDrawer,
        CameraProjector,
        DrawPayload,
        Mp4Recorder,
        draw_overlay_on_frame,
    )

    if args.camera_full_res:
        camera_width, camera_height = 1600, 900
    else:
        camera_width = int(args.camera_width)
        camera_height = int(args.camera_height)
    if camera_width <= 0 or camera_height <= 0:
        logger.error("camera 分辨率必须为正数，实际 %dx%d", camera_width, camera_height)
        return 1
    logger.info("相机 / MP4 分辨率: %dx%d", camera_width, camera_height)

    if args.seed is None:
        seed = int(time.time() * 1000) % (2**31)
        logger.info("未指定 --seed，本次运行 seed=%d（每次启动不同）", seed)
    else:
        seed = int(args.seed)
        logger.info("使用 --seed=%d（起点/终点随机可复现）", seed)
    rng = random.Random(seed)
    np.random.seed(seed)

    logger.info("连接 Carla server %s:%d ...", args.host, args.port)
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    try:
        server_version = client.get_server_version()
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "refused" in msg or "timeout" in msg or "timed out" in msg:
            logger.error(
                "无法连接 Carla server %s:%d —— %s\n"
                "请先在本机或对应主机启动 Carla 仿真端，例如：\n"
                "  cd <Carla安装目录> && ./CarlaUE4.sh -RenderOffScreen -nosound\n"
                "  # 或 UE5: ./CarlaUE5.sh -RenderOffScreen\n"
                "确认端口与 --host/--port 一致（默认 2000），再重试本脚本。",
                args.host,
                args.port,
                exc,
            )
            return 1
        raise
    logger.info("Carla server 版本: %s", server_version)

    if args.list_maps:
        available = list_available_maps(client)
        if not available:
            print("（无可用地图或无法查询）")
            return 1
        print("可用地图（短名 <= 完整路径）:")
        for m in available:
            print(f"  {map_short_name(m):20s}  <=  {m}")
        print("\n加载示例: python -m close_loop.monodrive.run_closed_loop --town", map_short_name(available[0]))
        return 0

    try:
        world = get_or_load_world(client, args.town)
    except (RuntimeError, ValueError) as exc:
        logger.error("%s", exc)
        available = list_available_maps(client)
        if available:
            logger.error("当前服务端可用地图:")
            for m in available:
                logger.error("  %s  <=  %s", map_short_name(m), m)
        return 1
    # 等待 server 完成加载
    world.tick() if world.get_settings().synchronous_mode else world.wait_for_tick()
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    if len(spawn_points) < 2:
        logger.error("地图 %s 的 spawn_points 数量过少 (%d)", args.town, len(spawn_points))
        return 1

    # TrafficManager（必须 sync）
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(seed)
    # 给所有交通车一些行为多样性
    tm.global_percentage_speed_difference(0.0)

    # World sync mode（必须在 TM sync 之后）
    old_settings = configure_world_sync(world, fixed_dt=0.125)

    ego_vehicle: Optional[carla.Vehicle] = None
    camera: Optional[carla.Sensor] = None
    traffic: List[carla.Vehicle] = []
    image_q: "queue.Queue[carla.Image]" = queue.Queue()
    recorder: Optional[Mp4Recorder] = None

    try:
        # ── 起点 / 终点 ──
        # 默认：未指定 idx 即随机；--no-random-* 才退回 spawn_points[0/1]
        rand_start = (args.start_idx is None) and (not args.no_random_start)
        rand_end = (args.end_idx is None) and (not args.no_random_target)
        start_idx = pick_spawn_point(spawn_points, args.start_idx, rand_start, rng, "start")
        end_idx = pick_spawn_point(spawn_points, args.end_idx, rand_end, rng, "end")
        if end_idx == start_idx:
            end_idx = (end_idx + len(spawn_points) // 2) % len(spawn_points)
            logger.info("终点 idx 与起点重合，调整为 spawn_points[%d]", end_idx)
        logger.info("起点 spawn_points[%d], 终点 spawn_points[%d]", start_idx, end_idx)

        start_tf = spawn_points[start_idx]
        end_tf = spawn_points[end_idx]

        # ── spawn ego ──
        ego_vehicle = spawn_ego(world, start_tf, blueprint_id="vehicle.mini.cooper")
        logger.info("ego 已 spawn: id=%s blueprint=%s", ego_vehicle.id, ego_vehicle.type_id)
        # 让物理稳定一帧
        world.tick()

        # ── 摄像头 ──
        camera = attach_front_camera(
            world, ego_vehicle,
            width=camera_width, height=camera_height, fov=120.0,
        )
        camera.listen(lambda image: image_q.put(image))
        # 摄像头第一帧会在第一个 tick 之后入队

        # ── 路径 ──
        route = DenseRoute(carla_map, sampling_resolution=args.sampling_resolution)
        route.compute(start_tf.location, end_tf.location)
        logger.info("路径长度 ~%.1f m, 航点数 %d", route.total_length, len(route))

        # ── 交通流 ──
        traffic = spawn_traffic(
            world, tm, n_target=args.n_traffic,
            skip_idx=[start_idx, end_idx], seed_rng=rng,
        )

        if args.all_lights_green:
            n_tl = set_all_traffic_lights_green(world, freeze=True)
            logger.info("已强制 %d 个红绿灯为绿灯（--all-lights-green）", n_tl)

        # ── Agent ──
        agent = MonoDriveAgent(
            vehicle=ego_vehicle,
            route=route,
            checkpoint=args.checkpoint,
            backbone_config_path=args.backbone_config,
            viz_top_k=args.viz_top_k,
            camera_width=camera_width,
            camera_height=camera_height,
            device=args.device,
            dt=0.125,
            long_mode=args.long_mode,
            accel_throttle_scale=args.accel_throttle_scale,
            accel_brake_scale=args.accel_brake_scale,
            ff_pid_correction_weight=args.ff_pid_correction_weight,
            precision=args.precision,
            use_pure_pursuit=not args.no_pure_pursuit,
            look_ahead_step=args.look_ahead_step,
            look_ahead_min_dist=args.look_ahead_min_dist,
            look_ahead_time=args.look_ahead_time,
            flip_y=args.flip_y,
            allow_cpu_fallback=args.allow_cpu_fallback,
            replan_every=args.replan_every,
            use_committed_tracking=not args.legacy_tracking,
            allow_reverse=args.allow_reverse,
            reverse_dx_threshold=args.reverse_dx_threshold,
            force_winner_idx=args.force_winner_idx,
            goal_min_dist_m=args.goal_min_dist_m,
            goal_hold_ticks=args.goal_hold_ticks,
            winner_hysteresis=args.winner_hysteresis,
            diagnostic_dir=args.diagnostic_dir,
            diagnostic_every=args.diagnostic_every,
        )
        logger.info(
            "Goal 选点: training_aligned（直线距离 >= %.1f m，路径前方最近点）| hold=%d tick (%.1fs)",
            args.goal_min_dist_m,
            args.goal_hold_ticks,
            args.goal_hold_ticks * 0.125,
        )
        if args.winner_hysteresis > 0:
            logger.info("Winner 迟滞: Δprob >= %.2f 才切换 mode", args.winner_hysteresis)
        if args.force_winner_idx is not None:
            logger.info("已强制 winner_idx=%d（忽略 probs.argmax）", args.force_winner_idx)
        if args.legacy_tracking:
            logger.info(
                "轨迹跟踪: legacy（每 tick 推理 + ego-local 预瞄，纵向 winner[0]）| long_mode=%s",
                args.long_mode,
            )
        else:
            logger.info(
                "轨迹跟踪: committed | replan_every=%d tick (%.1f Hz) | long_mode=%s",
                args.replan_every,
                8.0 / max(args.replan_every, 1),
                args.long_mode,
            )
        logger.info(
            "MonoDriveAgent long_mode=%s (acc_thr=%.2f, acc_brk=%.2f, ff_w=%.2f)",
            args.long_mode,
            args.accel_throttle_scale,
            args.accel_brake_scale,
            args.ff_pid_correction_weight,
        )

        # ── 可视化 ──
        # 关键: world.debug.draw_* 会被摄像头拍到 → 模型也会看见 → 默认关闭。
        enable_world_debug = (not args.no_viz) and args.world_debug
        drawer = BackgroundDebugDrawer(world, dt=0.125) if enable_world_debug else None
        if drawer is not None:
            drawer.start()
            logger.warning(
                "world.debug 已启用 (--world-debug)：模型摄像头会拍到调试线段，"
                "可能影响推理。仅当不需要模型干净输入时使用。",
            )

        # 2D overlay 投影器（无论是否录 MP4 都先备好；overlay 只在 --mp4 且未禁用时启用）
        projector = CameraProjector(width=camera_width, height=camera_height, fov_deg=120.0)
        do_mp4_overlay = bool(args.mp4) and not args.no_viz and not args.no_mp4_overlay

        if args.mp4:
            recorder = Mp4Recorder(
                args.mp4, fps=8, size=(camera_width, camera_height), fourcc="mp4v",
            )
            recorder.open()
            if do_mp4_overlay:
                logger.info("MP4 启用 2D overlay（模型看不到，只叠加在录像帧上）")
            else:
                logger.info("MP4 仅录原始相机帧（未做 overlay）")

        # ── 安全 Ctrl+C ──
        stop_flag = {"stop": False}

        def _on_signal(signum, frame):  # noqa: ARG001
            stop_flag["stop"] = True
            logger.warning("收到中断，准备退出")
        try:
            signal.signal(signal.SIGINT, _on_signal)
            signal.signal(signal.SIGTERM, _on_signal)
        except (ValueError, AttributeError):
            pass

        # ──── 主循环 ────
        t_start = time.perf_counter()
        # 各阶段 wall-clock 累积器（仅用于诊断）
        tt_tick = tt_sensor = tt_infer = tt_viz = tt_rec = 0.0
        n_model_steps = 0
        for tick_i in range(args.n_ticks):
            if stop_flag["stop"]:
                break

            _t0 = time.perf_counter()
            world.tick()
            tt_tick += time.perf_counter() - _t0

            # 仿真会周期性地把灯组切回红/黄，定期重新刷成绿灯
            if args.all_lights_green and tick_i % 8 == 0:
                set_all_traffic_lights_green(world, freeze=True)

            # 拉一帧摄像头（同步模式下每个 tick 必有一帧；超时打印警告）
            _t0 = time.perf_counter()
            try:
                image = image_q.get(timeout=2.0)
            except queue.Empty:
                logger.warning("tick %d: 摄像头超时无帧，跳过", tick_i)
                continue

            # 入 ring buffer
            # 注意：bgra 是 raw_data 的 numpy view（只读底层 buffer）。模型必须拿
            # **原始** 帧；后续若要在 MP4 上画 overlay，必须 .copy() 一份再画。
            bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
            agent.push_camera_bgra(bgra)
            # ego 状态（w 由 yaw 差分，与训练侧一致）
            agent.push_ego_snapshot()
            tt_sensor += time.perf_counter() - _t0

            # 推理 + 控制
            _t0 = time.perf_counter()
            agent.set_current_tick(tick_i)
            result = agent.run_step()
            ego_vehicle.apply_control(result.control)
            tt_infer += time.perf_counter() - _t0
            if result.used_model:
                n_model_steps += 1

            # ── 准备可视化通用数据（world.debug 和 2D overlay 共用） ──
            _t0 = time.perf_counter()
            z_ego = float(ego_vehicle.get_location().z)
            idx = agent._last_route_idx
            ahead_xyz: List[Tuple[float, float, float]] = [
                (float(wp.transform.location.x),
                 float(wp.transform.location.y),
                 float(wp.transform.location.z))
                for wp in route.waypoints_ahead(idx, max_distance=20.0)
            ]
            latest_ego = agent.ego_buf.latest()
            p_ref_viz = np.array([latest_ego.x, latest_ego.y], dtype=np.float64)
            goal_wp, _, goal_xy_viz = route.goal_at_training_aligned(
                idx, p_ref_viz, min_dist_m=agent.goal_min_dist_m,
            )
            goal_xyz = (
                float(goal_xy_viz[0]),
                float(goal_xy_viz[1]),
                float(goal_wp.transform.location.z),
            )

            # ── world.debug（默认关闭；摄像头能拍到） ──
            if drawer is not None:
                if result.used_model and result.committed_xy_world is not None:
                    payload = DrawPayload(
                        z_ego=z_ego,
                        winner_xy=result.committed_xy_world,
                        all_world=(result.all_trajs_world if args.draw_candidates else None),
                        winner_idx=result.winner_idx,
                        look_world=result.look_ahead_world,
                        route_ahead_xyz=ahead_xyz,
                        goal_xyz=goal_xyz,
                        is_replan=result.replanned,
                    )
                else:
                    payload = DrawPayload(
                        z_ego=z_ego,
                        winner_idx=-1,
                        route_ahead_xyz=ahead_xyz,
                        goal_xyz=goal_xyz,
                    )
                drawer.submit(payload)
            tt_viz += time.perf_counter() - _t0

            # ── MP4 录制（2D overlay 在写入前叠加，模型完全看不到） ──
            _t0 = time.perf_counter()
            if recorder is not None:
                if do_mp4_overlay:
                    # 拷贝一份避免污染 raw_data；overlay 直接画在副本上
                    bgra_overlay = bgra.copy()
                    projector.update_extrinsic(image.transform)   # 与本帧严格同步
                    if result.used_model and result.committed_xy_world is not None:
                        draw_overlay_on_frame(
                            bgra_overlay,
                            projector,
                            winner_xy_world=result.committed_xy_world,
                            z_traj=z_ego,
                            candidates_world=(
                                result.all_trajs_world if args.draw_candidates else None
                            ),
                            winner_idx=result.winner_idx,
                            look_world=result.look_ahead_world,
                            goal_xyz=goal_xyz,
                            route_ahead_xyz=ahead_xyz,
                            is_replan=result.replanned,
                        )
                    else:
                        draw_overlay_on_frame(
                            bgra_overlay,
                            projector,
                            goal_xyz=goal_xyz,
                            route_ahead_xyz=ahead_xyz,
                        )
                    recorder.write_bgra(bgra_overlay)
                else:
                    recorder.write_bgra(bgra)
            tt_rec += time.perf_counter() - _t0

            # 简单日志
            if tick_i % 8 == 0:
                speed = ego_vehicle.get_velocity()
                v_mag = math.hypot(speed.x, speed.y) * 3.6
                # L=legacy 每 tick 推理；R=replan；H<age>=持有 committed；F=fallback
                if result.used_model:
                    if args.legacy_tracking:
                        rp_tag = "L"
                    elif result.replanned:
                        rp_tag = "R"
                    else:
                        rp_tag = f"H{result.committed_age}"
                else:
                    rp_tag = "F"
                rev_flag = "R" if getattr(result.control, "reverse", False) else "-"
                logger.info(
                    "tick %4d | %s | win=%2d | tgt=%.1f km/h | accel=%.2f m/s² | v=%.1f km/h | thr=%.2f brk=%.2f rev=%s str=%+.2f"
                    " | goal_d=%.1fm%s"
                    " | times: tick=%.0fms sensor=%.0fms infer=%.0fms viz=%.0fms rec=%.0fms",
                    tick_i,
                    rp_tag,
                    result.winner_idx,
                    result.target_speed_kmh,
                    result.accel_cmd,
                    v_mag,
                    result.control.throttle,
                    result.control.brake,
                    rev_flag,
                    result.control.steer,
                    result.goal_dist_m,
                    " (goal↻)" if result.goal_refreshed else "",
                    tt_tick * 1000.0 / max(tick_i + 1, 1),
                    tt_sensor * 1000.0 / max(tick_i + 1, 1),
                    tt_infer * 1000.0 / max(tick_i + 1, 1),
                    tt_viz * 1000.0 / max(tick_i + 1, 1),
                    tt_rec * 1000.0 / max(tick_i + 1, 1),
                )

            if agent.is_done(threshold=3.0):
                logger.info("已到达终点 (tick %d)", tick_i)
                break

        wall = time.perf_counter() - t_start
        n_done = max(tick_i + 1, 1)
        logger.info(
            "主循环结束: ticks=%d (model=%d), wall=%.1fs (%.1f tick/s)\n"
            "  avg per-tick: tick=%.0fms sensor=%.0fms infer=%.0fms viz=%.0fms rec=%.0fms",
            n_done, n_model_steps, wall, n_done / max(wall, 1e-6),
            tt_tick * 1000.0 / n_done,
            tt_sensor * 1000.0 / n_done,
            tt_infer * 1000.0 / n_done,
            tt_viz * 1000.0 / n_done,
            tt_rec * 1000.0 / n_done,
        )
        return 0

    except Exception:    # noqa: BLE001
        logger.exception("主循环异常")
        return 2

    finally:
        # ── 清理 ──
        if recorder is not None:
            try:
                recorder.close()
            except Exception:    # noqa: BLE001
                logger.exception("Mp4Recorder.close 异常")
        # 关掉后台 debug 绘制线程
        try:
            if 'drawer' in locals() and drawer is not None:
                drawer.close(timeout=2.0)
        except Exception:    # noqa: BLE001
            logger.exception("BackgroundDebugDrawer.close 异常")
        if camera is not None:
            try:
                camera.stop()
                camera.destroy()
            except Exception:    # noqa: BLE001
                logger.exception("camera 清理异常")
        if traffic:
            client.apply_batch([carla.command.DestroyActor(v.id) for v in traffic])
            logger.info("销毁 %d 辆 NPC", len(traffic))
        if ego_vehicle is not None:
            try:
                ego_vehicle.destroy()
            except Exception:    # noqa: BLE001
                logger.exception("ego 销毁异常")
        try:
            tm.set_synchronous_mode(False)
        except Exception:    # noqa: BLE001
            pass
        restore_world_settings(world, old_settings)


if __name__ == "__main__":
    sys.exit(main())
