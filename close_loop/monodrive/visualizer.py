"""可视化 + MP4 录制。

- ``WorldDebugDrawer``：调用 ``world.debug.draw_line/draw_point`` 在 Carla 服务器端
  渲染 winner 轨迹（粗、绿）、其余 7 条候选（细、灰）、planner 前方 20m 航点
  （蓝）以及目标点（红色球）。``life_time`` 取略大于一个 tick，避免闪烁。
  ⚠️ 这些线段会被摄像头拍到，**模型会看到它们**；默认关闭，仅 ``--world-debug`` 开启。
- ``CameraProjector`` / ``draw_overlay_on_frame``：把世界系轨迹/航点投影到摄像头像素
  坐标，**直接在录制帧上用 cv2 画线**。模型 buffer 拿到的是原始干净图像，MP4 录像里
  才看得到叠加的轨迹。
- ``Mp4Recorder``：把前视摄像头的 BGRA 数据写成 ``mp4v`` 编码的 MP4 文件
  （fps=8，分辨率 = 摄像头分辨率，默认 1600×900）。**完全不依赖屏幕窗口**，
  与 Carla server 的 ``-RenderOffScreen`` 离屏模式兼容。
"""

from __future__ import annotations

import logging
import math
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

import carla

logger = logging.getLogger("monodrive_viz")


def _to_location(x: float, y: float, z: float) -> carla.Location:
    return carla.Location(x=float(x), y=float(y), z=float(z))


# ─────────────────────────────────────────────────────────────
# 2D 叠加：世界坐标 → 摄像头像素
# ─────────────────────────────────────────────────────────────
def _carla_rotation_matrix(rotation: "carla.Rotation") -> np.ndarray:
    """Carla / Unreal 的 (roll, pitch, yaw) → 3×3 旋转矩阵（左手系，z 上）。"""
    cy = math.cos(math.radians(rotation.yaw))
    sy = math.sin(math.radians(rotation.yaw))
    cp = math.cos(math.radians(rotation.pitch))
    sp = math.sin(math.radians(rotation.pitch))
    cr = math.cos(math.radians(rotation.roll))
    sr = math.sin(math.radians(rotation.roll))
    return np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
        [sp,      -cp * sr,                cp * cr],
    ], dtype=np.float64)


class CameraProjector:
    """把世界系点投影到摄像头像素 ``(u, v)``。

    使用拍摄当帧的 ``sensor.transform`` 作为外参（``carla.Image.transform`` 直接给出，
    与 ``world.tick()`` 时刻同步）。
    """

    def __init__(self, width: int, height: int, fov_deg: float) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fov_deg = float(fov_deg)
        self.fx = self.width / (2.0 * math.tan(math.radians(self.fov_deg) / 2.0))
        self.fy = self.fx
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

    def update_extrinsic(self, sensor_transform: "carla.Transform") -> None:
        loc = sensor_transform.location
        self.cam_pos = np.array([loc.x, loc.y, loc.z], dtype=np.float64)
        self.R_cam_to_world = _carla_rotation_matrix(sensor_transform.rotation)
        self.R_world_to_cam = self.R_cam_to_world.T

    def project(self, world_xyz: np.ndarray) -> Optional[Tuple[int, int]]:
        """``(3,)`` 世界点 → ``(u, v)`` 像素；点在相机后方或视口外返回 ``None``。"""
        p = self.R_world_to_cam @ (world_xyz - self.cam_pos)
        depth = p[0]
        if depth <= 0.1:
            return None
        u = self.cx + self.fx * (p[1] / depth)
        v = self.cy - self.fy * (p[2] / depth)
        if not (-self.width <= u <= 2 * self.width and -self.height <= v <= 2 * self.height):
            return None
        return int(round(u)), int(round(v))

    def project_xy(self, xy: np.ndarray, z: float) -> List[Optional[Tuple[int, int]]]:
        """``(T, 2)`` 世界 xy + 常 z → 像素 ``[(u,v), ...]``（不可见点为 ``None``）。"""
        out: List[Optional[Tuple[int, int]]] = []
        for i in range(xy.shape[0]):
            out.append(self.project(np.array([xy[i, 0], xy[i, 1], z], dtype=np.float64)))
        return out


def _polyline_clipped(pixels: List[Optional[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
    """把含 ``None`` 的像素序列切成多段连续 polyline，方便 ``cv2.polylines``。"""
    segs: List[List[Tuple[int, int]]] = []
    cur: List[Tuple[int, int]] = []
    for p in pixels:
        if p is None:
            if len(cur) >= 2:
                segs.append(cur)
            cur = []
        else:
            cur.append(p)
    if len(cur) >= 2:
        segs.append(cur)
    return segs


def draw_overlay_on_frame(
    bgra: np.ndarray,
    projector: CameraProjector,
    winner_xy_world: Optional[np.ndarray] = None,
    z_traj: float = 0.0,
    candidates_world: Optional[np.ndarray] = None,
    winner_idx: int = -1,
    look_world: Optional[Tuple[float, float, float]] = None,
    goal_xyz: Optional[Tuple[float, float, float]] = None,
    route_ahead_xyz: Optional[Sequence[Tuple[float, float, float]]] = None,
    is_replan: bool = False,
) -> None:
    """在 ``bgra``（**原地修改**）上叠加 2D 轨迹/航点。需要 ``cv2``。"""
    import cv2  # 延迟导入

    # 颜色 (BGR)
    c_winner = (60, 230, 60) if is_replan else (220, 220, 60)   # 重 plan 绿 / 持有青
    c_candidate = (160, 160, 160)
    c_route = (255, 140, 60)
    c_goal = (40, 40, 240)
    c_look = (40, 220, 240)

    # 候选轨迹（细灰）
    if candidates_world is not None:
        for k in range(candidates_world.shape[0]):
            if k == winner_idx:
                continue
            pix = projector.project_xy(candidates_world[k], z_traj + 0.05)
            for seg in _polyline_clipped(pix):
                cv2.polylines(bgra, [np.array(seg, dtype=np.int32)], False, c_candidate, 1, cv2.LINE_AA)

    # winner 轨迹（粗）
    if winner_xy_world is not None and len(winner_xy_world) >= 2:
        pix = projector.project_xy(winner_xy_world, z_traj + 0.10)
        for seg in _polyline_clipped(pix):
            cv2.polylines(bgra, [np.array(seg, dtype=np.int32)], False, c_winner, 2, cv2.LINE_AA)

    # planner 前方航点（连线 + 点）
    if route_ahead_xyz:
        route_pix = [projector.project(np.array(p, dtype=np.float64)) for p in route_ahead_xyz]
        for seg in _polyline_clipped(route_pix):
            cv2.polylines(bgra, [np.array(seg, dtype=np.int32)], False, c_route, 1, cv2.LINE_AA)

    # look-ahead 点
    if look_world is not None:
        p = projector.project(np.array(look_world, dtype=np.float64))
        if p is not None:
            cv2.circle(bgra, p, 4, c_look, -1, cv2.LINE_AA)

    # goal
    if goal_xyz is not None:
        p = projector.project(np.array(goal_xyz, dtype=np.float64))
        if p is not None:
            cv2.circle(bgra, p, 6, c_goal, 2, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────
# World debug 绘制
# ─────────────────────────────────────────────────────────────
class WorldDebugDrawer:
    """轻量包装 ``carla.DebugHelper``。

    所有绘制 API 都对 world 调一次 ``world.debug.draw_*``。``life_time`` 取
    ``2 * dt``，让线段在一个 tick 内既能完整显示也不会跨帧残留太久。
    """

    def __init__(self, world: carla.World, dt: float = 0.125) -> None:
        self.world = world
        self.debug = world.debug
        self.dt = float(dt)
        self.life_time = float(2.0 * self.dt)

        # 颜色（RGB）
        self.color_winner = carla.Color(50, 230, 50)      # 亮绿
        self.color_candidate = carla.Color(150, 150, 150) # 灰
        self.color_route = carla.Color(60, 120, 255)      # 蓝
        self.color_goal = carla.Color(255, 40, 40)        # 红
        self.color_look = carla.Color(255, 220, 40)       # 黄

    # ─────────────────────────────────────────────────────────────
    def draw_winner(
        self, traj_world_xy: np.ndarray, z: float, thickness: float = 0.25
    ) -> None:
        """``traj_world_xy``: ``(T, 2)`` world xy。""" 
        if traj_world_xy is None or len(traj_world_xy) < 2:
            return
        for i in range(len(traj_world_xy) - 1):
            p0 = _to_location(traj_world_xy[i, 0], traj_world_xy[i, 1], z + 0.5)
            p1 = _to_location(traj_world_xy[i + 1, 0], traj_world_xy[i + 1, 1], z + 0.5)
            self.debug.draw_line(p0, p1, thickness=thickness,
                                 color=self.color_winner, life_time=self.life_time)
        # winner 末端打点
        end = traj_world_xy[-1]
        self.debug.draw_point(
            _to_location(end[0], end[1], z + 0.6),
            size=0.12, color=self.color_winner, life_time=self.life_time,
        )

    def draw_candidates(
        self,
        all_traj_world_xy: np.ndarray,
        winner_idx: int,
        z: float,
        thickness: float = 0.07,
    ) -> None:
        """``all_traj_world_xy``: ``(n_traj, T, 2)``。winner 跳过（已经由 ``draw_winner`` 画过）。"""
        if all_traj_world_xy is None:
            return
        n_traj = all_traj_world_xy.shape[0]
        for k in range(n_traj):
            if k == winner_idx:
                continue
            traj = all_traj_world_xy[k]
            if len(traj) < 2:
                continue
            for i in range(len(traj) - 1):
                p0 = _to_location(traj[i, 0], traj[i, 1], z + 0.3)
                p1 = _to_location(traj[i + 1, 0], traj[i + 1, 1], z + 0.3)
                self.debug.draw_line(p0, p1, thickness=thickness,
                                     color=self.color_candidate, life_time=self.life_time)

    def draw_route_lookahead(
        self, waypoints: Sequence[carla.Waypoint], z_offset: float = 0.2
    ) -> None:
        if not waypoints or len(waypoints) < 2:
            return
        for i in range(len(waypoints) - 1):
            l0 = waypoints[i].transform.location
            l1 = waypoints[i + 1].transform.location
            p0 = _to_location(l0.x, l0.y, l0.z + z_offset)
            p1 = _to_location(l1.x, l1.y, l1.z + z_offset)
            self.debug.draw_line(p0, p1, thickness=0.15,
                                 color=self.color_route, life_time=self.life_time)

    def draw_goal(self, wp: carla.Waypoint, z_offset: float = 0.8) -> None:
        if wp is None:
            return
        loc = wp.transform.location
        self.debug.draw_point(
            _to_location(loc.x, loc.y, loc.z + z_offset),
            size=0.22, color=self.color_goal, life_time=self.life_time,
        )

    def draw_look_ahead(self, look_world: Optional[Tuple[float, float, float]]) -> None:
        if look_world is None:
            return
        self.debug.draw_point(
            _to_location(look_world[0], look_world[1], look_world[2] + 0.7),
            size=0.16, color=self.color_look, life_time=self.life_time,
        )


# ─────────────────────────────────────────────────────────────
# 后台线程版 debug 绘制（fire-and-forget）
# ─────────────────────────────────────────────────────────────
@dataclass
class DrawPayload:
    """一帧 debug 绘制要用到的全部数据，从主线程 pickle-friendly 地传到 worker。

    所有 carla.Waypoint 已经被预先抽成 ``(x, y, z)``，避免跨线程访问 carla actor。
    """

    z_ego: float = 0.0
    winner_xy: Optional[np.ndarray] = None           # (T, 2) world xy（committed 持有的实际跟踪路径）
    all_world: Optional[np.ndarray] = None           # (top_k, T, 2) 上次 re-plan 时的候选
    winner_idx: int = -1
    look_world: Optional[Tuple[float, float, float]] = None
    route_ahead_xyz: List[Tuple[float, float, float]] = field(default_factory=list)
    goal_xyz: Optional[Tuple[float, float, float]] = None
    is_replan: bool = False                          # 本帧是否刚刚 re-plan（用于高亮）


class BackgroundDebugDrawer:
    """把 ``WorldDebugDrawer`` 包到后台线程：主循环 ``submit(payload)`` 立即返回。

    一个 worker 线程串行调用 ``carla.DebugHelper.draw_*``。如果主循环节奏比 RPC
    快，新 payload 会**替换**未消费的旧 payload（``queue.Queue(maxsize=1)`` + 拒绝
    阻塞），保证主循环永远不会被绘制 backpressure 卡住。

    使用::

        drawer = BackgroundDebugDrawer(world, dt=0.125)
        drawer.start()
        drawer.submit(DrawPayload(...))
        ...
        drawer.close()
    """

    def __init__(self, world: carla.World, dt: float = 0.125) -> None:
        self._impl = WorldDebugDrawer(world, dt=dt)
        self._queue: "queue.Queue[Optional[DrawPayload]]" = queue.Queue(maxsize=1)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        # 主线程侧的统计（无锁，仅记录）
        self.dropped = 0
        self.drawn = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="monodrive-debug-drawer", daemon=True)
        self._thread.start()

    def submit(self, payload: DrawPayload) -> None:
        """非阻塞投递：若 worker 还没消费上一帧，丢弃旧的、放新的。"""
        if self._thread is None:
            self.start()
        # 替换语义：先 try_put，满了就先抽掉再 put
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            try:
                _ = self._queue.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(payload)
            except queue.Full:
                self.dropped += 1

    def close(self, timeout: float = 1.5) -> None:
        if self._thread is None:
            return
        self._stop.set()
        # 放一个 sentinel 唤醒 worker
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            try:
                _ = self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
        self._thread.join(timeout=timeout)
        self._thread = None
        logger.info(
            "BackgroundDebugDrawer 关闭: drawn=%d, dropped=%d",
            self.drawn, self.dropped,
        )

    # ─────────────────────────────────────────────────────────────
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                payload = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if payload is None:
                return
            try:
                self._draw_one(payload)
                self.drawn += 1
            except Exception:    # noqa: BLE001
                logger.exception("BackgroundDebugDrawer worker 异常，继续运行")

    def _draw_one(self, p: DrawPayload) -> None:
        # 顺序：候选 → winner（持有 / 新 plan 颜色不同）→ look-ahead → route → goal
        if p.all_world is not None and p.winner_idx >= 0:
            self._impl.draw_candidates(p.all_world, p.winner_idx, p.z_ego)
        if p.winner_xy is not None:
            # 刚 re-plan 的帧用粗绿（亮），持有帧用稍细的青色，便于肉眼看到「切换 vs 持有」
            thickness = 0.25 if p.is_replan else 0.18
            self._impl.draw_winner(p.winner_xy, p.z_ego, thickness=thickness)
        self._impl.draw_look_ahead(p.look_world)

        # route_ahead 是预抽出的 (x,y,z) list；为复用 WorldDebugDrawer.draw_route_lookahead
        # 我们把它包成最小化 fake waypoint 列表（仅暴露 transform.location）。
        if p.route_ahead_xyz:
            fake_wps = [_FakeRouteWp(x, y, z) for (x, y, z) in p.route_ahead_xyz]
            self._impl.draw_route_lookahead(fake_wps, z_offset=0.0)

        if p.goal_xyz is not None:
            fake_goal = _FakeRouteWp(*p.goal_xyz)
            self._impl.draw_goal(fake_goal, z_offset=0.0)


@dataclass
class _FakeRouteWp:
    """最小化 ``carla.Waypoint`` 替身：仅暴露 ``transform.location.{x,y,z}``。

    用于把主线程预抽的 ``(x, y, z)`` 喂给 ``WorldDebugDrawer.draw_route_lookahead`` /
    ``draw_goal``，避免在 worker 线程里访问真实 carla 对象。
    """
    x: float
    y: float
    z: float

    @property
    def transform(self) -> "_FakeRouteTf":   # noqa: F821 - forward ref
        return _FakeRouteTf(carla.Location(x=self.x, y=self.y, z=self.z))


@dataclass
class _FakeRouteTf:
    location: carla.Location


# ─────────────────────────────────────────────────────────────
# MP4 录制
# ─────────────────────────────────────────────────────────────
class Mp4Recorder:
    """前视摄像头帧 → MP4 文件（``mp4v`` 编码）。

    用法::

        rec = Mp4Recorder("out.mp4", fps=8, size=(1600, 900))
        rec.open()
        rec.write_bgra(bgra_uint8)         # 直接传 carla.Image.raw_data 的 numpy view
        ...
        rec.close()

    ``size`` = ``(width, height)``。所有帧都按这个尺寸写入；尺寸不一致时抛错。
    """

    def __init__(
        self,
        out_path: str | Path,
        fps: int = 8,
        size: Tuple[int, int] = (1600, 900),
        fourcc: str = "mp4v",
    ) -> None:
        self.out_path = Path(out_path)
        self.fps = int(fps)
        self.size = (int(size[0]), int(size[1]))
        self.fourcc = fourcc
        self._writer = None
        self._frame_count = 0

    # ─────────────────────────────────────────────────────────────
    def open(self) -> None:
        import cv2  # 延迟导入，避免没装 cv2 时影响其他流程

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*self.fourcc)
        self._writer = cv2.VideoWriter(
            str(self.out_path), fourcc, float(self.fps), self.size, True
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"无法打开 cv2.VideoWriter: {self.out_path}")
        logger.info("MP4 录制到 %s  (fps=%d, size=%s)", self.out_path, self.fps, self.size)

    def is_open(self) -> bool:
        return self._writer is not None

    def write_bgra(self, bgra: np.ndarray) -> None:
        """``bgra``: ``(H, W, 4)`` uint8 (Carla ``carla.Image.raw_data`` 的标准布局)。

        cv2.VideoWriter 期望 BGR；这里直接丢前 3 通道即可。
        """
        if self._writer is None:
            raise RuntimeError("Mp4Recorder 还未 open()")
        if bgra.ndim != 3 or bgra.shape[-1] not in (3, 4):
            raise ValueError(f"期望 (H,W,3|4) uint8，当前 {bgra.shape}")
        if bgra.shape[1] != self.size[0] or bgra.shape[0] != self.size[1]:
            raise ValueError(
                f"帧尺寸 (W,H)={bgra.shape[1]}x{bgra.shape[0]} != writer 配置 {self.size}"
            )
        bgr = bgra[..., :3].copy()
        # mp4v 编码器某些后端只接受 contiguous 数组
        self._writer.write(np.ascontiguousarray(bgr))
        self._frame_count += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info("MP4 写入完成: %s  (frames=%d)", self.out_path, self._frame_count)

    def __enter__(self) -> "Mp4Recorder":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
