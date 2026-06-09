"""Carla 仿真到 MonoDriveBackbone 输入张量的转换。

负责两个 ring buffer 与一个坐标变换工具：

- ``FrameBuffer``：保存最近 8 帧前视摄像头图像，双线性下采样到 (288, 512)
  并归一化到 [0, 1] 的 FP32。``stack()`` 返回 ``(8, 3, 288, 512)``，
  对应 ``MonoDriveBackbone.forward(images=...)`` 的 ``[B, T, C, H, W]`` 输入。
- ``EgoBuffer``：保存最近 8 帧的 ego 物理状态：世界系 xy / yaw / 速度 / 加速度 / yaw_rate。
- ``build_ego_motion`` / ``build_target_point``：anchor 帧 ego-local ``(vx, vy, w)`` 与
  ``(target_x, target_y)``，格式与 ``data/b2d_dataset.py`` 一致。

**全部保持物理值**——模型内部目标点嵌入与 ego_motion 编码会自动 Symlog。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# 图像 / ego buffer 常量（与训练对齐）
# ─────────────────────────────────────────────────────────────
SOURCE_HW = (900, 1600)  # B2D 训练采集分辨率 (H, W)，仅作参考
DEFAULT_CAMERA_HW = (450, 800)  # 闭环默认 Carla 相机 (H, W)，低于训练分辨率以减轻渲染负担
MODEL_HW = (288, 512)    # 模型输入分辨率 (H, W)，与 config/vision_embedding.toml 一致
FINAL_HW = MODEL_HW
PAST_FRAMES = 8
TRAJECTORY_DT = 0.5  # 2Hz 未来轨迹标签间隔 (s)


def resize_frame_chw(frame_chw: torch.Tensor, out_height: int, out_width: int) -> torch.Tensor:
    """双线性下采样单帧 ``(3, H, W)`` FP32 图像到 ``(3, out_height, out_width)``。"""
    if frame_chw.ndim != 3:
        raise ValueError(
            f"frame_chw 期望 shape 为 (3, H, W)，实际为 {tuple(frame_chw.shape)}。"
        )
    if int(frame_chw.shape[0]) != 3:
        raise ValueError(f"frame_chw 通道数必须为 3，实际为 {frame_chw.shape[0]}。")
    resized = F.interpolate(
        frame_chw.unsqueeze(0),
        size=(int(out_height), int(out_width)),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0).contiguous()



def wrap_pi(x: float | np.ndarray) -> float | np.ndarray:
    """把角度（弧度）规整到 ``[-π, π)``。"""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


# ─────────────────────────────────────────────────────────────
# Frame buffer
# ─────────────────────────────────────────────────────────────
class FrameBuffer:
    """最近 ``maxlen`` 帧前视图像，FP32 [0,1] (3, H, W)；内部统一 resize 到 ``MODEL_HW``。"""

    def __init__(
        self,
        maxlen: int = PAST_FRAMES,
        source_hw: tuple[int, int] | None = DEFAULT_CAMERA_HW,
    ) -> None:
        self.maxlen = maxlen
        self.source_hw = source_hw
        self._buf: Deque[torch.Tensor] = deque(maxlen=maxlen)

    def __len__(self) -> int:
        return len(self._buf)

    def is_full(self) -> bool:
        return len(self._buf) >= self.maxlen

    def clear(self) -> None:
        self._buf.clear()

    def push_bgra_uint8(self, bgra: np.ndarray) -> None:
        """``bgra``: ``(H, W, 4)`` uint8（Carla ``sensor.camera.rgb`` 原始数据）。

        ``carla.Image.raw_data`` 是 BGRA；这里取前三通道并翻转为 RGB，再双线性
        下采样到 (288, 512)，最后归一化到 [0, 1] 的 FP32。
        """
        if bgra.ndim == 3 and bgra.shape[-1] == 4:
            rgb = bgra[..., :3][..., ::-1]   # BGRA -> RGB
        elif bgra.ndim == 3 and bgra.shape[-1] == 3:
            rgb = bgra
        else:
            raise ValueError(f"期望 (H,W,3|4)，当前 shape={bgra.shape}")
        if self.source_hw is not None and rgb.shape[:2] != self.source_hw:
            raise ValueError(
                f"摄像头输出分辨率 {rgb.shape[:2]} != 期望 {self.source_hw}; "
                f"请与 attach_front_camera 的 height/width 一致"
            )
        t = (
            torch.from_numpy(rgb).to(torch.float32)
            .div_(255.0)
            .permute(2, 0, 1)
            .contiguous()
        )  # (3, 900, 1600)
        t = resize_frame_chw(t, FINAL_HW[0], FINAL_HW[1])
        self._buf.append(t)

    def stack(self) -> torch.Tensor:
        """返回 ``(T, 3, 288, 512)`` FP32（T = ``len(self)``）。"""
        if len(self._buf) == 0:
            raise RuntimeError("FrameBuffer 为空")
        return torch.stack(list(self._buf), dim=0).contiguous()  # (T, 3, H, W)


# ─────────────────────────────────────────────────────────────
# Ego buffer
# ─────────────────────────────────────────────────────────────
@dataclass
class EgoSnapshot:
    """一帧 ego 状态（全部世界系，FP64 numpy 标量）。"""

    x: float
    y: float
    yaw: float          # 弧度，世界系
    vx: float
    vy: float
    ax: float
    ay: float
    yaw_rate: float     # rad/s（世界系；旋转不变量，等于 ego-local 系下的 yaw_rate)


class EgoBuffer:
    """最近 ``maxlen`` 帧 ego 状态。"""

    def __init__(self, maxlen: int = PAST_FRAMES) -> None:
        self.maxlen = maxlen
        self._buf: Deque[EgoSnapshot] = deque(maxlen=maxlen)
        self._prev_yaw: Optional[float] = None  # 用于差分估 yaw_rate（rad/s）
        self._dt: float = 0.125  # 8 FPS

    def set_dt(self, dt: float) -> None:
        self._dt = float(dt)

    def __len__(self) -> int:
        return len(self._buf)

    def is_full(self) -> bool:
        return len(self._buf) >= self.maxlen

    def clear(self) -> None:
        self._buf.clear()
        self._prev_yaw = None

    def latest(self) -> EgoSnapshot:
        if not self._buf:
            raise RuntimeError("EgoBuffer 为空")
        return self._buf[-1]

    def push_from_vehicle(self, vehicle) -> None:
        """从 ``carla.Vehicle`` 抓一帧。

        - 世界坐标系：``vehicle.get_transform().location`` (x, y)，
          ``vehicle.get_transform().rotation.yaw`` (度) 转弧度
        - 速度/加速度：``vehicle.get_velocity()`` / ``vehicle.get_acceleration()`` 的 (x, y)
        - w：相邻帧 yaw 差分（与 ``data/b2d_preprocess.py`` 一致），不用 IMU 直出值

        Carla 坐标系是 *左手系*：z 朝上、x 前、y **右**。
        我们直接在 Carla 原生 xy 上操作，与 B2D 预处理一致。
        """
        tf = vehicle.get_transform()
        loc = tf.location
        rot = tf.rotation
        vel = vehicle.get_velocity()
        acc = vehicle.get_acceleration()

        x = float(loc.x)
        y = float(loc.y)
        yaw = math.radians(float(rot.yaw))
        vx = float(vel.x)
        vy = float(vel.y)
        ax = float(acc.x)
        ay = float(acc.y)

        if self._prev_yaw is not None and self._dt > 0:
            dyaw = wrap_pi(yaw - self._prev_yaw)
            yaw_rate = float(dyaw) / self._dt
        else:
            yaw_rate = 0.0
        self._prev_yaw = yaw

        self._buf.append(EgoSnapshot(x=x, y=y, yaw=yaw, vx=vx, vy=vy, ax=ax, ay=ay, yaw_rate=yaw_rate))

    def world_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """返回 ``(xy, yaw, v, a, yaw_rate)``，全部世界系，shape 分别为 (T,2)/(T,)/(T,2)/(T,2)/(T,)。"""
        t_count = len(self._buf)
        xy = np.zeros((t_count, 2), dtype=np.float64)
        yaw = np.zeros((t_count,), dtype=np.float64)
        v = np.zeros((t_count, 2), dtype=np.float64)
        a = np.zeros((t_count, 2), dtype=np.float64)
        yr = np.zeros((t_count,), dtype=np.float64)
        for i, s in enumerate(self._buf):
            xy[i, 0] = s.x
            xy[i, 1] = s.y
            yaw[i] = s.yaw
            v[i, 0] = s.vx
            v[i, 1] = s.vy
            a[i, 0] = s.ax
            a[i, 1] = s.ay
            yr[i] = s.yaw_rate
        return xy, yaw, v, a, yr


# ─────────────────────────────────────────────────────────────
# 世界系 -> ego-local 系
# ─────────────────────────────────────────────────────────────
def to_ego_local_xy(
    xy_world: np.ndarray, yaw_ref: float, p_ref: np.ndarray
) -> np.ndarray:
    """把若干个世界系 xy 点变换到 ego-local 系（以 ``p_ref`` 为原点、``yaw_ref`` 为前向）。

    ``xy_world``: ``(N, 2)`` 或 ``(2,)``；返回 shape 与输入一致。
    """
    c, s = math.cos(-yaw_ref), math.sin(-yaw_ref)
    r_inv = np.array([[c, -s], [s, c]], dtype=np.float64)
    arr = np.asarray(xy_world, dtype=np.float64)
    flat = arr.reshape(-1, 2)
    out = (flat - np.asarray(p_ref, dtype=np.float64)) @ r_inv.T
    return out.reshape(arr.shape)


def build_ego_motion(ego_buf: EgoBuffer) -> torch.Tensor:
    """anchor 帧 ego-local ``(vx, vy, w)`` → ``(3,)`` 物理值张量。"""
    if not ego_buf.is_full():
        raise RuntimeError(
            f"EgoBuffer 未填满: 当前 {len(ego_buf)}/{ego_buf.maxlen}"
        )
    _xy_w, yaw_w, v_w, _a_w, yr = ego_buf.world_arrays()
    if yaw_w.shape[0] != PAST_FRAMES:
        raise RuntimeError(f"期望 {PAST_FRAMES} 帧，实际 {yaw_w.shape[0]}")

    ref = PAST_FRAMES - 1
    yaw_ref = float(yaw_w[ref])
    v_local = to_ego_local_xy(v_w, yaw_ref, np.zeros(2, dtype=np.float64))
    w = float(yr[ref])
    return torch.tensor(
        [v_local[ref, 0], v_local[ref, 1], w],
        dtype=torch.float32,
    )


def build_target_point(
    ego_buf: EgoBuffer,
    goal_world_xy: np.ndarray,
) -> torch.Tensor:
    """anchor 帧 ego-local 目标点 ``(x, y)`` → ``(2,)`` 物理值张量。"""
    if not ego_buf.is_full():
        raise RuntimeError(
            f"EgoBuffer 未填满: 当前 {len(ego_buf)}/{ego_buf.maxlen}"
        )
    xy_w, yaw_w, _v_w, _a_w, _yr = ego_buf.world_arrays()
    ref = PAST_FRAMES - 1
    p_ref = xy_w[ref]
    yaw_ref = float(yaw_w[ref])
    goal_local = to_ego_local_xy(
        np.asarray(goal_world_xy, dtype=np.float64), yaw_ref, p_ref
    )
    return torch.tensor([goal_local[0], goal_local[1]], dtype=torch.float32)


def build_goal_dxy(
    ego_buf: EgoBuffer,
    goal_world_xy: np.ndarray,
) -> torch.Tensor:
    """``build_target_point`` 的兼容别名。"""
    return build_target_point(ego_buf, goal_world_xy)


def ego_local_to_world(
    xy_local: np.ndarray, yaw_ref: float, p_ref: np.ndarray
) -> np.ndarray:
    """ego-local 系 -> 世界系，``R(yaw_ref) @ xy + p_ref``。

    用于把模型输出的轨迹（ego-local 系）转回 Carla 世界系做控制 / 可视化。
    """
    c, s = math.cos(yaw_ref), math.sin(yaw_ref)
    r_mat = np.array([[c, -s], [s, c]], dtype=np.float64)
    arr = np.asarray(xy_local, dtype=np.float64)
    flat = arr.reshape(-1, 2)
    out = flat @ r_mat.T + np.asarray(p_ref, dtype=np.float64)
    return out.reshape(arr.shape)
