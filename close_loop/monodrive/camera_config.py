"""B2D ``CAM_FRONT`` 相机参数，与 ``datasets/scenes/*/anno/*.json.gz`` 中 ``sensors.CAM_FRONT`` 一致。

训练采集使用 1600×900、水平 FOV 70°；``cam2ego`` 平移为车体前方 0.8 m、离地 1.6 m
（``y=0`` 时与 Carla 车体坐标 x 前 / y 右 / z 上一致）。
"""

from __future__ import annotations

import math
from typing import Tuple

# sensors.CAM_FRONT @ 1600×900（B2D 原始前视分辨率）
B2D_CAMERA_FOV_DEG = 70.0
B2D_CAMERA_WIDTH = 1600
B2D_CAMERA_HEIGHT = 900
B2D_CAMERA_HW = (B2D_CAMERA_HEIGHT, B2D_CAMERA_WIDTH)  # (H, W)

# cam2ego 平移 (m)：B2D ego x 前 / y 左 / z 上
B2D_CAM2EGO_X_M = 0.8
B2D_CAM2EGO_Y_M = 0.0
B2D_CAM2EGO_Z_M = 1.6
B2D_CAM2EGO_XYZ = (B2D_CAM2EGO_X_M, B2D_CAM2EGO_Y_M, B2D_CAM2EGO_Z_M)

# 训练集 1600×900 针孔内参（annotation intrinsic 样例值）
B2D_CAMERA_FX_1600 = 1142.5184053936916
B2D_CAMERA_FY_1600 = 1142.5184053936916
B2D_CAMERA_CX_1600 = 800.0
B2D_CAMERA_CY_1600 = 450.0


def pinhole_intrinsics(
    width: int,
    height: int,
    fov_deg: float = B2D_CAMERA_FOV_DEG,
) -> Tuple[float, float, float, float]:
    """由水平 FOV 与分辨率计算 ``(fx, fy, cx, cy)``，与 B2D ``intrinsic`` 公式一致。"""
    fx = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy


def scale_b2d_intrinsics(width: int, height: int) -> Tuple[float, float, float, float]:
    """把 B2D 1600×900 内参线性缩放到任意同宽高比分辨率。"""
    sx = width / B2D_CAMERA_WIDTH
    sy = height / B2D_CAMERA_HEIGHT
    return (
        B2D_CAMERA_FX_1600 * sx,
        B2D_CAMERA_FY_1600 * sy,
        B2D_CAMERA_CX_1600 * sx,
        B2D_CAMERA_CY_1600 * sy,
    )
