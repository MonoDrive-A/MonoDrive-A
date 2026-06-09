"""密集 GlobalRoutePlanner 包装 + 与训练集对齐的 goal 查询。

直接复用 ``agents.navigation.global_route_planner.GlobalRoutePlanner``，
``sampling_resolution`` 取 0.5m 让路径上的航点足够密。

**Goal 选点（默认）** — ``goal_at_training_aligned``，与
``data/b2d_preprocess.py`` 的 ``_build_target_candidates`` 一致：

- 候选：当前路径索引 **之后** 的所有航点；
- 选取：相对 ``p_ref`` 直线欧氏距离落在 ``[GOAL_MIN_DIST_M, GOAL_MAX_DIST_M]``（默认 24–30 m）
  的**第一个**前方航点；
- 若均不在该区间：取**最远**点；无候选则路径终点。

旧接口 ``goal_at_arc_distance``（沿路径弧长）仍保留，供对比 / 调试。

``DenseRoute``::

    route = DenseRoute(carla_map, sampling_resolution=0.5)
    route.compute(start_loc, end_loc)
    idx = route.nearest_index(ego_location)
    goal_wp, idx_g, goal_xy = route.goal_at_training_aligned(
        idx, p_ref_world, min_dist_m=GOAL_MIN_DIST_M,
    )
"""

from __future__ import annotations

# 与 data/b2d_preprocess.py target_min/max_distance 保持一致
GOAL_MIN_DIST_M = 24.0
GOAL_MAX_DIST_M = 30.0

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption


@dataclass
class DenseRoute:
    """对 ``GlobalRoutePlanner`` 输出的密集 ``(carla.Waypoint, RoadOption)`` 路径做的薄封装。

    属性::

        waypoints  : list[carla.Waypoint]   全路径航点
        options    : list[RoadOption]       同长度的路径选项
        cum_arc    : np.ndarray (N,)        累计弧长 (m)，``cum_arc[0] = 0``
        xy         : np.ndarray (N, 2)      航点 (x, y) world
    """

    carla_map: carla.Map
    sampling_resolution: float = 0.5
    grp: Optional[GlobalRoutePlanner] = field(default=None, init=False)
    waypoints: List["carla.Waypoint"] = field(default_factory=list)
    options: List[RoadOption] = field(default_factory=list)
    cum_arc: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))

    def __post_init__(self) -> None:
        self.grp = GlobalRoutePlanner(self.carla_map, self.sampling_resolution)

    # ─────────────────────────────────────────────────────────────
    def compute(
        self,
        start_location: carla.Location,
        end_location: carla.Location,
    ) -> None:
        """在 ``[start_location, end_location]`` 间生成 0.5m 密集航点路径。"""
        assert self.grp is not None
        plan = self.grp.trace_route(start_location, end_location)
        if not plan:
            raise RuntimeError(
                f"GlobalRoutePlanner.trace_route 返回空路径 "
                f"(start={start_location}, end={end_location})"
            )
        self.waypoints = [wp for wp, _ in plan]
        self.options = [opt for _, opt in plan]
        n = len(self.waypoints)
        xy = np.zeros((n, 2), dtype=np.float64)
        for i, wp in enumerate(self.waypoints):
            loc = wp.transform.location
            xy[i, 0] = float(loc.x)
            xy[i, 1] = float(loc.y)
        self.xy = xy
        diffs = np.diff(xy, axis=0)
        seg_len = np.linalg.norm(diffs, axis=1)
        self.cum_arc = np.concatenate([[0.0], np.cumsum(seg_len)])

    # ─────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.waypoints)

    @property
    def total_length(self) -> float:
        if len(self) == 0:
            return 0.0
        return float(self.cum_arc[-1])

    def end_location(self) -> carla.Location:
        return self.waypoints[-1].transform.location

    # ─────────────────────────────────────────────────────────────
    def nearest_index(self, ego_location: carla.Location, search_from: int = 0) -> int:
        """返回路径上离 ego 最近的航点索引。

        ``search_from`` 用于在主循环里跨帧单调推进，避免在自交叉路口处回跳。
        """
        if len(self) == 0:
            raise RuntimeError("DenseRoute 还未 compute()")
        ex, ey = float(ego_location.x), float(ego_location.y)
        seg = self.xy[search_from:]
        d2 = (seg[:, 0] - ex) ** 2 + (seg[:, 1] - ey) ** 2
        return int(search_from + np.argmin(d2))

    # ─────────────────────────────────────────────────────────────
    def goal_at_training_aligned(
        self,
        idx_now: int,
        p_ref_world: np.ndarray,
        min_dist_m: float = GOAL_MIN_DIST_M,
        max_dist_m: float = GOAL_MAX_DIST_M,
    ) -> Tuple[carla.Waypoint, int, np.ndarray]:
        """与训练集 ``_build_target_candidates`` 相同的 goal 规则（直线距离，非弧长）。

        Args:
            idx_now: 路径上离 ego 最近的航点索引（``nearest_index`` 输出）。
            p_ref_world: 参考点世界系 xy，应与 ``build_target_point`` 的 ref 帧一致
                         （EgoBuffer 最新帧位置）。
            min_dist_m: 目标点最小直线距离 (m)，默认 24。
            max_dist_m: 目标点最大直线距离 (m)，默认 30。

        Returns:
            ``(goal_waypoint, goal_index, goal_xy)``，``goal_xy`` shape ``(2,)``。
        """
        if len(self) == 0:
            raise RuntimeError("DenseRoute 还未 compute()")
        if float(min_dist_m) > float(max_dist_m):
            raise ValueError(
                f"min_dist_m 必须 <= max_dist_m，实际为 {min_dist_m} > {max_dist_m}"
            )
        idx_now = max(0, min(int(idx_now), len(self) - 1))
        p_ref = np.asarray(p_ref_world, dtype=np.float64).reshape(2)

        # 候选 = 路径上严格在 idx_now 之后的航点（对齐训练 ego_ts > t_ref）
        idx_start = min(idx_now + 1, len(self) - 1)
        if idx_start >= len(self):
            idx = len(self) - 1
            return self.waypoints[idx], idx, self.xy[idx].copy()

        cand_xy = self.xy[idx_start:]
        dist = np.hypot(cand_xy[:, 0] - p_ref[0], cand_xy[:, 1] - p_ref[1])
        in_band = (dist >= float(min_dist_m)) & (dist <= float(max_dist_m))
        if np.any(in_band):
            local_idx = int(np.argmax(in_band))
        else:
            local_idx = int(np.argmax(dist))
        idx = int(idx_start + local_idx)
        return self.waypoints[idx], idx, self.xy[idx].copy()

    def goal_at_arc_distance(
        self, idx_now: int, dist_m: float = 16.0
    ) -> Tuple[carla.Waypoint, int]:
        """从 ``idx_now`` 向前沿路径累计弧长，返回第一个累计距离 ``>= dist_m`` 的航点。

        若路径剩余长度 < ``dist_m``，则返回路径末端航点（idx = N-1）。
        """
        if len(self) == 0:
            raise RuntimeError("DenseRoute 还未 compute()")
        idx_now = max(0, min(idx_now, len(self) - 1))
        base_arc = float(self.cum_arc[idx_now])
        target_arc = base_arc + float(dist_m)
        # 单调递增 → searchsorted
        idx = int(np.searchsorted(self.cum_arc, target_arc, side="left"))
        idx = min(idx, len(self) - 1)
        return self.waypoints[idx], idx

    def waypoints_ahead(
        self, idx_now: int, max_distance: float = 20.0
    ) -> List[carla.Waypoint]:
        """从 ``idx_now`` 向前取累计弧长 ``<= max_distance`` 的所有航点，用于可视化。"""
        if len(self) == 0:
            return []
        idx_now = max(0, min(idx_now, len(self) - 1))
        base_arc = float(self.cum_arc[idx_now])
        max_arc = base_arc + float(max_distance)
        idx_end = int(np.searchsorted(self.cum_arc, max_arc, side="right"))
        idx_end = min(idx_end, len(self))
        return self.waypoints[idx_now:idx_end]

    def is_done(self, idx_now: int, threshold: float = 3.0) -> bool:
        """ego 离路径终点的剩余弧长 < ``threshold`` 视为完成。"""
        if len(self) == 0:
            return True
        idx_now = max(0, min(idx_now, len(self) - 1))
        remaining = float(self.cum_arc[-1] - self.cum_arc[idx_now])
        return remaining < float(threshold)
