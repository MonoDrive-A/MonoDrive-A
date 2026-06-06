"""B2D 场景预处理与逐场景 H5 写入。"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import gc
import gzip
import hashlib
import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


AGENT_CLASS_TO_ID = {
    "car": 0,
    "bicycle": 1,
    "motorcycle": 2,
    "pedestrian": 3,
}
MAP_CLASS_TO_ID = {
    "lane_divider": 0,
    "road_edge": 1,
    "crosswalk": 2,
    "centerline": 3,
}
MAP_TYPE_TO_CLASS_ID = {
    "solid": MAP_CLASS_TO_ID["lane_divider"],
    "brokensolid": MAP_CLASS_TO_ID["lane_divider"],
    "solidbroken": MAP_CLASS_TO_ID["lane_divider"],
    "brokenbroken": MAP_CLASS_TO_ID["lane_divider"],
    "solidsolid": MAP_CLASS_TO_ID["lane_divider"],
    "broken": MAP_CLASS_TO_ID["lane_divider"],
    "none": MAP_CLASS_TO_ID["road_edge"],
    "center": MAP_CLASS_TO_ID["centerline"],
    "crosswalk": MAP_CLASS_TO_ID["crosswalk"],
}
LOGGER = logging.getLogger(__name__)
TRAFFIC_LIGHT_NONE_CLASS = 3
STOP_SIGN_NONE_CLASS = 0
STOP_SIGN_PRESENT_CLASS = 1
# Agent 标签顺序: [x, y, l, w, h, yaw, vx, vy, ax, ay]。
AGENT_STATE_DIM = 10
MAP_POINT_DIM = 2


@dataclass(frozen=True)
class ScenePaths:
    """一个 B2D 场景的实际数据目录。

    Args:
        name: 场景名，通常是场景目录名。
        root: 包含 `anno/` 和 `camera/` 的真实场景目录。
        annotation_dir: gzip JSON 标注目录。
        rgb_front_dir: 前视 RGB 图像目录。
        depth_front_dir: 前视深度图目录；缺失时为 None。
    """

    name: str
    root: Path
    annotation_dir: Path
    rgb_front_dir: Path
    depth_front_dir: Path | None = None


@dataclass(frozen=True)
class SampleWindow:
    """一个训练样本对应的历史帧和未来监督帧。"""

    current_frame_id: int
    input_frame_ids: tuple[int, ...]
    future_frame_ids: tuple[int, ...]


@dataclass(frozen=True)
class HDMapElement:
    """HD Map 中一条可用于监督的矢量元素。"""

    class_id: int
    world_xyz: np.ndarray


@dataclass(frozen=True)
class HDMapCandidate:
    """已扫描到的 HD Map 文件及其 Town 名。"""

    path: Path
    town_name: str
    town_base_name: str


@dataclass(frozen=True)
class SceneMapBounds:
    """场景级 HD Map 粗裁剪范围。"""

    minimum_xy: np.ndarray
    maximum_xy: np.ndarray


@dataclass
class B2DPreprocessConfig:
    """B2D 预处理配置。

    默认设置与 `doc/Model.md` 对齐：原始数据 10Hz，模型输入 5Hz 最近 8 帧，
    未来规划标签为 3 秒、2Hz、6 个点。
    """

    raw_dataset_root: Path | str = Path("datasets")
    output_dir: Path | str = Path("data/preprocessed")
    hd_map_root: Path | str = Path("datasets/hd_map")
    map_cache_dir: Path | str | None = None
    camera_name: str = "rgb_front"
    camera_sensor_name: str = "CAM_FRONT"
    image_size: tuple[int, int] = (288, 512)
    raw_fps: int = 10
    model_fps: int = 5
    input_frame_count: int = 8
    trajectory_fps: int = 2
    future_seconds: float = 3.0
    window_stride: int = 1
    max_agents: int = 194
    max_map_elements: int = 60
    map_point_count: int = 100
    detection_forward_range: float = 32.0
    detection_lateral_range: float = 32.0
    min_visible_agent_vertices: int = 2
    min_visible_agent_history_frames: int = 2
    map_min_visible_points: int = 2
    hd_map_min_point_spacing: float = 0.5
    target_min_distance: float = 24.0
    target_max_distance: float = 30.0
    max_target_points: int = 32
    target_search_seconds: float | None = None
    smooth_future_trajectory: bool = False
    trajectory_smoothing_iterations: int = 1
    compression: str | None = "gzip"
    compression_level: int = 4

    def __post_init__(self) -> None:
        self.raw_dataset_root = Path(self.raw_dataset_root)
        self.output_dir = Path(self.output_dir)
        self.hd_map_root = Path(self.hd_map_root)
        self.map_cache_dir = (
            self.output_dir / "map_cache"
            if self.map_cache_dir is None
            else Path(self.map_cache_dir)
        )
        if len(self.image_size) != 2:
            raise ValueError(f"image_size 期望包含 2 个整数，实际为 {self.image_size!r}。")
        image_height, image_width = self.image_size
        if image_height <= 0 or image_width <= 0:
            raise ValueError(f"image_size 期望为正数，实际为 {self.image_size!r}。")
        if self.raw_fps <= 0 or self.model_fps <= 0 or self.trajectory_fps <= 0:
            raise ValueError(
                "raw_fps、model_fps 和 trajectory_fps 都必须为正数，"
                f"实际为 raw_fps={self.raw_fps}, model_fps={self.model_fps}, "
                f"trajectory_fps={self.trajectory_fps}。"
            )
        if self.raw_fps % self.model_fps != 0:
            raise ValueError(
                f"raw_fps 必须能整除 model_fps，实际为 {self.raw_fps}/{self.model_fps}。"
            )
        if self.raw_fps % self.trajectory_fps != 0:
            raise ValueError(
                "raw_fps 必须能整除 trajectory_fps，"
                f"实际为 {self.raw_fps}/{self.trajectory_fps}。"
            )
        if self.input_frame_count <= 0:
            raise ValueError(f"input_frame_count 必须为正数，实际为 {self.input_frame_count}。")
        if self.future_points <= 0:
            raise ValueError(f"future_seconds 必须产生至少 1 个未来点，实际为 {self.future_seconds}。")
        if not math.isclose(self.future_points / self.trajectory_fps, self.future_seconds):
            raise ValueError(
                "future_seconds * trajectory_fps 必须是整数，"
                f"实际为 {self.future_seconds} * {self.trajectory_fps}。"
            )
        if self.window_stride <= 0:
            raise ValueError(f"window_stride 必须为正数，实际为 {self.window_stride}。")
        if self.max_agents <= 0:
            raise ValueError(f"max_agents 必须为正数，实际为 {self.max_agents}。")
        if self.max_map_elements <= 0:
            raise ValueError(f"max_map_elements 必须为正数，实际为 {self.max_map_elements}。")
        if self.map_point_count <= 1:
            raise ValueError(f"map_point_count 必须大于 1，实际为 {self.map_point_count}。")
        if self.detection_forward_range <= 0 or self.detection_lateral_range <= 0:
            raise ValueError(
                "detection_forward_range 和 detection_lateral_range 必须为正数，"
                f"实际为 {self.detection_forward_range}/{self.detection_lateral_range}。"
            )
        if self.min_visible_agent_vertices <= 0:
            raise ValueError(
                f"min_visible_agent_vertices 必须为正数，实际为 {self.min_visible_agent_vertices}。"
            )
        if (
            self.min_visible_agent_history_frames <= 0
            or self.min_visible_agent_history_frames > self.input_frame_count
        ):
            raise ValueError(
                "min_visible_agent_history_frames 必须在 [1, input_frame_count] 范围内，"
                f"实际为 {self.min_visible_agent_history_frames}/{self.input_frame_count}。"
            )
        if self.map_min_visible_points <= 0:
            raise ValueError(f"map_min_visible_points 必须为正数，实际为 {self.map_min_visible_points}。")
        if self.hd_map_min_point_spacing <= 0:
            raise ValueError(
                f"hd_map_min_point_spacing 必须为正数，实际为 {self.hd_map_min_point_spacing}。"
            )
        if self.target_min_distance < 0 or self.target_max_distance <= 0:
            raise ValueError(
                "target_min_distance 必须非负且 target_max_distance 必须为正数，"
                f"实际为 {self.target_min_distance}/{self.target_max_distance}。"
            )
        if self.target_min_distance > self.target_max_distance:
            raise ValueError(
                "target_min_distance 必须小于或等于 target_max_distance，"
                f"实际为 {self.target_min_distance} > {self.target_max_distance}。"
            )
        if self.max_target_points <= 0:
            raise ValueError(f"max_target_points 必须为正数，实际为 {self.max_target_points}。")
        if self.target_search_seconds is not None and self.target_search_seconds <= 0:
            raise ValueError(
                "target_search_seconds 必须为正数或 None，"
                f"实际为 {self.target_search_seconds}。"
            )
        if self.trajectory_smoothing_iterations < 0:
            raise ValueError(
                "trajectory_smoothing_iterations 必须为非负整数，"
                f"实际为 {self.trajectory_smoothing_iterations}。"
            )
        if self.compression not in {None, "gzip", "lzf"}:
            raise ValueError(f"compression 仅支持 None、'gzip' 或 'lzf'，实际为 {self.compression!r}。")
        if not (0 <= self.compression_level <= 9):
            raise ValueError(f"compression_level 必须在 [0, 9]，实际为 {self.compression_level}。")

    @property
    def raw_to_model_stride(self) -> int:
        """10Hz 原始帧到 5Hz 模型输入帧的下采样间隔。"""

        return self.raw_fps // self.model_fps

    @property
    def trajectory_stride(self) -> int:
        """10Hz 原始帧到 2Hz 轨迹标签帧的采样间隔。"""

        return self.raw_fps // self.trajectory_fps

    @property
    def window_stride_raw(self) -> int:
        """滑窗步长换算到原始 10Hz 帧后的间隔。"""

        return self.window_stride * self.raw_to_model_stride

    @property
    def future_points(self) -> int:
        """未来规划标签点数。"""

        return int(round(self.future_seconds * self.trajectory_fps))


def discover_b2d_scenes(raw_root: Path | str, camera_name: str = "rgb_front") -> list[ScenePaths]:
    """发现包含 `anno/` 与指定前视相机目录的 B2D 场景。

    该函数从 `raw_root` 递归查找 `anno` 目录，因此兼容：

    - `datasets/SceneName/anno`
    - `datasets/SceneName/SceneName/anno`

    Args:
        raw_root: 原始 B2D 数据根目录。
        camera_name: `camera/` 下的相机目录名。

    Returns:
        按路径排序后的场景列表。
    """

    root = Path(raw_root)
    if not root.exists():
        raise FileNotFoundError(f"raw_root 不存在：{root}")
    if not root.is_dir():
        raise NotADirectoryError(f"raw_root 必须是目录：{root}")

    scenes: list[ScenePaths] = []
    for annotation_dir in sorted(root.rglob("anno")):
        if not annotation_dir.is_dir():
            continue
        scene_root = annotation_dir.parent
        rgb_front_dir = scene_root / "camera" / camera_name
        if not rgb_front_dir.is_dir():
            continue
        depth_front_dir = scene_root / "camera" / _depth_camera_name(camera_name)
        scenes.append(
            ScenePaths(
                name=scene_root.name,
                root=scene_root,
                annotation_dir=annotation_dir,
                rgb_front_dir=rgb_front_dir,
                depth_front_dir=depth_front_dir if depth_front_dir.is_dir() else None,
            )
        )
    return scenes


def build_sample_windows(frame_ids: list[int] | tuple[int, ...], config: B2DPreprocessConfig) -> list[SampleWindow]:
    """根据原始帧号构造 5Hz 输入滑窗和 2Hz 未来轨迹标签索引。

    Shape:
        输入历史帧号: `[T]`，其中 `T=config.input_frame_count`。
        未来帧号: `[K]`，其中 `K=config.future_points`。
    """

    ordered_frame_ids = sorted({int(frame_id) for frame_id in frame_ids})
    if not ordered_frame_ids:
        return []

    available_frame_ids = set(ordered_frame_ids)
    min_frame_id = ordered_frame_ids[0]
    max_frame_id = ordered_frame_ids[-1]
    first_current_id = min_frame_id + (config.input_frame_count - 1) * config.raw_to_model_stride
    last_current_id = max_frame_id - config.future_points * config.trajectory_stride

    windows: list[SampleWindow] = []
    for current_frame_id in ordered_frame_ids:
        if current_frame_id < first_current_id or current_frame_id > last_current_id:
            continue
        if (current_frame_id - first_current_id) % config.window_stride_raw != 0:
            continue

        input_frame_ids = tuple(
            current_frame_id - offset * config.raw_to_model_stride
            for offset in range(config.input_frame_count - 1, -1, -1)
        )
        future_frame_ids = tuple(
            current_frame_id + (future_index + 1) * config.trajectory_stride
            for future_index in range(config.future_points)
        )
        required_frame_ids = (*input_frame_ids, *future_frame_ids)
        if all(frame_id in available_frame_ids for frame_id in required_frame_ids):
            windows.append(
                SampleWindow(
                    current_frame_id=current_frame_id,
                    input_frame_ids=input_frame_ids,
                    future_frame_ids=future_frame_ids,
                )
            )
    return windows


def preprocess_b2d_dataset(config: B2DPreprocessConfig, overwrite: bool = False) -> list[Path]:
    """批量预处理 `config.raw_dataset_root` 下的所有 B2D 场景。"""

    LOGGER.info(
        "开始 B2D 预处理：raw_root=%s, output_dir=%s, hd_map_root=%s, map_cache_dir=%s",
        config.raw_dataset_root,
        config.output_dir,
        config.hd_map_root,
        config.map_cache_dir,
    )
    scenes = discover_b2d_scenes(config.raw_dataset_root, camera_name=config.camera_name)
    if not scenes:
        raise FileNotFoundError(
            f"未在 {config.raw_dataset_root} 下发现包含 anno/ 与 camera/{config.camera_name}/ 的场景。"
        )
    LOGGER.info("发现 %d 个 B2D 场景。", len(scenes))

    preprocessor = B2DScenePreprocessor(config)
    output_paths: list[Path] = []
    used_names: set[str] = set()
    dataset_start_time = time.perf_counter()
    for scene_index, scene in enumerate(scenes, start=1):
        output_name = _scene_output_name(scene, config.raw_dataset_root, used_names)
        output_path = config.output_dir / output_name
        LOGGER.info(
            "处理场景 %d/%d：%s -> %s",
            scene_index,
            len(scenes),
            scene.root,
            output_path,
        )
        output_paths.append(preprocessor.preprocess_scene(scene, output_path, overwrite=overwrite))
    LOGGER.info(
        "B2D 预处理完成：%d 个 H5，耗时 %.2fs。",
        len(output_paths),
        time.perf_counter() - dataset_start_time,
    )
    return output_paths


class B2DScenePreprocessor:
    """将单个 B2D 场景预处理为逐场景 H5 文件。"""

    def __init__(self, config: B2DPreprocessConfig) -> None:
        self.config = config

    def build_scene_arrays(self, scene: ScenePaths) -> dict[str, Any]:
        """构造除图像像素外的全部预处理数组，便于单元测试和 H5 写入复用。"""

        build_start_time = time.perf_counter()
        LOGGER.info("构造场景数组：%s", scene.root)
        frame_files = _collect_frame_files(scene)
        windows = build_sample_windows(sorted(frame_files), self.config)
        if not windows:
            raise ValueError(
                f"场景 {scene.root} 没有足够帧构造样本：需要 {self.config.input_frame_count} 个历史输入帧和 "
                f"{self.config.future_points} 个未来轨迹点。"
            )
        LOGGER.info(
            "场景帧统计：frames=%d, samples=%d, input_frames=%d, future_points=%d。",
            len(frame_files),
            len(windows),
            self.config.input_frame_count,
            self.config.future_points,
        )

        annotations = {
            frame_id: _read_annotation(paths["annotation"])
            for frame_id, paths in frame_files.items()
        }
        agent_box_index = _build_agent_box_index(annotations)
        scene_map_bounds = _build_scene_map_bounds(annotations, self.config)
        LOGGER.debug(
            "场景 Map bbox：min=(%.3f, %.3f), max=(%.3f, %.3f)。",
            float(scene_map_bounds.minimum_xy[0]),
            float(scene_map_bounds.minimum_xy[1]),
            float(scene_map_bounds.maximum_xy[0]),
            float(scene_map_bounds.maximum_xy[1]),
        )
        hd_map_elements = self._load_scene_hd_map_elements(scene, scene_map_bounds)
        image_frame_ids = sorted({frame_id for window in windows for frame_id in window.input_frame_ids})
        image_frame_index = {frame_id: index for index, frame_id in enumerate(image_frame_ids)}
        LOGGER.info(
            "场景数组准备：annotations=%d, hd_map_elements=%d, unique_image_frames=%d。",
            len(annotations),
            len(hd_map_elements),
            len(image_frame_ids),
        )

        sample_count = len(windows)
        input_frame_indices = np.empty(
            (sample_count, self.config.input_frame_count),
            dtype=np.int32,
        )
        input_frame_ids = np.empty_like(input_frame_indices)
        future_frame_ids = np.empty((sample_count, self.config.future_points), dtype=np.int32)
        current_frame_ids = np.empty((sample_count,), dtype=np.int32)
        current_pose = np.empty((sample_count, 3), dtype=np.float32)
        ego_motion = np.empty((sample_count, 3), dtype=np.float32)
        target_point = np.empty((sample_count, 2), dtype=np.float32)
        target_points = np.zeros((sample_count, self.config.max_target_points, 2), dtype=np.float32)
        target_valid = np.zeros((sample_count, self.config.max_target_points), dtype=np.bool_)
        commands = np.empty((sample_count, 3), dtype=np.int16)
        control = np.empty((sample_count, 3), dtype=np.float32)
        future_trajectory = np.empty((sample_count, self.config.future_points, 2), dtype=np.float32)
        agent_boxes = np.zeros((sample_count, self.config.max_agents, AGENT_STATE_DIM), dtype=np.float32)
        agent_classes = np.full((sample_count, self.config.max_agents), -1, dtype=np.int16)
        agent_valid = np.zeros((sample_count, self.config.max_agents), dtype=np.bool_)
        agent_future_trajectory = np.zeros(
            (sample_count, self.config.max_agents, self.config.future_points, 2),
            dtype=np.float32,
        )
        agent_future_valid = np.zeros(
            (sample_count, self.config.max_agents, self.config.future_points),
            dtype=np.bool_,
        )
        map_points = np.zeros(
            (sample_count, self.config.max_map_elements, self.config.map_point_count, MAP_POINT_DIM),
            dtype=np.float32,
        )
        map_classes = np.full((sample_count, self.config.max_map_elements), -1, dtype=np.int16)
        map_valid = np.zeros((sample_count, self.config.max_map_elements), dtype=np.bool_)
        traffic_light_state = np.full((sample_count,), TRAFFIC_LIGHT_NONE_CLASS, dtype=np.int16)
        traffic_light_xy = np.zeros((sample_count, 2), dtype=np.float32)
        traffic_light_valid = np.zeros((sample_count,), dtype=np.bool_)
        stop_sign_state = np.full((sample_count,), STOP_SIGN_NONE_CLASS, dtype=np.int16)
        stop_sign_xy = np.zeros((sample_count, 2), dtype=np.float32)
        stop_sign_valid = np.zeros((sample_count,), dtype=np.bool_)
        depth_image_cache: OrderedDict[int, np.ndarray | None] = OrderedDict()
        max_cached_depth_images = max(self.config.input_frame_count * 2, 1)

        for sample_index, window in enumerate(windows):
            current_annotation = annotations[window.current_frame_id]
            current_frame_ids[sample_index] = window.current_frame_id
            input_frame_ids[sample_index] = np.asarray(window.input_frame_ids, dtype=np.int32)
            input_frame_indices[sample_index] = np.asarray(
                [image_frame_index[frame_id] for frame_id in window.input_frame_ids],
                dtype=np.int32,
            )
            future_frame_ids[sample_index] = np.asarray(window.future_frame_ids, dtype=np.int32)
            current_pose[sample_index] = np.asarray(
                [
                    _safe_float(current_annotation, "x"),
                    _safe_float(current_annotation, "y"),
                    _safe_float(current_annotation, "theta"),
                ],
                dtype=np.float32,
            )
            ego_motion[sample_index] = _compute_ego_motion(
                window.current_frame_id,
                annotations,
                self.config.raw_fps,
            )
            future_raw_annotations = _collect_future_raw_annotations(
                window.current_frame_id,
                annotations,
                max_future_frame_offset=(
                    None
                    if self.config.target_search_seconds is None
                    else int(round(self.config.target_search_seconds * self.config.raw_fps))
                ),
            )
            (
                target_point[sample_index],
                target_points[sample_index],
                target_valid[sample_index],
            ) = _build_target_candidates(
                current_annotation,
                future_raw_annotations,
                target_min_distance=self.config.target_min_distance,
                target_max_distance=self.config.target_max_distance,
                max_target_points=self.config.max_target_points,
                smooth=self.config.smooth_future_trajectory,
                smoothing_iterations=self.config.trajectory_smoothing_iterations,
            )
            commands[sample_index] = np.asarray(
                [
                    int(current_annotation.get("command_near", -1)),
                    int(current_annotation.get("command_far", -1)),
                    int(current_annotation.get("next_command", -1)),
                ],
                dtype=np.int16,
            )
            control[sample_index] = np.asarray(
                [
                    _safe_float(current_annotation, "throttle"),
                    _safe_float(current_annotation, "steer"),
                    _safe_float(current_annotation, "brake"),
                ],
                dtype=np.float32,
            )
            future_trajectory[sample_index] = _build_future_trajectory(
                current_annotation,
                [annotations[frame_id] for frame_id in window.future_frame_ids],
                smooth=self.config.smooth_future_trajectory,
                smoothing_iterations=self.config.trajectory_smoothing_iterations,
            )
            history_depth_images = {
                frame_id: _load_depth_image_cached(
                    frame_id,
                    frame_files,
                    depth_image_cache,
                    max_cached_depth_images,
                )
                for frame_id in window.input_frame_ids
            }
            (
                agent_boxes[sample_index],
                agent_classes[sample_index],
                agent_valid[sample_index],
                agent_future_trajectory[sample_index],
                agent_future_valid[sample_index],
            ) = _build_agent_labels(
                window.current_frame_id,
                window.input_frame_ids,
                window.future_frame_ids,
                annotations,
                agent_box_index,
                self.config.raw_fps,
                self.config.max_agents,
                self.config,
                history_depth_images,
            )
            (
                map_points[sample_index],
                map_classes[sample_index],
                map_valid[sample_index],
            ) = _build_map_labels(
                current_annotation,
                hd_map_elements,
                self.config,
            )
            (
                traffic_light_state[sample_index],
                traffic_light_xy[sample_index],
                traffic_light_valid[sample_index],
            ) = _build_traffic_light_label(current_annotation)
            (
                stop_sign_state[sample_index],
                stop_sign_xy[sample_index],
                stop_sign_valid[sample_index],
            ) = _build_stop_sign_label(current_annotation)
            if LOGGER.isEnabledFor(logging.DEBUG) and (
                sample_index + 1 == sample_count or (sample_index + 1) % 100 == 0
            ):
                LOGGER.debug(
                    "样本构造进度：scene=%s, %d/%d。",
                    scene.name,
                    sample_index + 1,
                    sample_count,
                )

        LOGGER.info(
            "场景数组完成：samples=%d, valid_agents=%d, valid_agent_future_points=%d, valid_map_elements=%d, 耗时 %.2fs。",
            sample_count,
            int(agent_valid.sum()),
            int(agent_future_valid.sum()),
            int(map_valid.sum()),
            time.perf_counter() - build_start_time,
        )
        return {
            "scene_name": scene.name,
            "scene_root": str(scene.root),
            "image_frame_ids": np.asarray(image_frame_ids, dtype=np.int32),
            "image_paths": [
                str(frame_files[frame_id]["rgb_front"].relative_to(scene.root))
                for frame_id in image_frame_ids
            ],
            "current_frame_ids": current_frame_ids,
            "input_frame_indices": input_frame_indices,
            "input_frame_ids": input_frame_ids,
            "future_frame_ids": future_frame_ids,
            "current_pose": current_pose,
            "ego_motion": ego_motion,
            "target_point": target_point,
            "target_points": target_points,
            "target_valid": target_valid,
            "commands": commands,
            "control": control,
            "future_trajectory": future_trajectory,
            "agent_boxes": agent_boxes,
            "agent_classes": agent_classes,
            "agent_valid": agent_valid,
            "agent_future_trajectory": agent_future_trajectory,
            "agent_future_valid": agent_future_valid,
            "map_points": map_points,
            "map_classes": map_classes,
            "map_valid": map_valid,
            "traffic_light_state": traffic_light_state,
            "traffic_light_xy": traffic_light_xy,
            "traffic_light_valid": traffic_light_valid,
            "stop_sign_state": stop_sign_state,
            "stop_sign_xy": stop_sign_xy,
            "stop_sign_valid": stop_sign_valid,
            "frame_files": frame_files,
        }

    def _load_scene_hd_map_elements(
        self,
        scene: ScenePaths,
        scene_map_bounds: SceneMapBounds,
    ) -> list[HDMapElement]:
        map_path = _resolve_scene_hd_map_path(scene, self.config.hd_map_root)
        if map_path is None:
            LOGGER.warning("场景未匹配到 HD Map：scene=%s, hd_map_root=%s。", scene.root, self.config.hd_map_root)
            return []
        resolved_path = map_path.resolve()
        cache_path = _compact_map_cache_path(scene, resolved_path, self.config, scene_map_bounds)
        cached_elements = _try_read_compact_map_cache(cache_path, "scene-level")
        if cached_elements is not None:
            return cached_elements

        LOGGER.info("scene-level Map 缓存未命中：%s。", cache_path)
        town_elements = _load_town_hd_map_elements(resolved_path, self.config)
        elements = _filter_hd_map_elements_by_bounds(town_elements, scene_map_bounds)
        _write_compact_map_cache(cache_path, elements)
        LOGGER.info(
            "写入 scene-level Map 缓存：%s, source_elements=%d, scene_elements=%d。",
            cache_path,
            len(town_elements),
            len(elements),
        )
        return elements

    def preprocess_scene(self, scene: ScenePaths, output_path: Path | str, overwrite: bool = False) -> Path:
        """将单个场景写入 H5。

        H5 布局:
            `frames/rgb_front`: `[F, H, W, 3]`，uint8，5Hz 输入帧。
            `samples/input_frame_indices`: `[S, 8]`，索引到 `frames/rgb_front`。
            `labels/future_trajectory`: `[S, 6, 2]`，ego 坐标系米制轨迹。
        """

        h5py = _require_h5py()
        scene_start_time = time.perf_counter()
        output_path = Path(output_path)
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"H5 输出已存在，若要覆盖请设置 overwrite=True：{output_path}")

        arrays = self.build_scene_arrays(scene)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image_height, image_width = self.config.image_size
        compression_kwargs = _compression_kwargs(self.config)
        LOGGER.info(
            "开始写入 H5：%s, samples=%d, image_frames=%d。",
            output_path,
            int(arrays["current_frame_ids"].shape[0]),
            int(arrays["image_frame_ids"].shape[0]),
        )

        with h5py.File(output_path, "w") as h5_file:
            self._write_attrs(h5_file, scene, arrays)

            frames_group = h5_file.create_group("frames")
            frames_group.create_dataset("frame_ids", data=arrays["image_frame_ids"])
            string_dtype = h5py.string_dtype(encoding="utf-8")
            frames_group.create_dataset(
                "rgb_front_path",
                data=np.asarray(arrays["image_paths"], dtype=object),
                dtype=string_dtype,
            )
            rgb_dataset = frames_group.create_dataset(
                "rgb_front",
                shape=(len(arrays["image_frame_ids"]), image_height, image_width, 3),
                dtype=np.uint8,
                chunks=(1, image_height, image_width, 3),
                **compression_kwargs,
            )
            frame_files: dict[int, dict[str, Path]] = arrays["frame_files"]
            for image_index, frame_id in enumerate(arrays["image_frame_ids"]):
                rgb_dataset[image_index] = _load_resized_rgb(
                    frame_files[int(frame_id)]["rgb_front"],
                    self.config.image_size,
                )
                if LOGGER.isEnabledFor(logging.DEBUG) and (
                    image_index + 1 == len(arrays["image_frame_ids"]) or (image_index + 1) % 100 == 0
                ):
                    LOGGER.debug(
                        "RGB 写入进度：%s, %d/%d。",
                        scene.name,
                        image_index + 1,
                        len(arrays["image_frame_ids"]),
                    )

            samples_group = h5_file.create_group("samples")
            samples_group.create_dataset("current_frame_id", data=arrays["current_frame_ids"])
            samples_group.create_dataset("input_frame_indices", data=arrays["input_frame_indices"])
            samples_group.create_dataset("input_frame_ids", data=arrays["input_frame_ids"])
            samples_group.create_dataset("future_frame_ids", data=arrays["future_frame_ids"])

            labels_group = h5_file.create_group("labels")
            labels_group.create_dataset("current_pose", data=arrays["current_pose"])
            labels_group.create_dataset("ego_motion", data=arrays["ego_motion"])
            labels_group.create_dataset("target_point", data=arrays["target_point"])
            labels_group.create_dataset("target_points", data=arrays["target_points"])
            labels_group.create_dataset("target_valid", data=arrays["target_valid"])
            labels_group.create_dataset("commands", data=arrays["commands"])
            labels_group.create_dataset("control", data=arrays["control"])
            labels_group.create_dataset(
                "future_trajectory",
                data=arrays["future_trajectory"],
                chunks=(1, self.config.future_points, 2),
                **compression_kwargs,
            )
            labels_group.create_dataset(
                "agent_boxes",
                data=arrays["agent_boxes"],
                chunks=(1, self.config.max_agents, AGENT_STATE_DIM),
                **compression_kwargs,
            )
            labels_group.create_dataset("agent_classes", data=arrays["agent_classes"])
            labels_group.create_dataset("agent_valid", data=arrays["agent_valid"])
            labels_group.create_dataset(
                "agent_future_trajectory",
                data=arrays["agent_future_trajectory"],
                chunks=(1, self.config.max_agents, self.config.future_points, 2),
                **compression_kwargs,
            )
            labels_group.create_dataset("agent_future_valid", data=arrays["agent_future_valid"])
            labels_group.create_dataset(
                "map_points",
                data=arrays["map_points"],
                chunks=(1, self.config.max_map_elements, self.config.map_point_count, MAP_POINT_DIM),
                **compression_kwargs,
            )
            labels_group.create_dataset("map_classes", data=arrays["map_classes"])
            labels_group.create_dataset("map_valid", data=arrays["map_valid"])
            labels_group.create_dataset("traffic_light_state", data=arrays["traffic_light_state"])
            labels_group.create_dataset("traffic_light_xy", data=arrays["traffic_light_xy"])
            labels_group.create_dataset("traffic_light_valid", data=arrays["traffic_light_valid"])
            labels_group.create_dataset("stop_sign_state", data=arrays["stop_sign_state"])
            labels_group.create_dataset("stop_sign_xy", data=arrays["stop_sign_xy"])
            labels_group.create_dataset("stop_sign_valid", data=arrays["stop_sign_valid"])

        LOGGER.info("H5 写入完成：%s, 耗时 %.2fs。", output_path, time.perf_counter() - scene_start_time)
        return output_path

    def _write_attrs(self, h5_file: Any, scene: ScenePaths, arrays: dict[str, Any]) -> None:
        h5_file.attrs["schema_version"] = "b2d_h5_v5"
        h5_file.attrs["scene_name"] = scene.name
        h5_file.attrs["scene_root"] = str(scene.root)
        h5_file.attrs["sample_count"] = int(arrays["current_frame_ids"].shape[0])
        h5_file.attrs["frame_count"] = int(arrays["image_frame_ids"].shape[0])
        h5_file.attrs["raw_fps"] = self.config.raw_fps
        h5_file.attrs["model_fps"] = self.config.model_fps
        h5_file.attrs["trajectory_fps"] = self.config.trajectory_fps
        h5_file.attrs["future_seconds"] = self.config.future_seconds
        h5_file.attrs["future_points"] = self.config.future_points
        h5_file.attrs["input_frame_count"] = self.config.input_frame_count
        h5_file.attrs["max_agents"] = self.config.max_agents
        h5_file.attrs["max_map_elements"] = self.config.max_map_elements
        h5_file.attrs["map_point_count"] = self.config.map_point_count
        h5_file.attrs["raw_to_model_stride"] = self.config.raw_to_model_stride
        h5_file.attrs["trajectory_stride"] = self.config.trajectory_stride
        h5_file.attrs["window_stride"] = self.config.window_stride
        h5_file.attrs["window_stride_raw"] = self.config.window_stride_raw
        h5_file.attrs["image_height"] = self.config.image_size[0]
        h5_file.attrs["image_width"] = self.config.image_size[1]
        h5_file.attrs["coordinate_system"] = "ego: x forward, y left, unit meter"
        h5_file.attrs["camera_sensor_name"] = self.config.camera_sensor_name
        h5_file.attrs["detection_forward_range"] = self.config.detection_forward_range
        h5_file.attrs["detection_lateral_range"] = self.config.detection_lateral_range
        h5_file.attrs["min_visible_agent_vertices"] = self.config.min_visible_agent_vertices
        h5_file.attrs["min_visible_agent_history_frames"] = self.config.min_visible_agent_history_frames
        h5_file.attrs["map_min_visible_points"] = self.config.map_min_visible_points
        h5_file.attrs["hd_map_root"] = str(self.config.hd_map_root)
        h5_file.attrs["map_cache_dir"] = str(self.config.map_cache_dir)
        h5_file.attrs["hd_map_min_point_spacing"] = self.config.hd_map_min_point_spacing
        h5_file.attrs["target_min_distance"] = self.config.target_min_distance
        h5_file.attrs["target_max_distance"] = self.config.target_max_distance
        h5_file.attrs["max_target_points"] = self.config.max_target_points
        h5_file.attrs["target_search_seconds"] = (
            "all_future" if self.config.target_search_seconds is None else self.config.target_search_seconds
        )
        h5_file.attrs["smooth_future_trajectory"] = self.config.smooth_future_trajectory
        h5_file.attrs["trajectory_smoothing_iterations"] = self.config.trajectory_smoothing_iterations


def _collect_frame_files(scene: ScenePaths) -> dict[int, dict[str, Path]]:
    annotation_files = {
        _frame_id_from_path(path): path
        for path in sorted(scene.annotation_dir.glob("*.json.gz"))
    }
    rgb_files = {
        _frame_id_from_path(path): path
        for path in sorted(scene.rgb_front_dir.glob("*.jpg"))
    }
    depth_files = (
        {
            _frame_id_from_path(path): path
            for path in sorted(scene.depth_front_dir.glob("*.png"))
        }
        if scene.depth_front_dir is not None
        else {}
    )
    common_frame_ids = sorted(set(annotation_files) & set(rgb_files))
    if not common_frame_ids:
        raise FileNotFoundError(
            f"场景 {scene.root} 没有可配对的 annotation 与 {scene.rgb_front_dir.name} 图像。"
        )
    frame_files: dict[int, dict[str, Path]] = {}
    for frame_id in common_frame_ids:
        paths = {
            "annotation": annotation_files[frame_id],
            "rgb_front": rgb_files[frame_id],
        }
        if frame_id in depth_files:
            paths["depth_front"] = depth_files[frame_id]
        frame_files[frame_id] = paths
    return frame_files


def _depth_camera_name(camera_name: str) -> str:
    if camera_name.startswith("rgb_"):
        return "depth_" + camera_name[len("rgb_") :]
    return f"depth_{camera_name}"


def _frame_id_from_path(path: Path) -> int:
    try:
        return int(path.name.split(".")[0])
    except ValueError as exc:
        raise ValueError(f"帧文件名必须以整数帧号开头，实际为 {path.name!r}。") from exc


def _read_annotation(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as annotation_file:
        data = json.load(annotation_file)
    if not isinstance(data, dict):
        raise TypeError(f"annotation 必须是 JSON object，实际文件为 {path}。")
    return data


def _safe_float(annotation: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = annotation.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"annotation 字段 {key!r} 期望可转为 float，实际为 {value!r}。") from exc


def _collect_future_raw_annotations(
    current_frame_id: int,
    annotations: dict[int, dict[str, Any]],
    max_future_frame_offset: int | None,
) -> list[dict[str, Any]]:
    last_frame_id = max(annotations) if max_future_frame_offset is None else current_frame_id + max_future_frame_offset
    return [
        annotations[frame_id]
        for frame_id in range(current_frame_id + 1, last_frame_id + 1)
        if frame_id in annotations
    ]


def _build_target_candidates(
    current_annotation: dict[str, Any],
    future_annotations: list[dict[str, Any]],
    target_min_distance: float,
    target_max_distance: float,
    max_target_points: int,
    smooth: bool,
    smoothing_iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从未来真实轨迹中构造 24-30m 可达目标点候选。

    候选点来自当前帧之后的 10Hz ego 轨迹点，默认搜索到场景结束，也可用
    `target_search_seconds` 限制搜索时长。若没有点落入
    `[target_min_distance, target_max_distance]`，则使用搜索范围内最远点作为唯一候选。
    """

    future_points = np.asarray(
        [
            _world_xy_to_ego_xy(
                current_annotation,
                _safe_float(future_annotation, "x"),
                _safe_float(future_annotation, "y"),
            )
            for future_annotation in future_annotations
        ],
        dtype=np.float32,
    )
    if future_points.size == 0:
        target_points = np.zeros((max_target_points, 2), dtype=np.float32)
        target_valid = np.zeros((max_target_points,), dtype=np.bool_)
        target_valid[0] = True
        return target_points[0].copy(), target_points, target_valid

    if smooth and smoothing_iterations > 0:
        future_points = _smooth_future_trajectory(future_points, smoothing_iterations=smoothing_iterations)

    distances = np.linalg.norm(future_points, axis=1)
    candidate_mask = (distances >= target_min_distance) & (distances <= target_max_distance)
    candidate_points = future_points[candidate_mask]
    if candidate_points.shape[0] == 0:
        candidate_points = future_points[[int(np.argmax(distances))]]

    candidate_points = candidate_points[:max_target_points]
    target_points = np.zeros((max_target_points, 2), dtype=np.float32)
    target_valid = np.zeros((max_target_points,), dtype=np.bool_)
    target_points[: candidate_points.shape[0]] = candidate_points
    target_valid[: candidate_points.shape[0]] = True
    return target_points[0].copy(), target_points, target_valid


def _build_future_trajectory(
    current_annotation: dict[str, Any],
    future_annotations: list[dict[str, Any]],
    smooth: bool,
    smoothing_iterations: int,
) -> np.ndarray:
    trajectory = [
        _world_xy_to_ego_xy(
            current_annotation,
            _safe_float(future_annotation, "x"),
            _safe_float(future_annotation, "y"),
        )
        for future_annotation in future_annotations
    ]
    trajectory_array = np.asarray(trajectory, dtype=np.float32)
    if smooth and smoothing_iterations > 0:
        trajectory_array = _smooth_future_trajectory(
            trajectory_array,
            smoothing_iterations=smoothing_iterations,
        )
    return trajectory_array


def _smooth_future_trajectory(
    future_trajectory: np.ndarray,
    smoothing_iterations: int,
) -> np.ndarray:
    """对未来轨迹做轻量平滑，抑制采样抖动。

    平滑只作用于 ego 坐标系 XY 标签。当前 ego 原点作为锚点参与第一个
    未来点的平滑，最后一个输入端点保持不变，避免改变终点语义。
    """

    smoothed_trajectory = future_trajectory.astype(np.float32, copy=True)
    if smoothed_trajectory.shape[0] <= 1:
        return smoothed_trajectory

    for _ in range(smoothing_iterations):
        anchored_trajectory = np.concatenate(
            [
                np.zeros((1, 2), dtype=np.float32),
                smoothed_trajectory,
            ],
            axis=0,
        )
        next_trajectory = smoothed_trajectory.copy()
        # anchored_trajectory[0] 是当前 ego 原点；最后一个未来端点不平滑。
        for anchored_index in range(1, anchored_trajectory.shape[0] - 1):
            next_trajectory[anchored_index - 1] = (
                0.25 * anchored_trajectory[anchored_index - 1]
                + 0.5 * anchored_trajectory[anchored_index]
                + 0.25 * anchored_trajectory[anchored_index + 1]
            )
        smoothed_trajectory = next_trajectory
    return smoothed_trajectory


def _compute_ego_motion(
    current_frame_id: int,
    annotations: dict[int, dict[str, Any]],
    raw_fps: int,
) -> np.ndarray:
    """通过轨迹差分构造当前自车状态 `[Vx, Vy, W]`。

    速度单位为 m/s，角速度单位为 rad/s，坐标系为当前 ego 坐标系。
    """

    previous_id = current_frame_id - 1 if current_frame_id - 1 in annotations else current_frame_id
    next_id = current_frame_id + 1 if current_frame_id + 1 in annotations else current_frame_id
    if previous_id == next_id:
        return np.zeros((3,), dtype=np.float32)

    previous_annotation = annotations[previous_id]
    next_annotation = annotations[next_id]
    current_annotation = annotations[current_frame_id]
    dt = (next_id - previous_id) / raw_fps
    dx_dt = (_safe_float(next_annotation, "x") - _safe_float(previous_annotation, "x")) / dt
    dy_dt = (_safe_float(next_annotation, "y") - _safe_float(previous_annotation, "y")) / dt
    current_yaw = _b2d_theta_to_yaw(_safe_float(current_annotation, "theta"))
    velocity_x = dx_dt * math.cos(current_yaw) + dy_dt * math.sin(current_yaw)
    velocity_y = -dx_dt * math.sin(current_yaw) + dy_dt * math.cos(current_yaw)

    previous_yaw = _b2d_theta_to_yaw(_safe_float(previous_annotation, "theta"))
    next_yaw = _b2d_theta_to_yaw(_safe_float(next_annotation, "theta"))
    yaw_rate = _wrap_angle(next_yaw - previous_yaw) / dt
    return np.asarray([velocity_x, velocity_y, yaw_rate], dtype=np.float32)


def _build_agent_box_index(
    annotations: dict[int, dict[str, Any]],
) -> dict[int, dict[str, dict[str, Any]]]:
    """为每帧标注建立 Agent ID 索引，避免逐样本反复线性扫描框列表。"""

    box_index: dict[int, dict[str, dict[str, Any]]] = {}
    for frame_id, annotation in annotations.items():
        frame_index: dict[str, dict[str, Any]] = {}
        for box in annotation.get("bounding_boxes", []):
            box_id = str(box.get("id", ""))
            if box_id:
                frame_index[box_id] = box
        box_index[frame_id] = frame_index
    return box_index


def _build_agent_labels(
    current_frame_id: int,
    history_frame_ids: tuple[int, ...],
    future_frame_ids: tuple[int, ...],
    annotations: dict[int, dict[str, Any]],
    agent_box_index: dict[int, dict[str, dict[str, Any]]],
    raw_fps: int,
    max_agents: int,
    config: B2DPreprocessConfig,
    history_depth_images: dict[int, np.ndarray | None],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    current_annotation = annotations[current_frame_id]
    agent_boxes = np.zeros((max_agents, AGENT_STATE_DIM), dtype=np.float32)
    agent_classes = np.full((max_agents,), -1, dtype=np.int16)
    agent_valid = np.zeros((max_agents,), dtype=np.bool_)
    agent_future_trajectory = np.zeros((max_agents, len(future_frame_ids), 2), dtype=np.float32)
    agent_future_valid = np.zeros((max_agents, len(future_frame_ids)), dtype=np.bool_)

    agent_index = 0
    for box in current_annotation.get("bounding_boxes", []):
        class_id = _agent_class_id(box)
        if class_id is None:
            continue
        if agent_index >= max_agents:
            break

        center = box.get("center") or box.get("location") or [0.0, 0.0, 0.0]
        extent = box.get("extent") or [0.0, 0.0, 0.0]
        rotation = box.get("rotation") or [0.0, 0.0, 0.0]
        center_x, center_y = _world_xy_to_ego_xy(current_annotation, float(center[0]), float(center[1]))
        if not _is_xy_in_detection_range(center_x, center_y, config):
            continue
        agent_id = str(box.get("id", ""))
        if not _is_box_visible_in_history(
            agent_id,
            history_frame_ids,
            annotations,
            agent_box_index,
            config,
            history_depth_images,
        ):
            continue

        current_yaw = _b2d_theta_to_yaw(_safe_float(current_annotation, "theta"))
        box_yaw = math.radians(float(rotation[2]))
        relative_yaw = _wrap_angle(box_yaw - current_yaw)
        velocity_xy, acceleration_xy = _compute_agent_motion(
            agent_id,
            current_frame_id,
            agent_box_index,
            raw_fps,
            current_yaw,
        )

        agent_boxes[agent_index] = np.asarray(
            [
                center_x,
                center_y,
                float(extent[0]) * 2.0 if len(extent) > 0 else 0.0,
                float(extent[1]) * 2.0 if len(extent) > 1 else 0.0,
                float(extent[2]) * 2.0 if len(extent) > 2 else 0.0,
                relative_yaw,
                velocity_xy[0],
                velocity_xy[1],
                acceleration_xy[0],
                acceleration_xy[1],
            ],
            dtype=np.float32,
        )
        agent_classes[agent_index] = class_id
        agent_valid[agent_index] = True
        (
            agent_future_trajectory[agent_index],
            agent_future_valid[agent_index],
        ) = _build_agent_future_trajectory(
            current_annotation,
            agent_id,
            (center_x, center_y),
            future_frame_ids,
            agent_box_index,
        )
        agent_index += 1

    return agent_boxes, agent_classes, agent_valid, agent_future_trajectory, agent_future_valid


def _is_box_visible_in_history(
    agent_id: str,
    history_frame_ids: tuple[int, ...],
    annotations: dict[int, dict[str, Any]],
    agent_box_index: dict[int, dict[str, dict[str, Any]]],
    config: B2DPreprocessConfig,
    history_depth_images: dict[int, np.ndarray | None],
) -> bool:
    """同一 Agent 在历史输入窗口内至少 N 帧满足前视可见性要求。"""

    if not agent_id:
        return False

    visible_frame_count = 0
    for frame_id in history_frame_ids:
        annotation = annotations.get(frame_id)
        box = agent_box_index.get(frame_id, {}).get(agent_id)
        if annotation is None or box is None:
            continue
        if _is_box_visible_in_camera(annotation, box, config, history_depth_images.get(frame_id)):
            visible_frame_count += 1
            if visible_frame_count >= config.min_visible_agent_history_frames:
                return True
    return False


def _compute_agent_motion(
    agent_id: str,
    current_frame_id: int,
    agent_box_index: dict[int, dict[str, dict[str, Any]]],
    raw_fps: int,
    current_yaw: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """通过同一 Agent ID 的相邻帧轨迹差分构造 ego 坐标系速度和加速度。"""

    if not agent_id:
        return (0.0, 0.0), (0.0, 0.0)

    current_box = agent_box_index.get(current_frame_id, {}).get(agent_id)
    if current_box is None:
        return (0.0, 0.0), (0.0, 0.0)

    previous_frame_id = current_frame_id - 1
    next_frame_id = current_frame_id + 1
    previous_box = agent_box_index.get(previous_frame_id, {}).get(agent_id)
    next_box = agent_box_index.get(next_frame_id, {}).get(agent_id)

    current_xy = _box_world_xy(current_box)
    previous_xy = _box_world_xy(previous_box) if previous_box is not None else None
    next_xy = _box_world_xy(next_box) if next_box is not None else None

    if previous_xy is not None and next_xy is not None:
        velocity_world = (
            (next_xy[0] - previous_xy[0]) * raw_fps / 2.0,
            (next_xy[1] - previous_xy[1]) * raw_fps / 2.0,
        )
    elif next_xy is not None:
        velocity_world = (
            (next_xy[0] - current_xy[0]) * raw_fps,
            (next_xy[1] - current_xy[1]) * raw_fps,
        )
    elif previous_xy is not None:
        velocity_world = (
            (current_xy[0] - previous_xy[0]) * raw_fps,
            (current_xy[1] - previous_xy[1]) * raw_fps,
        )
    else:
        velocity_world = (0.0, 0.0)

    if previous_xy is not None and next_xy is not None:
        acceleration_world = (
            (next_xy[0] - 2.0 * current_xy[0] + previous_xy[0]) * raw_fps * raw_fps,
            (next_xy[1] - 2.0 * current_xy[1] + previous_xy[1]) * raw_fps * raw_fps,
        )
    else:
        acceleration_world = (0.0, 0.0)

    return (
        _world_vector_to_ego_xy(velocity_world[0], velocity_world[1], current_yaw),
        _world_vector_to_ego_xy(acceleration_world[0], acceleration_world[1], current_yaw),
    )


def _box_world_xy(box: dict[str, Any]) -> tuple[float, float]:
    center = box.get("center") or box.get("location") or [0.0, 0.0]
    return float(center[0]), float(center[1])


def _agent_class_id(box: dict[str, Any]) -> int | None:
    class_name = str(box.get("class", "")).lower()
    type_id = str(box.get("type_id", "")).lower()
    base_type = str(box.get("base_type", "")).lower()
    if class_name in {"ego_vehicle", "traffic_light", "traffic_sign"}:
        return None
    if class_name in {"walker", "pedestrian"} or "walker" in type_id or "pedestrian" in type_id:
        return AGENT_CLASS_TO_ID["pedestrian"]
    if class_name == "vehicle":
        if "bicycle" in type_id or base_type == "bicycle":
            return AGENT_CLASS_TO_ID["bicycle"]
        if "motorcycle" in type_id or base_type == "motorcycle":
            return AGENT_CLASS_TO_ID["motorcycle"]
        return AGENT_CLASS_TO_ID["car"]
    return None


def _is_xy_in_detection_range(ego_x: float, ego_y: float, config: B2DPreprocessConfig) -> bool:
    return (
        0.0 <= ego_x <= config.detection_forward_range
        and abs(ego_y) <= config.detection_lateral_range
    )


def _is_box_visible_in_camera(
    current_annotation: dict[str, Any],
    box: dict[str, Any],
    config: B2DPreprocessConfig,
    depth_image: np.ndarray | None,
) -> bool:
    """按前视相机投影过滤不可见 Agent。

    B2D 样例的深度图是 8-bit PNG，不能无损恢复 CARLA 真实米制深度。
    因此这里把深度图作为“该像素附近是否有有效表面”的保守辅助信号，
    同时使用 annotation 中的 `num_points` 剔除没有可见点的框。缺少 8 个
    `world_cord` 角点或相机标定时不默认放行。
    """

    world_corners = np.asarray(box.get("world_cord", []), dtype=np.float32)
    if world_corners.shape != (8, 3):
        return False

    projected = _project_world_points_to_front_image(current_annotation, world_corners, config.camera_sensor_name)
    if projected.size == 0:
        return False

    image_width, image_height = _front_image_size(current_annotation, config.camera_sensor_name)
    in_image_mask = (
        (projected[:, 2] > 0.1)
        & (projected[:, 0] >= 0.0)
        & (projected[:, 0] < image_width)
        & (projected[:, 1] >= 0.0)
        & (projected[:, 1] < image_height)
    )
    if int(in_image_mask.sum()) < config.min_visible_agent_vertices:
        return False

    if "num_points" in box and float(box.get("num_points", 0.0)) <= 0.0:
        return False

    if depth_image is None:
        return True

    visible_depth_vertices = 0
    for projected_point in projected[in_image_mask]:
        if _has_depth_support(depth_image, float(projected_point[0]), float(projected_point[1])):
            visible_depth_vertices += 1
    return visible_depth_vertices >= config.min_visible_agent_vertices


def _build_agent_future_trajectory(
    current_annotation: dict[str, Any],
    agent_id: str,
    current_agent_ego_xy: tuple[float, float],
    future_frame_ids: tuple[int, ...],
    agent_box_index: dict[int, dict[str, dict[str, Any]]],
) -> tuple[np.ndarray, np.ndarray]:
    """构造单个 Agent 的未来 3 秒 2Hz 位移监督。

    Shape:
        位移: `[K, 2]`，以当前 Agent 为原点，坐标轴沿当前 ego 坐标系，单位 meter。
        mask: `[K]`，同一 `id` 在对应未来帧存在时为 True。
    """

    trajectory = np.zeros((len(future_frame_ids), 2), dtype=np.float32)
    valid = np.zeros((len(future_frame_ids),), dtype=np.bool_)
    if not agent_id:
        return trajectory, valid

    for future_index, future_frame_id in enumerate(future_frame_ids):
        future_box = agent_box_index.get(future_frame_id, {}).get(agent_id)
        if future_box is None:
            continue
        future_xy = _box_world_xy(future_box)
        future_ego_xy = _world_xy_to_ego_xy(current_annotation, future_xy[0], future_xy[1])
        trajectory[future_index] = np.asarray(
            [
                future_ego_xy[0] - current_agent_ego_xy[0],
                future_ego_xy[1] - current_agent_ego_xy[1],
            ],
            dtype=np.float32,
        )
        valid[future_index] = True
    return trajectory, valid


def _project_world_points_to_front_image(
    current_annotation: dict[str, Any],
    world_xyz: np.ndarray,
    camera_sensor_name: str,
) -> np.ndarray:
    """使用 annotation 中的前视相机内外参把 world 点投影到原始图像平面。

    B2D 标注提供的 `world2cam` 把 world 坐标变换到 CARLA 相机坐标，
    其中 camera x 为前向、y 为右向、z 为上向。针孔模型使用
    `[right, down, forward]`，因此投影前转换为 `[camera_y, -camera_z, camera_x]`。
    """

    world_points = np.asarray(world_xyz, dtype=np.float64)
    if world_points.ndim != 2 or world_points.shape[1] != 3:
        return np.zeros((0, 3), dtype=np.float32)

    sensors = current_annotation.get("sensors", {})
    sensor = sensors.get(camera_sensor_name)
    if sensor is None:
        return np.zeros((0, 3), dtype=np.float32)

    intrinsic = np.asarray(sensor.get("intrinsic", []), dtype=np.float64)
    world_to_camera = np.asarray(sensor.get("world2cam", []), dtype=np.float64)
    if intrinsic.shape != (3, 3) or world_to_camera.shape != (4, 4):
        return np.zeros((0, 3), dtype=np.float32)

    homogeneous = np.concatenate(
        [world_points, np.ones((world_points.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    camera_xyz = (world_to_camera @ homogeneous.T).T[:, :3]
    # B2D/CARLA 相机坐标以 camera_x 为前向；针孔投影使用 [right, down, forward]。
    projection_xyz = np.column_stack([camera_xyz[:, 1], -camera_xyz[:, 2], camera_xyz[:, 0]])
    forward = projection_xyz[:, 2]
    valid_forward = forward > 1e-6
    projected = np.zeros((projection_xyz.shape[0], 3), dtype=np.float32)
    if not np.any(valid_forward):
        projected[:, 2] = forward.astype(np.float32)
        return projected

    pixel_h = (intrinsic @ projection_xyz[valid_forward].T).T
    pixel_xy = pixel_h[:, :2] / pixel_h[:, 2:3]
    projected[valid_forward, :2] = pixel_xy.astype(np.float32)
    projected[:, 2] = forward.astype(np.float32)
    return projected


def _front_image_size(current_annotation: dict[str, Any], camera_sensor_name: str) -> tuple[int, int]:
    sensor = current_annotation.get("sensors", {}).get(camera_sensor_name, {})
    return int(sensor.get("image_size_x", 1600)), int(sensor.get("image_size_y", 900))


def _has_depth_support(depth_image: np.ndarray, pixel_x: float, pixel_y: float) -> bool:
    image_height, image_width = depth_image.shape[:2]
    center_x = int(round(pixel_x))
    center_y = int(round(pixel_y))
    for offset_y in (0, 1):
        for offset_x in (0, 1):
            sample_x = min(max(center_x + offset_x, 0), image_width - 1)
            sample_y = min(max(center_y + offset_y, 0), image_height - 1)
            if int(depth_image[sample_y, sample_x]) < 255:
                return True
    return False


def _build_scene_map_bounds(
    annotations: dict[int, dict[str, Any]],
    config: B2DPreprocessConfig,
) -> SceneMapBounds:
    """用整段 ego 轨迹构造场景级 HD Map 粗裁剪范围。"""

    ego_xy = np.asarray(
        [
            [_safe_float(annotation, "x"), _safe_float(annotation, "y")]
            for annotation in annotations.values()
        ],
        dtype=np.float32,
    )
    margin = max(config.detection_forward_range, config.detection_lateral_range) + 80.0
    return SceneMapBounds(
        minimum_xy=(ego_xy.min(axis=0) - margin).astype(np.float32),
        maximum_xy=(ego_xy.max(axis=0) + margin).astype(np.float32),
    )


def _polyline_intersects_bounds(world_xyz: np.ndarray, scene_bounds: SceneMapBounds | None) -> bool:
    if scene_bounds is None:
        return True
    element_xy = world_xyz[:, :2]
    element_min = element_xy.min(axis=0)
    element_max = element_xy.max(axis=0)
    return not (
        element_max[0] < scene_bounds.minimum_xy[0]
        or element_min[0] > scene_bounds.maximum_xy[0]
        or element_max[1] < scene_bounds.minimum_xy[1]
        or element_min[1] > scene_bounds.maximum_xy[1]
    )


def _compact_map_cache_path(
    scene: ScenePaths,
    map_path: Path,
    config: B2DPreprocessConfig,
    scene_bounds: SceneMapBounds,
) -> Path:
    del scene_bounds
    safe_scene_name = re.sub(r"[^0-9A-Za-z_.-]+", "_", scene.name).strip("._") or "scene"
    return config.map_cache_dir / f"{safe_scene_name}_{map_path.stem}.npz"


def _compact_town_map_cache_path(map_path: Path, config: B2DPreprocessConfig) -> Path:
    safe_map_name = re.sub(r"[^0-9A-Za-z_.-]+", "_", map_path.stem).strip("._") or "hd_map"
    return config.map_cache_dir / f"town_{safe_map_name}.npz"


def _load_town_hd_map_elements(map_path: Path, config: B2DPreprocessConfig) -> list[HDMapElement]:
    """加载或构建按 HD Map 文件复用的 town-level 紧凑缓存。"""

    cache_path = _compact_town_map_cache_path(map_path, config)
    cached_elements = _try_read_compact_map_cache(cache_path, "town-level")
    if cached_elements is not None:
        return cached_elements

    LOGGER.info("town-level Map 缓存未命中，开始解包 HD Map：%s。", map_path)
    elements = _load_hd_map_elements(
        map_path,
        min_point_spacing=config.hd_map_min_point_spacing,
        scene_bounds=None,
    )
    _write_compact_map_cache(cache_path, elements)
    LOGGER.info("写入 town-level Map 缓存：%s, elements=%d。", cache_path, len(elements))
    return elements


def _filter_hd_map_elements_by_bounds(
    elements: list[HDMapElement],
    scene_bounds: SceneMapBounds,
) -> list[HDMapElement]:
    """从 town-level 缓存中筛出与当前场景 bbox 相交的 Map 元素。"""

    return [
        element
        for element in elements
        if _polyline_intersects_bounds(element.world_xyz, scene_bounds)
    ]


def _try_read_compact_map_cache(cache_path: Path, cache_level: str) -> list[HDMapElement] | None:
    """读取 canonical 缓存；若不存在，则按前缀兼容旧 hash 缓存。"""

    candidates = _compact_map_cache_read_candidates(cache_path)
    if not candidates:
        return None

    for candidate_path in candidates:
        try:
            elements = _read_compact_map_cache(candidate_path)
            if candidate_path == cache_path:
                LOGGER.info(
                    "命中 %s Map 缓存：%s, elements=%d。",
                    cache_level,
                    candidate_path,
                    len(elements),
                )
            else:
                LOGGER.info(
                    "命中旧版 hash %s Map 缓存：%s, canonical=%s, elements=%d。",
                    cache_level,
                    candidate_path,
                    cache_path,
                    len(elements),
                )
            return elements
        except (OSError, KeyError, ValueError):
            LOGGER.warning("%s Map 缓存无效，将跳过并删除：%s。", cache_level, candidate_path)
            candidate_path.unlink(missing_ok=True)
    return None


def _compact_map_cache_read_candidates(cache_path: Path) -> list[Path]:
    candidates: list[Path] = []
    if cache_path.is_file():
        candidates.append(cache_path)

    legacy_candidates = [
        candidate
        for candidate in cache_path.parent.glob(f"{cache_path.stem}_*.npz")
        if candidate.is_file()
    ]
    if len(legacy_candidates) > 1:
        LOGGER.warning(
            "发现多个旧版 hash Map 缓存，使用最新可读文件：prefix=%s, count=%d。",
            cache_path.stem,
            len(legacy_candidates),
        )
    legacy_candidates.sort(key=lambda candidate: candidate.stat().st_mtime_ns, reverse=True)
    candidates.extend(legacy_candidates)
    return candidates


def _write_compact_map_cache(cache_path: Path, elements: list[HDMapElement]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    class_ids = np.asarray([element.class_id for element in elements], dtype=np.int16)
    offsets = np.zeros((len(elements) + 1,), dtype=np.int64)
    point_arrays: list[np.ndarray] = []
    for element_index, element in enumerate(elements):
        world_xyz = element.world_xyz.astype(np.float32, copy=False)
        point_arrays.append(world_xyz)
        offsets[element_index + 1] = offsets[element_index] + world_xyz.shape[0]
    points = (
        np.concatenate(point_arrays, axis=0).astype(np.float32, copy=False)
        if point_arrays
        else np.zeros((0, 3), dtype=np.float32)
    )

    temporary_path = cache_path.with_name(cache_path.name + ".tmp")
    with temporary_path.open("wb") as cache_file:
        np.savez_compressed(
            cache_file,
            cache_format_version=np.asarray([1], dtype=np.int16),
            class_ids=class_ids,
            offsets=offsets,
            points=points,
        )
    temporary_path.replace(cache_path)


def _read_compact_map_cache(cache_path: Path) -> list[HDMapElement]:
    with np.load(cache_path) as cache_file:
        class_ids = np.asarray(cache_file["class_ids"], dtype=np.int16)
        offsets = np.asarray(cache_file["offsets"], dtype=np.int64)
        points = np.asarray(cache_file["points"], dtype=np.float32)

    if offsets.shape != (class_ids.shape[0] + 1,):
        raise ValueError(f"HD Map 缓存 offsets 形状异常：{cache_path}")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"HD Map 缓存 points 形状异常：{cache_path}")

    elements: list[HDMapElement] = []
    for element_index, class_id in enumerate(class_ids):
        start = int(offsets[element_index])
        end = int(offsets[element_index + 1])
        if end - start >= 2:
            elements.append(HDMapElement(class_id=int(class_id), world_xyz=points[start:end]))
    return elements


def _resolve_scene_hd_map_path(scene: ScenePaths, hd_map_root: Path) -> Path | None:
    map_candidates = _scan_hd_map_candidates(hd_map_root)
    if not map_candidates:
        return None

    town_names = _scene_town_name_candidates(scene)
    if not town_names:
        return None

    exact_candidates: dict[str, HDMapCandidate] = {}
    base_candidates: dict[str, list[HDMapCandidate]] = {}
    for map_candidate in map_candidates:
        exact_candidates.setdefault(map_candidate.town_name, map_candidate)
        base_candidates.setdefault(map_candidate.town_base_name, []).append(map_candidate)

    for town_name in town_names:
        map_candidate = exact_candidates.get(town_name)
        if map_candidate is not None:
            return map_candidate.path

    for town_name in town_names:
        base_name = _town_base_name(town_name)
        candidates = base_candidates.get(base_name, [])
        if candidates:
            return candidates[0].path
    return None


def _scan_hd_map_candidates(hd_map_root: Path) -> list[HDMapCandidate]:
    """先扫描 HD Map 目录，建立可按 Town 名复用的匹配索引。"""

    if not hd_map_root.is_dir():
        return []

    map_candidates: list[HDMapCandidate] = []
    for map_path in sorted(hd_map_root.glob("*_HD_map.npz")):
        town_name = _extract_town_name(map_path.stem)
        if town_name is None:
            continue
        map_candidates.append(
            HDMapCandidate(
                path=map_path,
                town_name=town_name,
                town_base_name=_town_base_name(town_name),
            )
        )
    return map_candidates


def _scene_town_name_candidates(scene: ScenePaths) -> list[str]:
    """从场景名和路径中提取候选 Town 名，兼容 `Town10HD`。"""

    candidates: dict[str, None] = {}
    for text in (scene.name, *scene.root.parts):
        town_name = _extract_town_name(text)
        if town_name is not None:
            candidates[town_name] = None
    return list(candidates)


def _extract_town_name(text: str) -> str | None:
    match = re.search(r"Town\d+(?:HD)?", text)
    return match.group(0) if match is not None else None


def _town_base_name(town_name: str) -> str:
    match = re.search(r"Town\d+", town_name)
    return match.group(0) if match is not None else town_name


def _load_hd_map_elements(
    map_path: Path,
    min_point_spacing: float,
    scene_bounds: SceneMapBounds | None = None,
) -> list[HDMapElement]:
    """加载 B2D HD Map，并压缩为 town-level 或 scene-level 矢量元素列表。

    B2D 的 `*_HD_map.npz` 是较大的 object array。该函数只在预处理端调用，
    当 `scene_bounds` 为 None 时生成按地图文件复用的整城稀疏元素；否则
    先用场景 bbox 粗裁剪，再做 polyline 稀疏化。
    """

    elements: list[HDMapElement] = []
    map_array: np.ndarray | None = None
    load_start_time = time.perf_counter()
    LOGGER.info("加载原始 HD Map：%s。", map_path)
    try:
        with np.load(map_path, allow_pickle=True) as map_file:
            map_array = map_file["arr"]
        for _road_id, lane_dict in map_array:
            if not isinstance(lane_dict, dict):
                continue
            for _lane_id, lane_items in lane_dict.items():
                if not isinstance(lane_items, list):
                    continue
                for lane_item in lane_items:
                    if not isinstance(lane_item, dict):
                        continue
                    class_id = _map_class_id(lane_item)
                    if class_id is None:
                        continue
                    world_xyz = _extract_map_points_xyz(lane_item.get("Points", []))
                    if world_xyz.shape[0] < 2:
                        continue
                    if not _polyline_intersects_bounds(world_xyz, scene_bounds):
                        continue
                    world_xyz = _thin_polyline_by_spacing(world_xyz, min_point_spacing)
                    if world_xyz.shape[0] >= 2:
                        elements.append(HDMapElement(class_id=class_id, world_xyz=world_xyz))
        LOGGER.info(
            "原始 HD Map 加载完成：%s, elements=%d, 耗时 %.2fs。",
            map_path,
            len(elements),
            time.perf_counter() - load_start_time,
        )
        return elements
    finally:
        del map_array
        gc.collect()


def _map_class_id(map_item: dict[str, Any]) -> int | None:
    map_type = str(map_item.get("Type", "")).replace(" ", "").lower()
    class_id = MAP_TYPE_TO_CLASS_ID.get(map_type)
    if class_id is None:
        return None
    return class_id


def _extract_map_points_xyz(points: Any) -> np.ndarray:
    extracted_points: list[tuple[float, float, float]] = []
    for point in points:
        coordinate = point[0] if _looks_like_coordinate_pair(point) else point
        if not isinstance(coordinate, (list, tuple)) or len(coordinate) < 2:
            continue
        z_value = float(coordinate[2]) if len(coordinate) > 2 else 0.0
        extracted_points.append((float(coordinate[0]), float(coordinate[1]), z_value))
    return np.asarray(extracted_points, dtype=np.float32)


def _looks_like_coordinate_pair(point: Any) -> bool:
    if not isinstance(point, (list, tuple)) or len(point) == 0:
        return False
    first_value = point[0]
    return isinstance(first_value, (list, tuple)) and len(first_value) >= 2


def _thin_polyline_by_spacing(world_xyz: np.ndarray, min_point_spacing: float) -> np.ndarray:
    kept_indices = [0]
    last_point = world_xyz[0, :2]
    for point_index in range(1, world_xyz.shape[0] - 1):
        if float(np.linalg.norm(world_xyz[point_index, :2] - last_point)) >= min_point_spacing:
            kept_indices.append(point_index)
            last_point = world_xyz[point_index, :2]
    if kept_indices[-1] != world_xyz.shape[0] - 1:
        kept_indices.append(world_xyz.shape[0] - 1)
    return world_xyz[np.asarray(kept_indices, dtype=np.int64)]


def _build_map_labels(
    current_annotation: dict[str, Any],
    hd_map_elements: list[HDMapElement],
    config: B2DPreprocessConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    map_points = np.zeros((config.max_map_elements, config.map_point_count, MAP_POINT_DIM), dtype=np.float32)
    map_classes = np.full((config.max_map_elements,), -1, dtype=np.int16)
    map_valid = np.zeros((config.max_map_elements,), dtype=np.bool_)
    if not hd_map_elements:
        return map_points, map_classes, map_valid

    candidates: list[tuple[float, int, np.ndarray]] = []
    for map_element in hd_map_elements:
        ego_xy = _world_xyz_to_ego_xy_array(current_annotation, map_element.world_xyz)
        in_range_mask = (
            (ego_xy[:, 0] >= 0.0)
            & (ego_xy[:, 0] <= config.detection_forward_range)
            & (np.abs(ego_xy[:, 1]) <= config.detection_lateral_range)
        )
        if int(in_range_mask.sum()) < 2:
            continue

        local_ego_xy = ego_xy[in_range_mask]
        local_world_xyz = map_element.world_xyz[in_range_mask]
        if not _is_map_polyline_visible(current_annotation, local_world_xyz, config):
            continue

        resampled_points = _resample_polyline_xy(local_ego_xy, config.map_point_count)
        min_distance = float(np.min(np.linalg.norm(local_ego_xy, axis=1)))
        candidates.append((min_distance, map_element.class_id, resampled_points))

    candidates.sort(key=lambda item: item[0])
    for map_index, (_distance, class_id, resampled_points) in enumerate(candidates[: config.max_map_elements]):
        map_points[map_index] = resampled_points
        map_classes[map_index] = np.int16(class_id)
        map_valid[map_index] = True
    return map_points, map_classes, map_valid


def _world_xyz_to_ego_xy_array(current_annotation: dict[str, Any], world_xyz: np.ndarray) -> np.ndarray:
    current_x = _safe_float(current_annotation, "x")
    current_y = _safe_float(current_annotation, "y")
    current_yaw = _b2d_theta_to_yaw(_safe_float(current_annotation, "theta"))
    delta_x = world_xyz[:, 0] - current_x
    delta_y = world_xyz[:, 1] - current_y
    ego_x = delta_x * math.cos(current_yaw) + delta_y * math.sin(current_yaw)
    ego_y = -delta_x * math.sin(current_yaw) + delta_y * math.cos(current_yaw)
    return np.column_stack([ego_x, ego_y]).astype(np.float32)


def _is_map_polyline_visible(
    current_annotation: dict[str, Any],
    world_xyz: np.ndarray,
    config: B2DPreprocessConfig,
) -> bool:
    projected = _project_world_points_to_front_image(current_annotation, world_xyz, config.camera_sensor_name)
    if projected.size == 0:
        return False
    image_width, image_height = _front_image_size(current_annotation, config.camera_sensor_name)
    in_image_mask = (
        (projected[:, 2] > 0.1)
        & (projected[:, 0] >= 0.0)
        & (projected[:, 0] < image_width)
        & (projected[:, 1] >= 0.0)
        & (projected[:, 1] < image_height)
    )
    return int(in_image_mask.sum()) >= config.map_min_visible_points


def _resample_polyline_xy(polyline_xy: np.ndarray, point_count: int) -> np.ndarray:
    if polyline_xy.shape[0] == 0:
        return np.zeros((point_count, 2), dtype=np.float32)
    if polyline_xy.shape[0] == 1:
        return np.repeat(polyline_xy.astype(np.float32), point_count, axis=0)

    deltas = np.diff(polyline_xy, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = float(cumulative[-1])
    if total_length <= 1e-6:
        return np.repeat(polyline_xy[:1].astype(np.float32), point_count, axis=0)

    target_distances = np.linspace(0.0, total_length, point_count, dtype=np.float32)
    resampled_x = np.interp(target_distances, cumulative, polyline_xy[:, 0])
    resampled_y = np.interp(target_distances, cumulative, polyline_xy[:, 1])
    return np.column_stack([resampled_x, resampled_y]).astype(np.float32)


def _build_traffic_light_label(current_annotation: dict[str, Any]) -> tuple[np.int16, np.ndarray, np.bool_]:
    traffic_lights = [
        box
        for box in current_annotation.get("bounding_boxes", [])
        if str(box.get("class", "")).lower() == "traffic_light" and bool(box.get("affects_ego", False))
    ]
    if not traffic_lights:
        return (
            np.int16(TRAFFIC_LIGHT_NONE_CLASS),
            np.zeros((2,), dtype=np.float32),
            np.bool_(False),
        )

    box = min(traffic_lights, key=lambda item: float(item.get("distance", float("inf"))))
    location = box.get("trigger_volume_location") or box.get("center") or box.get("location") or [0.0, 0.0]
    return (
        np.int16(int(box.get("state", TRAFFIC_LIGHT_NONE_CLASS))),
        np.asarray(_world_xy_to_ego_xy(current_annotation, float(location[0]), float(location[1])), dtype=np.float32),
        np.bool_(True),
    )


def _build_stop_sign_label(current_annotation: dict[str, Any]) -> tuple[np.int16, np.ndarray, np.bool_]:
    stop_signs = [
        box
        for box in current_annotation.get("bounding_boxes", [])
        if str(box.get("class", "")).lower() == "traffic_sign"
        and "stop" in str(box.get("type_id", "")).lower()
        and bool(box.get("affects_ego", False))
    ]
    if not stop_signs:
        return (
            np.int16(STOP_SIGN_NONE_CLASS),
            np.zeros((2,), dtype=np.float32),
            np.bool_(False),
        )

    box = min(stop_signs, key=lambda item: float(item.get("distance", float("inf"))))
    location = box.get("trigger_volume_location") or box.get("center") or box.get("location") or [0.0, 0.0]
    return (
        np.int16(STOP_SIGN_PRESENT_CLASS),
        np.asarray(_world_xy_to_ego_xy(current_annotation, float(location[0]), float(location[1])), dtype=np.float32),
        np.bool_(True),
    )


def _world_xy_to_ego_xy(
    current_annotation: dict[str, Any],
    target_x: float,
    target_y: float,
) -> tuple[float, float]:
    """将 B2D world 坐标转为当前 ego 坐标。

    B2D annotation 中 `theta` 与 CARLA/world yaw 的关系为 `yaw = theta - pi/2`。
    输出约定为 `x` 前向、`y` 左向，单位 meter。
    """

    current_x = _safe_float(current_annotation, "x")
    current_y = _safe_float(current_annotation, "y")
    current_yaw = _b2d_theta_to_yaw(_safe_float(current_annotation, "theta"))
    delta_x = target_x - current_x
    delta_y = target_y - current_y
    ego_x = delta_x * math.cos(current_yaw) + delta_y * math.sin(current_yaw)
    ego_y = -delta_x * math.sin(current_yaw) + delta_y * math.cos(current_yaw)
    return ego_x, ego_y


def _world_vector_to_ego_xy(vector_x: float, vector_y: float, current_yaw: float) -> tuple[float, float]:
    ego_x = vector_x * math.cos(current_yaw) + vector_y * math.sin(current_yaw)
    ego_y = -vector_x * math.sin(current_yaw) + vector_y * math.cos(current_yaw)
    return ego_x, ego_y


def _b2d_theta_to_yaw(theta: float) -> float:
    """将 B2D annotation `theta` 转为 world yaw。

    该关系由样例帧的 `CAM_FRONT.world2cam` 反推验证：前视相机的 world
    前向轴与 `theta - pi/2` 对齐。统一使用该 yaw 后，当前 ego 坐标系为
    `x` 前向、`y` 左向。
    """

    return _wrap_angle(theta - math.pi / 2.0)


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _load_resized_rgb(image_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    image_height, image_width = image_size
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        rgb_image = rgb_image.resize((image_width, image_height), resampling)
        return np.asarray(rgb_image, dtype=np.uint8)


def _load_depth_image(image_path: Path | None) -> np.ndarray | None:
    if image_path is None:
        return None
    with Image.open(image_path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def _load_depth_image_cached(
    frame_id: int,
    frame_files: dict[int, dict[str, Path]],
    depth_image_cache: OrderedDict[int, np.ndarray | None],
    max_cached_depth_images: int,
) -> np.ndarray | None:
    """按帧号缓存前视 depth 图，供历史窗口可见性过滤复用。"""

    if frame_id in depth_image_cache:
        depth_image_cache.move_to_end(frame_id)
        return depth_image_cache[frame_id]

    depth_image = _load_depth_image(frame_files.get(frame_id, {}).get("depth_front"))
    depth_image_cache[frame_id] = depth_image
    while len(depth_image_cache) > max_cached_depth_images:
        depth_image_cache.popitem(last=False)
    return depth_image


def _require_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "B2D H5 预处理需要 h5py。请先在项目环境中安装 h5py，例如："
            ".\\.venv\\Scripts\\python.exe -m pip install h5py"
        ) from exc
    return h5py


def _compression_kwargs(config: B2DPreprocessConfig) -> dict[str, Any]:
    if config.compression is None:
        return {}
    kwargs: dict[str, Any] = {"compression": config.compression}
    if config.compression == "gzip":
        kwargs["compression_opts"] = config.compression_level
    return kwargs


def _scene_output_name(scene: ScenePaths, raw_root: Path, used_names: set[str]) -> str:
    safe_name = re.sub(r"[^0-9A-Za-z_.-]+", "_", scene.name).strip("._")
    if not safe_name:
        safe_name = "scene"
    output_name = f"{safe_name}.h5"
    if output_name in used_names:
        try:
            relative_path = scene.root.relative_to(raw_root)
        except ValueError:
            relative_path = scene.root
        digest = hashlib.sha1(str(relative_path).encode("utf-8")).hexdigest()[:8]
        output_name = f"{safe_name}_{digest}.h5"
    used_names.add(output_name)
    return output_name


def _configure_logging(log_level: str) -> None:
    logger_level = getattr(logging, log_level)
    handler = logging.StreamHandler()
    handler.setLevel(logger_level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    LOGGER.handlers.clear()
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logger_level)
    LOGGER.propagate = False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="将 B2D 原始场景预处理为逐场景 H5 文件。")
    parser.add_argument("--raw-root", type=Path, default=Path("datasets"), help="B2D 原始数据根目录。")
    parser.add_argument("--output-dir", type=Path, default=Path("data/preprocessed"), help="H5 输出目录。")
    parser.add_argument("--hd-map-root", type=Path, default=Path("datasets/hd_map"), help="B2D HD Map 根目录。")
    parser.add_argument(
        "--map-cache-dir",
        type=Path,
        default=None,
        help="HD Map 紧凑缓存目录；默认使用 output_dir/map_cache，可指定已有缓存目录直接复用。",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="预处理日志级别；DEBUG 会输出缓存和样本构造进度。",
    )
    parser.add_argument("--camera-name", default="rgb_front", help="camera/ 下使用的相机目录。")
    parser.add_argument("--camera-sensor-name", default="CAM_FRONT", help="annotation sensors 中使用的前视相机名。")
    parser.add_argument("--image-height", type=int, default=288, help="输出图像高度。")
    parser.add_argument("--image-width", type=int, default=512, help="输出图像宽度。")
    parser.add_argument("--window-stride", type=int, default=1, help="滑窗步长，单位为 5Hz 模型帧。")
    parser.add_argument("--max-map-elements", type=int, default=60, help="每个样本最多保存的局部 Map 元素数量。")
    parser.add_argument("--map-point-count", type=int, default=100, help="每条 Map 元素重采样点数。")
    parser.add_argument("--detection-forward-range", type=float, default=32.0, help="检测与 Map 前向保留范围。")
    parser.add_argument("--detection-lateral-range", type=float, default=32.0, help="检测与 Map 左右保留范围。")
    parser.add_argument(
        "--min-visible-agent-vertices",
        type=int,
        default=2,
        help="Agent 3D 框至少需要投影可见的顶点数量。",
    )
    parser.add_argument(
        "--min-visible-agent-history-frames",
        type=int,
        default=2,
        help="同一 Agent 在 8 帧历史输入窗口内至少需要满足可见性要求的帧数。",
    )
    parser.add_argument(
        "--map-min-visible-points",
        type=int,
        default=2,
        help="Map 元素至少需要投影到前视图内的点数。",
    )
    parser.add_argument(
        "--hd-map-min-point-spacing",
        type=float,
        default=0.5,
        help="HD Map 原始 polyline 载入后的最小保留点间距，单位 meter。",
    )
    parser.add_argument(
        "--target-min-distance",
        type=float,
        default=24.0,
        help="目标候选点最小可达距离，单位 meter。",
    )
    parser.add_argument(
        "--target-max-distance",
        type=float,
        default=30.0,
        help="目标候选点最大可达距离，单位 meter。",
    )
    parser.add_argument(
        "--max-target-points",
        type=int,
        default=32,
        help="每个样本最多保存的目标候选点数量。",
    )
    parser.add_argument(
        "--target-search-seconds",
        type=float,
        default=None,
        help="目标候选搜索时长；默认搜索当前帧之后全部未来帧。",
    )
    parser.add_argument(
        "--smooth-future-trajectory",
        action="store_true",
        help="开启未来轨迹轻量平滑；默认关闭，通常不建议开启。",
    )
    parser.add_argument(
        "--no-smooth-future-trajectory",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--trajectory-smoothing-iterations",
        type=int,
        default=1,
        help="启用平滑时的迭代次数；默认 1，但通常不建议开启平滑。",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的 H5 文件。")
    args = parser.parse_args(argv)
    if args.smooth_future_trajectory and args.no_smooth_future_trajectory:
        raise ValueError(
            "--smooth-future-trajectory 与 --no-smooth-future-trajectory 不能同时使用。"
        )
    _configure_logging(args.log_level)

    config = B2DPreprocessConfig(
        raw_dataset_root=args.raw_root,
        output_dir=args.output_dir,
        hd_map_root=args.hd_map_root,
        map_cache_dir=args.map_cache_dir,
        camera_name=args.camera_name,
        camera_sensor_name=args.camera_sensor_name,
        image_size=(args.image_height, args.image_width),
        window_stride=args.window_stride,
        max_map_elements=args.max_map_elements,
        map_point_count=args.map_point_count,
        detection_forward_range=args.detection_forward_range,
        detection_lateral_range=args.detection_lateral_range,
        min_visible_agent_vertices=args.min_visible_agent_vertices,
        min_visible_agent_history_frames=args.min_visible_agent_history_frames,
        map_min_visible_points=args.map_min_visible_points,
        hd_map_min_point_spacing=args.hd_map_min_point_spacing,
        target_min_distance=args.target_min_distance,
        target_max_distance=args.target_max_distance,
        max_target_points=args.max_target_points,
        target_search_seconds=args.target_search_seconds,
        smooth_future_trajectory=args.smooth_future_trajectory,
        trajectory_smoothing_iterations=args.trajectory_smoothing_iterations,
    )
    output_paths = preprocess_b2d_dataset(config, overwrite=args.overwrite)
    for output_path in output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
