"""B2D 预处理 H5 样本可视化工具。"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from data.b2d_dataset import B2DH5Dataset


AGENT_CLASS_NAMES = {
    -1: "pad",
    0: "car",
    1: "bike",
    2: "moto",
    3: "ped",
}
AGENT_CLASS_COLORS = {
    0: (245, 128, 42),
    1: (34, 168, 92),
    2: (142, 95, 220),
    3: (224, 64, 72),
}
TRAFFIC_LIGHT_NAMES = {
    0: "red",
    1: "yellow",
    2: "green",
    3: "none",
}
STOP_SIGN_NAMES = {
    0: "none",
    1: "present",
}
MAP_CLASS_NAMES = {
    -1: "pad",
    0: "lane_divider",
    1: "road_edge",
    2: "crosswalk",
    3: "centerline",
}
MAP_CLASS_COLORS = {
    0: (20, 184, 166),
    1: (100, 116, 139),
    2: (236, 72, 153),
    3: (14, 165, 233),
}


@dataclass(frozen=True)
class BevViewConfig:
    """BEV 面板配置。

    坐标约定为 ego 坐标系，`x` 前向、`y` 左向、单位 meter。
    """

    x_min: float = -10.0
    x_max: float = 90.0
    y_min: float = -40.0
    y_max: float = 40.0
    width: int = 640
    height: int = 640
    grid_step: float = 10.0
    velocity_seconds: float = 0.5

    def __post_init__(self) -> None:
        if self.x_min >= self.x_max:
            raise ValueError(f"x_min 必须小于 x_max，实际为 {self.x_min} >= {self.x_max}。")
        if self.y_min >= self.y_max:
            raise ValueError(f"y_min 必须小于 y_max，实际为 {self.y_min} >= {self.y_max}。")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"width/height 必须为正数，实际为 {self.width}/{self.height}。")
        if self.grid_step <= 0:
            raise ValueError(f"grid_step 必须为正数，实际为 {self.grid_step}。")
        if self.velocity_seconds < 0:
            raise ValueError(f"velocity_seconds 必须非负，实际为 {self.velocity_seconds}。")


@dataclass(frozen=True)
class H5SampleData:
    """一个 H5 样本可视化所需的数据。"""

    scene_name: str
    schema_version: str
    target_min_distance: float
    target_max_distance: float
    target_search_seconds: str
    smooth_future_trajectory: bool
    trajectory_smoothing_iterations: int
    detection_forward_range: float
    detection_lateral_range: float
    current_frame_id: int
    input_frame_ids: np.ndarray
    future_frame_ids: np.ndarray
    images: np.ndarray
    future_trajectory: np.ndarray
    target_point: np.ndarray
    target_point_index: int
    target_points: np.ndarray
    target_valid: np.ndarray
    ego_motion: np.ndarray
    current_pose: np.ndarray
    commands: np.ndarray
    control: np.ndarray
    agent_boxes: np.ndarray
    agent_classes: np.ndarray
    agent_valid: np.ndarray
    agent_future_trajectory: np.ndarray
    agent_future_valid: np.ndarray
    map_points: np.ndarray
    map_classes: np.ndarray
    map_valid: np.ndarray
    traffic_light_state: int
    traffic_light_xy: np.ndarray
    traffic_light_valid: bool
    stop_sign_state: int
    stop_sign_xy: np.ndarray
    stop_sign_valid: bool


def render_h5_sample(
    h5_path: str | Path,
    sample_index: int,
    output_path: str | Path,
    bev_config: BevViewConfig | None = None,
    agent_limit: int = 80,
    random_target_point: bool = False,
) -> Path:
    """将一个 B2D H5 样本渲染为 PNG 诊断图。

    Args:
        h5_path: 预处理后的逐场景 H5 文件。
        sample_index: 样本索引。
        output_path: PNG 输出路径。
        bev_config: BEV 面板范围和尺寸配置。
        agent_limit: 最多绘制的有效 Agent 数，按距离从近到远截断。
        random_target_point: 是否复用训练 Dataset 的随机目标点抽样逻辑。

    Returns:
        输出 PNG 路径。
    """

    if agent_limit <= 0:
        raise ValueError(f"agent_limit 必须为正数，实际为 {agent_limit}。")

    sample_data = load_h5_sample(h5_path, sample_index, random_target_point=random_target_point)
    rendered_image = render_sample(sample_data, bev_config or BevViewConfig(), agent_limit=agent_limit)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered_image.save(output)
    return output


def load_h5_sample(
    h5_path: str | Path,
    sample_index: int,
    random_target_point: bool = False,
) -> H5SampleData:
    """通过训练 Dataset 读取单个样本，并展开为可视化数据。"""

    path = Path(h5_path)
    if not path.is_file():
        raise FileNotFoundError(f"h5_path 必须是文件：{path}")

    dataset = B2DH5Dataset(
        path,
        normalize_images=False,
        image_dtype=torch.uint8,
        random_target_point=random_target_point,
    )
    try:
        sample = dataset[sample_index]
    finally:
        dataset.close()

    visual_attrs = _read_h5_visual_attrs(path)
    images = _tensor_to_numpy(sample["images"]).transpose(0, 2, 3, 1).astype(np.uint8, copy=False)
    agent_boxes = _tensor_to_numpy(sample["agent_boxes"]).astype(np.float32, copy=False)
    if agent_boxes.shape[-1] != 10:
        raise ValueError(
            "当前可视化工具期望 agent_boxes 最后一维为 10，"
            f"实际为 {agent_boxes.shape[-1]}。请重新运行 B2D 预处理生成新 H5。"
        )

    return H5SampleData(
        scene_name=str(sample["scene_name"]),
        schema_version=str(visual_attrs["schema_version"]),
        target_min_distance=float(visual_attrs["target_min_distance"]),
        target_max_distance=float(visual_attrs["target_max_distance"]),
        target_search_seconds=str(visual_attrs["target_search_seconds"]),
        smooth_future_trajectory=bool(visual_attrs["smooth_future_trajectory"]),
        trajectory_smoothing_iterations=int(visual_attrs["trajectory_smoothing_iterations"]),
        detection_forward_range=float(visual_attrs["detection_forward_range"]),
        detection_lateral_range=float(visual_attrs["detection_lateral_range"]),
        current_frame_id=int(sample["current_frame_id"]),
        input_frame_ids=_tensor_to_numpy(sample["input_frame_ids"]).astype(np.int32, copy=False),
        future_frame_ids=_tensor_to_numpy(sample["future_frame_ids"]).astype(np.int32, copy=False),
        images=images,
        future_trajectory=_tensor_to_numpy(sample["future_trajectory"]).astype(np.float32, copy=False),
        target_point=_tensor_to_numpy(sample["target_point"]).astype(np.float32, copy=False),
        target_point_index=_scalar_int(sample["target_point_index"]),
        target_points=_tensor_to_numpy(sample["target_points"]).astype(np.float32, copy=False),
        target_valid=_tensor_to_numpy(sample["target_valid"]).astype(np.bool_, copy=False),
        ego_motion=_tensor_to_numpy(sample["ego_motion"]).astype(np.float32, copy=False),
        current_pose=_tensor_to_numpy(sample["current_pose"]).astype(np.float32, copy=False),
        commands=_tensor_to_numpy(sample["commands"]).astype(np.int16, copy=False),
        control=_tensor_to_numpy(sample["control"]).astype(np.float32, copy=False),
        agent_boxes=agent_boxes,
        agent_classes=_tensor_to_numpy(sample["agent_classes"]).astype(np.int16, copy=False),
        agent_valid=_tensor_to_numpy(sample["agent_valid"]).astype(np.bool_, copy=False),
        agent_future_trajectory=_tensor_to_numpy(sample["agent_future_trajectory"]).astype(np.float32, copy=False),
        agent_future_valid=_tensor_to_numpy(sample["agent_future_valid"]).astype(np.bool_, copy=False),
        map_points=_tensor_to_numpy(sample["map_points"]).astype(np.float32, copy=False),
        map_classes=_tensor_to_numpy(sample["map_classes"]).astype(np.int16, copy=False),
        map_valid=_tensor_to_numpy(sample["map_valid"]).astype(np.bool_, copy=False),
        traffic_light_state=_scalar_int(sample["traffic_light_state"]),
        traffic_light_xy=_tensor_to_numpy(sample["traffic_light_xy"]).astype(np.float32, copy=False),
        traffic_light_valid=_scalar_bool(sample["traffic_light_valid"]),
        stop_sign_state=_scalar_int(sample["stop_sign_state"]),
        stop_sign_xy=_tensor_to_numpy(sample["stop_sign_xy"]).astype(np.float32, copy=False),
        stop_sign_valid=_scalar_bool(sample["stop_sign_valid"]),
    )


def _read_h5_visual_attrs(path: Path) -> dict[str, Any]:
    h5py = _require_h5py()
    with h5py.File(path, "r") as h5_file:
        attrs = h5_file.attrs
        return {
            "schema_version": _decode_h5_attr(attrs.get("schema_version", "unknown")),
            "target_min_distance": float(attrs.get("target_min_distance", 24.0)),
            "target_max_distance": float(attrs.get("target_max_distance", 30.0)),
            "target_search_seconds": _decode_h5_attr(attrs.get("target_search_seconds", "unknown")),
            "smooth_future_trajectory": bool(attrs.get("smooth_future_trajectory", False)),
            "trajectory_smoothing_iterations": int(attrs.get("trajectory_smoothing_iterations", 0)),
            "detection_forward_range": float(attrs.get("detection_forward_range", 32.0)),
            "detection_lateral_range": float(attrs.get("detection_lateral_range", 32.0)),
        }


def render_sample(
    sample_data: H5SampleData,
    bev_config: BevViewConfig,
    agent_limit: int = 80,
) -> Image.Image:
    """将已读取的 H5 样本渲染为一张诊断图。"""

    font = ImageFont.load_default()
    canvas_width = 1360
    canvas_height = 960
    margin = 24
    panel_gap = 20
    canvas = Image.new("RGB", (canvas_width, canvas_height), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)

    title = (
        f"{sample_data.scene_name} | schema={sample_data.schema_version} | "
        f"sample frame={sample_data.current_frame_id}"
    )
    draw.text((margin, 12), title, fill=(15, 23, 42), font=font)

    current_image = Image.fromarray(sample_data.images[-1]).resize((640, 360), Image.Resampling.BILINEAR)
    current_origin = (margin, 42)
    canvas.paste(current_image, current_origin)
    _draw_panel_title(draw, current_origin, "current rgb_front")

    history_origin = (margin, 430)
    _draw_history_strip(canvas, draw, history_origin, sample_data, font)

    metadata_origin = (margin, 600)
    _draw_metadata_panel(draw, metadata_origin, sample_data, font)

    bev_origin = (margin + 640 + panel_gap, 42)
    bev_image = _draw_bev_panel(sample_data, bev_config, agent_limit, font)
    canvas.paste(bev_image, bev_origin)
    _draw_panel_title(draw, bev_origin, "ego BEV: x forward, y left")

    legend_origin = (margin + 640 + panel_gap, 704)
    _draw_legend(draw, legend_origin, sample_data, agent_limit, font)
    return canvas


def _draw_history_strip(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    sample_data: H5SampleData,
    font: ImageFont.ImageFont,
) -> None:
    x0, y0 = origin
    draw.text((x0, y0), "history input frames (5Hz, oldest -> current)", fill=(15, 23, 42), font=font)
    thumbnail_width = 72
    thumbnail_height = 40
    gap = 8
    for frame_index, image_array in enumerate(sample_data.images):
        thumbnail = Image.fromarray(image_array).resize((thumbnail_width, thumbnail_height), Image.Resampling.BILINEAR)
        thumbnail_x = x0 + frame_index * (thumbnail_width + gap)
        thumbnail_y = y0 + 20
        canvas.paste(thumbnail, (thumbnail_x, thumbnail_y))
        draw.rectangle(
            [thumbnail_x, thumbnail_y, thumbnail_x + thumbnail_width, thumbnail_y + thumbnail_height],
            outline=(148, 163, 184),
            width=1,
        )
        label = f"t{frame_index}: {int(sample_data.input_frame_ids[frame_index])}"
        draw.text((thumbnail_x + 2, thumbnail_y + thumbnail_height + 4), label, fill=(51, 65, 85), font=font)


def _draw_metadata_panel(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    sample_data: H5SampleData,
    font: ImageFont.ImageFont,
) -> None:
    x0, y0 = origin
    valid_agent_count = int(sample_data.agent_valid.sum())
    valid_agent_future_count = int(np.logical_and(sample_data.agent_valid, sample_data.agent_future_valid.any(axis=1)).sum())
    valid_map_count = int(sample_data.map_valid.sum())
    lines = [
        "sample metadata",
        f"input frame ids : {_format_int_array(sample_data.input_frame_ids)}",
        f"future frame ids: {_format_int_array(sample_data.future_frame_ids)}",
        f"ego_motion [vx,vy,w] = {_format_float_array(sample_data.ego_motion, 3)}",
        (
            "target_point [x,y]  = "
            f"{_format_float_array(sample_data.target_point, 2)}, "
            f"idx={sample_data.target_point_index}"
        ),
        (
            "target candidates / smoothing = "
            f"{int(sample_data.target_valid.sum())} in "
            f"[{sample_data.target_min_distance:.0f},{sample_data.target_max_distance:.0f}]m, "
            f"search={sample_data.target_search_seconds}, "
            f"smooth={sample_data.smooth_future_trajectory}, "
            f"iters={sample_data.trajectory_smoothing_iterations}"
        ),
        f"current_pose [x,y,theta] = {_format_float_array(sample_data.current_pose, 3)}",
        f"commands [near,far,next] = {_format_int_array(sample_data.commands)}",
        f"control [throttle,steer,brake] = {_format_float_array(sample_data.control, 3)}",
        f"valid agents = {valid_agent_count} / {len(sample_data.agent_valid)}",
        f"agent futures = {valid_agent_future_count}, maps = {valid_map_count} / {len(sample_data.map_valid)}",
        (
            "filter range = "
            f"x<= {sample_data.detection_forward_range:.0f}m, "
            f"|y|<= {sample_data.detection_lateral_range:.0f}m"
        ),
        (
            "traffic_light = "
            f"{TRAFFIC_LIGHT_NAMES.get(sample_data.traffic_light_state, str(sample_data.traffic_light_state))}, "
            f"valid={sample_data.traffic_light_valid}, xy={_format_float_array(sample_data.traffic_light_xy, 1)}"
        ),
        (
            "stop_sign = "
            f"{STOP_SIGN_NAMES.get(sample_data.stop_sign_state, str(sample_data.stop_sign_state))}, "
            f"valid={sample_data.stop_sign_valid}, xy={_format_float_array(sample_data.stop_sign_xy, 1)}"
        ),
    ]
    draw.rounded_rectangle([x0, y0, x0 + 640, y0 + 340], radius=8, fill=(255, 255, 255), outline=(203, 213, 225))
    for line_index, line in enumerate(lines):
        fill = (15, 23, 42) if line_index == 0 else (51, 65, 85)
        draw.text((x0 + 14, y0 + 12 + line_index * 24), line, fill=fill, font=font)


def _draw_bev_panel(
    sample_data: H5SampleData,
    config: BevViewConfig,
    agent_limit: int,
    font: ImageFont.ImageFont,
) -> Image.Image:
    bev_image = Image.new("RGB", (config.width, config.height), (255, 255, 255))
    draw = ImageDraw.Draw(bev_image)
    transform = _BevTransform(config)

    _draw_bev_grid(draw, transform, config, font)
    _draw_ego_marker(draw, transform)
    _draw_map_elements(draw, transform, sample_data.map_points, sample_data.map_classes, sample_data.map_valid)
    _draw_future_trajectory(draw, transform, sample_data.future_trajectory)
    _draw_target_candidates(draw, transform, sample_data.target_points, sample_data.target_valid)
    _draw_target_marker(draw, transform, sample_data.target_point, "target", (22, 163, 74), font)

    if sample_data.traffic_light_valid:
        _draw_target_marker(draw, transform, sample_data.traffic_light_xy, "tl", (234, 179, 8), font)
    if sample_data.stop_sign_valid:
        _draw_target_marker(draw, transform, sample_data.stop_sign_xy, "stop", (220, 38, 38), font)

    agent_indices = _sorted_valid_agent_indices(sample_data.agent_boxes, sample_data.agent_valid, agent_limit)
    for agent_index in agent_indices:
        box = sample_data.agent_boxes[agent_index]
        class_id = int(sample_data.agent_classes[agent_index])
        color = AGENT_CLASS_COLORS.get(class_id, (71, 85, 105))
        _draw_agent_future_trajectory(
            draw,
            transform,
            box,
            sample_data.agent_future_trajectory[agent_index],
            sample_data.agent_future_valid[agent_index],
            color,
        )
        _draw_agent_box(draw, transform, box, color)
        _draw_agent_velocity(draw, transform, box, color, config.velocity_seconds)

    draw.rectangle([0, 0, config.width - 1, config.height - 1], outline=(100, 116, 139), width=2)
    return bev_image


class _BevTransform:
    def __init__(self, config: BevViewConfig) -> None:
        self.config = config

    def to_pixel(self, ego_x: float, ego_y: float) -> tuple[int, int]:
        x_ratio = (ego_y - self.config.y_min) / (self.config.y_max - self.config.y_min)
        y_ratio = (self.config.x_max - ego_x) / (self.config.x_max - self.config.x_min)
        pixel_x = int(round(x_ratio * (self.config.width - 1)))
        pixel_y = int(round(y_ratio * (self.config.height - 1)))
        return pixel_x, pixel_y

    def in_view(self, ego_x: float, ego_y: float) -> bool:
        return (
            self.config.x_min <= ego_x <= self.config.x_max
            and self.config.y_min <= ego_y <= self.config.y_max
        )


def _draw_bev_grid(
    draw: ImageDraw.ImageDraw,
    transform: _BevTransform,
    config: BevViewConfig,
    font: ImageFont.ImageFont,
) -> None:
    grid_color = (226, 232, 240)
    axis_color = (100, 116, 139)
    x_start = math.ceil(config.x_min / config.grid_step) * config.grid_step
    x_value = x_start
    while x_value <= config.x_max:
        p0 = transform.to_pixel(x_value, config.y_min)
        p1 = transform.to_pixel(x_value, config.y_max)
        color = axis_color if math.isclose(x_value, 0.0, abs_tol=1e-6) else grid_color
        draw.line([p0, p1], fill=color, width=1)
        draw.text((4, p0[1] - 8), f"x={x_value:.0f}", fill=(100, 116, 139), font=font)
        x_value += config.grid_step

    y_start = math.ceil(config.y_min / config.grid_step) * config.grid_step
    y_value = y_start
    while y_value <= config.y_max:
        p0 = transform.to_pixel(config.x_min, y_value)
        p1 = transform.to_pixel(config.x_max, y_value)
        color = axis_color if math.isclose(y_value, 0.0, abs_tol=1e-6) else grid_color
        draw.line([p0, p1], fill=color, width=1)
        draw.text((p0[0] + 2, config.height - 16), f"y={y_value:.0f}", fill=(100, 116, 139), font=font)
        y_value += config.grid_step


def _draw_ego_marker(draw: ImageDraw.ImageDraw, transform: _BevTransform) -> None:
    center = transform.to_pixel(0.0, 0.0)
    triangle = [
        transform.to_pixel(2.2, 0.0),
        transform.to_pixel(-1.4, -1.0),
        transform.to_pixel(-1.4, 1.0),
    ]
    draw.polygon(triangle, fill=(15, 23, 42), outline=(15, 23, 42))
    draw.ellipse([center[0] - 3, center[1] - 3, center[0] + 3, center[1] + 3], fill=(255, 255, 255))


def _draw_future_trajectory(
    draw: ImageDraw.ImageDraw,
    transform: _BevTransform,
    future_trajectory: np.ndarray,
) -> None:
    trajectory_points = [transform.to_pixel(float(point[0]), float(point[1])) for point in future_trajectory]
    if trajectory_points:
        # future_trajectory 的第一个点是 t+0.5s；诊断图需要从当前 ego 位置连过去。
        draw.line([transform.to_pixel(0.0, 0.0), *trajectory_points], fill=(37, 99, 235), width=3)
    for point_index, pixel_point in enumerate(trajectory_points):
        radius = 4
        draw.ellipse(
            [
                pixel_point[0] - radius,
                pixel_point[1] - radius,
                pixel_point[0] + radius,
                pixel_point[1] + radius,
            ],
            fill=(37, 99, 235),
        )
        draw.text((pixel_point[0] + 5, pixel_point[1] - 5), str(point_index + 1), fill=(37, 99, 235))


def _draw_target_marker(
    draw: ImageDraw.ImageDraw,
    transform: _BevTransform,
    point_xy: np.ndarray,
    label: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    point = transform.to_pixel(float(point_xy[0]), float(point_xy[1]))
    size = 7
    draw.line([point[0] - size, point[1], point[0] + size, point[1]], fill=color, width=2)
    draw.line([point[0], point[1] - size, point[0], point[1] + size], fill=color, width=2)
    draw.text((point[0] + 8, point[1] - 8), label, fill=color, font=font)


def _draw_target_candidates(
    draw: ImageDraw.ImageDraw,
    transform: _BevTransform,
    target_points: np.ndarray,
    target_valid: np.ndarray,
) -> None:
    for point_xy in target_points[target_valid]:
        point = transform.to_pixel(float(point_xy[0]), float(point_xy[1]))
        radius = 4
        draw.ellipse(
            [point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius],
            outline=(22, 163, 74),
            width=1,
        )


def _draw_map_elements(
    draw: ImageDraw.ImageDraw,
    transform: _BevTransform,
    map_points: np.ndarray,
    map_classes: np.ndarray,
    map_valid: np.ndarray,
) -> None:
    for points_xy, class_id in zip(map_points[map_valid], map_classes[map_valid], strict=False):
        color = MAP_CLASS_COLORS.get(int(class_id), (100, 116, 139))
        pixel_points = [
            transform.to_pixel(float(point_xy[0]), float(point_xy[1]))
            for point_xy in points_xy
            if transform.in_view(float(point_xy[0]), float(point_xy[1]))
        ]
        if len(pixel_points) >= 2:
            draw.line(pixel_points, fill=color, width=1)


def _draw_agent_future_trajectory(
    draw: ImageDraw.ImageDraw,
    transform: _BevTransform,
    box: np.ndarray,
    future_displacement: np.ndarray,
    future_valid: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    valid_displacements = future_displacement[future_valid]
    if valid_displacements.shape[0] == 0:
        return
    center_xy = np.asarray(box[:2], dtype=np.float32)
    valid_points = valid_displacements + center_xy
    start = transform.to_pixel(float(center_xy[0]), float(center_xy[1]))
    trajectory_points = [transform.to_pixel(float(point[0]), float(point[1])) for point in valid_points]
    draw.line([start, *trajectory_points], fill=color, width=2)
    for point in trajectory_points:
        radius = 3
        draw.ellipse(
            [point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius],
            fill=(255, 255, 255),
            outline=color,
            width=1,
        )


def _draw_agent_box(
    draw: ImageDraw.ImageDraw,
    transform: _BevTransform,
    box: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    center_x, center_y, length, width, _height, yaw = [float(value) for value in box[:6]]
    half_length = max(length, 0.2) / 2.0
    half_width = max(width, 0.2) / 2.0
    corners = [
        (half_length, half_width),
        (half_length, -half_width),
        (-half_length, -half_width),
        (-half_length, half_width),
    ]
    rotated_corners = []
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    for local_x, local_y in corners:
        ego_x = center_x + local_x * cos_yaw - local_y * sin_yaw
        ego_y = center_y + local_x * sin_yaw + local_y * cos_yaw
        rotated_corners.append(transform.to_pixel(ego_x, ego_y))
    draw.polygon(rotated_corners, outline=color)
    draw.line([rotated_corners[0], rotated_corners[3]], fill=color, width=3)


def _draw_agent_velocity(
    draw: ImageDraw.ImageDraw,
    transform: _BevTransform,
    box: np.ndarray,
    color: tuple[int, int, int],
    velocity_seconds: float,
) -> None:
    if velocity_seconds == 0:
        return
    center_x = float(box[0])
    center_y = float(box[1])
    velocity_x = float(box[6])
    velocity_y = float(box[7])
    speed = math.hypot(velocity_x, velocity_y)
    if speed < 0.05:
        return
    start = transform.to_pixel(center_x, center_y)
    end = transform.to_pixel(
        center_x + velocity_x * velocity_seconds,
        center_y + velocity_y * velocity_seconds,
    )
    draw.line([start, end], fill=color, width=2)
    _draw_arrow_head(draw, start, end, color)


def _draw_arrow_head(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 6
    left = (
        int(round(end[0] - size * math.cos(angle - math.pi / 6))),
        int(round(end[1] - size * math.sin(angle - math.pi / 6))),
    )
    right = (
        int(round(end[0] - size * math.cos(angle + math.pi / 6))),
        int(round(end[1] - size * math.sin(angle + math.pi / 6))),
    )
    draw.polygon([end, left, right], fill=color)


def _sorted_valid_agent_indices(
    agent_boxes: np.ndarray,
    agent_valid: np.ndarray,
    agent_limit: int,
) -> list[int]:
    valid_indices = np.flatnonzero(agent_valid)
    distances = np.linalg.norm(agent_boxes[valid_indices, :2], axis=1)
    sorted_order = np.argsort(distances)
    return [int(valid_indices[index]) for index in sorted_order[:agent_limit]]


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    sample_data: H5SampleData,
    agent_limit: int,
    font: ImageFont.ImageFont,
) -> None:
    x0, y0 = origin
    draw.rounded_rectangle([x0, y0, x0 + 640, y0 + 220], radius=8, fill=(255, 255, 255), outline=(203, 213, 225))
    draw.text((x0 + 14, y0 + 12), "BEV legend", fill=(15, 23, 42), font=font)
    legend_items = [
        ("future trajectory", (37, 99, 235)),
        ("agent future", (245, 128, 42)),
        ("target candidates", (22, 163, 74)),
        ("traffic light", (234, 179, 8)),
        ("stop sign", (220, 38, 38)),
    ]
    for item_index, (label, color) in enumerate(legend_items):
        item_y = y0 + 40 + item_index * 24
        draw.rectangle([x0 + 16, item_y + 4, x0 + 28, item_y + 16], fill=color)
        draw.text((x0 + 38, item_y), label, fill=(51, 65, 85), font=font)

    class_counts = _agent_class_counts(sample_data.agent_classes, sample_data.agent_valid)
    class_x = x0 + 270
    draw.text((class_x, y0 + 40), f"agents drawn <= {agent_limit}", fill=(51, 65, 85), font=font)
    for row_index, class_id in enumerate([0, 1, 2, 3]):
        item_y = y0 + 66 + row_index * 24
        color = AGENT_CLASS_COLORS[class_id]
        draw.rectangle([class_x, item_y + 4, class_x + 12, item_y + 16], fill=color)
        text = f"{AGENT_CLASS_NAMES[class_id]}: {class_counts.get(class_id, 0)}"
        draw.text((class_x + 22, item_y), text, fill=(51, 65, 85), font=font)

    map_counts = _map_class_counts(sample_data.map_classes, sample_data.map_valid)
    map_x = x0 + 430
    draw.text((map_x, y0 + 40), "maps", fill=(51, 65, 85), font=font)
    for row_index, class_id in enumerate([0, 1, 2, 3]):
        item_y = y0 + 66 + row_index * 24
        color = MAP_CLASS_COLORS[class_id]
        draw.rectangle([map_x, item_y + 4, map_x + 12, item_y + 16], fill=color)
        text = f"{MAP_CLASS_NAMES[class_id]}: {map_counts.get(class_id, 0)}"
        draw.text((map_x + 22, item_y), text, fill=(51, 65, 85), font=font)

    footer = "agent state: [x,y,l,w,h,yaw,vx,vy,ax,ay], velocity arrow = 0.5s"
    draw.text((x0 + 14, y0 + 186), footer, fill=(71, 85, 105), font=font)


def _agent_class_counts(agent_classes: np.ndarray, agent_valid: np.ndarray) -> dict[int, int]:
    counts: dict[int, int] = {}
    for class_id in agent_classes[agent_valid]:
        class_id_int = int(class_id)
        counts[class_id_int] = counts.get(class_id_int, 0) + 1
    return counts


def _map_class_counts(map_classes: np.ndarray, map_valid: np.ndarray) -> dict[int, int]:
    counts: dict[int, int] = {}
    for class_id in map_classes[map_valid]:
        class_id_int = int(class_id)
        counts[class_id_int] = counts.get(class_id_int, 0) + 1
    return counts


def _draw_panel_title(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    title: str,
) -> None:
    x0, y0 = origin
    draw.rectangle([x0, y0, x0 + 210, y0 + 20], fill=(15, 23, 42))
    draw.text((x0 + 6, y0 + 5), title, fill=(255, 255, 255), font=ImageFont.load_default())


def _format_int_array(values: np.ndarray) -> str:
    return "[" + ", ".join(str(int(value)) for value in values.tolist()) + "]"


def _format_float_array(values: np.ndarray, decimals: int) -> str:
    return "[" + ", ".join(f"{float(value):.{decimals}f}" for value in values.tolist()) + "]"


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar_int(value: Any) -> int:
    array = _tensor_to_numpy(value)
    return int(array.item() if array.shape == () else array)


def _scalar_bool(value: Any) -> bool:
    array = _tensor_to_numpy(value)
    return bool(array.item() if array.shape == () else array)


def _decode_h5_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _require_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "B2D H5 可视化需要 h5py。请先在项目环境中安装 h5py，例如："
            ".\\.venv\\Scripts\\python.exe -m pip install h5py"
        ) from exc
    return h5py


def _default_output_path(h5_path: Path, sample_index: int, output_dir: Path) -> Path:
    return output_dir / f"{h5_path.stem}_sample_{sample_index:06d}.png"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导出 B2D 预处理 H5 样本诊断图。")
    parser.add_argument("--h5", type=Path, required=True, help="预处理后的逐场景 H5 文件。")
    parser.add_argument("--sample-index", type=int, default=0, help="起始样本索引。")
    parser.add_argument("--count", type=int, default=1, help="连续导出的样本数量。")
    parser.add_argument("--stride", type=int, default=1, help="多样本导出时的样本间隔。")
    parser.add_argument("--output", type=Path, default=None, help="单样本输出 PNG 路径。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("visualization/outputs"),
        help="多样本或默认输出目录。",
    )
    parser.add_argument("--bev-x-min", type=float, default=-10.0, help="BEV 前向最小距离。")
    parser.add_argument("--bev-x-max", type=float, default=90.0, help="BEV 前向最大距离。")
    parser.add_argument("--bev-y-min", type=float, default=-40.0, help="BEV 左向最小距离。")
    parser.add_argument("--bev-y-max", type=float, default=40.0, help="BEV 左向最大距离。")
    parser.add_argument("--agent-limit", type=int, default=80, help="最多绘制的有效 Agent 数。")
    parser.add_argument(
        "--random-target-point",
        action="store_true",
        help="启用 Dataset 的随机目标点抽样；默认关闭以保持可视化可复现。",
    )
    args = parser.parse_args(argv)

    if args.count <= 0:
        raise ValueError(f"count 必须为正数，实际为 {args.count}。")
    if args.stride <= 0:
        raise ValueError(f"stride 必须为正数，实际为 {args.stride}。")
    if args.output is not None and args.count != 1:
        raise ValueError("--output 仅允许在 count=1 时使用。")

    bev_config = BevViewConfig(
        x_min=args.bev_x_min,
        x_max=args.bev_x_max,
        y_min=args.bev_y_min,
        y_max=args.bev_y_max,
    )

    for offset in range(args.count):
        sample_index = args.sample_index + offset * args.stride
        output_path = args.output or _default_output_path(args.h5, sample_index, args.output_dir)
        rendered_path = render_h5_sample(
            args.h5,
            sample_index,
            output_path,
            bev_config=bev_config,
            agent_limit=args.agent_limit,
            random_target_point=args.random_target_point,
        )
        print(rendered_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
