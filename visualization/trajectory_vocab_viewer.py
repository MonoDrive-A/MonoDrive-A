"""轨迹词表归一化反变换校验与可视化工具。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class TrajectoryVocabularyData:
    """从 `.npz` 读取出的轨迹词表数据。

    Shape:
        `physical_trajectories`: `[V, 6, 2]`，ego 坐标系米制轨迹。
        `normalized_trajectories`: `[V, 6, 2]`，Symlog 后共享缩放归一化轨迹。
        `reconstructed_trajectories`: `[V, 6, 2]`，由归一化轨迹和缩放系数反求的米制轨迹。
        `per_trajectory_mse`: `[V]`，每条轨迹在物理空间的 MSE。
    """

    physical_trajectories: np.ndarray
    normalized_trajectories: np.ndarray
    reconstructed_trajectories: np.ndarray
    symlog_scale: float
    per_trajectory_mse: np.ndarray
    global_mse: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TrajectoryVocabularyViewConfig:
    """轨迹词表 BEV 可视化配置。

    坐标约定为 ego 坐标系，`x` 前向、`y` 左向，单位 meter。
    """

    x_min: float = -5.0
    x_max: float = 70.0
    y_min: float = -35.0
    y_max: float = 35.0
    panel_width: int = 240
    panel_height: int = 240
    columns: int = 4
    grid_step: float = 10.0
    max_mse: float = 1e-8

    def __post_init__(self) -> None:
        if self.x_min >= self.x_max:
            raise ValueError(f"x_min 必须小于 x_max，实际为 {self.x_min} >= {self.x_max}。")
        if self.y_min >= self.y_max:
            raise ValueError(f"y_min 必须小于 y_max，实际为 {self.y_min} >= {self.y_max}。")
        if self.panel_width <= 0 or self.panel_height <= 0:
            raise ValueError(
                f"panel_width/panel_height 必须为正数，实际为 {self.panel_width}/{self.panel_height}。"
            )
        if self.columns <= 0:
            raise ValueError(f"columns 必须为正数，实际为 {self.columns}。")
        if self.grid_step <= 0:
            raise ValueError(f"grid_step 必须为正数，实际为 {self.grid_step}。")
        if self.max_mse < 0:
            raise ValueError(f"max_mse 必须非负，实际为 {self.max_mse}。")


def load_trajectory_vocabulary_npz(npz_path: str | Path, max_mse: float = 1e-8) -> TrajectoryVocabularyData:
    """读取轨迹词表 `.npz`，反求物理轨迹并校验 MSE。

    `.npz` 必须至少包含：

    - `trajectory_vocab_m`: `[V, 6, 2]` 物理空间词表。
    - `trajectory_vocab_normalized`: `[V, 6, 2]` 归一化词表。
    - `symlog_scale`: 共享缩放系数。
    """

    path = Path(npz_path)
    if not path.is_file():
        raise FileNotFoundError(f"npz_path 必须是文件：{path}")
    if max_mse < 0:
        raise ValueError(f"max_mse 必须非负，实际为 {max_mse}。")

    with np.load(path) as npz_file:
        _require_npz_fields(
            npz_file,
            required_fields=("trajectory_vocab_m", "trajectory_vocab_normalized", "symlog_scale"),
            npz_path=path,
        )
        physical_trajectories = np.asarray(npz_file["trajectory_vocab_m"], dtype=np.float32)
        normalized_trajectories = np.asarray(npz_file["trajectory_vocab_normalized"], dtype=np.float32)
        symlog_scale = float(np.asarray(npz_file["symlog_scale"]).item())
        metadata = _load_metadata(npz_file)

    _validate_vocab_shapes(physical_trajectories, normalized_trajectories, path)
    if symlog_scale <= 0 or not math.isfinite(symlog_scale):
        raise ValueError(f"symlog_scale 必须为有限正数，实际为 {symlog_scale}。")

    reconstructed_trajectories = normalized_to_physical_trajectories(
        normalized_trajectories,
        symlog_scale=symlog_scale,
    )
    per_trajectory_mse = np.mean(
        np.square(reconstructed_trajectories - physical_trajectories),
        axis=(1, 2),
        dtype=np.float64,
    ).astype(np.float64)
    global_mse = float(np.mean(per_trajectory_mse, dtype=np.float64))
    if global_mse > max_mse:
        raise ValueError(
            f"归一化反求物理轨迹与原始物理轨迹不一致：global_mse={global_mse:.6e}，"
            f"阈值为 {max_mse:.6e}。"
        )

    return TrajectoryVocabularyData(
        physical_trajectories=physical_trajectories,
        normalized_trajectories=normalized_trajectories,
        reconstructed_trajectories=reconstructed_trajectories,
        symlog_scale=symlog_scale,
        per_trajectory_mse=per_trajectory_mse,
        global_mse=global_mse,
        metadata=metadata,
    )


def normalized_to_physical_trajectories(
    normalized_trajectories: np.ndarray,
    symlog_scale: float,
) -> np.ndarray:
    """从归一化轨迹和共享缩放系数反求物理空间轨迹。"""

    if normalized_trajectories.ndim != 3:
        raise ValueError(
            "normalized_trajectories 期望 shape 为 [V, K, D]，"
            f"实际为 {normalized_trajectories.shape}。"
        )
    if symlog_scale <= 0 or not math.isfinite(symlog_scale):
        raise ValueError(f"symlog_scale 必须为有限正数，实际为 {symlog_scale}。")

    symlog_trajectories = normalized_trajectories.astype(np.float32) * np.float32(symlog_scale)
    return inverse_symlog(symlog_trajectories)


def inverse_symlog(values: np.ndarray) -> np.ndarray:
    """计算 Symlog 的反变换 `sign(y) * (exp(abs(y)) - 1)`。"""

    return (np.sign(values) * np.expm1(np.abs(values))).astype(np.float32)


def render_trajectory_vocabulary(
    vocabulary_data: TrajectoryVocabularyData,
    output_path: str | Path,
    config: TrajectoryVocabularyViewConfig | None = None,
    indices: list[int] | None = None,
    max_trajectories: int = 32,
) -> Path:
    """把轨迹词表渲染为 PNG，并在图中叠加物理轨迹与反归一化轨迹。"""

    if max_trajectories <= 0:
        raise ValueError(f"max_trajectories 必须为正数，实际为 {max_trajectories}。")

    view_config = config or TrajectoryVocabularyViewConfig()
    selected_indices = _resolve_visualized_indices(
        vocabulary_size=vocabulary_data.physical_trajectories.shape[0],
        indices=indices,
        max_trajectories=max_trajectories,
    )
    if vocabulary_data.global_mse > view_config.max_mse:
        raise ValueError(
            f"global_mse={vocabulary_data.global_mse:.6e} 超过阈值 {view_config.max_mse:.6e}。"
        )

    rendered_image = _render_vocabulary_image(vocabulary_data, view_config, selected_indices)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered_image.save(output)
    return output


def render_trajectory_vocabulary_overlay(
    vocabulary_data: TrajectoryVocabularyData,
    output_path: str | Path,
    config: TrajectoryVocabularyViewConfig | None = None,
    width: int = 900,
    height: int = 720,
    draw_reconstructed: bool = False,
) -> Path:
    """把词表中的全部轨迹叠加在同一张 BEV 图中。"""

    if width <= 0 or height <= 0:
        raise ValueError(f"width/height 必须为正数，实际为 {width}/{height}。")

    base_config = config or TrajectoryVocabularyViewConfig()
    if vocabulary_data.global_mse > base_config.max_mse:
        raise ValueError(
            f"global_mse={vocabulary_data.global_mse:.6e} 超过阈值 {base_config.max_mse:.6e}。"
        )

    overlay_config = TrajectoryVocabularyViewConfig(
        x_min=base_config.x_min,
        x_max=base_config.x_max,
        y_min=base_config.y_min,
        y_max=base_config.y_max,
        panel_width=width,
        panel_height=height,
        columns=base_config.columns,
        grid_step=base_config.grid_step,
        max_mse=base_config.max_mse,
    )
    rendered_image = _render_overlay_image(
        vocabulary_data,
        overlay_config,
        draw_reconstructed=draw_reconstructed,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered_image.save(output)
    return output


def _render_vocabulary_image(
    vocabulary_data: TrajectoryVocabularyData,
    config: TrajectoryVocabularyViewConfig,
    indices: list[int],
) -> Image.Image:
    font = ImageFont.load_default()
    header_height = 128
    footer_height = 70
    margin = 18
    gap = 14
    rows = math.ceil(len(indices) / config.columns)
    canvas_width = margin * 2 + config.columns * config.panel_width + (config.columns - 1) * gap
    canvas_height = header_height + rows * config.panel_height + max(rows - 1, 0) * gap + footer_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)

    _draw_header(draw, vocabulary_data, margin, font)
    transform = _TrajectoryTransform(config)
    for item_index, vocab_index in enumerate(indices):
        row = item_index // config.columns
        col = item_index % config.columns
        origin_x = margin + col * (config.panel_width + gap)
        origin_y = header_height + row * (config.panel_height + gap)
        _draw_trajectory_panel(draw, transform, vocabulary_data, vocab_index, (origin_x, origin_y), font)

    _draw_footer(
        draw,
        y=canvas_height - footer_height + 12,
        margin=margin,
        vocabulary_data=vocabulary_data,
        indices=indices,
        font=font,
    )
    return canvas


def _render_overlay_image(
    vocabulary_data: TrajectoryVocabularyData,
    config: TrajectoryVocabularyViewConfig,
    draw_reconstructed: bool,
) -> Image.Image:
    font = ImageFont.load_default()
    header_height = 118
    footer_height = 54
    margin = 20
    canvas_width = config.panel_width + margin * 2
    canvas_height = config.panel_height + header_height + footer_height
    canvas = Image.new("RGBA", (canvas_width, canvas_height), (248, 250, 252, 255))
    draw = ImageDraw.Draw(canvas, "RGBA")

    metadata = vocabulary_data.metadata
    title = "all trajectory vocabulary overlay"
    subtitle = (
        f"V={vocabulary_data.physical_trajectories.shape[0]}, "
        f"scale={vocabulary_data.symlog_scale:.6g}, "
        f"global_mse={vocabulary_data.global_mse:.6e}, "
        f"reconstructed_overlay={draw_reconstructed}"
    )
    source = (
        f"algorithm={metadata.get('sampling_algorithm', 'unknown')}, "
        f"static={metadata.get('static_trajectory_policy', 'unknown')}, "
        f"coord={metadata.get('coordinate_system', 'ego: x forward, y left, unit meter')}"
    )
    draw.text((margin, 16), title, fill=(15, 23, 42, 255), font=font)
    draw.text((margin, 42), subtitle, fill=(51, 65, 85, 255), font=font)
    draw.text((margin, 66), source, fill=(51, 65, 85, 255), font=font)
    draw.rectangle([margin, 94, margin + 16, 106], fill=(37, 99, 235, 110))
    draw.text((margin + 24, 92), "stored physical trajectories", fill=(51, 65, 85, 255), font=font)
    draw.rectangle([margin + 240, 94, margin + 256, 106], fill=(15, 23, 42, 255))
    draw.text((margin + 264, 92), "forced static trajectory", fill=(51, 65, 85, 255), font=font)
    if draw_reconstructed:
        draw.rectangle([margin + 450, 94, margin + 466, 106], fill=(220, 38, 38, 70))
        draw.text((margin + 474, 92), "reconstructed trajectories", fill=(51, 65, 85, 255), font=font)

    origin = (margin, header_height)
    transform = _TrajectoryTransform(config)
    draw.rectangle(
        [
            origin[0],
            origin[1],
            origin[0] + config.panel_width,
            origin[1] + config.panel_height,
        ],
        fill=(255, 255, 255, 255),
        outline=(203, 213, 225, 255),
    )
    _draw_grid(draw, transform, origin, font)
    _draw_ego(draw, transform, origin)

    for vocab_index, trajectory in enumerate(vocabulary_data.physical_trajectories):
        color = (15, 23, 42, 255) if vocab_index == 0 else (37, 99, 235, 78)
        width = 3 if vocab_index == 0 else 2
        _draw_polyline(draw, transform, origin, trajectory, color=color, width=width)

    if draw_reconstructed:
        for reconstructed in vocabulary_data.reconstructed_trajectories:
            _draw_polyline(draw, transform, origin, reconstructed, color=(220, 38, 38, 54), width=1)

    _draw_overlay_endpoints(draw, transform, origin, vocabulary_data.physical_trajectories)
    footer = (
        "All stored physical trajectories are overlaid in one ego BEV panel. "
        "The black line is vocabulary index 0, the forced zero static trajectory."
    )
    draw.text((margin, canvas_height - footer_height + 16), footer, fill=(51, 65, 85, 255), font=font)
    return canvas.convert("RGB")


def _draw_header(
    draw: ImageDraw.ImageDraw,
    vocabulary_data: TrajectoryVocabularyData,
    margin: int,
    font: ImageFont.ImageFont,
) -> None:
    metadata = vocabulary_data.metadata
    title = "trajectory vocabulary reconstruction check"
    subtitle = (
        f"V={vocabulary_data.physical_trajectories.shape[0]}, "
        f"shape={list(vocabulary_data.physical_trajectories.shape)}, "
        f"scale={vocabulary_data.symlog_scale:.6g}, "
        f"global_mse={vocabulary_data.global_mse:.6e}"
    )
    source = (
        f"algorithm={metadata.get('sampling_algorithm', 'unknown')}, "
        f"static={metadata.get('static_trajectory_policy', 'unknown')}, "
        f"coord={metadata.get('coordinate_system', 'ego: x forward, y left, unit meter')}"
    )
    draw.text((margin, 16), title, fill=(15, 23, 42), font=font)
    draw.text((margin, 42), subtitle, fill=(51, 65, 85), font=font)
    draw.text((margin, 66), source, fill=(51, 65, 85), font=font)
    draw.rectangle([margin, 96, margin + 16, 108], fill=(37, 99, 235))
    draw.text((margin + 24, 94), "stored physical trajectory", fill=(51, 65, 85), font=font)
    draw.rectangle([margin + 230, 96, margin + 246, 108], fill=(220, 38, 38))
    draw.text((margin + 254, 94), "reconstructed from normalized + symlog_scale", fill=(51, 65, 85), font=font)


def _draw_trajectory_panel(
    draw: ImageDraw.ImageDraw,
    transform: "_TrajectoryTransform",
    vocabulary_data: TrajectoryVocabularyData,
    vocab_index: int,
    origin: tuple[int, int],
    font: ImageFont.ImageFont,
) -> None:
    x0, y0 = origin
    x1 = x0 + transform.config.panel_width
    y1 = y0 + transform.config.panel_height
    draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255), outline=(203, 213, 225))
    _draw_grid(draw, transform, origin, font)
    _draw_ego(draw, transform, origin)

    physical = vocabulary_data.physical_trajectories[vocab_index]
    reconstructed = vocabulary_data.reconstructed_trajectories[vocab_index]
    _draw_polyline(draw, transform, origin, physical, color=(37, 99, 235), width=3)
    _draw_polyline(draw, transform, origin, reconstructed, color=(220, 38, 38), width=1)
    _draw_points(draw, transform, origin, physical, color=(37, 99, 235), radius=3)
    _draw_points(draw, transform, origin, reconstructed, color=(220, 38, 38), radius=2)

    label = f"#{vocab_index:03d} mse={vocabulary_data.per_trajectory_mse[vocab_index]:.2e}"
    draw.rectangle([x0, y0, x0 + 140, y0 + 20], fill=(15, 23, 42))
    draw.text((x0 + 6, y0 + 5), label, fill=(255, 255, 255), font=font)


def _draw_grid(
    draw: ImageDraw.ImageDraw,
    transform: "_TrajectoryTransform",
    origin: tuple[int, int],
    font: ImageFont.ImageFont,
) -> None:
    config = transform.config
    grid_color = (226, 232, 240)
    axis_color = (100, 116, 139)
    x_start = math.ceil(config.x_min / config.grid_step) * config.grid_step
    x_value = x_start
    while x_value <= config.x_max:
        p0 = transform.to_canvas(origin, x_value, config.y_min)
        p1 = transform.to_canvas(origin, x_value, config.y_max)
        color = axis_color if math.isclose(x_value, 0.0, abs_tol=1e-6) else grid_color
        draw.line([p0, p1], fill=color, width=1)
        x_value += config.grid_step

    y_start = math.ceil(config.y_min / config.grid_step) * config.grid_step
    y_value = y_start
    while y_value <= config.y_max:
        p0 = transform.to_canvas(origin, config.x_min, y_value)
        p1 = transform.to_canvas(origin, config.x_max, y_value)
        color = axis_color if math.isclose(y_value, 0.0, abs_tol=1e-6) else grid_color
        draw.line([p0, p1], fill=color, width=1)
        y_value += config.grid_step

    x_label = transform.to_canvas(origin, config.x_max, config.y_min)
    y_label = transform.to_canvas(origin, config.x_min, config.y_max)
    draw.text((origin[0] + 4, x_label[1] - 16), "x forward", fill=(100, 116, 139), font=font)
    draw.text((y_label[0] + 4, origin[1] + config.panel_height - 16), "y left", fill=(100, 116, 139), font=font)


def _draw_ego(
    draw: ImageDraw.ImageDraw,
    transform: "_TrajectoryTransform",
    origin: tuple[int, int],
) -> None:
    if not transform.in_view(0.0, 0.0):
        return
    triangle = [
        transform.to_canvas(origin, 2.0, 0.0),
        transform.to_canvas(origin, -1.2, -0.8),
        transform.to_canvas(origin, -1.2, 0.8),
    ]
    draw.polygon(triangle, fill=(15, 23, 42), outline=(15, 23, 42))


def _draw_polyline(
    draw: ImageDraw.ImageDraw,
    transform: "_TrajectoryTransform",
    origin: tuple[int, int],
    trajectory: np.ndarray,
    color: tuple[int, int, int],
    width: int,
) -> None:
    points = [transform.to_canvas(origin, 0.0, 0.0)]
    points.extend(transform.to_canvas(origin, float(point[0]), float(point[1])) for point in trajectory)
    if len(points) >= 2:
        draw.line(points, fill=color, width=width)


def _draw_points(
    draw: ImageDraw.ImageDraw,
    transform: "_TrajectoryTransform",
    origin: tuple[int, int],
    trajectory: np.ndarray,
    color: tuple[int, int, int],
    radius: int,
) -> None:
    for point_index, point_xy in enumerate(trajectory):
        point = transform.to_canvas(origin, float(point_xy[0]), float(point_xy[1]))
        draw.ellipse(
            [point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius],
            fill=color,
        )
        if point_index == trajectory.shape[0] - 1:
            draw.text((point[0] + 4, point[1] - 8), "end", fill=color)


def _draw_overlay_endpoints(
    draw: ImageDraw.ImageDraw,
    transform: "_TrajectoryTransform",
    origin: tuple[int, int],
    trajectories: np.ndarray,
) -> None:
    for vocab_index, trajectory in enumerate(trajectories):
        end_point = trajectory[-1]
        point = transform.to_canvas(origin, float(end_point[0]), float(end_point[1]))
        if vocab_index == 0:
            radius = 4
            color = (15, 23, 42, 255)
        else:
            radius = 2
            color = (37, 99, 235, 110)
        draw.ellipse(
            [point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius],
            fill=color,
        )


def _draw_footer(
    draw: ImageDraw.ImageDraw,
    y: int,
    margin: int,
    vocabulary_data: TrajectoryVocabularyData,
    indices: list[int],
    font: ImageFont.ImageFont,
) -> None:
    selected_mse = vocabulary_data.per_trajectory_mse[np.asarray(indices, dtype=np.int64)]
    footer = (
        f"shown={len(indices)}, shown_mse_max={float(np.max(selected_mse)):.6e}, "
        f"shown_mse_mean={float(np.mean(selected_mse)):.6e}. "
        "Blue is stored physical trajectory; red is inverse(normalized * symlog_scale)."
    )
    draw.text((margin, y), footer, fill=(51, 65, 85), font=font)


class _TrajectoryTransform:
    def __init__(self, config: TrajectoryVocabularyViewConfig) -> None:
        self.config = config

    def to_canvas(self, origin: tuple[int, int], ego_x: float, ego_y: float) -> tuple[int, int]:
        x_ratio = (ego_y - self.config.y_min) / (self.config.y_max - self.config.y_min)
        y_ratio = (self.config.x_max - ego_x) / (self.config.x_max - self.config.x_min)
        pixel_x = origin[0] + int(round(x_ratio * (self.config.panel_width - 1)))
        pixel_y = origin[1] + int(round(y_ratio * (self.config.panel_height - 1)))
        return pixel_x, pixel_y

    def in_view(self, ego_x: float, ego_y: float) -> bool:
        return (
            self.config.x_min <= ego_x <= self.config.x_max
            and self.config.y_min <= ego_y <= self.config.y_max
        )


def _require_npz_fields(npz_file: Any, required_fields: tuple[str, ...], npz_path: Path) -> None:
    missing_fields = [field for field in required_fields if field not in npz_file.files]
    if missing_fields:
        raise KeyError(f"轨迹词表缺少字段 {missing_fields}：{npz_path}")


def _load_metadata(npz_file: Any) -> dict[str, Any]:
    if "metadata_json" not in npz_file.files:
        return {}
    metadata_json = str(npz_file["metadata_json"])
    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"metadata_json 不是合法 JSON：{exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata_json 期望为 JSON object，实际为 {type(metadata).__name__}。")
    return metadata


def _validate_vocab_shapes(
    physical_trajectories: np.ndarray,
    normalized_trajectories: np.ndarray,
    npz_path: Path,
) -> None:
    if physical_trajectories.ndim != 3:
        raise ValueError(
            f"trajectory_vocab_m 期望 shape 为 [V, K, D]，实际为 {physical_trajectories.shape}：{npz_path}"
        )
    if normalized_trajectories.shape != physical_trajectories.shape:
        raise ValueError(
            "trajectory_vocab_normalized 必须与 trajectory_vocab_m shape 一致，"
            f"实际为 {normalized_trajectories.shape} vs {physical_trajectories.shape}：{npz_path}"
        )
    if physical_trajectories.shape[-1] != 2:
        raise ValueError(
            f"当前可视化工具期望轨迹点维度为 2，实际为 {physical_trajectories.shape[-1]}：{npz_path}"
        )
    if not np.isfinite(physical_trajectories).all():
        raise ValueError(f"trajectory_vocab_m 包含 NaN 或 Inf：{npz_path}")
    if not np.isfinite(normalized_trajectories).all():
        raise ValueError(f"trajectory_vocab_normalized 包含 NaN 或 Inf：{npz_path}")


def _resolve_visualized_indices(
    vocabulary_size: int,
    indices: list[int] | None,
    max_trajectories: int,
) -> list[int]:
    if vocabulary_size <= 0:
        raise ValueError(f"轨迹词表不能为空，实际 V={vocabulary_size}。")
    if indices is None:
        return list(range(min(vocabulary_size, max_trajectories)))

    resolved_indices: list[int] = []
    for vocab_index in indices:
        if vocab_index < 0:
            vocab_index += vocabulary_size
        if vocab_index < 0 or vocab_index >= vocabulary_size:
            raise IndexError(f"轨迹索引超出范围：{vocab_index}，词表大小为 {vocabulary_size}。")
        if vocab_index not in resolved_indices:
            resolved_indices.append(vocab_index)
    if not resolved_indices:
        raise ValueError("indices 不能为空。")
    return resolved_indices[:max_trajectories]


def _parse_indices(indices_text: str | None) -> list[int] | None:
    if indices_text is None or not indices_text.strip():
        return None
    indices: list[int] = []
    for part in indices_text.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        indices.append(int(stripped))
    return indices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="可视化轨迹词表，并校验归一化轨迹可反求物理轨迹。")
    parser.add_argument("--npz", type=Path, required=True, help="轨迹词表 .npz 文件。")
    parser.add_argument("--output", type=Path, required=True, help="输出 PNG 路径。")
    parser.add_argument(
        "--overlay-output",
        type=Path,
        default=None,
        help="可选输出所有轨迹叠加在同一 BEV 图中的 PNG 路径。",
    )
    parser.add_argument(
        "--indices",
        default=None,
        help="要绘制的词表索引，逗号分隔；默认从 0 开始绘制。",
    )
    parser.add_argument("--max-trajectories", type=int, default=32, help="最多绘制的轨迹数量。")
    parser.add_argument("--columns", type=int, default=4, help="输出图中的面板列数。")
    parser.add_argument("--max-mse", type=float, default=1e-8, help="允许的全局 MSE 上限。")
    parser.add_argument("--bev-x-min", type=float, default=-5.0, help="BEV 前向最小距离。")
    parser.add_argument("--bev-x-max", type=float, default=70.0, help="BEV 前向最大距离。")
    parser.add_argument("--bev-y-min", type=float, default=-35.0, help="BEV 左向最小距离。")
    parser.add_argument("--bev-y-max", type=float, default=35.0, help="BEV 左向最大距离。")
    parser.add_argument("--panel-width", type=int, default=240, help="单轨迹面板宽度。")
    parser.add_argument("--panel-height", type=int, default=240, help="单轨迹面板高度。")
    parser.add_argument("--overlay-width", type=int, default=900, help="所有轨迹叠图宽度。")
    parser.add_argument("--overlay-height", type=int, default=720, help="所有轨迹叠图高度。")
    parser.add_argument(
        "--overlay-reconstructed",
        action="store_true",
        help="在所有轨迹叠图中同时绘制反归一化轨迹；默认只绘制原始物理轨迹。",
    )
    args = parser.parse_args(argv)

    indices = _parse_indices(args.indices)
    view_config = TrajectoryVocabularyViewConfig(
        x_min=args.bev_x_min,
        x_max=args.bev_x_max,
        y_min=args.bev_y_min,
        y_max=args.bev_y_max,
        panel_width=args.panel_width,
        panel_height=args.panel_height,
        columns=args.columns,
        max_mse=args.max_mse,
    )
    vocabulary_data = load_trajectory_vocabulary_npz(args.npz, max_mse=args.max_mse)
    output_path = render_trajectory_vocabulary(
        vocabulary_data,
        args.output,
        config=view_config,
        indices=indices,
        max_trajectories=args.max_trajectories,
    )
    print(output_path)
    if args.overlay_output is not None:
        overlay_output_path = render_trajectory_vocabulary_overlay(
            vocabulary_data,
            args.overlay_output,
            config=view_config,
            width=args.overlay_width,
            height=args.overlay_height,
            draw_reconstructed=args.overlay_reconstructed,
        )
        print(overlay_output_path)
    print(f"global_mse={vocabulary_data.global_mse:.12e}")
    print(f"max_trajectory_mse={float(np.max(vocabulary_data.per_trajectory_mse)):.12e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
