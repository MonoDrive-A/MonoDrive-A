"""目标点嵌入策略可视化工具。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from model.target_point_embedding import (
    TargetPointEmbedding,
    TargetPointEmbeddingConfig,
    load_target_point_embedding_config,
)


@dataclass(frozen=True)
class TargetPointEmbeddingVisualizationData:
    """目标点嵌入诊断图所需的数据。

    Shape:
        `target_point`: `[2]`，ego 坐标系米制 `[x, y]`。
        `grid_xy`: `[H, W, 2]`，目标点嵌入层真实栅格中心。
        `meter_vector_field`: `[H, W, 2]`，目标点到栅格中心的米制向量场。
        `symlog_vector_field`: `[H, W, 2]`，真实送入卷积的 Symlog 向量场。
        `embedded_feature_norm`: `[H_out, W_out]`，三层卷积输出在通道维的 L2 norm。
        `tokens`: `[goal_token_count, hidden_dim]`，目标导航点 Token。
    """

    target_point: np.ndarray
    config_path: Path
    config: TargetPointEmbeddingConfig
    grid_xy: np.ndarray
    meter_vector_field: np.ndarray
    symlog_vector_field: np.ndarray
    embedded_feature_norm: np.ndarray
    tokens: np.ndarray
    token_dtype: str
    parameter_dtypes: tuple[str, ...]
    buffer_dtypes: tuple[str, ...]


@dataclass(frozen=True)
class _BevTransform:
    """ego 坐标到画布像素的变换。"""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    width: int
    height: int

    def to_pixel(self, ego_x: float, ego_y: float) -> tuple[int, int]:
        x_ratio = (ego_y - self.y_min) / (self.y_max - self.y_min)
        y_ratio = (self.x_max - ego_x) / (self.x_max - self.x_min)
        pixel_x = int(round(x_ratio * (self.width - 1)))
        pixel_y = int(round(y_ratio * (self.height - 1)))
        return pixel_x, pixel_y

    def in_view(self, ego_x: float, ego_y: float) -> bool:
        return self.x_min <= ego_x <= self.x_max and self.y_min <= ego_y <= self.y_max


def run_target_point_embedding(
    target_x: float,
    target_y: float,
    config_path: str | Path,
    project_root: str | Path,
    device: str | torch.device = "cpu",
) -> TargetPointEmbeddingVisualizationData:
    """调用真实目标点嵌入层，收集向量场、卷积特征和输出 Token。"""

    if not math.isfinite(target_x) or not math.isfinite(target_y):
        raise ValueError(f"target_x/target_y 必须为有限数，实际为 {target_x}, {target_y}。")

    resolved_project_root = Path(project_root).resolve()
    resolved_config_path = _resolve_project_path(config_path, resolved_project_root, "config_path")
    config = load_target_point_embedding_config(resolved_config_path, resolved_project_root)
    target_device = torch.device(device)
    module = TargetPointEmbedding(config).to(device=target_device)
    module.eval()

    target_points = torch.tensor([[target_x, target_y]], dtype=torch.float32, device=target_device)
    with torch.no_grad():
        tokens = module(target_points)
        # 中间结果来自 model 实现，避免可视化脚本复制目标点嵌入逻辑。
        meter_vector_field = module._build_meter_vector_field(target_points)
        symlog_vector_features = module._build_vector_features(target_points)
        embedded_features = module.downsample(module.conv2(module.conv1(symlog_vector_features)))

    symlog_vector_field = (
        symlog_vector_features[0]
        .detach()
        .to(dtype=torch.float32)
        .cpu()
        .permute(1, 2, 0)
        .numpy()
        .astype(np.float32, copy=False)
    )
    embedded_feature_norm = (
        embedded_features[0].detach().to(dtype=torch.float32).cpu().norm(dim=0).numpy().astype(np.float32, copy=False)
    )
    return TargetPointEmbeddingVisualizationData(
        target_point=np.asarray([target_x, target_y], dtype=np.float32),
        config_path=resolved_config_path,
        config=config,
        grid_xy=module.grid_xy.detach().to(dtype=torch.float32).cpu().numpy().astype(np.float32, copy=False),
        meter_vector_field=meter_vector_field[0]
        .detach()
        .to(dtype=torch.float32)
        .cpu()
        .numpy()
        .astype(np.float32, copy=False),
        symlog_vector_field=symlog_vector_field,
        embedded_feature_norm=embedded_feature_norm,
        tokens=tokens[0].detach().to(dtype=torch.float32).cpu().numpy().astype(np.float32, copy=False),
        token_dtype=str(tokens.dtype),
        parameter_dtypes=tuple(sorted({str(parameter.dtype) for parameter in module.parameters()})),
        buffer_dtypes=tuple(sorted({str(buffer.dtype) for buffer in module.buffers() if buffer.is_floating_point()})),
    )


def render_target_point_embedding(
    target_x: float,
    target_y: float,
    config_path: str | Path,
    output_path: str | Path,
    project_root: str | Path,
    device: str | torch.device = "cpu",
) -> Path:
    """运行真实目标点嵌入层，并把嵌入策略渲染为 PNG。"""

    resolved_project_root = Path(project_root).resolve()
    output = _resolve_project_path(output_path, resolved_project_root, "output_path")
    visualization_data = run_target_point_embedding(
        target_x=target_x,
        target_y=target_y,
        config_path=config_path,
        project_root=resolved_project_root,
        device=device,
    )
    rendered_image = render_visualization(visualization_data)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered_image.save(output)
    return output


def render_visualization(data: TargetPointEmbeddingVisualizationData) -> Image.Image:
    """把目标点向量场、卷积特征和输出 Token 渲染为一张诊断图。"""

    font = ImageFont.load_default()
    canvas_width = 1280
    canvas_height = 850
    margin = 24
    canvas = Image.new("RGB", (canvas_width, canvas_height), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)

    title = (
        "target point embedding | "
        f"target=[{float(data.target_point[0]):.2f}, {float(data.target_point[1]):.2f}] m | "
        f"tokens={tuple(data.tokens.shape)} {data.token_dtype}"
    )
    draw.text((margin, 14), title, fill=(15, 23, 42), font=font)
    draw.text(
        (margin, 38),
        "The viewer calls model.target_point_embedding.TargetPointEmbedding for grid, vector field, convs, and tokens.",
        fill=(71, 85, 105),
        font=font,
    )

    field_origin = (margin, 86)
    _draw_vector_field_panel(canvas, draw, field_origin, data, width=680, height=382, font=font)

    heatmap_origin = (margin + 720, 86)
    _draw_component_heatmaps(canvas, draw, heatmap_origin, data, font)

    metadata_origin = (margin, 506)
    _draw_metadata_panel(draw, metadata_origin, data, font)

    token_origin = (margin + 720, 506)
    _draw_token_panel(draw, token_origin, data, font)
    return canvas


def _draw_vector_field_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    data: TargetPointEmbeddingVisualizationData,
    width: int,
    height: int,
    font: ImageFont.ImageFont,
) -> None:
    config = data.config
    transform = _BevTransform(
        x_min=config.x_min_m,
        x_max=config.x_max_m,
        y_min=config.y_min_m,
        y_max=config.y_max_m,
        width=width,
        height=height,
    )
    panel = Image.new("RGB", (width, height), (255, 255, 255))
    panel_draw = ImageDraw.Draw(panel)
    vector_norm = np.linalg.norm(data.symlog_vector_field, axis=-1)
    max_norm = max(float(vector_norm.max()), 1e-6)

    x_edges = np.linspace(config.x_min_m, config.x_max_m, config.grid_height + 1, dtype=np.float32)
    y_edges = np.linspace(config.y_min_m, config.y_max_m, config.grid_width + 1, dtype=np.float32)
    for row_index in range(config.grid_height):
        for col_index in range(config.grid_width):
            top_left = transform.to_pixel(float(x_edges[row_index + 1]), float(y_edges[col_index]))
            bottom_right = transform.to_pixel(float(x_edges[row_index]), float(y_edges[col_index + 1]))
            color = _sequential_color(float(vector_norm[row_index, col_index]) / max_norm)
            panel_draw.rectangle([top_left, bottom_right], fill=color)

    _draw_grid_lines(panel_draw, transform, x_edges, y_edges)
    _draw_vector_arrows(panel_draw, transform, data.grid_xy, data.symlog_vector_field)
    _draw_ego_marker(panel_draw, transform)
    _draw_target_marker(panel_draw, transform, data.target_point, font)
    panel_draw.rectangle([0, 0, width - 1, height - 1], outline=(100, 116, 139), width=2)

    canvas.paste(panel, origin)
    _draw_panel_label(draw, origin, "18x16 Symlog vector grid: symlog(grid_xy - target_point)", font)
    legend_y = origin[1] + height + 8
    draw.text(
        (origin[0], legend_y),
        f"x range=[{config.x_min_m:.0f},{config.x_max_m:.0f}]m, "
        f"y range=[{config.y_min_m:.0f},{config.y_max_m:.0f}]m, "
        f"symlog vector norm max={max_norm:.2f}",
        fill=(51, 65, 85),
        font=font,
    )


def _draw_component_heatmaps(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    data: TargetPointEmbeddingVisualizationData,
    font: ImageFont.ImageFont,
) -> None:
    x0, y0 = origin
    panel_width = 230
    panel_height = 128
    gap_x = 22
    gap_y = 58
    components = [
        ("symlog vector x", data.symlog_vector_field[..., 0], True),
        ("symlog vector y", data.symlog_vector_field[..., 1], True),
        ("symlog vector norm", np.linalg.norm(data.symlog_vector_field, axis=-1), False),
        ("conv output norm 9x8", data.embedded_feature_norm, False),
    ]
    for item_index, (title, values, diverging) in enumerate(components):
        row = item_index // 2
        col = item_index % 2
        panel_origin = (x0 + col * (panel_width + gap_x), y0 + row * (panel_height + gap_y))
        heatmap = _array_to_heatmap(values, diverging=diverging).resize(
            (panel_width, panel_height),
            Image.Resampling.NEAREST,
        )
        canvas.paste(heatmap, panel_origin)
        draw.rectangle(
            [
                panel_origin[0],
                panel_origin[1],
                panel_origin[0] + panel_width,
                panel_origin[1] + panel_height,
            ],
            outline=(100, 116, 139),
            width=1,
        )
        _draw_panel_label(draw, panel_origin, title, font)
        value_min = float(np.min(values))
        value_max = float(np.max(values))
        draw.text(
            (panel_origin[0], panel_origin[1] + panel_height + 6),
            f"min={value_min:.3f}, max={value_max:.3f}",
            fill=(51, 65, 85),
            font=font,
        )


def _draw_metadata_panel(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    data: TargetPointEmbeddingVisualizationData,
    font: ImageFont.ImageFont,
) -> None:
    x0, y0 = origin
    width = 680
    height = 292
    config = data.config
    lines = [
        "embedding configuration",
        f"config_path = {data.config_path}",
        f"grid = {config.grid_height}x{config.grid_width}, vector_order = {config.vector_order}",
        f"vector_transform = {config.vector_transform}",
        (
            "conv = "
            f"2->{config.feature_channels}, "
            f"k1={config.conv1_kernel_size}, k2={config.conv2_kernel_size}, "
            f"down={config.downsample_kernel_size}/{config.downsample_stride}"
        ),
        f"conv output = [{config.feature_channels}, {config.output_height}, {config.output_width}]",
        f"flattened_dim = {config.flattened_dim}, projected_dim = {config.projected_dim}",
        f"goal tokens = {config.goal_token_count}, hidden_dim = {config.hidden_dim}",
        f"parameter dtypes = {', '.join(data.parameter_dtypes)}",
        f"buffer dtypes = {', '.join(data.buffer_dtypes)}",
        "all intermediate tensors shown here are produced through the model module implementation",
    ]
    draw.rounded_rectangle([x0, y0, x0 + width, y0 + height], radius=8, fill=(255, 255, 255), outline=(203, 213, 225))
    for line_index, line in enumerate(lines):
        fill = (15, 23, 42) if line_index == 0 else (51, 65, 85)
        draw.text((x0 + 14, y0 + 14 + line_index * 26), line, fill=fill, font=font)


def _draw_token_panel(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    data: TargetPointEmbeddingVisualizationData,
    font: ImageFont.ImageFont,
) -> None:
    x0, y0 = origin
    width = 510
    height = 292
    tokens = data.tokens
    token_norms = np.linalg.norm(tokens, axis=1)
    token_means = tokens.mean(axis=1)
    token_stds = tokens.std(axis=1)
    draw.rounded_rectangle([x0, y0, x0 + width, y0 + height], radius=8, fill=(255, 255, 255), outline=(203, 213, 225))
    draw.text((x0 + 14, y0 + 14), "output goal token statistics", fill=(15, 23, 42), font=font)
    for token_index in range(tokens.shape[0]):
        row_y = y0 + 44 + token_index * 44
        draw.text(
            (x0 + 14, row_y),
            (
                f"token {token_index}: norm={float(token_norms[token_index]):.4f}, "
                f"mean={float(token_means[token_index]):.4f}, std={float(token_stds[token_index]):.4f}"
            ),
            fill=(51, 65, 85),
            font=font,
        )

    chart_origin = (x0 + 18, y0 + 148)
    chart_width = width - 36
    chart_height = 112
    draw.rectangle(
        [chart_origin[0], chart_origin[1], chart_origin[0] + chart_width, chart_origin[1] + chart_height],
        outline=(100, 116, 139),
        width=1,
    )
    preview_dim = min(48, tokens.shape[1])
    max_abs = max(float(np.max(np.abs(tokens[:, :preview_dim]))), 1e-6)
    bar_group_width = chart_width / preview_dim
    zero_y = chart_origin[1] + chart_height // 2
    draw.line(
        [(chart_origin[0], zero_y), (chart_origin[0] + chart_width, zero_y)],
        fill=(148, 163, 184),
        width=1,
    )
    for dim_index in range(preview_dim):
        for token_index in range(tokens.shape[0]):
            value = float(tokens[token_index, dim_index])
            bar_height = int(round((chart_height / 2 - 4) * abs(value) / max_abs))
            left = int(round(chart_origin[0] + dim_index * bar_group_width + token_index * bar_group_width / 2))
            right = int(round(left + max(bar_group_width / 2 - 1, 1)))
            if value >= 0:
                top = zero_y - bar_height
                bottom = zero_y
            else:
                top = zero_y
                bottom = zero_y + bar_height
            color = (37, 99, 235) if token_index == 0 else (220, 38, 38)
            draw.rectangle([left, top, right, bottom], fill=color)
    draw.text(
        (chart_origin[0], chart_origin[1] + chart_height + 8),
        f"first {preview_dim} dims, blue=token0, red=token1, max_abs={max_abs:.4f}",
        fill=(51, 65, 85),
        font=font,
    )


def _draw_grid_lines(
    draw: ImageDraw.ImageDraw,
    transform: _BevTransform,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> None:
    for x_value in x_edges:
        p0 = transform.to_pixel(float(x_value), transform.y_min)
        p1 = transform.to_pixel(float(x_value), transform.y_max)
        color = (100, 116, 139) if math.isclose(float(x_value), 0.0, abs_tol=1e-6) else (226, 232, 240)
        draw.line([p0, p1], fill=color, width=1)
    for y_value in y_edges:
        p0 = transform.to_pixel(transform.x_min, float(y_value))
        p1 = transform.to_pixel(transform.x_max, float(y_value))
        color = (100, 116, 139) if math.isclose(float(y_value), 0.0, abs_tol=1e-6) else (226, 232, 240)
        draw.line([p0, p1], fill=color, width=1)


def _draw_vector_arrows(
    draw: ImageDraw.ImageDraw,
    transform: _BevTransform,
    grid_xy: np.ndarray,
    vector_field: np.ndarray,
) -> None:
    cell_x_span = (transform.x_max - transform.x_min) / grid_xy.shape[0]
    cell_y_span = (transform.y_max - transform.y_min) / grid_xy.shape[1]
    arrow_scale = 0.32 * min(cell_x_span, cell_y_span)
    for row_index in range(grid_xy.shape[0]):
        for col_index in range(grid_xy.shape[1]):
            center_x = float(grid_xy[row_index, col_index, 0])
            center_y = float(grid_xy[row_index, col_index, 1])
            vector_x = float(vector_field[row_index, col_index, 0])
            vector_y = float(vector_field[row_index, col_index, 1])
            vector_length = math.hypot(vector_x, vector_y)
            if vector_length < 1e-6:
                continue
            unit_x = vector_x / vector_length
            unit_y = vector_y / vector_length
            start = transform.to_pixel(
                center_x - unit_x * arrow_scale * 0.5,
                center_y - unit_y * arrow_scale * 0.5,
            )
            end = transform.to_pixel(
                center_x + unit_x * arrow_scale * 0.5,
                center_y + unit_y * arrow_scale * 0.5,
            )
            draw.line([start, end], fill=(15, 23, 42), width=1)
            _draw_arrow_head(draw, start, end, (15, 23, 42), size=4)


def _draw_ego_marker(draw: ImageDraw.ImageDraw, transform: _BevTransform) -> None:
    if not transform.in_view(0.0, 0.0):
        return
    triangle = [
        transform.to_pixel(2.0, 0.0),
        transform.to_pixel(-1.2, -0.8),
        transform.to_pixel(-1.2, 0.8),
    ]
    draw.polygon(triangle, fill=(15, 23, 42), outline=(15, 23, 42))


def _draw_target_marker(
    draw: ImageDraw.ImageDraw,
    transform: _BevTransform,
    target_point: np.ndarray,
    font: ImageFont.ImageFont,
) -> None:
    target_x = float(target_point[0])
    target_y = float(target_point[1])
    if not transform.in_view(target_x, target_y):
        return
    point = transform.to_pixel(target_x, target_y)
    size = 8
    color = (22, 163, 74)
    draw.line([point[0] - size, point[1], point[0] + size, point[1]], fill=color, width=3)
    draw.line([point[0], point[1] - size, point[0], point[1] + size], fill=color, width=3)
    draw.text((point[0] + 10, point[1] - 10), "target", fill=color, font=font)


def _draw_arrow_head(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    size: int,
) -> None:
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    left = (
        int(round(end[0] - size * math.cos(angle - math.pi / 6))),
        int(round(end[1] - size * math.sin(angle - math.pi / 6))),
    )
    right = (
        int(round(end[0] - size * math.cos(angle + math.pi / 6))),
        int(round(end[1] - size * math.sin(angle + math.pi / 6))),
    )
    draw.polygon([end, left, right], fill=color)


def _draw_panel_label(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    title: str,
    font: ImageFont.ImageFont,
) -> None:
    x0, y0 = origin
    label_width = min(max(len(title) * 7 + 14, 160), 430)
    draw.rectangle([x0, y0, x0 + label_width, y0 + 20], fill=(15, 23, 42))
    draw.text((x0 + 6, y0 + 5), title, fill=(255, 255, 255), font=font)


def _array_to_heatmap(values: np.ndarray, diverging: bool) -> Image.Image:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"values 必须为 2D 数组，实际 shape 为 {array.shape}。")
    if diverging:
        max_abs = max(float(np.max(np.abs(array))), 1e-6)
        color_array = np.zeros((*array.shape, 3), dtype=np.uint8)
        for index, value in np.ndenumerate(array):
            color_array[index] = _diverging_color(float(value) / max_abs)
    else:
        low = float(np.min(array))
        high = float(np.max(array))
        scale = max(high - low, 1e-6)
        normalized = (array - low) / scale
        color_array = np.zeros((*array.shape, 3), dtype=np.uint8)
        for index, value in np.ndenumerate(normalized):
            color_array[index] = _sequential_color(float(value))
    return Image.fromarray(color_array, mode="RGB")


def _sequential_color(value: float) -> tuple[int, int, int]:
    value = min(max(value, 0.0), 1.0)
    if value < 0.5:
        ratio = value / 0.5
        return (
            int(round(219 + ratio * (96 - 219))),
            int(round(234 + ratio * (165 - 234))),
            int(round(254 + ratio * (250 - 254))),
        )
    ratio = (value - 0.5) / 0.5
    return (
        int(round(96 + ratio * (234 - 96))),
        int(round(165 + ratio * (179 - 165))),
        int(round(250 + ratio * (8 - 250))),
    )


def _diverging_color(value: float) -> tuple[int, int, int]:
    value = min(max(value, -1.0), 1.0)
    if value < 0.0:
        ratio = abs(value)
        return (
            int(round(241 + ratio * (37 - 241))),
            int(round(245 + ratio * (99 - 245))),
            int(round(249 + ratio * (235 - 249))),
        )
    ratio = value
    return (
        int(round(241 + ratio * (220 - 241))),
        int(round(245 + ratio * (38 - 245))),
        int(round(249 + ratio * (38 - 249))),
    )


def _resolve_project_path(path: str | Path, project_root: Path, field_name: str) -> Path:
    raw_path = Path(path)
    resolved_path = raw_path.resolve() if raw_path.is_absolute() else (project_root / raw_path).resolve()
    try:
        resolved_path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} 必须位于项目目录内，项目根目录为 {project_root}，实际为 {resolved_path}。"
        ) from exc
    return resolved_path


def _format_coord_for_filename(value: float) -> str:
    sign = "p" if value >= 0 else "m"
    return sign + f"{abs(value):.2f}".replace(".", "p")


def _default_output_path(target_x: float, target_y: float, output_dir: Path) -> Path:
    x_text = _format_coord_for_filename(target_x)
    y_text = _format_coord_for_filename(target_y)
    return output_dir / f"target_point_embedding_x{x_text}_y{y_text}.png"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="可视化目标点嵌入层的 18x16 栅格、卷积特征和输出 Token。")
    parser.add_argument("--x", type=float, required=True, help="目标点 ego x 坐标，单位 meter，前向为正。")
    parser.add_argument("--y", type=float, required=True, help="目标点 ego y 坐标，单位 meter，左向为正。")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/target_point_embedding.toml"),
        help="目标点嵌入层配置文件。",
    )
    parser.add_argument("--output", type=Path, default=None, help="输出 PNG 路径，必须位于项目目录内。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("visualization/outputs/target_point_embedding"),
        help="默认输出目录，必须位于项目目录内。",
    )
    parser.add_argument("--device", default="cpu", help="运行设备，例如 cpu 或 cuda。")
    args = parser.parse_args(argv)

    project_root = PROJECT_ROOT.resolve()
    output_path = args.output or _default_output_path(args.x, args.y, args.output_dir)
    rendered_path = render_target_point_embedding(
        target_x=args.x,
        target_y=args.y,
        config_path=args.config,
        output_path=output_path,
        project_root=project_root,
        device=args.device,
    )
    print(rendered_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
