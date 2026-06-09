"""统一主干每层视觉特征 PCA 可视化工具。"""

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

from data.b2d_dataset import B2DH5Dataset
from model.backbone import (
    BackboneConfig,
    MonoDriveBackbone,
    MonoDriveBackboneOutput,
    load_backbone_config,
    override_backbone_precision,
)
from model.vision_embedding import (
    load_vision_embedding_config,
    override_vision_embedding_precision,
)

AGENT_CLASS_COLORS = {
    0: (245, 128, 42),
    1: (34, 168, 92),
    2: (224, 64, 72),
}
MAP_CLASS_COLORS = {
    0: (20, 184, 166),
    1: (100, 116, 139),
    2: (14, 165, 233),
}
TRAJECTORY_COLORS = [
    (37, 99, 235),
    (14, 165, 233),
    (22, 163, 74),
    (234, 179, 8),
    (220, 38, 38),
]
DEFAULT_TRAJECTORY_TOP_K = 5
DEFAULT_AGENT_TOP_K = 16
DEFAULT_MAP_TOP_K = 32
DEFAULT_AGENT_CONFIDENCE_THRESHOLD = 0.0
DEFAULT_MAP_CONFIDENCE_THRESHOLD = 0.0


@dataclass(frozen=True)
class ModelOutputVisualizationData:
    """模型输出可视化所需的数据。

    Shape:
        `top_trajectory_points`: `[K, 6, 2]`，ego 坐标系米制轨迹。
        `agent_boxes`: `[A, 6]`，`[x, y, l, w, h, yaw]`。
        `agent_none_scores`: `[A]`，完整类别 softmax 上的 `none` 概率。
        `agent_future_points`: `[A, 6, 2]`，ego 坐标系米制 Agent future。
        `map_points`: `[M, 100, 2]`，ego 坐标系米制 Map 点。
    """

    target_point: np.ndarray
    future_trajectory: np.ndarray
    trajectory_vocab_probabilities: np.ndarray
    top_trajectory_indices: np.ndarray
    top_trajectory_scores: np.ndarray
    top_trajectory_vocab_points: np.ndarray
    top_trajectory_residuals: np.ndarray
    top_trajectory_corrections: np.ndarray
    top_trajectory_points: np.ndarray
    agent_scores: np.ndarray
    agent_class_ids: np.ndarray
    agent_class_labels: np.ndarray
    agent_none_scores: np.ndarray
    agent_boxes: np.ndarray
    agent_mode_ids: np.ndarray
    agent_future_points: np.ndarray
    map_scores: np.ndarray
    map_class_ids: np.ndarray
    map_class_labels: np.ndarray
    map_points: np.ndarray
    model_weight_source: str
    agent_confidence_threshold: float
    map_confidence_threshold: float


@dataclass(frozen=True)
class BackboneFeaturePCAVisualizationData:
    """主干 PCA 诊断图所需的数据。

    Shape:
        `images`: `[8, H, W, 3]`，RGB uint8 图像。
        `layer_pca_images`: `[L, T_latent, H, W, 3]`，每层视觉 Token PCA RGB 上采样图。
        `layer_token_norms`: `[L, T_latent, H_patch, W_patch]`，每层视觉 Token L2 norm。
        `model_outputs`: 检测、轨迹和地图预测的 BEV 可视化数据。
    """

    scene_name: str
    h5_path: Path
    config_path: Path
    sample_index: int
    current_frame_id: int
    input_frame_ids: np.ndarray
    images: np.ndarray
    backbone_config: BackboneConfig
    sequence_shape: tuple[int, ...]
    vision_shape: tuple[int, ...]
    detection_shape: tuple[int, ...]
    trajectory_shape: tuple[int, ...]
    goal_shape: tuple[int, ...]
    detection_agent_logits_shape: tuple[int, ...]
    trajectory_logits_shape: tuple[int, ...]
    latent_grid_shape: tuple[int, int, int]
    backbone_dtype: str
    attention_dtype: str
    dinov3_dtype: str
    conv_dtype: str
    layer_pca_images: np.ndarray
    layer_token_norms: np.ndarray
    model_outputs: ModelOutputVisualizationData


def run_backbone_feature_pca_sample(
    h5_path: str | Path,
    sample_index: int,
    config_path: str | Path,
    project_root: str | Path,
    device: str | torch.device = "cpu",
    checkpoint_path: str | Path | None = None,
    trajectory_top_k: int = DEFAULT_TRAJECTORY_TOP_K,
    agent_top_k: int = DEFAULT_AGENT_TOP_K,
    map_top_k: int = DEFAULT_MAP_TOP_K,
    agent_confidence_threshold: float = DEFAULT_AGENT_CONFIDENCE_THRESHOLD,
    map_confidence_threshold: float = DEFAULT_MAP_CONFIDENCE_THRESHOLD,
) -> BackboneFeaturePCAVisualizationData:
    """调用真实统一主干，收集每层视觉 Token PCA 数据。"""

    _validate_positive_int(trajectory_top_k, "trajectory_top_k")
    _validate_positive_int(agent_top_k, "agent_top_k")
    _validate_positive_int(map_top_k, "map_top_k")
    _validate_confidence_threshold(agent_confidence_threshold, "agent_confidence_threshold")
    _validate_confidence_threshold(map_confidence_threshold, "map_confidence_threshold")

    resolved_project_root = Path(project_root).resolve()
    resolved_h5_path = _resolve_project_path(h5_path, resolved_project_root, "h5_path")
    if not resolved_h5_path.is_file():
        raise FileNotFoundError(f"h5_path 必须是文件：{resolved_h5_path}")
    resolved_config_path = _resolve_project_path(config_path, resolved_project_root, "config_path")
    resolved_checkpoint_path = (
        _resolve_project_path(checkpoint_path, resolved_project_root, "checkpoint_path")
        if checkpoint_path is not None
        else None
    )
    if resolved_checkpoint_path is not None and not resolved_checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint_path 必须是文件：{resolved_checkpoint_path}")

    backbone_config = load_backbone_config(resolved_config_path, resolved_project_root)
    fp32_backbone_config = override_backbone_precision(
        backbone_config,
        backbone_dtype="float32",
        attention_dtype="float32",
    )
    vision_config = load_vision_embedding_config(
        fp32_backbone_config.vision_config_path,
        resolved_project_root,
    )
    fp32_vision_config = override_vision_embedding_precision(
        vision_config,
        dinov3_dtype="float32",
        conv_dtype="float32",
    )

    dataset = B2DH5Dataset(
        resolved_h5_path,
        normalize_images=True,
        image_dtype=torch.float32,
        random_target_point=False,
    )
    try:
        sample = dataset[sample_index]
    finally:
        dataset.close()

    target_device = torch.device(device)
    model = MonoDriveBackbone(
        fp32_backbone_config,
        vision_config=fp32_vision_config,
    ).to(device=target_device)
    model_weight_source = _load_model_weights_if_requested(
        model,
        resolved_checkpoint_path,
        target_device,
    )
    model.eval()

    images = sample["images"].unsqueeze(0).to(device=target_device)
    target_points = sample["target_point"].unsqueeze(0).to(device=target_device)
    ego_motion = sample["ego_motion"].unsqueeze(0).to(device=target_device)
    with torch.no_grad():
        backbone_output = model(
            images=images,
            target_points=target_points,
            ego_motion=ego_motion,
            return_layer_features=True,
        )

    display_images = _images_to_uint8(sample["images"])
    layer_pca_images, layer_token_norms = _summarize_backbone_layers(
        backbone_output,
        output_size=display_images.shape[1:3],
    )
    model_outputs = _summarize_model_outputs(
        backbone_output,
        model,
        sample,
        model_weight_source=model_weight_source,
        trajectory_top_k=trajectory_top_k,
        agent_top_k=agent_top_k,
        map_top_k=map_top_k,
        agent_confidence_threshold=agent_confidence_threshold,
        map_confidence_threshold=map_confidence_threshold,
    )
    return BackboneFeaturePCAVisualizationData(
        scene_name=str(sample["scene_name"]),
        h5_path=resolved_h5_path,
        config_path=resolved_config_path,
        sample_index=sample_index,
        current_frame_id=int(sample["current_frame_id"]),
        input_frame_ids=_tensor_to_numpy(sample["input_frame_ids"]).astype(np.int32, copy=False),
        images=display_images,
        backbone_config=fp32_backbone_config,
        sequence_shape=tuple(int(dim) for dim in backbone_output.sequence_features.shape),
        vision_shape=tuple(int(dim) for dim in backbone_output.vision_features.shape),
        detection_shape=tuple(int(dim) for dim in backbone_output.detection_features.shape),
        trajectory_shape=tuple(int(dim) for dim in backbone_output.trajectory_features.shape),
        goal_shape=tuple(int(dim) for dim in backbone_output.goal_features.shape),
        detection_agent_logits_shape=tuple(
            int(dim) for dim in backbone_output.detection_output.agent_class_logits.shape
        ),
        trajectory_logits_shape=tuple(int(dim) for dim in backbone_output.trajectory_output.logits.shape),
        latent_grid_shape=backbone_output.vision_embedding_output.latent_grid_shape,
        backbone_dtype=fp32_backbone_config.backbone_dtype,
        attention_dtype=fp32_backbone_config.attention_dtype,
        dinov3_dtype=fp32_vision_config.dinov3_dtype,
        conv_dtype=fp32_vision_config.conv_dtype,
        layer_pca_images=layer_pca_images,
        layer_token_norms=layer_token_norms,
        model_outputs=model_outputs,
    )


def render_backbone_feature_pca_sample(
    h5_path: str | Path,
    sample_index: int,
    config_path: str | Path,
    output_path: str | Path,
    project_root: str | Path,
    device: str | torch.device = "cpu",
    checkpoint_path: str | Path | None = None,
    trajectory_top_k: int = DEFAULT_TRAJECTORY_TOP_K,
    agent_top_k: int = DEFAULT_AGENT_TOP_K,
    map_top_k: int = DEFAULT_MAP_TOP_K,
    agent_confidence_threshold: float = DEFAULT_AGENT_CONFIDENCE_THRESHOLD,
    map_confidence_threshold: float = DEFAULT_MAP_CONFIDENCE_THRESHOLD,
) -> Path:
    """运行真实统一主干并导出每层 PCA 诊断 PNG。"""

    resolved_project_root = Path(project_root).resolve()
    output = _resolve_project_path(output_path, resolved_project_root, "output_path")
    visualization_data = run_backbone_feature_pca_sample(
        h5_path=h5_path,
        sample_index=sample_index,
        config_path=config_path,
        project_root=resolved_project_root,
        device=device,
        checkpoint_path=checkpoint_path,
        trajectory_top_k=trajectory_top_k,
        agent_top_k=agent_top_k,
        map_top_k=map_top_k,
        agent_confidence_threshold=agent_confidence_threshold,
        map_confidence_threshold=map_confidence_threshold,
    )
    rendered_image = render_visualization(visualization_data)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered_image.save(output)
    return output


def render_visualization(data: BackboneFeaturePCAVisualizationData) -> Image.Image:
    """把主干每层 PCA 特征渲染为 PNG。"""

    font = ImageFont.load_default()
    canvas_width = 1780
    canvas_height = 1680
    margin = 24
    canvas = Image.new("RGB", (canvas_width, canvas_height), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)

    title = (
        f"backbone layer PCA | scene={data.scene_name} | "
        f"sample={data.sample_index} | frame={data.current_frame_id}"
    )
    draw.text((margin, 14), title, fill=(15, 23, 42), font=font)
    draw.text(
        (margin, 38),
        "The viewer calls MonoDriveBackbone in FP32 and visualizes visual-token PCA after each Transformer layer.",
        fill=(71, 85, 105),
        font=font,
    )

    current_origin = (margin, 70)
    current_image = Image.fromarray(data.images[-1]).resize((512, 288), Image.Resampling.BILINEAR)
    canvas.paste(current_image, current_origin)
    _draw_panel_label(draw, current_origin, "current rgb_front", font)

    history_origin = (margin, 392)
    _draw_history_strip(canvas, draw, history_origin, data, font)

    metadata_origin = (margin, 520)
    _draw_metadata_panel(draw, metadata_origin, data, font)

    layer_grid_origin = (margin + 560, 70)
    _draw_layer_grid(canvas, draw, layer_grid_origin, data, font)

    outputs_origin = (margin, 970)
    _draw_model_outputs_panel(canvas, draw, outputs_origin, data.model_outputs, font)

    norm_origin = (margin + 560, 1290)
    _draw_norm_summary(draw, norm_origin, data, font)

    trajectory_origin = (margin + 1100, 1290)
    _draw_trajectory_diagnostics(draw, trajectory_origin, data.model_outputs, font)
    return canvas


@dataclass(frozen=True)
class _OutputBevConfig:
    """模型输出 BEV 面板配置，坐标为 ego 米制坐标系。"""

    x_min: float = -10.0
    x_max: float = 90.0
    y_min: float = -40.0
    y_max: float = 40.0
    width: int = 512
    height: int = 512
    grid_step: float = 10.0


class _OutputBevTransform:
    def __init__(self, config: _OutputBevConfig) -> None:
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


def _draw_model_outputs_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    outputs: ModelOutputVisualizationData,
    font: ImageFont.ImageFont,
) -> None:
    x0, y0 = origin
    config = _OutputBevConfig()
    transform = _OutputBevTransform(config)
    panel = Image.new("RGB", (config.width, config.height), (255, 255, 255))
    panel_draw = ImageDraw.Draw(panel)

    _draw_output_grid(panel_draw, transform, config, font)
    _draw_output_ego(panel_draw, transform)
    _draw_output_maps(panel_draw, transform, outputs)
    _draw_output_trajectories(panel_draw, transform, outputs)
    _draw_output_agents(panel_draw, transform, outputs)
    _draw_output_target(panel_draw, transform, outputs.target_point, "target", (22, 163, 74), font)

    panel_draw.rectangle([0, 0, config.width - 1, config.height - 1], outline=(100, 116, 139), width=2)
    canvas.paste(panel, origin)
    _draw_panel_label(draw, origin, "model outputs BEV", font)

    legend_y = y0 + config.height + 12
    draw.rounded_rectangle(
        [x0, legend_y, x0 + config.width, legend_y + 172],
        radius=8,
        fill=(255, 255, 255),
        outline=(203, 213, 225),
    )
    draw.text((x0 + 14, legend_y + 12), "output summary", fill=(15, 23, 42), font=font)
    lines = [
        _format_top_trajectory_line(outputs),
        f"agents shown = {len(outputs.agent_scores)}, maps shown = {len(outputs.map_scores)}",
        (
            "detection filters: argmax != none, "
            f"agent p>={outputs.agent_confidence_threshold:.3f}, "
            f"map p>={outputs.map_confidence_threshold:.3f}"
        ),
        "trajectory = vocab_symlog + residual * symlog_scale, then inverse Symlog",
        "agent/map coordinates are inverse Symlog or expm1 decoded for visualization only",
        f"model weights = {outputs.model_weight_source}",
    ]
    for line_index, line in enumerate(lines):
        draw.text((x0 + 14, legend_y + 40 + line_index * 24), line, fill=(51, 65, 85), font=font)


def _draw_output_grid(
    draw: ImageDraw.ImageDraw,
    transform: _OutputBevTransform,
    config: _OutputBevConfig,
    font: ImageFont.ImageFont,
) -> None:
    grid_color = (226, 232, 240)
    axis_color = (100, 116, 139)
    x_value = math.ceil(config.x_min / config.grid_step) * config.grid_step
    while x_value <= config.x_max:
        p0 = transform.to_pixel(x_value, config.y_min)
        p1 = transform.to_pixel(x_value, config.y_max)
        color = axis_color if math.isclose(x_value, 0.0, abs_tol=1e-6) else grid_color
        draw.line([p0, p1], fill=color, width=1)
        draw.text((4, p0[1] - 8), f"x={x_value:.0f}", fill=(100, 116, 139), font=font)
        x_value += config.grid_step

    y_value = math.ceil(config.y_min / config.grid_step) * config.grid_step
    while y_value <= config.y_max:
        p0 = transform.to_pixel(config.x_min, y_value)
        p1 = transform.to_pixel(config.x_max, y_value)
        color = axis_color if math.isclose(y_value, 0.0, abs_tol=1e-6) else grid_color
        draw.line([p0, p1], fill=color, width=1)
        y_value += config.grid_step


def _draw_output_ego(draw: ImageDraw.ImageDraw, transform: _OutputBevTransform) -> None:
    center = transform.to_pixel(0.0, 0.0)
    triangle = [
        transform.to_pixel(2.2, 0.0),
        transform.to_pixel(-1.4, -1.0),
        transform.to_pixel(-1.4, 1.0),
    ]
    draw.polygon(triangle, fill=(15, 23, 42), outline=(15, 23, 42))
    draw.ellipse([center[0] - 3, center[1] - 3, center[0] + 3, center[1] + 3], fill=(255, 255, 255))


def _draw_output_trajectories(
    draw: ImageDraw.ImageDraw,
    transform: _OutputBevTransform,
    outputs: ModelOutputVisualizationData,
) -> None:
    gt_points = [transform.to_pixel(float(point[0]), float(point[1])) for point in outputs.future_trajectory]
    if gt_points:
        draw.line([transform.to_pixel(0.0, 0.0), *gt_points], fill=(15, 23, 42), width=3)
    for trajectory_index, trajectory_points in enumerate(outputs.top_trajectory_points):
        color = TRAJECTORY_COLORS[trajectory_index % len(TRAJECTORY_COLORS)]
        pixels = [transform.to_pixel(float(point[0]), float(point[1])) for point in trajectory_points]
        if pixels:
            draw.line([transform.to_pixel(0.0, 0.0), *pixels], fill=color, width=2)
        for point in pixels:
            draw.ellipse([point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3], fill=color)


def _draw_output_agents(
    draw: ImageDraw.ImageDraw,
    transform: _OutputBevTransform,
    outputs: ModelOutputVisualizationData,
) -> None:
    for agent_index, box in enumerate(outputs.agent_boxes):
        class_id = int(outputs.agent_class_ids[agent_index])
        color = AGENT_CLASS_COLORS.get(class_id, (71, 85, 105))
        future_points = outputs.agent_future_points[agent_index]
        center_xy = box[:2]
        future_pixels = [
            transform.to_pixel(float(point[0]), float(point[1]))
            for point in future_points
            if transform.in_view(float(point[0]), float(point[1]))
        ]
        if future_pixels:
            start = transform.to_pixel(float(center_xy[0]), float(center_xy[1]))
            draw.line([start, *future_pixels], fill=color, width=1)
        _draw_output_agent_box(draw, transform, box, color)
        label_point = transform.to_pixel(float(center_xy[0]), float(center_xy[1]))
        draw.text(
            (label_point[0] + 4, label_point[1] - 8),
            (
                f"{str(outputs.agent_class_labels[agent_index])}:{float(outputs.agent_scores[agent_index]):.2f} "
                f"none:{float(outputs.agent_none_scores[agent_index]):.2f}"
            ),
            fill=color,
        )


def _draw_output_agent_box(
    draw: ImageDraw.ImageDraw,
    transform: _OutputBevTransform,
    box: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    center_x, center_y, length, width, _height, yaw = [float(value) for value in box[:6]]
    if not transform.in_view(center_x, center_y):
        return
    half_length = max(length, 0.2) / 2.0
    half_width = max(width, 0.2) / 2.0
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    corners = []
    for local_x, local_y in (
        (half_length, half_width),
        (half_length, -half_width),
        (-half_length, -half_width),
        (-half_length, half_width),
    ):
        ego_x = center_x + local_x * cos_yaw - local_y * sin_yaw
        ego_y = center_y + local_x * sin_yaw + local_y * cos_yaw
        corners.append(transform.to_pixel(ego_x, ego_y))
    draw.polygon(corners, outline=color)
    draw.line([corners[0], corners[3]], fill=color, width=2)


def _draw_output_maps(
    draw: ImageDraw.ImageDraw,
    transform: _OutputBevTransform,
    outputs: ModelOutputVisualizationData,
) -> None:
    for map_index, points_xy in enumerate(outputs.map_points):
        class_id = int(outputs.map_class_ids[map_index])
        color = MAP_CLASS_COLORS.get(class_id, (100, 116, 139))
        pixels = [
            transform.to_pixel(float(point_xy[0]), float(point_xy[1]))
            for point_xy in points_xy
            if transform.in_view(float(point_xy[0]), float(point_xy[1]))
        ]
        if len(pixels) >= 2:
            draw.line(pixels, fill=color, width=1)
            draw.text(
                (pixels[0][0] + 4, pixels[0][1] - 8),
                f"{str(outputs.map_class_labels[map_index])}:{float(outputs.map_scores[map_index]):.2f}",
                fill=color,
            )


def _draw_output_target(
    draw: ImageDraw.ImageDraw,
    transform: _OutputBevTransform,
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


def _format_top_trajectory_line(outputs: ModelOutputVisualizationData) -> str:
    items = []
    for trajectory_index, score in zip(outputs.top_trajectory_indices, outputs.top_trajectory_scores, strict=False):
        items.append(f"{int(trajectory_index)}:{float(score):.3f}")
    return "top trajectories = " + ", ".join(items)


def _draw_history_strip(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    data: BackboneFeaturePCAVisualizationData,
    font: ImageFont.ImageFont,
) -> None:
    x0, y0 = origin
    draw.text((x0, y0), "history input frames (oldest -> current)", fill=(15, 23, 42), font=font)
    thumbnail_width = 58
    thumbnail_height = 33
    gap = 7
    for frame_index, image_array in enumerate(data.images):
        thumbnail = Image.fromarray(image_array).resize(
            (thumbnail_width, thumbnail_height),
            Image.Resampling.BILINEAR,
        )
        thumbnail_x = x0 + frame_index * (thumbnail_width + gap)
        thumbnail_y = y0 + 22
        canvas.paste(thumbnail, (thumbnail_x, thumbnail_y))
        draw.rectangle(
            [thumbnail_x, thumbnail_y, thumbnail_x + thumbnail_width, thumbnail_y + thumbnail_height],
            outline=(148, 163, 184),
            width=1,
        )
        label = str(int(data.input_frame_ids[frame_index]))
        draw.text((thumbnail_x + 2, thumbnail_y + thumbnail_height + 4), label, fill=(51, 65, 85), font=font)


def _draw_metadata_panel(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    data: BackboneFeaturePCAVisualizationData,
    font: ImageFont.ImageFont,
) -> None:
    x0, y0 = origin
    width = 512
    height = 430
    config = data.backbone_config
    lines = [
        "backbone metadata",
        f"h5_path = {data.h5_path}",
        f"config_path = {data.config_path}",
        f"precision override = backbone:{data.backbone_dtype}, attention:{data.attention_dtype}",
        f"vision precision override = dinov3:{data.dinov3_dtype}, conv:{data.conv_dtype}",
        f"sequence shape = {data.sequence_shape}",
        f"vision/detection/trajectory/goal = {data.vision_shape}, {data.detection_shape}, {data.trajectory_shape}, {data.goal_shape}",
        f"agent logits shape = {data.detection_agent_logits_shape}",
        f"trajectory logits shape = {data.trajectory_logits_shape}",
        f"latent grid [T,H,W] = {data.latent_grid_shape}",
        f"layers = {config.layer_count}, heads = {config.attention_head_count}",
        f"visual RoPE heads = {config.rope_head_count}, theta = {config.rope_theta:g}",
        f"modal FFN 0-based layers = {config.modal_ffn_layer_indices}",
        f"token order = {config.token_order}",
        "PCA is fitted per layer on all visual tokens from the sample.",
    ]
    draw.rounded_rectangle([x0, y0, x0 + width, y0 + height], radius=8, fill=(255, 255, 255), outline=(203, 213, 225))
    for line_index, line in enumerate(lines):
        fill = (15, 23, 42) if line_index == 0 else (51, 65, 85)
        draw.text((x0 + 14, y0 + 14 + line_index * 26), line, fill=fill, font=font)


def _draw_layer_grid(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    data: BackboneFeaturePCAVisualizationData,
    font: ImageFont.ImageFont,
) -> None:
    x0, y0 = origin
    draw.text((x0, y0), "Transformer layer visual-token PCA RGB", fill=(15, 23, 42), font=font)
    panel_width = 360
    panel_height = 240
    gap_x = 28
    gap_y = 48
    frame_width = 164
    frame_height = 92
    frame_gap = 10
    for layer_index, layer_images in enumerate(data.layer_pca_images):
        row = layer_index // 3
        col = layer_index % 3
        panel_x = x0 + col * (panel_width + gap_x)
        panel_y = y0 + 30 + row * (panel_height + gap_y)
        draw.rounded_rectangle(
            [panel_x, panel_y, panel_x + panel_width, panel_y + panel_height],
            radius=8,
            fill=(255, 255, 255),
            outline=(203, 213, 225),
        )
        layer_label = f"layer {layer_index} visual PCA"
        draw.text((panel_x + 12, panel_y + 10), layer_label, fill=(15, 23, 42), font=font)
        for latent_index, pca_image in enumerate(layer_images):
            frame_row = latent_index // 2
            frame_col = latent_index % 2
            frame_x = panel_x + 12 + frame_col * (frame_width + frame_gap)
            frame_y = panel_y + 34 + frame_row * (frame_height + 30)
            frame = Image.fromarray(pca_image).resize(
                (frame_width, frame_height),
                Image.Resampling.BILINEAR,
            )
            canvas.paste(frame, (frame_x, frame_y))
            draw.rectangle(
                [frame_x, frame_y, frame_x + frame_width, frame_y + frame_height],
                outline=(100, 116, 139),
                width=1,
            )
            draw.text((frame_x + 6, frame_y + 6), f"t={latent_index}", fill=(255, 255, 255), font=font)
        layer_norms = data.layer_token_norms[layer_index].reshape(-1)
        stats = (
            f"norm mean={float(layer_norms.mean()):.3f}, "
            f"max={float(layer_norms.max()):.3f}"
        )
        draw.text((panel_x + 12, panel_y + panel_height - 24), stats, fill=(51, 65, 85), font=font)


def _draw_norm_summary(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    data: BackboneFeaturePCAVisualizationData,
    font: ImageFont.ImageFont,
) -> None:
    x0, y0 = origin
    width = 512
    height = 300
    draw.rounded_rectangle([x0, y0, x0 + width, y0 + height], radius=8, fill=(255, 255, 255), outline=(203, 213, 225))
    draw.text((x0 + 14, y0 + 12), "visual token norm by layer", fill=(15, 23, 42), font=font)
    means = data.layer_token_norms.reshape(data.layer_token_norms.shape[0], -1).mean(axis=1)
    max_value = max(float(means.max()), 1e-6)
    chart_x = x0 + 24
    chart_y = y0 + 48
    chart_width = width - 48
    chart_height = 190
    bar_width = chart_width / len(means)
    for layer_index, mean_value in enumerate(means):
        bar_height = int(round(chart_height * float(mean_value) / max_value))
        left = int(round(chart_x + layer_index * bar_width))
        right = int(round(chart_x + (layer_index + 1) * bar_width)) - 2
        bottom = chart_y + chart_height
        top = bottom - bar_height
        draw.rectangle([left, top, right, bottom], fill=(37, 99, 235))
        draw.text((left, bottom + 6), str(layer_index), fill=(51, 65, 85), font=font)
    draw.rectangle([chart_x, chart_y, chart_x + chart_width, chart_y + chart_height], outline=(100, 116, 139), width=1)
    draw.text(
        (x0 + 14, y0 + height - 34),
        f"mean range=[{float(means.min()):.4f}, {float(means.max()):.4f}]",
        fill=(51, 65, 85),
        font=font,
    )


def _draw_trajectory_diagnostics(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    outputs: ModelOutputVisualizationData,
    font: ImageFont.ImageFont,
) -> None:
    x0, y0 = origin
    width = 632
    height = 300
    draw.rounded_rectangle([x0, y0, x0 + width, y0 + height], radius=8, fill=(255, 255, 255), outline=(203, 213, 225))
    draw.text((x0 + 14, y0 + 12), "trajectory vocab probability / residual correction", fill=(15, 23, 42), font=font)

    probabilities = outputs.trajectory_vocab_probabilities.astype(np.float64, copy=False)
    positive_probabilities = probabilities[probabilities > 0.0]
    entropy = float(-(positive_probabilities * np.log(positive_probabilities)).sum())
    normalized_entropy = entropy / max(math.log(max(int(probabilities.size), 2)), 1e-12)
    top_mass = float(outputs.top_trajectory_scores.sum())
    draw.text(
        (x0 + 14, y0 + 36),
        f"vocab={int(probabilities.size)} top_mass={top_mass:.3f} entropy_norm={normalized_entropy:.3f}",
        fill=(51, 65, 85),
        font=font,
    )

    display_count = min(int(outputs.top_trajectory_scores.shape[0]), 5)
    if display_count == 0:
        draw.text((x0 + 14, y0 + 72), "no trajectory candidates", fill=(100, 116, 139), font=font)
        return

    draw.text((x0 + 14, y0 + 66), "rank/id", fill=(100, 116, 139), font=font)
    draw.text((x0 + 90, y0 + 66), "probability", fill=(100, 116, 139), font=font)
    draw.text((x0 + 250, y0 + 66), "top residual correction", fill=(100, 116, 139), font=font)
    max_score = max(float(outputs.top_trajectory_scores.max()), 1e-12)
    for row_index in range(display_count):
        row_y = y0 + 90 + row_index * 40
        color = TRAJECTORY_COLORS[row_index % len(TRAJECTORY_COLORS)]
        trajectory_id = int(outputs.top_trajectory_indices[row_index])
        score = float(outputs.top_trajectory_scores[row_index])
        residual = outputs.top_trajectory_residuals[row_index]
        correction = outputs.top_trajectory_corrections[row_index]
        correction_norms = np.linalg.norm(correction, axis=-1)
        final_correction = correction[-1]
        residual_abs_max = float(np.abs(residual).max())
        correction_mean = float(correction_norms.mean())
        correction_max = float(correction_norms.max())

        draw.rectangle([x0 + 14, row_y + 5, x0 + 24, row_y + 15], fill=color)
        draw.text((x0 + 32, row_y), f"#{row_index + 1} id={trajectory_id}", fill=(15, 23, 42), font=font)

        bar_left = x0 + 90
        bar_top = row_y + 3
        bar_width = 130
        draw.rectangle([bar_left, bar_top, bar_left + bar_width, bar_top + 12], outline=(203, 213, 225))
        fill_width = int(round(bar_width * score / max_score))
        draw.rectangle([bar_left, bar_top, bar_left + fill_width, bar_top + 12], fill=color)
        draw.text((bar_left, row_y + 18), f"p={score:.4f}", fill=(51, 65, 85), font=font)

        draw.text(
            (x0 + 250, row_y),
            f"raw|max={residual_abs_max:.3f} meter mean/max={correction_mean:.2f}/{correction_max:.2f}",
            fill=(51, 65, 85),
            font=font,
        )
        draw.text(
            (x0 + 250, row_y + 18),
            f"final delta=({float(final_correction[0]):+.2f},{float(final_correction[1]):+.2f}) m",
            fill=(51, 65, 85),
            font=font,
        )

    if int(outputs.top_trajectory_scores.shape[0]) > display_count:
        draw.text(
            (x0 + 14, y0 + height - 24),
            f"showing first {display_count}/{int(outputs.top_trajectory_scores.shape[0])} trajectory rows",
            fill=(100, 116, 139),
            font=font,
        )


def _summarize_model_outputs(
    backbone_output: MonoDriveBackboneOutput,
    model: MonoDriveBackbone,
    sample: dict[str, Any],
    model_weight_source: str,
    trajectory_top_k: int = DEFAULT_TRAJECTORY_TOP_K,
    agent_top_k: int = DEFAULT_AGENT_TOP_K,
    map_top_k: int = DEFAULT_MAP_TOP_K,
    agent_confidence_threshold: float = DEFAULT_AGENT_CONFIDENCE_THRESHOLD,
    map_confidence_threshold: float = DEFAULT_MAP_CONFIDENCE_THRESHOLD,
) -> ModelOutputVisualizationData:
    """把模型空间输出转换为 BEV 诊断图使用的米制数据。"""

    (
        trajectory_probabilities,
        trajectory_indices,
        trajectory_scores,
        trajectory_vocab_points,
        trajectory_residuals,
        trajectory_corrections,
        trajectory_points,
    ) = _summarize_trajectory_outputs(
        backbone_output,
        model,
        trajectory_top_k,
    )
    (
        agent_scores,
        agent_class_ids,
        agent_class_labels,
        agent_none_scores,
        agent_boxes,
        agent_mode_ids,
        agent_future_points,
    ) = _summarize_agent_outputs(
        backbone_output,
        model,
        agent_top_k,
        agent_confidence_threshold,
    )
    map_scores, map_class_ids, map_class_labels, map_points = _summarize_map_outputs(
        backbone_output,
        model,
        map_top_k,
        map_confidence_threshold,
    )
    return ModelOutputVisualizationData(
        target_point=_tensor_to_numpy(sample["target_point"]).astype(np.float32, copy=False),
        future_trajectory=_tensor_to_numpy(sample["future_trajectory"]).astype(np.float32, copy=False),
        trajectory_vocab_probabilities=trajectory_probabilities,
        top_trajectory_indices=trajectory_indices,
        top_trajectory_scores=trajectory_scores,
        top_trajectory_vocab_points=trajectory_vocab_points,
        top_trajectory_residuals=trajectory_residuals,
        top_trajectory_corrections=trajectory_corrections,
        top_trajectory_points=trajectory_points,
        agent_scores=agent_scores,
        agent_class_ids=agent_class_ids,
        agent_class_labels=agent_class_labels,
        agent_none_scores=agent_none_scores,
        agent_boxes=agent_boxes,
        agent_mode_ids=agent_mode_ids,
        agent_future_points=agent_future_points,
        map_scores=map_scores,
        map_class_ids=map_class_ids,
        map_class_labels=map_class_labels,
        map_points=map_points,
        model_weight_source=model_weight_source,
        agent_confidence_threshold=float(agent_confidence_threshold),
        map_confidence_threshold=float(map_confidence_threshold),
    )


def _summarize_trajectory_outputs(
    backbone_output: MonoDriveBackboneOutput,
    model: MonoDriveBackbone,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    logits = backbone_output.trajectory_output.logits[0].detach().to(dtype=torch.float32).cpu()
    residuals = backbone_output.trajectory_output.residuals[0].detach().to(dtype=torch.float32).cpu()
    probabilities = torch.softmax(logits, dim=-1)
    selected_count = min(top_k, int(probabilities.numel()))
    scores, indices = torch.topk(probabilities, k=selected_count)
    vocab_symlog = model.vocabulary.trajectory_vocab_symlog.detach().to(dtype=torch.float32).cpu()
    symlog_scale = model.vocabulary.symlog_scale.detach().to(dtype=torch.float32).cpu()
    selected_vocab_symlog = vocab_symlog[indices]
    selected_residuals = residuals[indices]
    selected_symlog = selected_vocab_symlog + selected_residuals * symlog_scale
    vocab_points = _inverse_symlog(selected_vocab_symlog)
    trajectory_points = _inverse_symlog(selected_symlog)
    trajectory_corrections = trajectory_points - vocab_points
    return (
        probabilities.numpy().astype(np.float32, copy=False),
        indices.numpy().astype(np.int32, copy=False),
        scores.numpy().astype(np.float32, copy=False),
        vocab_points.numpy().astype(np.float32, copy=False),
        selected_residuals.numpy().astype(np.float32, copy=False),
        trajectory_corrections.numpy().astype(np.float32, copy=False),
        trajectory_points.numpy().astype(np.float32, copy=False),
    )


def _summarize_agent_outputs(
    backbone_output: MonoDriveBackboneOutput,
    model: MonoDriveBackbone,
    top_k: int,
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    detection_output = backbone_output.detection_output
    class_logits = detection_output.agent_class_logits[0].detach().to(dtype=torch.float32).cpu()
    states = detection_output.agent_states[0].detach().to(dtype=torch.float32).cpu()
    mode_logits = detection_output.agent_mode_logits[0].detach().to(dtype=torch.float32).cpu()
    futures = detection_output.agent_future_trajectories[0].detach().to(dtype=torch.float32).cpu()

    class_probabilities = torch.softmax(class_logits, dim=-1)
    query_scores, class_ids = class_probabilities.max(dim=-1)
    none_class_id = len(model.detection_config.agent_class_names)
    none_scores = class_probabilities[:, none_class_id]
    non_none_indices = torch.nonzero(class_ids != none_class_id, as_tuple=False).flatten()
    if int(non_none_indices.numel()) == 0:
        return (
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=np.float32),
            np.empty((0, 6), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0, model.detection_config.agent_future_points, 2), dtype=np.float32),
        )
    candidate_scores = query_scores[non_none_indices]
    confidence_mask = candidate_scores >= confidence_threshold
    filtered_indices = non_none_indices[confidence_mask]
    filtered_scores = candidate_scores[confidence_mask]
    if int(filtered_scores.numel()) == 0:
        return (
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=np.float32),
            np.empty((0, 6), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0, model.detection_config.agent_future_points, 2), dtype=np.float32),
        )
    selected_count = min(top_k, int(filtered_scores.numel()))
    selected_scores, candidate_order = torch.topk(filtered_scores, k=selected_count)
    selected_indices = filtered_indices[candidate_order]
    selected_class_ids = class_ids[selected_indices]
    class_labels = _select_class_labels(
        (*model.detection_config.agent_class_names, model.detection_config.agent_none_class_name),
        selected_class_ids,
    )

    selected_states = states[selected_indices]
    centers_xy = _inverse_symlog(selected_states[:, 0:2])
    sizes_lwh = torch.expm1(selected_states[:, 2:5]).clamp(min=0.2, max=20.0)
    yaw = torch.atan2(selected_states[:, 5], selected_states[:, 6]).unsqueeze(-1)
    agent_boxes = torch.cat((centers_xy, sizes_lwh, yaw), dim=-1)

    mode_probabilities = torch.softmax(mode_logits[selected_indices], dim=-1)
    mode_ids = torch.argmax(mode_probabilities, dim=-1)
    selected_futures = futures[selected_indices, mode_ids]
    future_displacements = _inverse_symlog(selected_futures)
    future_points = future_displacements + centers_xy[:, None, :]
    selected_none_scores = none_scores[selected_indices]
    return (
        selected_scores.numpy().astype(np.float32, copy=False),
        selected_class_ids.numpy().astype(np.int32, copy=False),
        class_labels,
        selected_none_scores.numpy().astype(np.float32, copy=False),
        agent_boxes.numpy().astype(np.float32, copy=False),
        mode_ids.numpy().astype(np.int32, copy=False),
        future_points.numpy().astype(np.float32, copy=False),
    )


def _summarize_map_outputs(
    backbone_output: MonoDriveBackboneOutput,
    model: MonoDriveBackbone,
    top_k: int,
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    detection_output = backbone_output.detection_output
    class_logits = detection_output.map_class_logits[0].detach().to(dtype=torch.float32).cpu()
    points = detection_output.map_points[0].detach().to(dtype=torch.float32).cpu()
    class_probabilities = torch.softmax(class_logits, dim=-1)
    query_scores, class_ids = class_probabilities.max(dim=-1)
    none_class_id = len(model.detection_config.map_class_names)
    non_none_indices = torch.nonzero(class_ids != none_class_id, as_tuple=False).flatten()
    if int(non_none_indices.numel()) == 0:
        return (
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=object),
            np.empty((0, model.detection_config.map_point_count, 2), dtype=np.float32),
        )
    candidate_scores = query_scores[non_none_indices]
    confidence_mask = candidate_scores >= confidence_threshold
    filtered_indices = non_none_indices[confidence_mask]
    filtered_scores = candidate_scores[confidence_mask]
    if int(filtered_scores.numel()) == 0:
        return (
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=object),
            np.empty((0, model.detection_config.map_point_count, 2), dtype=np.float32),
        )
    selected_count = min(top_k, int(filtered_scores.numel()))
    selected_scores, candidate_order = torch.topk(filtered_scores, k=selected_count)
    selected_indices = filtered_indices[candidate_order]
    selected_class_ids = class_ids[selected_indices]
    class_labels = _select_class_labels(
        (*model.detection_config.map_class_names, model.detection_config.map_none_class_name),
        selected_class_ids,
    )
    selected_points = _inverse_symlog(points[selected_indices])
    return (
        selected_scores.numpy().astype(np.float32, copy=False),
        selected_class_ids.numpy().astype(np.int32, copy=False),
        class_labels,
        selected_points.numpy().astype(np.float32, copy=False),
    )


def _select_class_labels(class_names: tuple[str, ...], class_ids: torch.Tensor) -> np.ndarray:
    labels = [class_names[int(class_id)] for class_id in class_ids]
    return np.asarray(labels, dtype=object)


def _inverse_symlog(values: torch.Tensor) -> torch.Tensor:
    """把 Symlog 空间张量反变换到米制物理空间。"""

    return torch.sign(values) * torch.expm1(torch.abs(values))


def _summarize_backbone_layers(
    backbone_output: MonoDriveBackboneOutput,
    output_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if not backbone_output.layer_vision_features:
        raise ValueError("backbone_output.layer_vision_features 为空，无法执行每层 PCA 可视化。")
    latent_t, grid_h, grid_w = backbone_output.vision_embedding_output.latent_grid_shape
    layer_pca_images = []
    layer_token_norms = []
    for layer_features in backbone_output.layer_vision_features:
        visual_features = layer_features[0].detach().to(dtype=torch.float32).cpu()
        expected_token_count = latent_t * grid_h * grid_w
        if int(visual_features.shape[0]) != expected_token_count:
            raise ValueError(
                "视觉层特征 token 数与 latent grid 不一致："
                f"期望 {expected_token_count}，实际为 {visual_features.shape[0]}。"
            )
        # [T*H*W, D] -> [D, T, H, W]
        feature_map = visual_features.reshape(latent_t, grid_h, grid_w, -1).permute(3, 0, 1, 2)
        layer_pca_images.append(_feature_map_to_pca_images(feature_map, output_size))
        layer_token_norms.append(
            visual_features.norm(dim=-1).reshape(latent_t, grid_h, grid_w).numpy().astype(np.float32, copy=False)
        )
    return np.stack(layer_pca_images, axis=0), np.stack(layer_token_norms, axis=0)


def _feature_map_to_pca_images(
    feature_map: torch.Tensor,
    output_size: tuple[int, int],
) -> np.ndarray:
    """将 `[C, T, H, W]` 特征图 PCA 到 RGB 并上采样到原始图像尺寸。"""

    if feature_map.ndim != 4:
        raise ValueError(
            "feature_map 期望 shape 为 [C, T, H, W]，"
            f"实际为 {tuple(feature_map.shape)}。"
        )
    output_height, output_width = output_size
    channel_count = int(feature_map.shape[0])
    frame_count = int(feature_map.shape[1])
    grid_height = int(feature_map.shape[2])
    grid_width = int(feature_map.shape[3])
    # [C, T, H, W] -> [T*H*W, C]
    samples = feature_map.permute(1, 2, 3, 0).reshape(-1, channel_count)
    samples = samples - samples.mean(dim=0, keepdim=True)
    if not torch.isfinite(samples).all():
        raise ValueError("feature_map 中存在 NaN 或 Inf，无法执行 PCA 可视化。")
    try:
        _u, _s, vh = torch.linalg.svd(samples, full_matrices=False)
    except RuntimeError:
        samples = samples.cpu()
        _u, _s, vh = torch.linalg.svd(samples, full_matrices=False)
    component_count = min(3, int(vh.shape[0]))
    projected = samples @ vh[:component_count].transpose(0, 1).to(device=samples.device)
    if component_count < 3:
        padding = torch.zeros(
            projected.shape[0],
            3 - component_count,
            device=projected.device,
            dtype=projected.dtype,
        )
        projected = torch.cat((projected, padding), dim=1)

    pca_grid = projected.reshape(frame_count, grid_height, grid_width, 3)
    pca_grid = _normalize_pca_grid(pca_grid)
    # [T, H, W, 3] -> [T, 3, H, W] -> [T, H_out, W_out, 3]
    pca_grid = pca_grid.permute(0, 3, 1, 2)
    upsampled = torch.nn.functional.interpolate(
        pca_grid,
        size=(output_height, output_width),
        mode="bilinear",
        align_corners=False,
    )
    upsampled = upsampled.permute(0, 2, 3, 1).clamp(0.0, 1.0)
    return (upsampled.numpy() * 255.0).round().astype(np.uint8)


def _normalize_pca_grid(pca_grid: torch.Tensor) -> torch.Tensor:
    normalized_channels = []
    for channel_index in range(3):
        channel = pca_grid[..., channel_index]
        low = torch.quantile(channel, 0.02)
        high = torch.quantile(channel, 0.98)
        if bool(high <= low):
            normalized_channels.append(torch.zeros_like(channel))
        else:
            normalized_channels.append(((channel - low) / (high - low)).clamp(0.0, 1.0))
    return torch.stack(normalized_channels, dim=-1)


def _images_to_uint8(images: torch.Tensor) -> np.ndarray:
    image_array = images.detach().cpu().clamp(0.0, 1.0)
    # [T, C, H, W] -> [T, H, W, C]
    image_array = image_array.permute(0, 2, 3, 1).numpy()
    return (image_array * 255.0).round().astype(np.uint8)


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_model_weights_if_requested(
    model: MonoDriveBackbone,
    checkpoint_path: Path | None,
    device: torch.device,
) -> str:
    if checkpoint_path is None:
        return "initialized"
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=device)
    state_dict = _extract_model_state_dict(payload, checkpoint_path)
    model.load_state_dict(state_dict, strict=True)
    checkpoint_label = _format_display_path(checkpoint_path)
    step = payload.get("global_step") if isinstance(payload, dict) else None
    if step is None:
        return checkpoint_label
    return f"{checkpoint_label} @ step {int(step)}"


def _extract_model_state_dict(payload: Any, checkpoint_path: Path) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint payload 必须是 dict：{checkpoint_path}")
    candidate: Any
    if "model_state" in payload:
        candidate = payload["model_state"]
    elif "state_dict" in payload:
        candidate = payload["state_dict"]
    elif "model" in payload:
        candidate = payload["model"]
    else:
        candidate = payload
    if not isinstance(candidate, dict) or not candidate:
        raise TypeError(f"checkpoint 未包含可加载的模型 state_dict：{checkpoint_path}")
    if not all(isinstance(key, str) for key in candidate):
        raise TypeError(f"checkpoint state_dict key 必须全部为字符串：{checkpoint_path}")
    if not all(isinstance(value, torch.Tensor) for value in candidate.values()):
        raise TypeError(f"checkpoint state_dict value 必须全部为 torch.Tensor：{checkpoint_path}")
    state_dict = dict(candidate)
    if all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def _format_display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


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


def _validate_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} 必须为整数，实际为 {value!r}。")
    if value <= 0:
        raise ValueError(f"{field_name} 必须为正整数，实际为 {value}。")


def _validate_confidence_threshold(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} 必须为 [0, 1] 区间内的浮点数，实际为 {value!r}。")
    threshold = float(value)
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"{field_name} 必须在 [0, 1] 区间内，实际为 {threshold}。")


def _default_output_path(h5_path: Path, sample_index: int, output_dir: Path) -> Path:
    return output_dir / f"{h5_path.stem}_backbone_feature_pca_{sample_index:06d}.png"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行统一主干并导出每层视觉特征 PCA 诊断图。")
    parser.add_argument("--h5", type=Path, required=True, help="预处理后的逐场景 H5 文件。")
    parser.add_argument("--sample-index", type=int, default=0, help="样本索引。")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/backbone.toml"),
        help="统一主干配置文件。",
    )
    parser.add_argument("--output", type=Path, default=None, help="输出 PNG 路径，必须位于项目目录内。")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="可选模型 checkpoint，支持训练 payload 的 model_state 或直接保存的 state_dict。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("visualization/outputs/backbone_feature_pca"),
        help="默认输出目录，必须位于项目目录内。",
    )
    parser.add_argument("--device", default="cpu", help="运行设备，例如 cpu 或 cuda。")
    parser.add_argument(
        "--trajectory-top-k",
        type=int,
        default=DEFAULT_TRAJECTORY_TOP_K,
        help="模型输出 BEV 面板绘制的轨迹 top-k 数量。",
    )
    parser.add_argument(
        "--agent-top-k",
        type=int,
        default=DEFAULT_AGENT_TOP_K,
        help="模型输出 BEV 面板绘制的 Agent 数量，默认显示完整 16 个查询。",
    )
    parser.add_argument(
        "--map-top-k",
        type=int,
        default=DEFAULT_MAP_TOP_K,
        help="模型输出 BEV 面板绘制的 Map 数量，默认显示完整 32 个查询。",
    )
    parser.add_argument(
        "--agent-confidence-threshold",
        type=float,
        default=DEFAULT_AGENT_CONFIDENCE_THRESHOLD,
        help="Agent 检测 query 的最低 argmax 类别概率，低于该阈值的非 none query 不绘制。",
    )
    parser.add_argument(
        "--map-confidence-threshold",
        type=float,
        default=DEFAULT_MAP_CONFIDENCE_THRESHOLD,
        help="Map 检测 query 的最低 argmax 类别概率，低于该阈值的非 none query 不绘制。",
    )
    args = parser.parse_args(argv)

    project_root = PROJECT_ROOT.resolve()
    output_path = args.output or _default_output_path(args.h5, args.sample_index, args.output_dir)
    rendered_path = render_backbone_feature_pca_sample(
        h5_path=args.h5,
        sample_index=args.sample_index,
        config_path=args.config,
        output_path=output_path,
        project_root=project_root,
        device=args.device,
        checkpoint_path=args.checkpoint,
        trajectory_top_k=args.trajectory_top_k,
        agent_top_k=args.agent_top_k,
        map_top_k=args.map_top_k,
        agent_confidence_threshold=args.agent_confidence_threshold,
        map_confidence_threshold=args.map_confidence_threshold,
    )
    print(rendered_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
