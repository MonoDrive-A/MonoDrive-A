"""统一主干每层视觉特征 PCA 可视化工具。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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


@dataclass(frozen=True)
class BackboneFeaturePCAVisualizationData:
    """主干 PCA 诊断图所需的数据。

    Shape:
        `images`: `[8, H, W, 3]`，RGB uint8 图像。
        `layer_pca_images`: `[L, T_latent, H, W, 3]`，每层视觉 Token PCA RGB 上采样图。
        `layer_token_norms`: `[L, T_latent, H_patch, W_patch]`，每层视觉 Token L2 norm。
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


def run_backbone_feature_pca_sample(
    h5_path: str | Path,
    sample_index: int,
    config_path: str | Path,
    project_root: str | Path,
    device: str | torch.device = "cpu",
) -> BackboneFeaturePCAVisualizationData:
    """调用真实统一主干，收集每层视觉 Token PCA 数据。"""

    resolved_project_root = Path(project_root).resolve()
    resolved_h5_path = _resolve_project_path(h5_path, resolved_project_root, "h5_path")
    if not resolved_h5_path.is_file():
        raise FileNotFoundError(f"h5_path 必须是文件：{resolved_h5_path}")
    resolved_config_path = _resolve_project_path(config_path, resolved_project_root, "config_path")

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
    )


def render_backbone_feature_pca_sample(
    h5_path: str | Path,
    sample_index: int,
    config_path: str | Path,
    output_path: str | Path,
    project_root: str | Path,
    device: str | torch.device = "cpu",
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
    )
    rendered_image = render_visualization(visualization_data)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered_image.save(output)
    return output


def render_visualization(data: BackboneFeaturePCAVisualizationData) -> Image.Image:
    """把主干每层 PCA 特征渲染为 PNG。"""

    font = ImageFont.load_default()
    canvas_width = 1780
    canvas_height = 1360
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

    norm_origin = (margin, 1010)
    _draw_norm_summary(draw, norm_origin, data, font)
    return canvas


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
        "--output-dir",
        type=Path,
        default=Path("visualization/outputs/backbone_feature_pca"),
        help="默认输出目录，必须位于项目目录内。",
    )
    parser.add_argument("--device", default="cpu", help="运行设备，例如 cpu 或 cuda。")
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
    )
    print(rendered_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
