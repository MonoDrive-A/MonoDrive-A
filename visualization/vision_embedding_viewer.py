"""骨干视觉嵌入层 FP32 诊断可视化工具。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from data.b2d_dataset import B2DH5Dataset
from model.vision_embedding import (
    BackboneVisionEmbedding,
    VisionEmbeddingOutput,
    load_vision_embedding_config,
    override_vision_embedding_precision,
)


@dataclass(frozen=True)
class VisionEmbeddingVisualizationData:
    """视觉嵌入诊断图所需的数据。

    Shape:
        `images`: `[8, H, W, 3]`，RGB uint8 图像。
        `dinov3_pca_images`: `[8, H, W, 3]`，DINOv3 Patch 特征 PCA RGB 上采样图。
        `latent_pca_images`: `[T_latent, H, W, 3]`，卷积压缩后 latent 特征 PCA RGB 上采样图。
        `token_norms`: `[T_latent, H_patch, W_patch]`，输出 token L2 norm。
    """

    scene_name: str
    h5_path: Path
    config_path: Path
    sample_index: int
    current_frame_id: int
    input_frame_ids: np.ndarray
    images: np.ndarray
    tokens_shape: tuple[int, ...]
    dinov3_feature_map_shape: tuple[int, ...]
    feature_map_shape: tuple[int, ...]
    patch_grid_shape: tuple[int, int]
    latent_grid_shape: tuple[int, int, int]
    dinov3_dtype: str
    conv_dtype: str
    dinov3_pca_images: np.ndarray
    latent_pca_images: np.ndarray
    token_norms: np.ndarray


def render_vision_embedding_sample(
    h5_path: str | Path,
    sample_index: int,
    config_path: str | Path,
    output_path: str | Path,
    device: str | torch.device = "cpu",
) -> Path:
    """运行视觉嵌入层并渲染 PNG 诊断图。

    可视化固定使用 FP32 精度覆盖配置中的精度字段，但其余结构配置全部来自
    `config_path`。模型调用路径直接复用 `BackboneVisionEmbedding`。
    """

    sample_data = run_vision_embedding_sample(
        h5_path=h5_path,
        sample_index=sample_index,
        config_path=config_path,
        device=device,
    )
    rendered_image = render_visualization(sample_data)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered_image.save(output)
    return output


def run_vision_embedding_sample(
    h5_path: str | Path,
    sample_index: int,
    config_path: str | Path,
    device: str | torch.device = "cpu",
) -> VisionEmbeddingVisualizationData:
    """通过真实实现运行一个 H5 样本的视觉嵌入。"""

    h5_file_path = Path(h5_path)
    if not h5_file_path.is_file():
        raise FileNotFoundError(f"h5_path 必须是文件：{h5_file_path}")
    resolved_config_path = Path(config_path)
    config = load_vision_embedding_config(resolved_config_path)
    fp32_config = override_vision_embedding_precision(
        config,
        dinov3_dtype="float32",
        conv_dtype="float32",
    )
    target_device = torch.device(device)

    dataset = B2DH5Dataset(
        h5_file_path,
        normalize_images=True,
        image_dtype=torch.float32,
        random_target_point=False,
    )
    try:
        sample = dataset[sample_index]
    finally:
        dataset.close()

    images = sample["images"].unsqueeze(0).to(device=target_device)
    model = BackboneVisionEmbedding(fp32_config).to(device=target_device)
    model.eval()
    with torch.no_grad():
        embedding_output = model(images)

    display_images = _images_to_uint8(sample["images"])
    dinov3_pca_images, latent_pca_images, token_norms = _summarize_embedding_output(
        embedding_output,
        output_size=display_images.shape[1:3],
    )
    return VisionEmbeddingVisualizationData(
        scene_name=str(sample["scene_name"]),
        h5_path=h5_file_path,
        config_path=resolved_config_path,
        sample_index=sample_index,
        current_frame_id=int(sample["current_frame_id"]),
        input_frame_ids=_tensor_to_numpy(sample["input_frame_ids"]).astype(np.int32, copy=False),
        images=display_images,
        tokens_shape=tuple(int(dim) for dim in embedding_output.tokens.shape),
        dinov3_feature_map_shape=tuple(int(dim) for dim in embedding_output.dinov3_feature_map.shape),
        feature_map_shape=tuple(int(dim) for dim in embedding_output.feature_map.shape),
        patch_grid_shape=embedding_output.patch_grid_shape,
        latent_grid_shape=embedding_output.latent_grid_shape,
        dinov3_dtype=fp32_config.dinov3_dtype,
        conv_dtype=fp32_config.conv_dtype,
        dinov3_pca_images=dinov3_pca_images,
        latent_pca_images=latent_pca_images,
        token_norms=token_norms,
    )


def render_visualization(sample_data: VisionEmbeddingVisualizationData) -> Image.Image:
    """把输入图像和视觉嵌入统计渲染为一张诊断图。"""

    font = ImageFont.load_default()
    canvas_width = 1460
    canvas_height = 980
    margin = 24
    canvas = Image.new("RGB", (canvas_width, canvas_height), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)

    title = (
        f"vision embedding | scene={sample_data.scene_name} | "
        f"sample={sample_data.sample_index} | frame={sample_data.current_frame_id}"
    )
    draw.text((margin, 12), title, fill=(15, 23, 42), font=font)
    draw.text(
        (margin, 34),
        "DINOv3 preprocessing: tensor mean/std normalization only; PCA maps are upsampled to 288x512.",
        fill=(71, 85, 105),
        font=font,
    )

    current_origin = (margin, 64)
    current_image = Image.fromarray(sample_data.images[-1]).resize((512, 288), Image.Resampling.BILINEAR)
    canvas.paste(current_image, current_origin)
    _draw_panel_title(draw, current_origin, "current rgb_front")

    history_origin = (margin, 376)
    _draw_history_strip(canvas, draw, history_origin, sample_data, font)

    metadata_origin = (margin, 500)
    _draw_metadata_panel(draw, metadata_origin, sample_data, font)

    heatmap_origin = (margin + 560, 64)
    _draw_dinov3_pca_panel(canvas, draw, heatmap_origin, sample_data, font)

    compressed_origin = (margin + 560, 355)
    _draw_latent_pca_panel(canvas, draw, compressed_origin, sample_data, font)

    histogram_origin = (margin + 560, 780)
    _draw_histogram_panel(draw, histogram_origin, sample_data.token_norms, font)
    return canvas


def _draw_history_strip(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    sample_data: VisionEmbeddingVisualizationData,
    font: ImageFont.ImageFont,
) -> None:
    x0, y0 = origin
    draw.text((x0, y0), "history input frames (oldest -> current)", fill=(15, 23, 42), font=font)
    thumbnail_width = 58
    thumbnail_height = 33
    gap = 7
    for frame_index, image_array in enumerate(sample_data.images):
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
        label = str(int(sample_data.input_frame_ids[frame_index]))
        draw.text((thumbnail_x + 2, thumbnail_y + thumbnail_height + 4), label, fill=(51, 65, 85), font=font)


def _draw_metadata_panel(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    sample_data: VisionEmbeddingVisualizationData,
    font: ImageFont.ImageFont,
) -> None:
    x0, y0 = origin
    lines = [
        "embedding metadata",
        f"h5_path = {sample_data.h5_path}",
        f"config_path = {sample_data.config_path}",
        f"precision override = dinov3:{sample_data.dinov3_dtype}, conv:{sample_data.conv_dtype}",
        f"tokens shape = {sample_data.tokens_shape}",
        f"dinov3_feature_map shape = {sample_data.dinov3_feature_map_shape}",
        f"feature_map shape = {sample_data.feature_map_shape}",
        f"patch grid [H,W] = {sample_data.patch_grid_shape}",
        f"latent grid [T,H,W] = {sample_data.latent_grid_shape}",
        "visualized feature = PCA RGB over channels, upsampled to input resolution",
    ]
    draw.rounded_rectangle([x0, y0, x0 + 512, y0 + 320], radius=8, fill=(255, 255, 255), outline=(203, 213, 225))
    for line_index, line in enumerate(lines):
        fill = (15, 23, 42) if line_index == 0 else (51, 65, 85)
        draw.text((x0 + 14, y0 + 14 + line_index * 30), line, fill=fill, font=font)


def _draw_dinov3_pca_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    sample_data: VisionEmbeddingVisualizationData,
    font: ImageFont.ImageFont,
) -> None:
    x0, y0 = origin
    draw.text((x0, y0), "DINOv3 patch feature PCA RGB", fill=(15, 23, 42), font=font)
    heatmap_width = 150
    heatmap_height = 84
    gap_x = 16
    gap_y = 30
    for frame_index, pca_image in enumerate(sample_data.dinov3_pca_images):
        row = frame_index // 4
        col = frame_index % 4
        panel_x = x0 + col * (heatmap_width + gap_x)
        panel_y = y0 + 28 + row * (heatmap_height + gap_y)
        pca_panel = Image.fromarray(pca_image).resize(
            (heatmap_width, heatmap_height),
            Image.Resampling.BILINEAR,
        )
        canvas.paste(pca_panel, (panel_x, panel_y))
        draw.rectangle(
            [panel_x, panel_y, panel_x + heatmap_width, panel_y + heatmap_height],
            outline=(100, 116, 139),
            width=1,
        )
        draw.text((panel_x + 6, panel_y + 6), f"input t={frame_index}", fill=(255, 255, 255), font=font)

    stats_y = y0 + 28 + 2 * (heatmap_height + gap_y)
    stats = "PCA fitted on all DINOv3 patch tokens in this sample; maps are 18x32 -> 288x512."
    draw.text((x0, stats_y), stats, fill=(51, 65, 85), font=font)


def _draw_latent_pca_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    sample_data: VisionEmbeddingVisualizationData,
    font: ImageFont.ImageFont,
) -> None:
    x0, y0 = origin
    draw.text((x0, y0), "compressed latent feature PCA RGB", fill=(15, 23, 42), font=font)
    heatmap_width = 260
    heatmap_height = 146
    gap_x = 24
    gap_y = 34
    for latent_index, pca_image in enumerate(sample_data.latent_pca_images):
        row = latent_index // 2
        col = latent_index % 2
        panel_x = x0 + col * (heatmap_width + gap_x)
        panel_y = y0 + 28 + row * (heatmap_height + gap_y)
        pca_panel = Image.fromarray(pca_image).resize(
            (heatmap_width, heatmap_height),
            Image.Resampling.BILINEAR,
        )
        canvas.paste(pca_panel, (panel_x, panel_y))
        draw.rectangle(
            [panel_x, panel_y, panel_x + heatmap_width, panel_y + heatmap_height],
            outline=(100, 116, 139),
            width=1,
        )
        draw.text((panel_x + 8, panel_y + 8), f"latent t={latent_index}", fill=(255, 255, 255), font=font)

    stats = "PCA fitted on all compressed latent tokens in this sample; maps are 18x32 -> 288x512."
    draw.text((x0, y0 + 28 + 2 * (heatmap_height + gap_y)), stats, fill=(51, 65, 85), font=font)


def _draw_histogram_panel(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    token_norms: np.ndarray,
    font: ImageFont.ImageFont,
) -> None:
    x0, y0 = origin
    width = 700
    height = 180
    draw.rounded_rectangle([x0, y0, x0 + width, y0 + height], radius=8, fill=(255, 255, 255), outline=(203, 213, 225))
    draw.text((x0 + 14, y0 + 12), "token norm histogram", fill=(15, 23, 42), font=font)

    flattened_norms = token_norms.reshape(-1)
    counts, bin_edges = np.histogram(flattened_norms, bins=32)
    max_count = max(int(counts.max()), 1)
    chart_x = x0 + 22
    chart_y = y0 + 44
    chart_width = width - 44
    chart_height = height - 74
    bar_width = chart_width / len(counts)
    for bin_index, count in enumerate(counts):
        bar_height = int(round(chart_height * int(count) / max_count))
        left = int(round(chart_x + bin_index * bar_width))
        right = int(round(chart_x + (bin_index + 1) * bar_width)) - 1
        bottom = chart_y + chart_height
        top = bottom - bar_height
        draw.rectangle([left, top, right, bottom], fill=(37, 99, 235))
    draw.rectangle(
        [chart_x, chart_y, chart_x + chart_width, chart_y + chart_height],
        outline=(100, 116, 139),
        width=1,
    )
    label = (
        f"min={float(flattened_norms.min()):.4f}, "
        f"mean={float(flattened_norms.mean()):.4f}, "
        f"max={float(flattened_norms.max()):.4f}, "
        f"bins=[{float(bin_edges[0]):.3f}, {float(bin_edges[-1]):.3f}]"
    )
    draw.text((x0 + 14, y0 + height - 24), label, fill=(51, 65, 85), font=font)


def _summarize_embedding_output(
    embedding_output: VisionEmbeddingOutput,
    output_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dinov3_feature_map = embedding_output.dinov3_feature_map[0].detach().to(dtype=torch.float32).cpu()
    dinov3_pca_images = _feature_map_to_pca_images(
        dinov3_feature_map,
        output_size=output_size,
    )
    feature_map = embedding_output.feature_map[0].detach().to(dtype=torch.float32).cpu()
    latent_pca_images = _feature_map_to_pca_images(
        feature_map,
        output_size=output_size,
    )
    token_norms = (
        embedding_output.tokens[0]
        .detach()
        .to(dtype=torch.float32)
        .cpu()
        .norm(dim=-1)
        .reshape(embedding_output.latent_grid_shape)
        .numpy()
        .astype(np.float32, copy=False)
    )
    return dinov3_pca_images, latent_pca_images, token_norms


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
    if output_height <= 0 or output_width <= 0:
        raise ValueError(f"output_size 必须为正整数，实际为 {output_size}。")

    channel_count = int(feature_map.shape[0])
    frame_count = int(feature_map.shape[1])
    grid_height = int(feature_map.shape[2])
    grid_width = int(feature_map.shape[3])
    if channel_count <= 0 or frame_count <= 0 or grid_height <= 0 or grid_width <= 0:
        raise ValueError(f"feature_map 各维度必须为正数，实际为 {tuple(feature_map.shape)}。")

    # [C, T, H, W] -> [T*H*W, C]
    samples = feature_map.permute(1, 2, 3, 0).reshape(-1, channel_count)
    samples = samples - samples.mean(dim=0, keepdim=True)
    if not torch.isfinite(samples).all():
        raise ValueError("feature_map 中存在 NaN 或 Inf，无法执行 PCA 可视化。")

    try:
        _u, _s, vh = torch.linalg.svd(samples, full_matrices=False)
    except RuntimeError:
        _u, _s, vh = torch.linalg.svd(samples.cpu(), full_matrices=False)
        samples = samples.cpu()
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


def _draw_panel_title(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    title: str,
) -> None:
    x0, y0 = origin
    draw.rectangle([x0, y0, x0 + 210, y0 + 20], fill=(15, 23, 42))
    draw.text((x0 + 6, y0 + 5), title, fill=(255, 255, 255), font=ImageFont.load_default())


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _default_output_path(h5_path: Path, sample_index: int, output_dir: Path) -> Path:
    return output_dir / f"{h5_path.stem}_vision_embedding_{sample_index:06d}.png"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行骨干视觉嵌入层并导出 FP32 诊断图。")
    parser.add_argument("--h5", type=Path, required=True, help="预处理后的逐场景 H5 文件。")
    parser.add_argument("--sample-index", type=int, default=0, help="样本索引。")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/vision_embedding.toml"),
        help="视觉嵌入配置文件。",
    )
    parser.add_argument("--output", type=Path, default=None, help="输出 PNG 路径。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("visualization/outputs/vision_embedding"),
        help="默认输出目录。",
    )
    parser.add_argument("--device", default="cpu", help="运行设备，例如 cpu 或 cuda。")
    args = parser.parse_args(argv)

    output_path = args.output or _default_output_path(args.h5, args.sample_index, args.output_dir)
    rendered_path = render_vision_embedding_sample(
        h5_path=args.h5,
        sample_index=args.sample_index,
        config_path=args.config,
        output_path=output_path,
        device=args.device,
    )
    print(rendered_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
