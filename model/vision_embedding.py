"""骨干视觉嵌入层。

本文件负责加载冻结的 DINOv3-ViT-B、抽取 Patch 序列，并在 DINOv3 后接
时空 3D 卷积压缩模块。DINOv3 前处理只做张量归一化，不做 resize、crop
或其他会改变图像几何尺寸的操作。
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
import tomllib
from typing import Any, Mapping, NamedTuple

import torch
import torch.nn as nn

from model.residual_block import SpatioTemporalResidualBlock3d


__all__ = [
    "BackboneVisionEmbedding",
    "VisionEmbeddingConfig",
    "VisionEmbeddingOutput",
    "load_vision_embedding_config",
    "override_vision_embedding_precision",
]


SUPPORTED_DTYPE_NAMES = {"float32", "bfloat16"}


@dataclass(frozen=True)
class VisionEmbeddingConfig:
    """骨干视觉嵌入层配置。

    Args:
        model_path: DINOv3 本地模型目录，必须解析到项目目录内。
        local_files_only: 是否禁止从网络下载模型文件。
        trust_remote_code: 是否允许 Transformers 加载模型目录中的远程代码声明。
        freeze_dinov3: 是否冻结 DINOv3 参数并在前向时使用 `no_grad`。
        input_frames: 历史输入帧数。
        input_channels: RGB 通道数。
        input_height: 输入图像高度，单位 pixel。
        input_width: 输入图像宽度，单位 pixel。
        input_value_range: 输入值域，当前仅支持 `zero_one`。
        image_mean: DINOv3 归一化均值。
        image_std: DINOv3 归一化标准差。
        residual_block_count: DINOv3 后接的 3D 残差卷积块数量。
        temporal_compression_after_block: 时间压缩卷积插入在第几个残差块之后。
        temporal_compression_kernel: 时间压缩卷积核，格式为 `[T, H, W]`。
        temporal_compression_stride: 时间压缩卷积步长，格式为 `[T, H, W]`。
        output_hidden_dim: 输出视觉 token 维度。
        dinov3_dtype: DINOv3 前向精度，支持 `float32` / `bfloat16`。
        conv_dtype: 3D 卷积压缩前向精度，支持 `float32` / `bfloat16`。
        token_order: 输出 token 展平顺序，当前仅支持 `time_height_width`。
        expected_token_count: 期望输出视觉 token 数。
    """

    model_path: Path
    local_files_only: bool
    trust_remote_code: bool
    freeze_dinov3: bool
    input_frames: int
    input_channels: int
    input_height: int
    input_width: int
    input_value_range: str
    image_mean: tuple[float, ...]
    image_std: tuple[float, ...]
    residual_block_count: int
    temporal_compression_after_block: int
    temporal_compression_kernel: tuple[int, int, int]
    temporal_compression_stride: tuple[int, int, int]
    output_hidden_dim: int
    dinov3_dtype: str
    conv_dtype: str
    token_order: str
    expected_token_count: int

    def __post_init__(self) -> None:
        if self.input_frames <= 0:
            raise ValueError(f"input_frames 必须为正整数，实际为 {self.input_frames}。")
        if self.input_channels <= 0:
            raise ValueError(f"input_channels 必须为正整数，实际为 {self.input_channels}。")
        if self.input_height <= 0:
            raise ValueError(f"input_height 必须为正整数，实际为 {self.input_height}。")
        if self.input_width <= 0:
            raise ValueError(f"input_width 必须为正整数，实际为 {self.input_width}。")
        if self.input_value_range != "zero_one":
            raise ValueError(
                "input_value_range 当前仅支持 'zero_one'，"
                f"实际为 {self.input_value_range!r}。"
            )
        if len(self.image_mean) != self.input_channels:
            raise ValueError(
                "image_mean 长度必须等于 input_channels，"
                f"实际为 {len(self.image_mean)} 和 {self.input_channels}。"
            )
        if len(self.image_std) != self.input_channels:
            raise ValueError(
                "image_std 长度必须等于 input_channels，"
                f"实际为 {len(self.image_std)} 和 {self.input_channels}。"
            )
        if any(std <= 0.0 for std in self.image_std):
            raise ValueError(f"image_std 每一项都必须为正数，实际为 {self.image_std}。")
        if self.residual_block_count <= 0:
            raise ValueError(
                f"residual_block_count 必须为正整数，实际为 {self.residual_block_count}。"
            )
        if not 1 <= self.temporal_compression_after_block < self.residual_block_count:
            raise ValueError(
                "temporal_compression_after_block 必须位于 "
                f"[1, {self.residual_block_count - 1}]，"
                f"实际为 {self.temporal_compression_after_block}。"
            )
        _validate_3d_int_tuple(
            self.temporal_compression_kernel,
            "temporal_compression_kernel",
        )
        _validate_3d_int_tuple(
            self.temporal_compression_stride,
            "temporal_compression_stride",
        )
        if self.output_hidden_dim <= 0:
            raise ValueError(f"output_hidden_dim 必须为正整数，实际为 {self.output_hidden_dim}。")
        _validate_dtype_name(self.dinov3_dtype, "dinov3_dtype")
        _validate_dtype_name(self.conv_dtype, "conv_dtype")
        if self.token_order != "time_height_width":
            raise ValueError(
                "token_order 当前仅支持 'time_height_width'，"
                f"实际为 {self.token_order!r}。"
            )
        if self.expected_token_count <= 0:
            raise ValueError(
                f"expected_token_count 必须为正整数，实际为 {self.expected_token_count}。"
            )

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        """单样本输入图像 shape，格式为 `[T, C, H, W]`。"""

        return (self.input_frames, self.input_channels, self.input_height, self.input_width)

    @property
    def dinov3_torch_dtype(self) -> torch.dtype:
        """DINOv3 前向使用的 torch dtype。"""

        return _dtype_from_name(self.dinov3_dtype)

    @property
    def conv_torch_dtype(self) -> torch.dtype:
        """3D 卷积压缩前向使用的 torch dtype。"""

        return _dtype_from_name(self.conv_dtype)


class VisionEmbeddingOutput(NamedTuple):
    """骨干视觉嵌入层输出。

    Shape:
        `tokens`: `[B, N, output_hidden_dim]`。
        `dinov3_feature_map`: `[B, C_dino, T, H_patch, W_patch]`。
        `feature_map`: `[B, output_hidden_dim, T_latent, H_patch, W_patch]`。
        `patch_grid_shape`: `(H_patch, W_patch)`。
        `latent_grid_shape`: `(T_latent, H_patch, W_patch)`。
    """

    tokens: torch.Tensor
    dinov3_feature_map: torch.Tensor
    feature_map: torch.Tensor
    patch_grid_shape: tuple[int, int]
    latent_grid_shape: tuple[int, int, int]


class BackboneVisionEmbedding(nn.Module):
    """DINOv3 后接 3D 卷积压缩的骨干视觉嵌入层。

    DINOv3 前只执行 mean/std 归一化，不做 resize。卷积模块严格接在
    DINOv3 Patch 序列之后，用于把 8 帧时序特征压缩为 4 帧 latent。

    Args:
        config: `load_vision_embedding_config` 读取并校验后的视觉嵌入配置。

    Shape:
        输入: `[B, input_frames, input_channels, input_height, input_width]`。
        输出: `VisionEmbeddingOutput.tokens` 为 `[B, expected_token_count, output_hidden_dim]`。
    """

    def __init__(self, config: VisionEmbeddingConfig) -> None:
        super().__init__()
        self.config = config
        self.dinov3 = _load_dinov3_model(config)
        self.dinov3_hidden_dim = _require_dinov3_int_config(
            self.dinov3,
            "hidden_size",
        )
        self.patch_size = _require_dinov3_int_config(self.dinov3, "patch_size")
        if self.patch_size <= 0:
            raise ValueError(f"DINOv3 patch_size 必须为正整数，实际为 {self.patch_size}。")
        if config.input_height % self.patch_size != 0 or config.input_width % self.patch_size != 0:
            raise ValueError(
                "input_height/input_width 必须能被 DINOv3 patch_size 整除，"
                f"实际为 {(config.input_height, config.input_width)} 和 patch_size={self.patch_size}。"
            )

        if config.freeze_dinov3:
            for parameter in self.dinov3.parameters():
                parameter.requires_grad_(False)
            self.dinov3.eval()

        self.register_buffer(
            "image_mean",
            torch.tensor(config.image_mean, dtype=torch.float32).view(1, 1, config.input_channels, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor(config.image_std, dtype=torch.float32).view(1, 1, config.input_channels, 1, 1),
        )
        self.residual_blocks = nn.ModuleList(
            SpatioTemporalResidualBlock3d(self.dinov3_hidden_dim)
            for _ in range(config.residual_block_count)
        )
        self.temporal_compression = nn.Conv3d(
            in_channels=self.dinov3_hidden_dim,
            out_channels=self.dinov3_hidden_dim,
            kernel_size=config.temporal_compression_kernel,
            stride=config.temporal_compression_stride,
        )
        self.output_projection = nn.Conv3d(
            in_channels=self.dinov3_hidden_dim,
            out_channels=config.output_hidden_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def train(self, mode: bool = True) -> "BackboneVisionEmbedding":
        """切换训练模式，并保持冻结 DINOv3 处于 eval。"""

        super().train(mode)
        if self.config.freeze_dinov3:
            self.dinov3.eval()
        return self

    def forward(self, images: torch.Tensor) -> VisionEmbeddingOutput:
        """抽取视觉 token。

        输入图像必须已经由 Dataset 缩放到 `[0, 1]`，本函数只执行 DINOv3
        mean/std 归一化。整个前向过程不会 resize、crop 或 padding 图像。
        """

        self._validate_images(images)
        dinov3_feature_map = self._extract_dinov3_patch_feature_map(images)
        compressed_feature_map = self._compress_feature_map(dinov3_feature_map)
        tokens = self._flatten_feature_map(compressed_feature_map)
        patch_grid_shape = (
            self.config.input_height // self.patch_size,
            self.config.input_width // self.patch_size,
        )
        latent_grid_shape = (
            int(compressed_feature_map.shape[2]),
            int(compressed_feature_map.shape[3]),
            int(compressed_feature_map.shape[4]),
        )
        return VisionEmbeddingOutput(
            tokens=tokens,
            dinov3_feature_map=dinov3_feature_map,
            feature_map=compressed_feature_map,
            patch_grid_shape=patch_grid_shape,
            latent_grid_shape=latent_grid_shape,
        )

    def _validate_images(self, images: torch.Tensor) -> None:
        if not torch.is_floating_point(images):
            raise TypeError(f"images 必须为浮点张量，实际 dtype 为 {images.dtype}。")
        if images.ndim != 5:
            raise ValueError(
                "images 期望 shape 为 [B, T, C, H, W]，"
                f"实际为 {tuple(images.shape)}。"
            )
        expected_shape = self.config.input_shape
        actual_shape = tuple(int(dim) for dim in images.shape[1:])
        if actual_shape != expected_shape:
            raise ValueError(
                "images 的单样本 shape 与配置不一致："
                f"期望 {expected_shape}，实际为 {actual_shape}。"
            )

    def _extract_dinov3_patch_feature_map(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = int(images.shape[0])
        frame_count = self.config.input_frames
        patch_height = self.config.input_height // self.patch_size
        patch_width = self.config.input_width // self.patch_size
        patch_count = patch_height * patch_width

        normalized_images = self._normalize_images(images)
        # [B, T, C, H, W] -> [B * T, C, H, W]
        flat_images = normalized_images.reshape(
            batch_size * frame_count,
            self.config.input_channels,
            self.config.input_height,
            self.config.input_width,
        )

        grad_context = torch.no_grad() if self.config.freeze_dinov3 else nullcontext()
        with grad_context:
            with _precision_context(flat_images.device, self.config.dinov3_torch_dtype):
                dinov3_output = self.dinov3(pixel_values=flat_images)
        last_hidden_state = _extract_last_hidden_state(dinov3_output)
        if last_hidden_state.ndim != 3:
            raise ValueError(
                "DINOv3 last_hidden_state 期望 shape 为 [B*T, N, C]，"
                f"实际为 {tuple(last_hidden_state.shape)}。"
            )
        if int(last_hidden_state.shape[0]) != batch_size * frame_count:
            raise ValueError(
                "DINOv3 输出 batch 维与输入不一致："
                f"期望 {batch_size * frame_count}，实际为 {last_hidden_state.shape[0]}。"
            )
        if int(last_hidden_state.shape[1]) < patch_count:
            raise ValueError(
                "DINOv3 输出 token 数少于输入图像应有的 patch 数："
                f"期望至少 {patch_count}，实际为 {last_hidden_state.shape[1]}。"
            )
        if int(last_hidden_state.shape[2]) != self.dinov3_hidden_dim:
            raise ValueError(
                "DINOv3 输出通道数与配置不一致："
                f"期望 {self.dinov3_hidden_dim}，实际为 {last_hidden_state.shape[2]}。"
            )

        # DINOv3 的 Patch token 位于序列末尾；只取 Patch 序列，丢弃 CLS / register token。
        patch_tokens = last_hidden_state[:, -patch_count:, :]
        # [B*T, H_patch*W_patch, C] -> [B, T, H_patch, W_patch, C]
        patch_grid = patch_tokens.reshape(
            batch_size,
            frame_count,
            patch_height,
            patch_width,
            self.dinov3_hidden_dim,
        )
        # [B, T, H_patch, W_patch, C] -> [B, C, T, H_patch, W_patch]
        return patch_grid.permute(0, 4, 1, 2, 3).contiguous()

    def _normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        image_mean = self.image_mean.to(device=images.device, dtype=torch.float32)
        image_std = self.image_std.to(device=images.device, dtype=torch.float32)
        return (images.to(dtype=torch.float32) - image_mean) / image_std

    def _compress_feature_map(self, feature_map: torch.Tensor) -> torch.Tensor:
        compressed_features = feature_map
        with _precision_context(compressed_features.device, self.config.conv_torch_dtype):
            for block_index, residual_block in enumerate(self.residual_blocks, start=1):
                compressed_features = residual_block(compressed_features)
                if block_index == self.config.temporal_compression_after_block:
                    # [B, C, 8, H, W] -> [B, C, 4, H, W]
                    compressed_features = self.temporal_compression(compressed_features)
            compressed_features = self.output_projection(compressed_features)
        return compressed_features

    def _flatten_feature_map(self, feature_map: torch.Tensor) -> torch.Tensor:
        if self.config.token_order != "time_height_width":
            raise ValueError(f"不支持的 token_order：{self.config.token_order!r}。")
        # [B, C, T, H, W] -> [B, T, H, W, C] -> [B, T*H*W, C]
        tokens = feature_map.permute(0, 2, 3, 4, 1).reshape(
            int(feature_map.shape[0]),
            -1,
            self.config.output_hidden_dim,
        )
        if int(tokens.shape[1]) != self.config.expected_token_count:
            raise ValueError(
                "输出视觉 token 数与配置不一致："
                f"期望 {self.config.expected_token_count}，实际为 {tokens.shape[1]}。"
            )
        return tokens


def load_vision_embedding_config(
    config_path: str | Path,
    project_root: str | Path | None = None,
) -> VisionEmbeddingConfig:
    """读取视觉嵌入 TOML 配置。"""

    resolved_config_path = Path(config_path).resolve()
    resolved_project_root = (
        Path(project_root).resolve() if project_root is not None else resolved_config_path.parent.parent
    )
    with resolved_config_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)

    dinov3_config = _require_table(raw_config, "dinov3")
    input_config = _require_table(raw_config, "input")
    compression_config = _require_table(raw_config, "compression")
    precision_config = _require_table(raw_config, "precision")
    output_config = _require_table(raw_config, "output")

    model_path_text = _require_string(dinov3_config, "model_path")
    raw_model_path = Path(model_path_text)
    if raw_model_path.is_absolute():
        raise ValueError(f"model_path 必须是项目内相对路径，实际为 {raw_model_path}。")
    model_path = (resolved_project_root / raw_model_path).resolve()
    _ensure_project_relative_path(model_path, resolved_project_root, "model_path")

    return VisionEmbeddingConfig(
        model_path=model_path,
        local_files_only=_require_bool(dinov3_config, "local_files_only"),
        trust_remote_code=_require_bool(dinov3_config, "trust_remote_code"),
        freeze_dinov3=_require_bool(dinov3_config, "freeze_dinov3"),
        input_frames=_require_int(input_config, "input_frames"),
        input_channels=_require_int(input_config, "input_channels"),
        input_height=_require_int(input_config, "input_height"),
        input_width=_require_int(input_config, "input_width"),
        input_value_range=_require_string(input_config, "input_value_range"),
        image_mean=_require_float_tuple(input_config, "image_mean"),
        image_std=_require_float_tuple(input_config, "image_std"),
        residual_block_count=_require_int(compression_config, "residual_block_count"),
        temporal_compression_after_block=_require_int(
            compression_config,
            "temporal_compression_after_block",
        ),
        temporal_compression_kernel=_require_3d_int_tuple(
            compression_config,
            "temporal_compression_kernel",
        ),
        temporal_compression_stride=_require_3d_int_tuple(
            compression_config,
            "temporal_compression_stride",
        ),
        output_hidden_dim=_require_int(compression_config, "output_hidden_dim"),
        dinov3_dtype=_require_string(precision_config, "dinov3_dtype"),
        conv_dtype=_require_string(precision_config, "conv_dtype"),
        token_order=_require_string(output_config, "token_order"),
        expected_token_count=_require_int(output_config, "expected_token_count"),
    )


def override_vision_embedding_precision(
    config: VisionEmbeddingConfig,
    dinov3_dtype: str,
    conv_dtype: str,
) -> VisionEmbeddingConfig:
    """返回只替换精度字段的新配置。"""

    _validate_dtype_name(dinov3_dtype, "dinov3_dtype")
    _validate_dtype_name(conv_dtype, "conv_dtype")
    return replace(config, dinov3_dtype=dinov3_dtype, conv_dtype=conv_dtype)


def _load_dinov3_model(config: VisionEmbeddingConfig) -> nn.Module:
    if not config.model_path.is_dir():
        raise FileNotFoundError(f"DINOv3 模型目录不存在：{config.model_path}")
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError("BackboneVisionEmbedding 需要 transformers。请先安装项目依赖。") from exc

    return AutoModel.from_pretrained(
        config.model_path,
        local_files_only=config.local_files_only,
        trust_remote_code=config.trust_remote_code,
    )


def _precision_context(device: torch.device, dtype: torch.dtype) -> Any:
    if dtype == torch.float32:
        return nullcontext()
    if dtype != torch.bfloat16:
        raise ValueError(f"当前仅支持 float32 和 bfloat16，实际为 {dtype}。")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"当前设备不支持 bfloat16 autocast：device.type={device.type!r}。")
    return torch.autocast(device_type=device.type, dtype=dtype)


def _extract_last_hidden_state(dinov3_output: Any) -> torch.Tensor:
    if hasattr(dinov3_output, "last_hidden_state"):
        return dinov3_output.last_hidden_state
    if isinstance(dinov3_output, tuple) and dinov3_output:
        first_output = dinov3_output[0]
        if isinstance(first_output, torch.Tensor):
            return first_output
    raise TypeError("DINOv3 输出缺少 last_hidden_state。")


def _require_dinov3_int_config(model: nn.Module, field_name: str) -> int:
    model_config = getattr(model, "config", None)
    value = getattr(model_config, field_name, None)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"DINOv3 config.{field_name} 必须为整数，实际为 {value!r}。")
    return value


def _dtype_from_name(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.lower()
    if normalized == "float32":
        return torch.float32
    if normalized == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"dtype 仅支持 {sorted(SUPPORTED_DTYPE_NAMES)}，实际为 {dtype_name!r}。")


def _validate_dtype_name(dtype_name: str, field_name: str) -> None:
    if dtype_name not in SUPPORTED_DTYPE_NAMES:
        raise ValueError(
            f"{field_name} 仅支持 {sorted(SUPPORTED_DTYPE_NAMES)}，实际为 {dtype_name!r}。"
        )


def _validate_3d_int_tuple(values: tuple[int, int, int], field_name: str) -> None:
    if len(values) != 3:
        raise ValueError(f"{field_name} 必须包含 3 个整数，实际为 {values}。")
    for index, value in enumerate(values):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{field_name}[{index}] 必须为整数，实际为 {value!r}。")
        if value <= 0:
            raise ValueError(f"{field_name}[{index}] 必须为正整数，实际为 {value}。")


def _require_table(raw_config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw_config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"配置缺少 [{key}] 表。")
    return value


def _require_string(table: Mapping[str, Any], key: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"配置项 {key} 必须为非空字符串，实际为 {value!r}。")
    return value


def _require_bool(table: Mapping[str, Any], key: str) -> bool:
    value = table.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"配置项 {key} 必须为布尔值，实际为 {value!r}。")
    return value


def _require_int(table: Mapping[str, Any], key: str) -> int:
    value = table.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"配置项 {key} 必须为整数，实际为 {value!r}。")
    return value


def _require_float_tuple(table: Mapping[str, Any], key: str) -> tuple[float, ...]:
    value = table.get(key)
    if not isinstance(value, list):
        raise ValueError(f"配置项 {key} 必须为数值列表，实际为 {value!r}。")
    converted_values = []
    for index, item in enumerate(value):
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise ValueError(f"配置项 {key}[{index}] 必须为数值，实际为 {item!r}。")
        converted_values.append(float(item))
    return tuple(converted_values)


def _require_3d_int_tuple(table: Mapping[str, Any], key: str) -> tuple[int, int, int]:
    value = table.get(key)
    if not isinstance(value, list):
        raise ValueError(f"配置项 {key} 必须为整数列表，实际为 {value!r}。")
    converted_values = []
    for index, item in enumerate(value):
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(f"配置项 {key}[{index}] 必须为整数，实际为 {item!r}。")
        converted_values.append(item)
    values_tuple = tuple(converted_values)
    if len(values_tuple) != 3:
        raise ValueError(f"配置项 {key} 必须包含 3 个整数，实际为 {values_tuple}。")
    return values_tuple


def _ensure_project_relative_path(path: Path, project_root: Path, config_key: str) -> None:
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(
            f"{config_key} 必须解析到项目目录内，项目根目录为 {project_root}，实际为 {path}。"
        ) from exc
