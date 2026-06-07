"""目标点嵌入层。

本文件负责把 ego 坐标系下的单个目标点转换为目标导航点 Token。目标点先
转换为覆盖车辆前后和左右范围的栅格向量场，再通过三层卷积下采样，最后
展平并用线性层投影为 2 个目标导航点 Token。
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any, Mapping

import torch
import torch.nn as nn


__all__ = [
    "TargetPointEmbedding",
    "TargetPointEmbeddingConfig",
    "load_target_point_embedding_config",
]


SUPPORTED_VECTOR_ORDERS = {"grid_minus_target", "target_minus_grid"}
SUPPORTED_VECTOR_TRANSFORMS = {"symlog"}
SUPPORTED_FLATTEN_ORDERS = {"channel_height_width"}
SUPPORTED_DTYPE_NAMES = {"float32"}


@dataclass(frozen=True)
class TargetPointEmbeddingConfig:
    """目标点嵌入层配置。

    Args:
        coordinate_dim: 目标点坐标维度，当前为 ego XY。
        grid_height: 栅格高度，对应 x 前后方向。
        grid_width: 栅格宽度，对应 y 左右方向。
        x_min_m: 栅格后向边界，单位 meter。
        x_max_m: 栅格前向边界，单位 meter。
        y_min_m: 栅格右向边界，单位 meter。
        y_max_m: 栅格左向边界，单位 meter。
        vector_order: 栅格向量场方向。
        vector_transform: 送入卷积前的向量变换，当前为 `symlog`。
        feature_channels: 卷积中间通道数。
        conv1_kernel_size: 第一层卷积核 `[H, W]`。
        conv1_stride: 第一层卷积步长 `[H, W]`。
        conv1_padding: 第一层卷积 padding `[H, W]`。
        conv2_kernel_size: 第二层卷积核 `[H, W]`。
        conv2_stride: 第二层卷积步长 `[H, W]`。
        conv2_padding: 第二层卷积 padding `[H, W]`。
        downsample_kernel_size: 下采样卷积核 `[H, W]`。
        downsample_stride: 下采样卷积步长 `[H, W]`。
        downsample_padding: 下采样卷积 padding `[H, W]`。
        output_height: 下采样后的栅格高度。
        output_width: 下采样后的栅格宽度。
        goal_token_count: 输出目标导航点 Token 数。
        hidden_dim: 每个目标导航点 Token 的特征维度。
        flatten_order: 卷积特征展平顺序。
        dtype: 目标点嵌入层强制运行精度，当前仅支持 `float32`。
    """

    coordinate_dim: int
    grid_height: int
    grid_width: int
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    vector_order: str
    vector_transform: str
    feature_channels: int
    conv1_kernel_size: tuple[int, int]
    conv1_stride: tuple[int, int]
    conv1_padding: tuple[int, int]
    conv2_kernel_size: tuple[int, int]
    conv2_stride: tuple[int, int]
    conv2_padding: tuple[int, int]
    downsample_kernel_size: tuple[int, int]
    downsample_stride: tuple[int, int]
    downsample_padding: tuple[int, int]
    output_height: int
    output_width: int
    goal_token_count: int
    hidden_dim: int
    flatten_order: str
    dtype: str

    def __post_init__(self) -> None:
        if self.coordinate_dim != 2:
            raise ValueError(
                "coordinate_dim 必须为 2，以表示 ego 坐标系 [x, y]，"
                f"实际为 {self.coordinate_dim}。"
            )
        for field_name in (
            "grid_height",
            "grid_width",
            "feature_channels",
            "output_height",
            "output_width",
            "goal_token_count",
            "hidden_dim",
        ):
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} 必须为正整数，实际为 {value}。")
        if self.x_min_m >= self.x_max_m:
            raise ValueError(
                "x_min_m 必须小于 x_max_m，"
                f"实际为 {self.x_min_m} 和 {self.x_max_m}。"
            )
        if self.y_min_m >= self.y_max_m:
            raise ValueError(
                "y_min_m 必须小于 y_max_m，"
                f"实际为 {self.y_min_m} 和 {self.y_max_m}。"
            )
        if self.vector_order not in SUPPORTED_VECTOR_ORDERS:
            raise ValueError(
                f"vector_order 仅支持 {sorted(SUPPORTED_VECTOR_ORDERS)}，"
                f"实际为 {self.vector_order!r}。"
            )
        if self.vector_transform not in SUPPORTED_VECTOR_TRANSFORMS:
            raise ValueError(
                f"vector_transform 仅支持 {sorted(SUPPORTED_VECTOR_TRANSFORMS)}，"
                f"实际为 {self.vector_transform!r}。"
            )
        if self.flatten_order not in SUPPORTED_FLATTEN_ORDERS:
            raise ValueError(
                f"flatten_order 仅支持 {sorted(SUPPORTED_FLATTEN_ORDERS)}，"
                f"实际为 {self.flatten_order!r}。"
            )
        if self.dtype not in SUPPORTED_DTYPE_NAMES:
            raise ValueError(f"dtype 仅支持 {sorted(SUPPORTED_DTYPE_NAMES)}，实际为 {self.dtype!r}。")

        for field_name in (
            "conv1_kernel_size",
            "conv1_stride",
            "conv1_padding",
            "conv2_kernel_size",
            "conv2_stride",
            "conv2_padding",
            "downsample_kernel_size",
            "downsample_stride",
            "downsample_padding",
        ):
            _validate_2d_int_tuple(getattr(self, field_name), field_name)
        for field_name in ("conv1_padding", "conv2_padding", "downsample_padding"):
            values = getattr(self, field_name)
            if values[0] < 0 or values[1] < 0:
                raise ValueError(f"{field_name} 不能为负数，实际为 {values}。")

        conv1_shape = _conv2d_output_shape(
            (self.grid_height, self.grid_width),
            self.conv1_kernel_size,
            self.conv1_stride,
            self.conv1_padding,
        )
        conv2_shape = _conv2d_output_shape(
            conv1_shape,
            self.conv2_kernel_size,
            self.conv2_stride,
            self.conv2_padding,
        )
        downsample_shape = _conv2d_output_shape(
            conv2_shape,
            self.downsample_kernel_size,
            self.downsample_stride,
            self.downsample_padding,
        )
        expected_shape = (self.output_height, self.output_width)
        if downsample_shape != expected_shape:
            raise ValueError(
                "卷积配置推导出的输出空间尺寸与 output_height/output_width 不一致："
                f"推导为 {downsample_shape}，配置为 {expected_shape}。"
            )

    @property
    def flattened_dim(self) -> int:
        """卷积输出展平后的特征维度。"""

        return self.feature_channels * self.output_height * self.output_width

    @property
    def projected_dim(self) -> int:
        """线性层输出维度，随后 reshape 为目标导航点 Token。"""

        return self.goal_token_count * self.hidden_dim


class TargetPointEmbedding(nn.Module):
    """将目标点编码为目标导航点 Token。

    Args:
        config: `load_target_point_embedding_config` 读取并校验后的配置。

    Shape:
        输入: `[B, 2]`，ego 坐标系目标点，单位 meter，坐标为 `[x, y]`。
        输出: `[B, goal_token_count, hidden_dim]`。
    """

    def __init__(self, config: TargetPointEmbeddingConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("grid_xy", _build_grid_xy(config))
        self.conv1 = nn.Conv2d(
            in_channels=config.coordinate_dim,
            out_channels=config.feature_channels,
            kernel_size=config.conv1_kernel_size,
            stride=config.conv1_stride,
            padding=config.conv1_padding,
        )
        self.conv2 = nn.Conv2d(
            in_channels=config.feature_channels,
            out_channels=config.feature_channels,
            kernel_size=config.conv2_kernel_size,
            stride=config.conv2_stride,
            padding=config.conv2_padding,
        )
        self.downsample = nn.Conv2d(
            in_channels=config.feature_channels,
            out_channels=config.feature_channels,
            kernel_size=config.downsample_kernel_size,
            stride=config.downsample_stride,
            padding=config.downsample_padding,
        )
        self.output_projection = nn.Linear(config.flattened_dim, config.projected_dim)
        _force_floating_tensors_to_float32(self)

    def _apply(self, fn: Any) -> "TargetPointEmbedding":
        super()._apply(fn)
        _force_floating_tensors_to_float32(self)
        return self

    def forward(self, target_points: torch.Tensor) -> torch.Tensor:
        """输出目标导航点 Token。

        `target_points` 必须是 ego 坐标系下的米制 `[x, y]`，shape 为
        `[B, 2]`。目标点到栅格中心的米制向量先做 Symlog 变换，再进入卷积。
        内部向量场、卷积和输出投影都强制在 FP32 中执行。
        """

        self._validate_target_points(target_points)
        with _disabled_autocast(self.grid_xy):
            target_points_fp32 = target_points.to(dtype=torch.float32)
            vector_features = self._build_vector_features(target_points_fp32)
            embedded_features = self.downsample(self.conv2(self.conv1(vector_features)))
            self._validate_embedded_features(embedded_features)
            flattened_features = self._flatten_embedded_features(embedded_features)
            projected_tokens = self.output_projection(flattened_features)
            return projected_tokens.reshape(
                int(target_points.shape[0]),
                self.config.goal_token_count,
                self.config.hidden_dim,
            )

    def _validate_target_points(self, target_points: torch.Tensor) -> None:
        if not torch.is_floating_point(target_points):
            raise TypeError(f"target_points 必须为浮点张量，实际 dtype 为 {target_points.dtype}。")
        if target_points.ndim != 2:
            raise ValueError(
                "target_points 期望 shape 为 [B, coordinate_dim]，"
                f"实际为 {tuple(target_points.shape)}。"
            )
        if int(target_points.shape[1]) != self.config.coordinate_dim:
            raise ValueError(
                "target_points 最后一维必须等于 coordinate_dim，"
                f"期望 {self.config.coordinate_dim}，实际为 {target_points.shape[1]}。"
            )

    def _build_vector_features(self, target_points_fp32: torch.Tensor) -> torch.Tensor:
        meter_vector_field = self._build_meter_vector_field(target_points_fp32)
        normalized_vector_field = self._normalize_vector_field(meter_vector_field)
        # [B, H, W, 2] -> [B, 2, H, W]
        return normalized_vector_field.permute(0, 3, 1, 2).contiguous()

    def _build_meter_vector_field(self, target_points_fp32: torch.Tensor) -> torch.Tensor:
        """构造目标点到栅格中心的米制向量场。

        Shape:
            输入: `[B, 2]`。
            输出: `[B, H, W, 2]`，单位 meter。
        """

        grid_xy = self.grid_xy.to(device=target_points_fp32.device, dtype=torch.float32)
        # [H, W, 2] 和 [B, 2] -> [B, H, W, 2]。
        if self.config.vector_order == "grid_minus_target":
            return grid_xy.unsqueeze(0) - target_points_fp32[:, None, None, :]
        elif self.config.vector_order == "target_minus_grid":
            return target_points_fp32[:, None, None, :] - grid_xy.unsqueeze(0)
        else:
            raise ValueError(f"不支持的 vector_order：{self.config.vector_order!r}。")

    def _normalize_vector_field(self, meter_vector_field: torch.Tensor) -> torch.Tensor:
        """把米制向量场变换到模型输入数值空间。

        Shape:
            输入: `[B, H, W, 2]`，单位 meter。
            输出: `[B, H, W, 2]`，Symlog 空间。
        """

        if self.config.vector_transform == "symlog":
            return torch.sign(meter_vector_field) * torch.log1p(torch.abs(meter_vector_field))
        raise ValueError(f"不支持的 vector_transform：{self.config.vector_transform!r}。")

    def _validate_embedded_features(self, embedded_features: torch.Tensor) -> None:
        expected_shape = (
            self.config.feature_channels,
            self.config.output_height,
            self.config.output_width,
        )
        actual_shape = tuple(int(dim) for dim in embedded_features.shape[1:])
        if actual_shape != expected_shape:
            raise ValueError(
                "目标点卷积输出 shape 与配置不一致："
                f"期望 [B, {expected_shape[0]}, {expected_shape[1]}, {expected_shape[2]}]，"
                f"实际为 {tuple(embedded_features.shape)}。"
            )

    def _flatten_embedded_features(self, embedded_features: torch.Tensor) -> torch.Tensor:
        if self.config.flatten_order != "channel_height_width":
            raise ValueError(f"不支持的 flatten_order：{self.config.flatten_order!r}。")
        # [B, C, H, W] -> [B, C * H * W]
        return embedded_features.reshape(int(embedded_features.shape[0]), -1)


def load_target_point_embedding_config(
    config_path: str | Path,
    project_root: str | Path | None = None,
) -> TargetPointEmbeddingConfig:
    """读取目标点嵌入层 TOML 配置。"""

    resolved_config_path = Path(config_path).resolve()
    resolved_project_root = (
        Path(project_root).resolve() if project_root is not None else resolved_config_path.parent.parent
    )
    _ensure_project_relative_path(resolved_config_path, resolved_project_root, "config_path")
    with resolved_config_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)

    target_point_config = _require_table(raw_config, "target_point")
    grid_config = _require_table(raw_config, "grid")
    normalization_config = _require_table(raw_config, "normalization")
    convolution_config = _require_table(raw_config, "convolution")
    output_config = _require_table(raw_config, "output")
    precision_config = _require_table(raw_config, "precision")

    return TargetPointEmbeddingConfig(
        coordinate_dim=_require_int(target_point_config, "coordinate_dim"),
        grid_height=_require_int(grid_config, "height"),
        grid_width=_require_int(grid_config, "width"),
        x_min_m=_require_float(grid_config, "x_min_m"),
        x_max_m=_require_float(grid_config, "x_max_m"),
        y_min_m=_require_float(grid_config, "y_min_m"),
        y_max_m=_require_float(grid_config, "y_max_m"),
        vector_order=_require_string(grid_config, "vector_order"),
        vector_transform=_require_string(normalization_config, "vector_transform"),
        feature_channels=_require_int(convolution_config, "feature_channels"),
        conv1_kernel_size=_require_2d_int_tuple(convolution_config, "conv1_kernel_size"),
        conv1_stride=_require_2d_int_tuple(convolution_config, "conv1_stride"),
        conv1_padding=_require_2d_int_tuple(convolution_config, "conv1_padding"),
        conv2_kernel_size=_require_2d_int_tuple(convolution_config, "conv2_kernel_size"),
        conv2_stride=_require_2d_int_tuple(convolution_config, "conv2_stride"),
        conv2_padding=_require_2d_int_tuple(convolution_config, "conv2_padding"),
        downsample_kernel_size=_require_2d_int_tuple(
            convolution_config,
            "downsample_kernel_size",
        ),
        downsample_stride=_require_2d_int_tuple(convolution_config, "downsample_stride"),
        downsample_padding=_require_2d_int_tuple(convolution_config, "downsample_padding"),
        output_height=_require_int(convolution_config, "output_height"),
        output_width=_require_int(convolution_config, "output_width"),
        goal_token_count=_require_int(output_config, "goal_token_count"),
        hidden_dim=_require_int(output_config, "hidden_dim"),
        flatten_order=_require_string(output_config, "flatten_order"),
        dtype=_require_string(precision_config, "dtype"),
    )


def _build_grid_xy(config: TargetPointEmbeddingConfig) -> torch.Tensor:
    x_cell_size = (config.x_max_m - config.x_min_m) / float(config.grid_height)
    y_cell_size = (config.y_max_m - config.y_min_m) / float(config.grid_width)
    x_positions = config.x_min_m + (torch.arange(config.grid_height, dtype=torch.float32) + 0.5) * x_cell_size
    y_positions = config.y_min_m + (torch.arange(config.grid_width, dtype=torch.float32) + 0.5) * y_cell_size
    grid_x, grid_y = torch.meshgrid(x_positions, y_positions, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1)


def _disabled_autocast(reference_tensor: torch.Tensor) -> Any:
    """根据参考张量设备构造禁用 autocast 的上下文。"""

    if reference_tensor.device.type == "meta":
        return nullcontext()
    try:
        return torch.autocast(device_type=reference_tensor.device.type, enabled=False)
    except (RuntimeError, ValueError):
        return nullcontext()


def _force_floating_tensors_to_float32(module: nn.Module) -> None:
    """将模块内所有浮点参数、buffer 和已有梯度恢复为 FP32。"""

    with torch.no_grad():
        for parameter in module.parameters(recurse=True):
            if parameter.is_floating_point() and parameter.dtype != torch.float32:
                parameter.data = parameter.data.to(dtype=torch.float32)
            if parameter.grad is not None and parameter.grad.is_floating_point():
                parameter.grad.data = parameter.grad.data.to(dtype=torch.float32)

        for buffer in module.buffers(recurse=True):
            if buffer.is_floating_point() and buffer.dtype != torch.float32:
                buffer.data = buffer.data.to(dtype=torch.float32)


def _conv2d_output_shape(
    input_shape: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> tuple[int, int]:
    output_height = (input_shape[0] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
    output_width = (input_shape[1] + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
    if output_height <= 0 or output_width <= 0:
        raise ValueError(
            "卷积配置产生了非正输出尺寸："
            f"input_shape={input_shape}, kernel_size={kernel_size}, "
            f"stride={stride}, padding={padding}, output_shape={(output_height, output_width)}。"
        )
    return (output_height, output_width)


def _validate_2d_int_tuple(values: tuple[int, int], field_name: str) -> None:
    if len(values) != 2:
        raise ValueError(f"{field_name} 必须包含 2 个整数，实际为 {values}。")
    for index, value in enumerate(values):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{field_name}[{index}] 必须为整数，实际为 {value!r}。")
        if value <= 0 and "padding" not in field_name:
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


def _require_int(table: Mapping[str, Any], key: str) -> int:
    value = table.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"配置项 {key} 必须为整数，实际为 {value!r}。")
    return value


def _require_float(table: Mapping[str, Any], key: str) -> float:
    value = table.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"配置项 {key} 必须为数值，实际为 {value!r}。")
    return float(value)


def _require_2d_int_tuple(table: Mapping[str, Any], key: str) -> tuple[int, int]:
    value = table.get(key)
    if not isinstance(value, list):
        raise ValueError(f"配置项 {key} 必须为整数列表，实际为 {value!r}。")
    converted_values = []
    for index, item in enumerate(value):
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(f"配置项 {key}[{index}] 必须为整数，实际为 {item!r}。")
        converted_values.append(item)
    values_tuple = tuple(converted_values)
    if len(values_tuple) != 2:
        raise ValueError(f"配置项 {key} 必须包含 2 个整数，实际为 {values_tuple}。")
    return values_tuple


def _ensure_project_relative_path(path: Path, project_root: Path, config_key: str) -> None:
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(
            f"{config_key} 必须解析到项目目录内，项目根目录为 {project_root}，实际为 {path}。"
        ) from exc
