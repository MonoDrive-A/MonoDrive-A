"""与词表文件同目录的模型侧轨迹词表加载、嵌入和解码模块。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import tomllib
from typing import Any, Mapping, NamedTuple

import numpy as np
import torch
import torch.nn as nn

from model.swiglu import SwiGLU


__all__ = [
    "TrajectoryDecoderOutput",
    "TrajectoryVocabData",
    "TrajectoryVocabModelConfig",
    "TrajectoryVocabularyDecoder",
    "TrajectoryVocabularyEmbedding",
    "load_trajectory_vocab_config",
    "load_trajectory_vocabulary",
]


@dataclass(frozen=True)
class TrajectoryVocabModelConfig:
    """轨迹词表模型侧配置。

    Args:
        vocab_path: 轨迹词表 `.npz` 文件路径，读取自 TOML 配置并解析为项目内路径。
        physical_key: `.npz` 中米制物理词表字段名。
        symlog_key: `.npz` 中 Symlog 词表字段名。
        normalized_key: `.npz` 中已归一化词表字段名，嵌入层只使用该字段。
        scale_key: `.npz` 中共享 Symlog 缩放系数字段名。
        num_trajectories: 词表轨迹数量。
        future_points: 每条轨迹未来点数。
        trajectory_dim: 每个未来点坐标维度。
        hidden_dim: 轨迹查询和解码输入特征维度。
        frequency_count: 每个归一化坐标使用的高频编码频率数。
        frequency_base: 高频编码分母底数。
        frequency_scale: 高频编码角度前置系数。
        swiglu_hidden_dim: SwiGLU 激活后的中间特征维度。
        logit_init_value: 解码层 logit 初始输出值。
        residual_output_init_value: Tanh 后残差初始输出值。
        residual_activation: 残差激活函数名称，当前为 `tanh`。
    """

    vocab_path: Path
    physical_key: str
    symlog_key: str
    normalized_key: str
    scale_key: str
    num_trajectories: int
    future_points: int
    trajectory_dim: int
    hidden_dim: int
    frequency_count: int
    frequency_base: float
    frequency_scale: float
    swiglu_hidden_dim: int
    logit_init_value: float
    residual_output_init_value: float
    residual_activation: str

    def __post_init__(self) -> None:
        if self.num_trajectories <= 0:
            raise ValueError(f"num_trajectories 必须为正整数，实际为 {self.num_trajectories}。")
        if self.future_points <= 0:
            raise ValueError(f"future_points 必须为正整数，实际为 {self.future_points}。")
        if self.trajectory_dim <= 0:
            raise ValueError(f"trajectory_dim 必须为正整数，实际为 {self.trajectory_dim}。")
        if self.trajectory_dim != 2:
            raise ValueError(
                "trajectory_dim 必须为 2，"
                f"以按 [phi_y(y), phi_x(x)] 形式编码 ego XY 轨迹，实际为 {self.trajectory_dim}。"
            )
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim 必须为正整数，实际为 {self.hidden_dim}。")
        if self.frequency_count <= 0:
            raise ValueError(f"frequency_count 必须为正整数，实际为 {self.frequency_count}。")
        if self.frequency_base <= 0:
            raise ValueError(f"frequency_base 必须为正数，实际为 {self.frequency_base}。")
        if self.frequency_scale <= 0:
            raise ValueError(f"frequency_scale 必须为正数，实际为 {self.frequency_scale}。")
        if self.swiglu_hidden_dim <= 0:
            raise ValueError(
                f"swiglu_hidden_dim 必须为正整数，实际为 {self.swiglu_hidden_dim}。"
            )
        if self.residual_activation != "tanh":
            raise ValueError(
                "residual_activation 当前仅支持 'tanh'，"
                f"实际为 {self.residual_activation!r}。"
            )
        if not -1.0 < self.residual_output_init_value < 1.0:
            raise ValueError(
                "residual_output_init_value 必须位于 (-1, 1)，"
                f"实际为 {self.residual_output_init_value}。"
            )
        for field_name in ("physical_key", "symlog_key", "normalized_key", "scale_key"):
            value = getattr(self, field_name)
            if not value:
                raise ValueError(f"{field_name} 不能为空。")

    @property
    def trajectory_shape(self) -> tuple[int, int, int]:
        """词表轨迹张量 shape。"""

        return (self.num_trajectories, self.future_points, self.trajectory_dim)

    @property
    def residual_dim(self) -> int:
        """每条轨迹残差的展平维度。"""

        return self.future_points * self.trajectory_dim

    @property
    def high_frequency_encoding_dim(self) -> int:
        """每条轨迹经高频编码后的展平维度。"""

        return self.residual_dim * self.frequency_count * 2


@dataclass(frozen=True)
class TrajectoryVocabData:
    """加载后的轨迹词表张量。

    Shape:
        `trajectory_vocab_m`: `[V, K, D]`，ego 坐标系米制轨迹，单位 meter。
        `trajectory_vocab_symlog`: `[V, K, D]`，Symlog 空间轨迹。
        `trajectory_vocab_normalized`: `[V, K, D]`，共享缩放后的归一化轨迹。
        `symlog_scale`: scalar，共享 Symlog 缩放系数。
    """

    trajectory_vocab_m: torch.Tensor
    trajectory_vocab_symlog: torch.Tensor
    trajectory_vocab_normalized: torch.Tensor
    symlog_scale: torch.Tensor
    metadata_json: str | None


class TrajectoryDecoderOutput(NamedTuple):
    """轨迹词表解码结果。

    Shape:
        `logits`: `[B, V]`，轨迹词表概率的未激活 logit。
        `residuals`: `[B, V, K, D]`，经 Tanh 约束后的 Symlog 残差。
    """

    logits: torch.Tensor
    residuals: torch.Tensor


class TrajectoryVocabularyEmbedding(nn.Module):
    """将归一化轨迹词表编码为轨迹查询特征。

    Args:
        config: 模型侧轨迹词表配置。
        vocabulary: `load_trajectory_vocabulary` 加载得到的词表数据。

    Shape:
        输出: `[V, hidden_dim]`，其中 `V` 为词表轨迹数量。
    """

    def __init__(
        self,
        config: TrajectoryVocabModelConfig,
        vocabulary: TrajectoryVocabData,
    ) -> None:
        super().__init__()
        self.config = config
        normalized_shape = tuple(vocabulary.trajectory_vocab_normalized.shape)
        if normalized_shape != config.trajectory_shape:
            raise ValueError(
                "trajectory_vocab_normalized shape 与配置不一致："
                f"期望 {config.trajectory_shape}，实际为 {normalized_shape}。"
            )

        self.register_buffer(
            "trajectory_vocab_normalized",
            vocabulary.trajectory_vocab_normalized.detach().clone().to(dtype=torch.float32),
        )
        frequency_exponents = torch.arange(config.frequency_count, dtype=torch.float32)
        frequency_exponent_ratios = frequency_exponents / float(config.frequency_count)
        frequency_bands = config.frequency_scale / torch.pow(
            torch.tensor(config.frequency_base, dtype=torch.float32),
            frequency_exponent_ratios,
        )
        self.register_buffer("frequency_bands", frequency_bands)
        self.linear_in = nn.Linear(
            config.high_frequency_encoding_dim,
            config.swiglu_hidden_dim * 2,
        )
        self.activation = SwiGLU()
        self.linear_out = nn.Linear(config.swiglu_hidden_dim, config.hidden_dim)

    def forward(self) -> torch.Tensor:
        """输出 256 条轨迹查询嵌入。

        归一化词表来自 `.npz` 的已归一化字段，shape 为 `[V, K, 2]`，
        最后一维按 `[x, y]` 解释。高频编码后每个时间步按
        `[phi_y(y), phi_x(x)]` 拼接，并展平为
        `[V, K * 2 * frequency_count * 2]`。
        """

        trajectory_x = self.trajectory_vocab_normalized[..., 0]
        trajectory_y = self.trajectory_vocab_normalized[..., 1]
        y_features = self._encode_coordinate(trajectory_y)
        x_features = self._encode_coordinate(trajectory_x)
        # [V, K, 2F] + [V, K, 2F] -> [V, K, 4F]，每步顺序为 [phi_y, phi_x]。
        per_step_features = torch.cat((y_features, x_features), dim=-1)
        # [V, K, 4F] -> [V, K * 2 * 2F]
        high_frequency_features = per_step_features.reshape(self.config.num_trajectories, -1)
        hidden_features = self.activation(self.linear_in(high_frequency_features))
        trajectory_queries = self.linear_out(hidden_features)
        return trajectory_queries

    def _encode_coordinate(self, coordinates: torch.Tensor) -> torch.Tensor:
        """将单个 ego 坐标序列编码为交错 sin/cos 高频特征。

        Shape:
            输入: `[V, K]`。
            输出: `[V, K, frequency_count * 2]`，顺序为
            `[sin_0, cos_0, ..., sin_{F-1}, cos_{F-1}]`。
        """

        # [V, K] -> [V, K, F]
        encoded_angles = coordinates[..., None] * self.frequency_bands
        # [V, K, F] -> [V, K, F, 2] -> [V, K, 2F]
        encoded_features = torch.stack(
            (torch.sin(encoded_angles), torch.cos(encoded_angles)),
            dim=-1,
        )
        return encoded_features.reshape(
            *coordinates.shape,
            self.config.frequency_count * 2,
        )


class TrajectoryVocabularyDecoder(nn.Module):
    """从轨迹 token 特征解码轨迹词表 logit 和残差。

    Args:
        config: 模型侧轨迹词表配置。

    Shape:
        输入: `[B, V, hidden_dim]`。
        输出:
            `logits`: `[B, V]`。
            `residuals`: `[B, V, K, D]`。
    """

    def __init__(self, config: TrajectoryVocabModelConfig) -> None:
        super().__init__()
        self.config = config
        self.output_linear = nn.Linear(config.hidden_dim, 1 + config.residual_dim)
        self.residual_activation = nn.Tanh()
        self._reset_output_initialization()

    def forward(self, trajectory_features: torch.Tensor) -> TrajectoryDecoderOutput:
        """解码轨迹词表输出。

        `logits` 不做激活，由训练或推理流程在需要概率时自行执行 Softmax。
        `residuals` 使用 Tanh 激活，表示 Symlog 空间中对每条候选轨迹的逐坐标残差。
        """

        if trajectory_features.ndim != 3:
            raise ValueError(
                "trajectory_features 期望 shape 为 [B, V, hidden_dim]，"
                f"实际为 {tuple(trajectory_features.shape)}。"
            )
        if int(trajectory_features.shape[1]) != self.config.num_trajectories:
            raise ValueError(
                "trajectory_features 的轨迹数量与配置不一致："
                f"期望 {self.config.num_trajectories}，实际为 {trajectory_features.shape[1]}。"
            )
        if int(trajectory_features.shape[2]) != self.config.hidden_dim:
            raise ValueError(
                "trajectory_features 的特征维度与配置不一致："
                f"期望 {self.config.hidden_dim}，实际为 {trajectory_features.shape[2]}。"
            )

        decoded_features = self.output_linear(trajectory_features)
        logits = decoded_features[..., 0]
        residual_features = decoded_features[..., 1:].reshape(
            trajectory_features.shape[0],
            self.config.num_trajectories,
            self.config.future_points,
            self.config.trajectory_dim,
        )
        residuals = self.residual_activation(residual_features)
        return TrajectoryDecoderOutput(logits=logits, residuals=residuals)

    def _reset_output_initialization(self) -> None:
        residual_bias_value = math.atanh(self.config.residual_output_init_value)
        with torch.no_grad():
            self.output_linear.weight.zero_()
            self.output_linear.bias[0].fill_(self.config.logit_init_value)
            self.output_linear.bias[1:].fill_(residual_bias_value)


def load_trajectory_vocab_config(
    config_path: str | Path,
    project_root: str | Path | None = None,
) -> TrajectoryVocabModelConfig:
    """读取轨迹词表 TOML 配置。

    Args:
        config_path: TOML 配置文件路径。
        project_root: 项目根目录。若为 `None`，按配置文件父目录的父目录解析。

    Returns:
        `TrajectoryVocabModelConfig`。
    """

    config_path = Path(config_path)
    resolved_config_path = config_path.resolve()
    resolved_project_root = (
        Path(project_root).resolve() if project_root is not None else resolved_config_path.parent.parent
    )
    with resolved_config_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)

    vocabulary_config = _require_table(raw_config, "vocabulary")
    embedding_config = _require_table(raw_config, "embedding")
    decoder_config = _require_table(raw_config, "decoder")

    vocab_path_text = _require_string(vocabulary_config, "path")
    raw_vocab_path = Path(vocab_path_text)
    if raw_vocab_path.is_absolute():
        raise ValueError(f"vocabulary.path 必须是项目内相对路径，实际为 {raw_vocab_path}。")
    vocab_path = (resolved_project_root / raw_vocab_path).resolve()
    _ensure_project_relative_path(vocab_path, resolved_project_root, "vocabulary.path")

    return TrajectoryVocabModelConfig(
        vocab_path=vocab_path,
        physical_key=_require_string(vocabulary_config, "physical_key"),
        symlog_key=_require_string(vocabulary_config, "symlog_key"),
        normalized_key=_require_string(vocabulary_config, "normalized_key"),
        scale_key=_require_string(vocabulary_config, "scale_key"),
        num_trajectories=_require_int(vocabulary_config, "num_trajectories"),
        future_points=_require_int(vocabulary_config, "future_points"),
        trajectory_dim=_require_int(vocabulary_config, "trajectory_dim"),
        hidden_dim=_require_int(embedding_config, "hidden_dim"),
        frequency_count=_require_int(embedding_config, "frequency_count"),
        frequency_base=_require_float(embedding_config, "frequency_base"),
        frequency_scale=_require_float(embedding_config, "frequency_scale"),
        swiglu_hidden_dim=_require_int(embedding_config, "swiglu_hidden_dim"),
        logit_init_value=_require_float(decoder_config, "logit_init_value"),
        residual_output_init_value=_require_float(decoder_config, "residual_output_init_value"),
        residual_activation=_require_string(decoder_config, "residual_activation"),
    )


def load_trajectory_vocabulary(
    config: TrajectoryVocabModelConfig,
    device: torch.device | str | None = None,
) -> TrajectoryVocabData:
    """从 `.npz` 加载轨迹词表。

    模型嵌入只使用 `.npz` 中的已归一化字段；物理空间和 Symlog 字段保留给
    loss、可视化或推理阶段反查使用。
    """

    if not config.vocab_path.exists():
        raise FileNotFoundError(f"轨迹词表文件不存在：{config.vocab_path}")
    if not config.vocab_path.is_file():
        raise FileNotFoundError(f"轨迹词表路径不是文件：{config.vocab_path}")

    with np.load(config.vocab_path, allow_pickle=False) as npz_file:
        required_keys = (
            config.physical_key,
            config.symlog_key,
            config.normalized_key,
            config.scale_key,
        )
        missing_keys = [key for key in required_keys if key not in npz_file.files]
        if missing_keys:
            raise KeyError(f"轨迹词表缺少字段 {missing_keys}：{config.vocab_path}")

        trajectory_vocab_m = _load_vocab_array(npz_file, config.physical_key, config)
        trajectory_vocab_symlog = _load_vocab_array(npz_file, config.symlog_key, config)
        trajectory_vocab_normalized = _load_vocab_array(npz_file, config.normalized_key, config)
        symlog_scale = np.asarray(npz_file[config.scale_key], dtype=np.float32)
        if symlog_scale.shape != ():
            raise ValueError(
                f"{config.scale_key} 期望为标量，实际 shape 为 {symlog_scale.shape}。"
            )
        metadata_json = (
            str(np.asarray(npz_file["metadata_json"]).item()) if "metadata_json" in npz_file.files else None
        )

    target_device = torch.device(device) if device is not None else None
    return TrajectoryVocabData(
        trajectory_vocab_m=torch.as_tensor(
            trajectory_vocab_m,
            dtype=torch.float32,
            device=target_device,
        ),
        trajectory_vocab_symlog=torch.as_tensor(
            trajectory_vocab_symlog,
            dtype=torch.float32,
            device=target_device,
        ),
        trajectory_vocab_normalized=torch.as_tensor(
            trajectory_vocab_normalized,
            dtype=torch.float32,
            device=target_device,
        ),
        symlog_scale=torch.as_tensor(float(symlog_scale), dtype=torch.float32, device=target_device),
        metadata_json=metadata_json,
    )


def _load_vocab_array(
    npz_file: Any,
    key: str,
    config: TrajectoryVocabModelConfig,
) -> np.ndarray:
    trajectory_array = np.asarray(npz_file[key], dtype=np.float32)
    if trajectory_array.shape != config.trajectory_shape:
        raise ValueError(
            f"{key} 期望 shape 为 {config.trajectory_shape}，实际为 {trajectory_array.shape}。"
        )
    if not np.isfinite(trajectory_array).all():
        raise ValueError(f"{key} 中存在 NaN 或 Inf：{config.vocab_path}")
    return trajectory_array


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


def _ensure_project_relative_path(path: Path, project_root: Path, config_key: str) -> None:
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(
            f"{config_key} 必须解析到项目目录内，项目根目录为 {project_root}，实际为 {path}。"
        ) from exc
