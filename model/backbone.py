"""MonoDrive 统一序列 Transformer 主干。"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
import tomllib
from typing import Any, Mapping, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.detection_head import (
    DetectionDecoderOutput,
    DetectionHeadConfig,
    DetectionHeadDecoder,
    DetectionQueryEmbedding,
    load_detection_head_config,
)
from model.rope_3d import apply_rope_3d
from model.swiglu import SwiGLU
from model.target_point_embedding import (
    TargetPointEmbedding,
    TargetPointEmbeddingConfig,
    load_target_point_embedding_config,
)
from model.trajectory_vocab import (
    TrajectoryDecoderOutput,
    TrajectoryVocabData,
    TrajectoryVocabModelConfig,
    TrajectoryVocabularyDecoder,
    TrajectoryVocabularyEmbedding,
    load_trajectory_vocab_config,
    load_trajectory_vocabulary,
)
from model.vision_embedding import (
    BackboneVisionEmbedding,
    VisionEmbeddingConfig,
    VisionEmbeddingOutput,
    load_vision_embedding_config,
)


__all__ = [
    "BackboneConfig",
    "BackboneTokenSlices",
    "MonoDriveBackbone",
    "MonoDriveBackboneOutput",
    "override_backbone_precision",
    "load_backbone_config",
]


SUPPORTED_DTYPE_NAMES = {"float32", "bfloat16"}
SUPPORTED_TOKEN_ORDERS = {"vision_register_detection_trajectory_goal"}
SUPPORTED_TOKEN_TYPES = {"vision", "register", "agent", "map", "trajectory", "goal"}
SUPPORTED_ACTIVATIONS = {"swiglu"}
SUPPORTED_POSITION_ORDERS = {"height_width_time"}
SUPPORTED_VECTOR_TRANSFORMS = {"symlog"}
GOAL_TOKEN_INSERT_LAYER_INDEX = 12
DETECTION_OUTPUT_LAYER_INDEX = GOAL_TOKEN_INSERT_LAYER_INDEX - 1


@dataclass(frozen=True)
class BackboneConfig:
    """统一主干配置。

    Args:
        project_root: 项目根目录。
        vision_config_path: 视觉嵌入配置路径。
        target_point_config_path: 目标点嵌入配置路径。
        trajectory_vocab_config_path: 轨迹词表配置路径。
        detection_head_config_path: 检测头配置路径。
        hidden_dim: 统一序列特征维度。
        layer_count: Transformer 层数。
        attention_head_count: 注意力头数。
        register_token_count: 寄存器 Token 数。
        expected_sequence_length: 统一序列期望长度。
        token_order: Token 拼接顺序。
        modal_ffn_layer_indices: 使用模态独立 FFN 的 0-based 层索引。
        rms_norm_eps: RMSNorm 数值稳定项。
        rope_head_count: 前多少个注意力头对视觉 Token 应用 RoPE。
        attention_dropout: SDPA dropout 概率。
        ffn_layer1_output_dim: FFN 第一层输出维度，SwiGLU 后会变为该值的一半。
        ffn_activation: FFN 激活名称。
        rope_theta: 3D RoPE 基频。
        rope_axis_dims: RoPE 在 `[H, W, T]` 三轴上的通道划分。
        visual_position_order: 视觉位置坐标顺序。
        position_min: 归一化位置最小值。
        position_max: 归一化位置最大值。
        token_type_order: 身份嵌入表的 Token 类型顺序。
        ego_motion_input_dim: 自车运动输入维度。
        ego_motion_vector_transform: 自车运动进入线性层前的数值变换。
        backbone_dtype: Transformer 主干线性层和 FFN 的 autocast 精度。
        attention_dtype: SDPA 输入精度。
        register_token_std: 寄存器 Token 初始化标准差。
        token_type_embedding_std: 身份嵌入初始化标准差。
        ego_motion_linear_std: 自车运动线性层初始化标准差。
    """

    project_root: Path
    vision_config_path: Path
    target_point_config_path: Path
    trajectory_vocab_config_path: Path
    detection_head_config_path: Path
    hidden_dim: int
    layer_count: int
    attention_head_count: int
    register_token_count: int
    expected_sequence_length: int
    token_order: str
    modal_ffn_layer_indices: tuple[int, ...]
    rms_norm_eps: float
    rope_head_count: int
    attention_dropout: float
    ffn_layer1_output_dim: int
    ffn_activation: str
    rope_theta: float
    rope_axis_dims: tuple[int, int, int]
    visual_position_order: str
    position_min: float
    position_max: float
    token_type_order: tuple[str, ...]
    ego_motion_input_dim: int
    ego_motion_vector_transform: str
    backbone_dtype: str
    attention_dtype: str
    register_token_std: float
    token_type_embedding_std: float
    ego_motion_linear_std: float

    def __post_init__(self) -> None:
        _validate_positive_int(self.hidden_dim, "hidden_dim")
        _validate_positive_int(self.layer_count, "layer_count")
        if self.layer_count <= GOAL_TOKEN_INSERT_LAYER_INDEX:
            raise ValueError(
                "layer_count 必须至少为 13，"
                f"以便第 13 层输入前加入目标点 Token，实际为 {self.layer_count}。"
            )
        _validate_positive_int(self.attention_head_count, "attention_head_count")
        if self.hidden_dim % self.attention_head_count != 0:
            raise ValueError(
                "hidden_dim 必须能被 attention_head_count 整除，"
                f"实际为 {self.hidden_dim} 和 {self.attention_head_count}。"
            )
        _validate_positive_int(self.register_token_count, "register_token_count")
        _validate_positive_int(self.expected_sequence_length, "expected_sequence_length")
        if self.token_order not in SUPPORTED_TOKEN_ORDERS:
            raise ValueError(
                f"token_order 仅支持 {sorted(SUPPORTED_TOKEN_ORDERS)}，"
                f"实际为 {self.token_order!r}。"
            )
        if len(set(self.modal_ffn_layer_indices)) != len(self.modal_ffn_layer_indices):
            raise ValueError(
                "modal_ffn_layer_indices 不能包含重复项，"
                f"实际为 {self.modal_ffn_layer_indices}。"
            )
        for layer_index in self.modal_ffn_layer_indices:
            if layer_index < 0 or layer_index >= self.layer_count:
                raise ValueError(
                    "modal_ffn_layer_indices 必须位于 "
                    f"[0, {self.layer_count - 1}]，实际包含 {layer_index}。"
                )
        if self.rms_norm_eps <= 0.0:
            raise ValueError(f"rms_norm_eps 必须为正数，实际为 {self.rms_norm_eps}。")
        _validate_positive_int(self.rope_head_count, "rope_head_count")
        if self.rope_head_count > self.attention_head_count:
            raise ValueError(
                "rope_head_count 不能超过 attention_head_count，"
                f"实际为 {self.rope_head_count} 和 {self.attention_head_count}。"
            )
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError(f"attention_dropout 必须位于 [0, 1)，实际为 {self.attention_dropout}。")
        _validate_positive_int(self.ffn_layer1_output_dim, "ffn_layer1_output_dim")
        if self.ffn_layer1_output_dim % 2 != 0:
            raise ValueError(
                "ffn_layer1_output_dim 必须为偶数，"
                f"以便 SwiGLU 二等分，实际为 {self.ffn_layer1_output_dim}。"
            )
        expected_layer1_dim = self.hidden_dim * 4
        if self.ffn_layer1_output_dim != expected_layer1_dim:
            raise ValueError(
                "ffn_layer1_output_dim 必须等于 4 * hidden_dim，"
                f"期望 {expected_layer1_dim}，实际为 {self.ffn_layer1_output_dim}。"
            )
        if self.ffn_activation not in SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"ffn_activation 仅支持 {sorted(SUPPORTED_ACTIVATIONS)}，"
                f"实际为 {self.ffn_activation!r}。"
            )
        if self.rope_theta <= 0.0:
            raise ValueError(f"rope_theta 必须为正数，实际为 {self.rope_theta}。")
        _validate_axis_dims(self.rope_axis_dims, self.head_dim)
        if self.visual_position_order not in SUPPORTED_POSITION_ORDERS:
            raise ValueError(
                f"visual_position_order 仅支持 {sorted(SUPPORTED_POSITION_ORDERS)}，"
                f"实际为 {self.visual_position_order!r}。"
            )
        if self.position_min >= self.position_max:
            raise ValueError(
                "position_min 必须小于 position_max，"
                f"实际为 {self.position_min} 和 {self.position_max}。"
            )
        if set(self.token_type_order) != SUPPORTED_TOKEN_TYPES:
            raise ValueError(
                "token_type_order 必须且只能包含 "
                f"{sorted(SUPPORTED_TOKEN_TYPES)}，实际为 {self.token_type_order}。"
            )
        _validate_positive_int(self.ego_motion_input_dim, "ego_motion_input_dim")
        if self.ego_motion_vector_transform not in SUPPORTED_VECTOR_TRANSFORMS:
            raise ValueError(
                "ego_motion_vector_transform 仅支持 "
                f"{sorted(SUPPORTED_VECTOR_TRANSFORMS)}，实际为 {self.ego_motion_vector_transform!r}。"
            )
        _validate_dtype_name(self.backbone_dtype, "backbone_dtype")
        _validate_dtype_name(self.attention_dtype, "attention_dtype")
        for field_name in (
            "register_token_std",
            "token_type_embedding_std",
            "ego_motion_linear_std",
        ):
            value = getattr(self, field_name)
            if value < 0.0:
                raise ValueError(f"{field_name} 不能为负数，实际为 {value}。")

    @property
    def head_dim(self) -> int:
        """单个注意力头的通道维度。"""

        return self.hidden_dim // self.attention_head_count

    @property
    def backbone_torch_dtype(self) -> torch.dtype:
        """Transformer 主干 autocast 使用的 dtype。"""

        return _dtype_from_name(self.backbone_dtype)

    @property
    def attention_torch_dtype(self) -> torch.dtype:
        """SDPA 输入使用的 dtype。"""

        return _dtype_from_name(self.attention_dtype)


class BackboneTokenSlices(NamedTuple):
    """统一序列中各 Token 分段位置。"""

    vision: slice
    register: slice
    detection: slice
    agent: slice
    map: slice
    trajectory: slice
    goal: slice

    @property
    def total_length(self) -> int:
        """统一序列长度。"""

        return int(self.goal.stop)

    @property
    def pre_goal_length(self) -> int:
        """第 13 层输入前、不含目标点 Token 的序列长度。"""

        return int(self.goal.start)


class MonoDriveBackboneOutput(NamedTuple):
    """统一主干前向输出。

    Shape:
        `sequence_features`: `[B, 2614, D]`。
        `vision_features`: `[B, 2304, D]`。
        `register_features`: `[B, 4, D]`。
        `detection_features`: 第 12 层旁路累积结果 Acc_11，`[B, 48, D]`，骨干精度。
        `trajectory_features`: `[B, 256, D]`。
        `trajectory_decoder_features`: `[B, 256, D]`。
        `goal_features`: `[B, 2, D]`。
        `layer_vision_features`: 每项 `[B, 2304, D]`。
    """

    sequence_features: torch.Tensor
    vision_features: torch.Tensor
    register_features: torch.Tensor
    detection_features: torch.Tensor
    trajectory_features: torch.Tensor
    trajectory_decoder_features: torch.Tensor
    goal_features: torch.Tensor
    vision_embedding_output: VisionEmbeddingOutput
    detection_output: DetectionDecoderOutput
    trajectory_output: TrajectoryDecoderOutput
    token_slices: BackboneTokenSlices
    layer_vision_features: tuple[torch.Tensor, ...]


class _BackboneInputSequence(NamedTuple):
    """第 1-12 层输入序列及其未加身份嵌入的检测查询基线。

    Shape:
        `token_features`: `[B, 2612, D]`。
        `initial_detection_queries`: `[B, 48, D]`。
    """

    token_features: torch.Tensor
    initial_detection_queries: torch.Tensor


class TokenRMSNorm(nn.Module):
    """适用于 `[B, N, D]` Token 序列的 RMSNorm。"""

    def __init__(self, hidden_dim: int, eps: float) -> None:
        super().__init__()
        _validate_positive_int(hidden_dim, "hidden_dim")
        if eps <= 0.0:
            raise ValueError(f"eps 必须为正数，实际为 {eps}。")
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, token_features: torch.Tensor) -> torch.Tensor:
        """沿最后一维执行 RMSNorm。"""

        rms = token_features.pow(2).mean(dim=-1, keepdim=True).sqrt()
        return self.weight * token_features / (rms + self.eps)


class SwiGLUFeedForward(nn.Module):
    """统一主干使用的 SwiGLU FFN。"""

    def __init__(self, hidden_dim: int, ffn_layer1_output_dim: int) -> None:
        super().__init__()
        if ffn_layer1_output_dim % 2 != 0:
            raise ValueError(
                "ffn_layer1_output_dim 必须为偶数，"
                f"实际为 {ffn_layer1_output_dim}。"
            )
        self.linear_in = nn.Linear(hidden_dim, ffn_layer1_output_dim)
        self.activation = SwiGLU()
        self.linear_out = nn.Linear(ffn_layer1_output_dim // 2, hidden_dim)

    def forward(self, token_features: torch.Tensor) -> torch.Tensor:
        """执行 `D -> 4D -> SwiGLU(4D -> 2D) -> D` 的 FFN。"""

        return self.linear_out(self.activation(self.linear_in(token_features)))


class ModalIndependentFeedForward(nn.Module):
    """按视觉相关和驾驶相关 Token 拆分的独立 FFN。"""

    def __init__(self, hidden_dim: int, ffn_layer1_output_dim: int) -> None:
        super().__init__()
        self.visual_ffn = SwiGLUFeedForward(hidden_dim, ffn_layer1_output_dim)
        self.driving_ffn = SwiGLUFeedForward(hidden_dim, ffn_layer1_output_dim)

    def forward(
        self,
        token_features: torch.Tensor,
        token_slices: BackboneTokenSlices,
    ) -> torch.Tensor:
        """分别处理视觉相关和驾驶相关 Token。"""

        output_features = torch.empty_like(token_features)
        visual_related = slice(token_slices.vision.start, token_slices.register.stop)
        driving_related = slice(token_slices.detection.start, token_slices.goal.stop)
        output_features[:, visual_related, :] = self.visual_ffn(token_features[:, visual_related, :])
        output_features[:, driving_related, :] = self.driving_ffn(token_features[:, driving_related, :])
        return output_features


class VisualRoPESelfAttention(nn.Module):
    """只对视觉 Token 应用 RoPE 的 SDPA 自注意力。"""

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.config = config
        self.qkv_projection = nn.Linear(config.hidden_dim, config.hidden_dim * 3)
        self.output_projection = nn.Linear(config.hidden_dim, config.hidden_dim)

    def forward(
        self,
        token_features: torch.Tensor,
        visual_positions: torch.Tensor,
        token_slices: BackboneTokenSlices,
    ) -> torch.Tensor:
        """执行全序列 SDPA，且只旋转视觉 Token 的前若干头 Q/K。"""

        if token_features.ndim != 3:
            raise ValueError(
                "token_features 期望 shape 为 [B, N, D]，"
                f"实际为 {tuple(token_features.shape)}。"
            )
        batch_size, sequence_length, hidden_dim = token_features.shape
        if int(hidden_dim) != self.config.hidden_dim:
            raise ValueError(
                "token_features 最后一维必须等于 hidden_dim，"
                f"期望 {self.config.hidden_dim}，实际为 {hidden_dim}。"
            )

        qkv_features = self.qkv_projection(token_features)
        # [B, N, 3D] -> [3, B, H, N, Dh]
        qkv_features = qkv_features.reshape(
            batch_size,
            sequence_length,
            3,
            self.config.attention_head_count,
            self.config.head_dim,
        )
        qkv_features = qkv_features.permute(2, 0, 3, 1, 4).contiguous()
        query, key, value = qkv_features.unbind(dim=0)
        query, key = self._apply_visual_rope(query, key, visual_positions, token_slices.vision)

        attention_dtype = self.config.attention_torch_dtype
        query = query.to(dtype=attention_dtype)
        key = key.to(dtype=attention_dtype)
        value = value.to(dtype=attention_dtype)
        attended_features = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        # [B, H, N, Dh] -> [B, N, D]
        attended_features = attended_features.transpose(1, 2).reshape(
            batch_size,
            sequence_length,
            self.config.hidden_dim,
        )
        return self.output_projection(attended_features)

    def _apply_visual_rope(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        visual_positions: torch.Tensor,
        vision_slice: slice,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_fp32 = query.to(dtype=torch.float32)
        key_fp32 = key.to(dtype=torch.float32)
        rope_head_slice = slice(0, self.config.rope_head_count)
        visual_query = query_fp32[:, rope_head_slice, vision_slice, :]
        visual_key = key_fp32[:, rope_head_slice, vision_slice, :]
        visual_positions = visual_positions.to(device=query.device, dtype=torch.float32)
        rotated_query = apply_rope_3d(
            visual_query,
            visual_positions,
            self.config.rope_axis_dims,
            self.config.rope_theta,
        )
        rotated_key = apply_rope_3d(
            visual_key,
            visual_positions,
            self.config.rope_axis_dims,
            self.config.rope_theta,
        )
        query_fp32 = query_fp32.clone()
        key_fp32 = key_fp32.clone()
        query_fp32[:, rope_head_slice, vision_slice, :] = rotated_query
        key_fp32[:, rope_head_slice, vision_slice, :] = rotated_key
        return query_fp32, key_fp32


class BackboneTransformerBlock(nn.Module):
    """Pre-Norm Transformer Block。"""

    def __init__(self, config: BackboneConfig, use_modal_independent_ffn: bool) -> None:
        super().__init__()
        self.attention_norm = TokenRMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.attention = VisualRoPESelfAttention(config)
        self.ffn_norm = TokenRMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.ffn = (
            ModalIndependentFeedForward(config.hidden_dim, config.ffn_layer1_output_dim)
            if use_modal_independent_ffn
            else SwiGLUFeedForward(config.hidden_dim, config.ffn_layer1_output_dim)
        )
        self.use_modal_independent_ffn = use_modal_independent_ffn

    def forward(
        self,
        token_features: torch.Tensor,
        visual_positions: torch.Tensor,
        token_slices: BackboneTokenSlices,
    ) -> torch.Tensor:
        """执行一层主干 Transformer。"""

        attention_features = self.attention(
            self.attention_norm(token_features),
            visual_positions,
            token_slices,
        )
        token_features = token_features + attention_features
        normalized_features = self.ffn_norm(token_features)
        if self.use_modal_independent_ffn:
            ffn_features = self.ffn(normalized_features, token_slices)
        else:
            ffn_features = self.ffn(normalized_features)
        return token_features + ffn_features


class MonoDriveBackbone(nn.Module):
    """MonoDrive 统一序列主干。

    Args:
        config: `load_backbone_config` 读取并校验后的主干配置。
        vision_config: 可选的视觉嵌入配置覆盖，主要用于 FP32 可视化。
        target_point_config: 可选的目标点嵌入配置覆盖。
        trajectory_config: 可选的轨迹词表配置覆盖。
        detection_config: 可选的检测头配置覆盖。
        vocabulary: 可选的轨迹词表数据覆盖。

    Shape:
        `images`: `[B, 8, 3, 288, 512]`。
        `target_points`: `[B, 2]`，ego 坐标系米制目标点。
        `ego_motion`: `[B, 3]`，`[V_x, V_y, W]`。
    """

    def __init__(
        self,
        config: BackboneConfig,
        vision_config: VisionEmbeddingConfig | None = None,
        target_point_config: TargetPointEmbeddingConfig | None = None,
        trajectory_config: TrajectoryVocabModelConfig | None = None,
        detection_config: DetectionHeadConfig | None = None,
        vocabulary: TrajectoryVocabData | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.vision_config = vision_config or load_vision_embedding_config(
            config.vision_config_path,
            config.project_root,
        )
        self.target_point_config = target_point_config or load_target_point_embedding_config(
            config.target_point_config_path,
            config.project_root,
        )
        self.trajectory_config = trajectory_config or load_trajectory_vocab_config(
            config.trajectory_vocab_config_path,
            config.project_root,
        )
        self.detection_config = detection_config or load_detection_head_config(
            config.detection_head_config_path,
            config.project_root,
        )
        self.vocabulary = vocabulary or load_trajectory_vocabulary(self.trajectory_config)
        self._validate_subconfigs()

        self.vision_embedding = BackboneVisionEmbedding(self.vision_config)
        self.target_point_embedding = TargetPointEmbedding(self.target_point_config)
        self.trajectory_embedding = TrajectoryVocabularyEmbedding(
            self.trajectory_config,
            self.vocabulary,
        )
        self.detection_query_embedding = DetectionQueryEmbedding(self.detection_config)
        self.trajectory_decoder = TrajectoryVocabularyDecoder(self.trajectory_config)
        self.detection_decoder = DetectionHeadDecoder(self.detection_config)
        self.detection_residual_norms = nn.ModuleList(
            TokenRMSNorm(config.hidden_dim, config.rms_norm_eps)
            for _ in range(GOAL_TOKEN_INSERT_LAYER_INDEX)
        )
        self.detection_residual_projections = nn.ModuleList(
            nn.Linear(config.hidden_dim, config.hidden_dim)
            for _ in range(GOAL_TOKEN_INSERT_LAYER_INDEX)
        )
        self.detection_layer_identity_embeddings = nn.Parameter(
            torch.zeros(GOAL_TOKEN_INSERT_LAYER_INDEX, 2, config.hidden_dim)
        )
        self.token_type_to_index = {
            token_type: index for index, token_type in enumerate(config.token_type_order)
        }
        self.token_type_embeddings = nn.Parameter(
            torch.empty(len(config.token_type_order), config.hidden_dim, dtype=torch.float32)
        )
        self.register_tokens = nn.Parameter(
            torch.empty(config.register_token_count, config.hidden_dim, dtype=torch.float32)
        )
        self.ego_motion_encoder = nn.Linear(config.ego_motion_input_dim, config.hidden_dim)
        self.transformer_blocks = nn.ModuleList(
            BackboneTransformerBlock(
                config,
                use_modal_independent_ffn=layer_index in config.modal_ffn_layer_indices,
            )
            for layer_index in range(config.layer_count)
        )
        self._reset_input_initialization()

    def _apply(self, fn: Any) -> "MonoDriveBackbone":
        super()._apply(fn)
        _force_parameter_to_float32(self.token_type_embeddings)
        _force_parameter_to_float32(self.register_tokens)
        _force_floating_tensors_to_float32(self.ego_motion_encoder)
        return self

    def forward(
        self,
        images: torch.Tensor,
        target_points: torch.Tensor,
        ego_motion: torch.Tensor,
        return_layer_features: bool = False,
    ) -> MonoDriveBackboneOutput:
        """运行完整主干并解码检测和轨迹输出。"""

        self._validate_forward_inputs(images, target_points, ego_motion)
        vision_output = self.vision_embedding(images)
        token_slices = self._build_token_slices(vision_output)
        visual_positions = self._build_visual_positions(
            vision_output.latent_grid_shape,
            device=images.device,
        )
        input_sequence = self._build_input_sequence(
            vision_output.tokens,
            images.device,
            token_slices,
        )
        backbone_dtype = self.config.backbone_torch_dtype
        token_features = input_sequence.token_features.to(dtype=backbone_dtype)
        accumulated_detection_queries = input_sequence.initial_detection_queries.to(
            dtype=backbone_dtype,
        )

        detection_features: torch.Tensor | None = None
        layer_vision_features = []
        with _precision_context(images.device, backbone_dtype):
            for layer_index, transformer_block in enumerate(self.transformer_blocks):
                if layer_index == GOAL_TOKEN_INSERT_LAYER_INDEX:
                    token_features = self._append_goal_tokens(
                        token_features,
                        target_points,
                        images.device,
                        token_slices,
                    )
                token_features = transformer_block(token_features, visual_positions, token_slices)
                if layer_index < GOAL_TOKEN_INSERT_LAYER_INDEX:
                    token_features, accumulated_detection_queries = (
                        self._apply_detection_layer_residual(
                            token_features,
                            accumulated_detection_queries,
                            layer_index,
                            token_slices,
                        )
                    )
                if layer_index == DETECTION_OUTPUT_LAYER_INDEX:
                    detection_features = accumulated_detection_queries
                if return_layer_features:
                    layer_vision_features.append(token_features[:, token_slices.vision, :].detach())

        if detection_features is None:
            raise RuntimeError(
                "未能取得第 12 层检测 Token 旁路累积结果，"
                f"layer_count={self.config.layer_count}。"
            )
        split_features = self._split_sequence_features(token_features, token_slices)
        trajectory_decoder_features = self._prepare_trajectory_decoder_features(
            split_features["trajectory"],
            ego_motion,
        )
        detection_decoder_features = accumulated_detection_queries.to(dtype=torch.float32)
        detection_output = self.detection_decoder(detection_decoder_features)
        trajectory_output = self.trajectory_decoder(trajectory_decoder_features)
        return MonoDriveBackboneOutput(
            sequence_features=token_features,
            vision_features=split_features["vision"],
            register_features=split_features["register"],
            detection_features=detection_features,
            trajectory_features=split_features["trajectory"],
            trajectory_decoder_features=trajectory_decoder_features,
            goal_features=split_features["goal"],
            vision_embedding_output=vision_output,
            detection_output=detection_output,
            trajectory_output=trajectory_output,
            token_slices=token_slices,
            layer_vision_features=tuple(layer_vision_features),
        )

    def _reset_input_initialization(self) -> None:
        with torch.no_grad():
            self.token_type_embeddings.normal_(
                mean=0.0,
                std=self.config.token_type_embedding_std,
            )
            self.register_tokens.normal_(mean=0.0, std=self.config.register_token_std)
            self.ego_motion_encoder.weight.normal_(
                mean=0.0,
                std=self.config.ego_motion_linear_std,
            )
            self.ego_motion_encoder.bias.zero_()
            for detection_residual_projection in self.detection_residual_projections:
                detection_residual_projection.weight.zero_()
                detection_residual_projection.bias.zero_()
            self.detection_layer_identity_embeddings.zero_()

    def _validate_subconfigs(self) -> None:
        expected_hidden_dims = {
            "backbone.hidden_dim": self.config.hidden_dim,
            "vision.output_hidden_dim": self.vision_config.output_hidden_dim,
            "target_point.hidden_dim": self.target_point_config.hidden_dim,
            "trajectory.hidden_dim": self.trajectory_config.hidden_dim,
            "detection.hidden_dim": self.detection_config.hidden_dim,
        }
        if len(set(expected_hidden_dims.values())) != 1:
            raise ValueError(f"各模块 hidden_dim 必须一致，实际为 {expected_hidden_dims}。")

        expected_sequence_length = (
            self.vision_config.expected_token_count
            + self.config.register_token_count
            + self.detection_config.total_query_count
            + self.trajectory_config.num_trajectories
            + self.target_point_config.goal_token_count
        )
        if expected_sequence_length != self.config.expected_sequence_length:
            raise ValueError(
                "expected_sequence_length 与子模块 token 数不一致："
                f"配置为 {self.config.expected_sequence_length}，"
                f"按子模块推导为 {expected_sequence_length}。"
            )

    def _validate_forward_inputs(
        self,
        images: torch.Tensor,
        target_points: torch.Tensor,
        ego_motion: torch.Tensor,
    ) -> None:
        if images.ndim != 5:
            raise ValueError(f"images 期望 shape 为 [B, T, C, H, W]，实际为 {tuple(images.shape)}。")
        batch_size = int(images.shape[0])
        if target_points.ndim != 2:
            raise ValueError(
                "target_points 期望 shape 为 [B, 2]，"
                f"实际为 {tuple(target_points.shape)}。"
            )
        if int(target_points.shape[0]) != batch_size or int(target_points.shape[1]) != 2:
            raise ValueError(
                "target_points 必须与 images batch 对齐且最后一维为 2，"
                f"images batch={batch_size}，target_points shape={tuple(target_points.shape)}。"
            )
        if ego_motion.ndim != 2:
            raise ValueError(
                "ego_motion 期望 shape 为 [B, ego_motion_input_dim]，"
                f"实际为 {tuple(ego_motion.shape)}。"
            )
        if int(ego_motion.shape[0]) != batch_size or int(ego_motion.shape[1]) != self.config.ego_motion_input_dim:
            raise ValueError(
                "ego_motion 必须与 images batch 对齐且最后一维等于 ego_motion_input_dim，"
                f"images batch={batch_size}，期望维度={self.config.ego_motion_input_dim}，"
                f"实际 shape={tuple(ego_motion.shape)}。"
            )

    def _build_token_slices(self, vision_output: VisionEmbeddingOutput) -> BackboneTokenSlices:
        vision_start = 0
        vision_stop = vision_start + int(vision_output.tokens.shape[1])
        register_stop = vision_stop + self.config.register_token_count
        detection_stop = register_stop + self.detection_config.total_query_count
        agent_stop = register_stop + self.detection_config.agent_query_count
        trajectory_stop = detection_stop + self.trajectory_config.num_trajectories
        goal_stop = trajectory_stop + self.target_point_config.goal_token_count
        token_slices = BackboneTokenSlices(
            vision=slice(vision_start, vision_stop),
            register=slice(vision_stop, register_stop),
            detection=slice(register_stop, detection_stop),
            agent=slice(register_stop, agent_stop),
            map=slice(agent_stop, detection_stop),
            trajectory=slice(detection_stop, trajectory_stop),
            goal=slice(trajectory_stop, goal_stop),
        )
        if token_slices.total_length != self.config.expected_sequence_length:
            raise ValueError(
                "统一序列长度与配置不一致："
                f"期望 {self.config.expected_sequence_length}，实际为 {token_slices.total_length}。"
            )
        return token_slices

    def _build_visual_positions(
        self,
        latent_grid_shape: tuple[int, int, int],
        device: torch.device,
    ) -> torch.Tensor:
        if self.config.visual_position_order != "height_width_time":
            raise ValueError(f"不支持的 visual_position_order：{self.config.visual_position_order!r}。")
        latent_t, grid_h, grid_w = latent_grid_shape
        time_positions = _normalized_positions(
            latent_t,
            self.config.position_min,
            self.config.position_max,
            device,
        )
        height_positions = _normalized_positions(
            grid_h,
            self.config.position_min,
            self.config.position_max,
            device,
        )
        width_positions = _normalized_positions(
            grid_w,
            self.config.position_min,
            self.config.position_max,
            device,
        )
        # 视觉 token 顺序为 [T, H, W]，RoPE 位置最后一维按 [H, W, T] 解释。
        time_grid, height_grid, width_grid = torch.meshgrid(
            time_positions,
            height_positions,
            width_positions,
            indexing="ij",
        )
        return torch.stack((height_grid, width_grid, time_grid), dim=-1).reshape(-1, 3)

    def _build_input_sequence(
        self,
        vision_tokens: torch.Tensor,
        device: torch.device,
        token_slices: BackboneTokenSlices,
    ) -> _BackboneInputSequence:
        batch_size = int(vision_tokens.shape[0])
        register_tokens = self.register_tokens.to(device=device, dtype=torch.float32)
        detection_query_tokens = self.detection_query_embedding().to(
            device=device,
            dtype=torch.float32,
        )
        trajectory_tokens = self.trajectory_embedding().to(device=device, dtype=torch.float32)

        register_tokens = register_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        initial_detection_queries = detection_query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        trajectory_tokens = trajectory_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        vision_tokens = vision_tokens.to(dtype=torch.float32)
        vision_tokens = self._add_type_embedding(vision_tokens, "vision")
        register_tokens = self._add_type_embedding(register_tokens, "register")
        detection_tokens = self._add_detection_layer_identity_embeddings(
            initial_detection_queries.to(dtype=self.config.backbone_torch_dtype),
            layer_index=0,
            token_slices=token_slices,
        )
        trajectory_tokens = self._add_type_embedding(trajectory_tokens, "trajectory")
        return _BackboneInputSequence(
            token_features=torch.cat(
                (
                    vision_tokens,
                    register_tokens,
                    detection_tokens,
                    trajectory_tokens,
                ),
                dim=1,
            ),
            initial_detection_queries=initial_detection_queries,
        )

    def _append_goal_tokens(
        self,
        token_features: torch.Tensor,
        target_points: torch.Tensor,
        device: torch.device,
        token_slices: BackboneTokenSlices,
    ) -> torch.Tensor:
        if int(token_features.shape[1]) != token_slices.pre_goal_length:
            raise ValueError(
                "加入目标点 Token 前的序列长度必须等于 pre_goal_length，"
                f"期望 {token_slices.pre_goal_length}，实际为 {token_features.shape[1]}。"
            )
        goal_tokens = self.target_point_embedding(target_points.to(device=device))
        goal_tokens = self._add_type_embedding(goal_tokens, "goal")
        goal_tokens = goal_tokens.to(device=token_features.device, dtype=token_features.dtype)
        token_features = torch.cat((token_features, goal_tokens), dim=1)
        if int(token_features.shape[1]) != token_slices.total_length:
            raise ValueError(
                "加入目标点 Token 后的序列长度必须等于 total_length，"
                f"期望 {token_slices.total_length}，实际为 {token_features.shape[1]}。"
            )
        return token_features

    def _add_type_embedding(self, token_features: torch.Tensor, token_type: str) -> torch.Tensor:
        type_index = self.token_type_to_index[token_type]
        type_embedding = self.token_type_embeddings[type_index].to(
            device=token_features.device,
            dtype=torch.float32,
        )
        return token_features.to(dtype=torch.float32) + type_embedding

    def _add_detection_layer_identity_embeddings(
        self,
        detection_queries: torch.Tensor,
        layer_index: int,
        token_slices: BackboneTokenSlices,
    ) -> torch.Tensor:
        """为检测查询叠加指定层的 agent/map 专用身份嵌入。

        Args:
            detection_queries: `[B, 48, hidden_dim]`，不含身份嵌入的检测查询。
            layer_index: 检测旁路层号，取值范围 `[0, GOAL_TOKEN_INSERT_LAYER_INDEX)`。
            token_slices: 统一序列切片。

        Returns:
            `[B, 48, hidden_dim]`，叠加本层身份嵌入后的检测 Token。
        """
        if layer_index < 0 or layer_index >= GOAL_TOKEN_INSERT_LAYER_INDEX:
            raise ValueError(
                "layer_index 必须满足 0 <= layer_index < GOAL_TOKEN_INSERT_LAYER_INDEX，"
                f"实际为 {layer_index}。"
            )
        agent_count = token_slices.agent.stop - token_slices.agent.start
        map_count = token_slices.map.stop - token_slices.map.start
        layer_embeddings = self.detection_layer_identity_embeddings[layer_index].to(
            device=detection_queries.device,
            dtype=detection_queries.dtype,
        )
        agent_tokens = detection_queries[:, :agent_count, :] + layer_embeddings[0]
        map_tokens = (
            detection_queries[:, agent_count : agent_count + map_count, :]
            + layer_embeddings[1]
        )
        return torch.cat((agent_tokens, map_tokens), dim=1)

    def _apply_detection_layer_residual(
        self,
        token_features: torch.Tensor,
        accumulated_detection_queries: torch.Tensor,
        layer_index: int,
        token_slices: BackboneTokenSlices,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """在第 layer_index 层 Transformer 输出后执行检测 Token 旁路残差更新。

        Args:
            token_features: `[B, L, hidden_dim]`，当前层 Transformer 输出。
            accumulated_detection_queries: `[B, 48, hidden_dim]`，不含身份嵌入的旁路累积状态。
            layer_index: 当前层号，取值范围 `[0, GOAL_TOKEN_INSERT_LAYER_INDEX)`。
            token_slices: 统一序列切片。

        Returns:
            写回检测切片后的 `token_features` 与更新后的 `accumulated_detection_queries`。
        """
        detection_output = token_features[:, token_slices.detection, :]
        detection_output = self.detection_residual_norms[layer_index](detection_output)
        detection_residual = self.detection_residual_projections[layer_index](detection_output)
        accumulated_detection_queries = accumulated_detection_queries + detection_residual
        updated_detection_tokens = self._add_detection_layer_identity_embeddings(
            accumulated_detection_queries,
            layer_index,
            token_slices,
        )
        token_features = token_features.clone()
        token_features[:, token_slices.detection, :] = updated_detection_tokens
        return token_features, accumulated_detection_queries

    def _split_sequence_features(
        self,
        sequence_features: torch.Tensor,
        token_slices: BackboneTokenSlices,
    ) -> dict[str, torch.Tensor]:
        return {
            "vision": sequence_features[:, token_slices.vision, :],
            "register": sequence_features[:, token_slices.register, :],
            "detection": sequence_features[:, token_slices.detection, :],
            "trajectory": sequence_features[:, token_slices.trajectory, :],
            "goal": sequence_features[:, token_slices.goal, :],
        }

    def _prepare_trajectory_decoder_features(
        self,
        trajectory_features: torch.Tensor,
        ego_motion: torch.Tensor,
    ) -> torch.Tensor:
        with _disabled_autocast(trajectory_features):
            ego_motion_fp32 = ego_motion.to(device=trajectory_features.device, dtype=torch.float32)
            if self.config.ego_motion_vector_transform == "symlog":
                ego_motion_fp32 = torch.sign(ego_motion_fp32) * torch.log1p(torch.abs(ego_motion_fp32))
            else:
                raise ValueError(
                    f"不支持的 ego_motion_vector_transform：{self.config.ego_motion_vector_transform!r}。"
                )
            ego_motion_features = self.ego_motion_encoder(ego_motion_fp32)
            return trajectory_features.to(dtype=torch.float32) + ego_motion_features[:, None, :]

def load_backbone_config(
    config_path: str | Path,
    project_root: str | Path | None = None,
) -> BackboneConfig:
    """读取统一主干 TOML 配置。"""

    resolved_config_path = Path(config_path).resolve()
    resolved_project_root = (
        Path(project_root).resolve() if project_root is not None else resolved_config_path.parent.parent
    )
    _ensure_project_relative_path(resolved_config_path, resolved_project_root, "config_path")
    with resolved_config_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)

    modules_config = _require_table(raw_config, "modules")
    architecture_config = _require_table(raw_config, "architecture")
    attention_config = _require_table(raw_config, "attention")
    feed_forward_config = _require_table(raw_config, "feed_forward")
    rope_config = _require_table(raw_config, "rope")
    identity_config = _require_table(raw_config, "identity")
    ego_motion_config = _require_table(raw_config, "ego_motion")
    precision_config = _require_table(raw_config, "precision")
    initialization_config = _require_table(raw_config, "initialization")

    return BackboneConfig(
        project_root=resolved_project_root,
        vision_config_path=_resolve_project_relative_config_path(
            modules_config,
            "vision_config_path",
            resolved_project_root,
        ),
        target_point_config_path=_resolve_project_relative_config_path(
            modules_config,
            "target_point_config_path",
            resolved_project_root,
        ),
        trajectory_vocab_config_path=_resolve_project_relative_config_path(
            modules_config,
            "trajectory_vocab_config_path",
            resolved_project_root,
        ),
        detection_head_config_path=_resolve_project_relative_config_path(
            modules_config,
            "detection_head_config_path",
            resolved_project_root,
        ),
        hidden_dim=_require_int(architecture_config, "hidden_dim"),
        layer_count=_require_int(architecture_config, "layer_count"),
        attention_head_count=_require_int(architecture_config, "attention_head_count"),
        register_token_count=_require_int(architecture_config, "register_token_count"),
        expected_sequence_length=_require_int(architecture_config, "expected_sequence_length"),
        token_order=_require_string(architecture_config, "token_order"),
        modal_ffn_layer_indices=_require_int_tuple(architecture_config, "modal_ffn_layer_indices"),
        rms_norm_eps=_require_float(architecture_config, "rms_norm_eps"),
        rope_head_count=_require_int(attention_config, "rope_head_count"),
        attention_dropout=_require_float(attention_config, "attention_dropout"),
        ffn_layer1_output_dim=_require_int(feed_forward_config, "ffn_layer1_output_dim"),
        ffn_activation=_require_string(feed_forward_config, "ffn_activation"),
        rope_theta=_require_float(rope_config, "theta"),
        rope_axis_dims=_require_3d_int_tuple(rope_config, "axis_dims"),
        visual_position_order=_require_string(rope_config, "visual_position_order"),
        position_min=_require_float(rope_config, "position_min"),
        position_max=_require_float(rope_config, "position_max"),
        token_type_order=_require_string_tuple(identity_config, "token_type_order"),
        ego_motion_input_dim=_require_int(ego_motion_config, "input_dim"),
        ego_motion_vector_transform=_require_string(ego_motion_config, "vector_transform"),
        backbone_dtype=_require_string(precision_config, "backbone_dtype"),
        attention_dtype=_require_string(precision_config, "attention_dtype"),
        register_token_std=_require_float(initialization_config, "register_token_std"),
        token_type_embedding_std=_require_float(initialization_config, "token_type_embedding_std"),
        ego_motion_linear_std=_require_float(initialization_config, "ego_motion_linear_std"),
    )


def override_backbone_precision(
    config: BackboneConfig,
    backbone_dtype: str,
    attention_dtype: str,
) -> BackboneConfig:
    """返回只替换主干和注意力精度的新配置。"""

    _validate_dtype_name(backbone_dtype, "backbone_dtype")
    _validate_dtype_name(attention_dtype, "attention_dtype")
    return replace(config, backbone_dtype=backbone_dtype, attention_dtype=attention_dtype)


def _precision_context(device: torch.device, dtype: torch.dtype) -> Any:
    if dtype == torch.float32:
        return nullcontext()
    if dtype != torch.bfloat16:
        raise ValueError(f"当前仅支持 float32 和 bfloat16，实际为 {dtype}。")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"当前设备不支持 bfloat16 autocast：device.type={device.type!r}。")
    return torch.autocast(device_type=device.type, dtype=dtype)


def _disabled_autocast(reference_tensor: torch.Tensor) -> Any:
    if reference_tensor.device.type == "meta":
        return nullcontext()
    try:
        return torch.autocast(device_type=reference_tensor.device.type, enabled=False)
    except (RuntimeError, ValueError):
        return nullcontext()


def _normalized_positions(
    count: int,
    position_min: float,
    position_max: float,
    device: torch.device,
) -> torch.Tensor:
    if count <= 0:
        raise ValueError(f"位置数量必须为正整数，实际为 {count}。")
    if count == 1:
        return torch.tensor([(position_min + position_max) * 0.5], device=device, dtype=torch.float32)
    return torch.linspace(position_min, position_max, count, device=device, dtype=torch.float32)


def _validate_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} 必须为整数，实际为 {value!r}。")
    if value <= 0:
        raise ValueError(f"{field_name} 必须为正整数，实际为 {value}。")


def _validate_dtype_name(dtype_name: str, field_name: str) -> None:
    if dtype_name not in SUPPORTED_DTYPE_NAMES:
        raise ValueError(
            f"{field_name} 仅支持 {sorted(SUPPORTED_DTYPE_NAMES)}，实际为 {dtype_name!r}。"
        )


def _dtype_from_name(dtype_name: str) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"dtype 仅支持 {sorted(SUPPORTED_DTYPE_NAMES)}，实际为 {dtype_name!r}。")


def _validate_axis_dims(axis_dims: tuple[int, int, int], head_dim: int) -> None:
    if len(axis_dims) != 3:
        raise ValueError(f"rope_axis_dims 必须包含 3 个整数，实际为 {axis_dims}。")
    for axis_index, axis_dim in enumerate(axis_dims):
        if not isinstance(axis_dim, int) or isinstance(axis_dim, bool):
            raise TypeError(f"rope_axis_dims[{axis_index}] 必须为整数，实际为 {axis_dim!r}。")
        if axis_dim <= 0 or axis_dim % 2 != 0:
            raise ValueError(f"rope_axis_dims[{axis_index}] 必须为正偶数，实际为 {axis_dim}。")
    if sum(axis_dims) > head_dim:
        raise ValueError(
            "rope_axis_dims 总和不能超过单头通道数，"
            f"实际为 {sum(axis_dims)} > {head_dim}。"
        )


def _force_parameter_to_float32(parameter: nn.Parameter) -> None:
    if parameter.is_floating_point() and parameter.dtype != torch.float32:
        parameter.data = parameter.data.to(dtype=torch.float32)
    if parameter.grad is not None and parameter.grad.is_floating_point():
        parameter.grad.data = parameter.grad.data.to(dtype=torch.float32)


def _force_floating_tensors_to_float32(module: nn.Module) -> None:
    with torch.no_grad():
        for parameter in module.parameters(recurse=True):
            _force_parameter_to_float32(parameter)
        for buffer in module.buffers(recurse=True):
            if buffer.is_floating_point() and buffer.dtype != torch.float32:
                buffer.data = buffer.data.to(dtype=torch.float32)


def _resolve_project_relative_config_path(
    table: Mapping[str, Any],
    key: str,
    project_root: Path,
) -> Path:
    path_text = _require_string(table, key)
    raw_path = Path(path_text)
    if raw_path.is_absolute():
        raise ValueError(f"{key} 必须是项目内相对路径，实际为 {raw_path}。")
    resolved_path = (project_root / raw_path).resolve()
    _ensure_project_relative_path(resolved_path, project_root, key)
    return resolved_path


def _ensure_project_relative_path(path: Path, project_root: Path, config_key: str) -> None:
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(
            f"{config_key} 必须解析到项目目录内，项目根目录为 {project_root}，实际为 {path}。"
        ) from exc


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


def _require_int_tuple(table: Mapping[str, Any], key: str) -> tuple[int, ...]:
    value = table.get(key)
    if not isinstance(value, list):
        raise ValueError(f"配置项 {key} 必须为整数列表，实际为 {value!r}。")
    values = []
    for index, item in enumerate(value):
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(f"配置项 {key}[{index}] 必须为整数，实际为 {item!r}。")
        values.append(item)
    return tuple(values)


def _require_3d_int_tuple(table: Mapping[str, Any], key: str) -> tuple[int, int, int]:
    values = _require_int_tuple(table, key)
    if len(values) != 3:
        raise ValueError(f"配置项 {key} 必须包含 3 个整数，实际为 {values}。")
    return (values[0], values[1], values[2])


def _require_string_tuple(table: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = table.get(key)
    if not isinstance(value, list):
        raise ValueError(f"配置项 {key} 必须为字符串列表，实际为 {value!r}。")
    values = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise ValueError(f"配置项 {key}[{index}] 必须为非空字符串，实际为 {item!r}。")
        values.append(item)
    return tuple(values)
