"""训练阶段的数据读取、校验、匹配和标签构造。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any, Mapping, NamedTuple, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from data.b2d_dataset import B2DH5Dataset
from model.detection_head import (
    DetectionDecoderOutput,
    DetectionHeadConfig,
    load_detection_head_config,
)
from model.trajectory_vocab import (
    TrajectoryDecoderOutput,
    TrajectoryVocabData,
    TrajectoryVocabModelConfig,
    load_trajectory_vocab_config,
    load_trajectory_vocabulary,
)


__all__ = [
    "AgentMatchingTargets",
    "MapMatchingTargets",
    "TrainingBatchLabels",
    "TrainingDataConfig",
    "TrajectoryVocabLabels",
    "ValidatedTrainingDataset",
    "build_agent_matching_targets",
    "build_map_matching_targets",
    "build_training_batch_labels",
    "build_training_dataset",
    "build_trajectory_vocab_labels",
    "inverse_symlog",
    "load_training_data_config",
    "symlog",
    "training_collate",
]


@dataclass(frozen=True)
class TrainingDataConfig:
    """训练数据处理配置。

    Args:
        project_root: 项目根目录。
        h5_paths: 逐场景 H5 文件路径。目录配置会在加载时展开为当前目录下的 `*.h5`。
        normalize_images: 是否让底层 `B2DH5Dataset` 将图像归一化到 `[0, 1]`。
        random_target_point: 是否从有效目标点候选中随机采样。
        scan_on_init: 是否在数据集初始化时扫描并剔除无效样本。
        detection_head_config_path: 检测头配置路径，仅引用已有配置。
        trajectory_vocab_config_path: 轨迹词表配置路径，仅引用已有配置。

    Shape:
        数据集样本沿用 `B2DH5Dataset` 输出约定；匹配和标签构造函数统一使用 batch 维 `[B, ...]`。
    """

    project_root: Path
    h5_paths: tuple[Path, ...]
    normalize_images: bool
    random_target_point: bool
    scan_on_init: bool
    detection_head_config_path: Path
    trajectory_vocab_config_path: Path
    image_min: float
    image_max: float
    future_trajectory_abs_max_m: float
    ego_motion_abs_max: float
    agent_position_abs_max_m: float
    agent_size_min_m: float
    agent_size_max_m: float
    agent_motion_abs_max: float
    map_position_abs_max_m: float
    inverse_mse_eps: float
    inverse_mse_max_logit: float
    label_normalize_max: bool
    agent_class_cost_weight: float
    agent_center_cost_weight: float
    agent_size_cost_weight: float
    agent_yaw_cost_weight: float
    agent_future_cost_weight: float
    map_class_cost_weight: float
    map_point_cost_weight: float
    map_bidirectional_class_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.h5_paths:
            raise ValueError("h5_paths 不能为空，至少需要一个 H5 文件。")
        for h5_path in self.h5_paths:
            _ensure_project_relative_path(h5_path.resolve(), self.project_root, "h5_paths")
            if h5_path.suffix.lower() != ".h5":
                raise ValueError(f"h5_paths 只能包含 .h5 文件，实际为 {h5_path}。")
        _ensure_project_relative_path(
            self.detection_head_config_path.resolve(),
            self.project_root,
            "detection_head_config_path",
        )
        _ensure_project_relative_path(
            self.trajectory_vocab_config_path.resolve(),
            self.project_root,
            "trajectory_vocab_config_path",
        )
        if self.image_min >= self.image_max:
            raise ValueError(
                f"image_min 必须小于 image_max，实际为 {self.image_min} 和 {self.image_max}。"
            )
        for field_name in (
            "future_trajectory_abs_max_m",
            "ego_motion_abs_max",
            "agent_position_abs_max_m",
            "agent_size_min_m",
            "agent_size_max_m",
            "agent_motion_abs_max",
            "map_position_abs_max_m",
            "inverse_mse_eps",
            "inverse_mse_max_logit",
            "agent_class_cost_weight",
            "agent_center_cost_weight",
            "agent_size_cost_weight",
            "agent_yaw_cost_weight",
            "agent_future_cost_weight",
            "map_class_cost_weight",
            "map_point_cost_weight",
        ):
            value = getattr(self, field_name)
            if value <= 0.0:
                raise ValueError(f"{field_name} 必须为正数，实际为 {value}。")
        if self.agent_size_min_m >= self.agent_size_max_m:
            raise ValueError(
                "agent_size_min_m 必须小于 agent_size_max_m，"
                f"实际为 {self.agent_size_min_m} 和 {self.agent_size_max_m}。"
            )


class TrajectoryVocabLabels(NamedTuple):
    """轨迹词表监督标签。

    Shape:
        `soft_labels`: `[B, V]`。
        `winner_indices`: `[B]`。
        `residual_targets`: `[B, V, K, 2]`，只在 winner 位置有效。
        `residual_mask`: `[B, V]`。
        `predicted_trajectories_m`: `[B, V, K, 2]`，模型输出反变换后的物理空间轨迹。
    """

    soft_labels: torch.Tensor
    winner_indices: torch.Tensor
    residual_targets: torch.Tensor
    residual_mask: torch.Tensor
    predicted_trajectories_m: torch.Tensor


class AgentMatchingTargets(NamedTuple):
    """Agent 匈牙利匹配和监督目标。

    Shape:
        `class_targets`: `[B, Q_agent]`。
        `state_targets`: `[B, Q_agent, state_dim]`。
        `state_mask`: `[B, Q_agent]`。
        `mode_targets`: `[B, Q_agent]`。
        `future_targets`: `[B, Q_agent, M, K, 2]`。
        `future_mask`: `[B, Q_agent, M, K]`，只在匹配 query 的 winner mode 和有效未来点为真。
    """

    class_targets: torch.Tensor
    state_targets: torch.Tensor
    state_mask: torch.Tensor
    mode_targets: torch.Tensor
    future_targets: torch.Tensor
    future_mask: torch.Tensor
    matched_query_indices: tuple[torch.Tensor, ...]
    matched_gt_indices: tuple[torch.Tensor, ...]


class MapMatchingTargets(NamedTuple):
    """Map 匈牙利匹配和监督目标。

    Shape:
        `class_targets`: `[B, Q_map]`。
        `point_targets`: `[B, Q_map, P, 2]`。
        `point_mask`: `[B, Q_map]`。
    """

    class_targets: torch.Tensor
    point_targets: torch.Tensor
    point_mask: torch.Tensor
    matched_query_indices: tuple[torch.Tensor, ...]
    matched_gt_indices: tuple[torch.Tensor, ...]


class TrainingBatchLabels(NamedTuple):
    """训练 batch 的全部数据处理输出。"""

    trajectory: TrajectoryVocabLabels
    agent: AgentMatchingTargets
    map: MapMatchingTargets


class ValidatedTrainingDataset(Dataset[dict[str, Any]]):
    """包装 `B2DH5Dataset`，剔除含 NaN、Inf 或明显越界的训练样本。"""

    def __init__(self, base_dataset: B2DH5Dataset, config: TrainingDataConfig) -> None:
        self.base_dataset = base_dataset
        self.config = config
        if config.scan_on_init:
            valid_indices = []
            for sample_index in range(len(base_dataset)):
                sample = base_dataset[sample_index]
                if _is_valid_sample(sample, config):
                    valid_indices.append(sample_index)
            if not valid_indices:
                raise ValueError("扫描 H5 后没有发现任何有效训练样本。")
            self.valid_indices = tuple(valid_indices)
        else:
            self.valid_indices = tuple(range(len(base_dataset)))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base_dataset[self.valid_indices[index]]
        if not _is_valid_sample(sample, self.config):
            raise ValueError(
                "训练样本在读取时未通过数值校验，"
                f"index={index}，source_index={self.valid_indices[index]}，"
                f"h5_path={sample.get('h5_path')}，current_frame_id={sample.get('current_frame_id')}。"
            )
        return sample

    def close(self) -> None:
        """关闭底层 H5 句柄。"""

        self.base_dataset.close()


def load_training_data_config(
    config_path: str | Path,
    project_root: str | Path | None = None,
) -> TrainingDataConfig:
    """读取训练数据处理 TOML 配置。"""

    resolved_config_path = Path(config_path).resolve()
    resolved_project_root = (
        Path(project_root).resolve() if project_root is not None else resolved_config_path.parent.parent
    )
    _ensure_project_relative_path(resolved_config_path, resolved_project_root, "config_path")
    with resolved_config_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)

    dataset_config = _require_table(raw_config, "dataset")
    modules_config = _require_table(raw_config, "modules")
    validation_config = _require_table(raw_config, "validation")
    trajectory_label_config = _require_table(raw_config, "trajectory_label")
    agent_matching_config = _require_table(raw_config, "agent_matching")
    map_matching_config = _require_table(raw_config, "map_matching")

    h5_paths = _resolve_h5_paths(dataset_config, resolved_project_root)
    return TrainingDataConfig(
        project_root=resolved_project_root,
        h5_paths=tuple(h5_paths),
        normalize_images=_require_bool(dataset_config, "normalize_images"),
        random_target_point=_require_bool(dataset_config, "random_target_point"),
        scan_on_init=_require_bool(dataset_config, "scan_on_init"),
        detection_head_config_path=_resolve_project_relative_path(
            modules_config,
            "detection_head_config_path",
            resolved_project_root,
        ),
        trajectory_vocab_config_path=_resolve_project_relative_path(
            modules_config,
            "trajectory_vocab_config_path",
            resolved_project_root,
        ),
        image_min=_require_float(validation_config, "image_min"),
        image_max=_require_float(validation_config, "image_max"),
        future_trajectory_abs_max_m=_require_float(
            validation_config,
            "future_trajectory_abs_max_m",
        ),
        ego_motion_abs_max=_require_float(validation_config, "ego_motion_abs_max"),
        agent_position_abs_max_m=_require_float(validation_config, "agent_position_abs_max_m"),
        agent_size_min_m=_require_float(validation_config, "agent_size_min_m"),
        agent_size_max_m=_require_float(validation_config, "agent_size_max_m"),
        agent_motion_abs_max=_require_float(validation_config, "agent_motion_abs_max"),
        map_position_abs_max_m=_require_float(validation_config, "map_position_abs_max_m"),
        inverse_mse_eps=_require_float(trajectory_label_config, "inverse_mse_eps"),
        inverse_mse_max_logit=_require_float(trajectory_label_config, "inverse_mse_max_logit"),
        label_normalize_max=_require_bool(trajectory_label_config, "label_normalize_max"),
        agent_class_cost_weight=_require_float(agent_matching_config, "class_cost_weight"),
        agent_center_cost_weight=_require_float(agent_matching_config, "center_cost_weight"),
        agent_size_cost_weight=_require_float(agent_matching_config, "size_cost_weight"),
        agent_yaw_cost_weight=_require_float(agent_matching_config, "yaw_cost_weight"),
        agent_future_cost_weight=_require_float(agent_matching_config, "future_cost_weight"),
        map_class_cost_weight=_require_float(map_matching_config, "class_cost_weight"),
        map_point_cost_weight=_require_float(map_matching_config, "point_cost_weight"),
        map_bidirectional_class_names=_require_string_tuple(
            map_matching_config,
            "bidirectional_class_names",
        ),
    )


def build_training_dataset(config: TrainingDataConfig) -> ValidatedTrainingDataset:
    """按配置构建经过校验的训练数据集。"""

    base_dataset = B2DH5Dataset(
        list(config.h5_paths),
        normalize_images=config.normalize_images,
        random_target_point=config.random_target_point,
    )
    return ValidatedTrainingDataset(base_dataset, config)


def training_collate(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """将样本列表合并为 batch，张量字段堆叠，元信息字段保留为列表。"""

    if not samples:
        raise ValueError("samples 不能为空。")
    batch: dict[str, Any] = {}
    keys = samples[0].keys()
    for key in keys:
        values = [sample[key] for sample in samples]
        if all(isinstance(value, torch.Tensor) for value in values):
            batch[key] = torch.stack(values, dim=0)
        else:
            batch[key] = values
    return batch


def build_training_batch_labels(
    detection_output: DetectionDecoderOutput,
    trajectory_output: TrajectoryDecoderOutput,
    batch: Mapping[str, Any],
    data_config: TrainingDataConfig,
    detection_config: DetectionHeadConfig | None = None,
    trajectory_config: TrajectoryVocabModelConfig | None = None,
    vocabulary: TrajectoryVocabData | None = None,
) -> TrainingBatchLabels:
    """从模型输出和 H5 batch 构造训练监督标签。"""

    detection_config = detection_config or load_detection_head_config(
        data_config.detection_head_config_path,
        project_root=data_config.project_root,
    )
    trajectory_config = trajectory_config or load_trajectory_vocab_config(
        data_config.trajectory_vocab_config_path,
        project_root=data_config.project_root,
    )
    vocabulary = vocabulary or load_trajectory_vocabulary(
        trajectory_config,
        device=trajectory_output.logits.device,
    )
    return TrainingBatchLabels(
        trajectory=build_trajectory_vocab_labels(
            trajectory_output,
            batch["future_trajectory"],
            vocabulary,
            data_config,
        ),
        agent=build_agent_matching_targets(
            detection_output,
            batch,
            detection_config,
            data_config,
        ),
        map=build_map_matching_targets(
            detection_output,
            batch,
            detection_config,
            data_config,
        ),
    )


def build_trajectory_vocab_labels(
    trajectory_output: TrajectoryDecoderOutput,
    future_trajectory_m: torch.Tensor,
    vocabulary: TrajectoryVocabData,
    config: TrainingDataConfig,
) -> TrajectoryVocabLabels:
    """构造轨迹词表 soft label 和 winner-only 残差标签。"""

    logits = trajectory_output.logits.to(dtype=torch.float32)
    residuals = trajectory_output.residuals.to(device=logits.device, dtype=torch.float32)
    gt_trajectory_m = future_trajectory_m.to(device=logits.device, dtype=torch.float32)
    vocab_m = vocabulary.trajectory_vocab_m.to(device=logits.device, dtype=torch.float32)
    vocab_symlog = vocabulary.trajectory_vocab_symlog.to(device=logits.device, dtype=torch.float32)

    _validate_shape(logits, 2, "trajectory_output.logits")
    _validate_shape(residuals, 4, "trajectory_output.residuals")
    if residuals.shape[:2] != logits.shape:
        raise ValueError(
            "trajectory_output.residuals 前两维必须等于 logits，"
            f"实际为 {tuple(residuals.shape)} 和 {tuple(logits.shape)}。"
        )
    if gt_trajectory_m.shape != (logits.shape[0], residuals.shape[2], residuals.shape[3]):
        raise ValueError(
            "future_trajectory_m shape 与模型输出不一致，"
            f"期望 {(logits.shape[0], residuals.shape[2], residuals.shape[3])}，"
            f"实际为 {tuple(gt_trajectory_m.shape)}。"
        )

    predicted_trajectories_m = inverse_symlog(vocab_symlog.unsqueeze(0) + residuals)
    mse = (vocab_m.unsqueeze(0) - gt_trajectory_m[:, None, :, :]).square().mean(dim=(2, 3))
    inverse_mse = 1.0 / (mse + config.inverse_mse_eps)
    normalized_logits = config.inverse_mse_max_logit * inverse_mse / inverse_mse.max(
        dim=1,
        keepdim=True,
    ).values.clamp_min(config.inverse_mse_eps)
    soft_labels = torch.softmax(normalized_logits, dim=1)
    if config.label_normalize_max:
        soft_labels = soft_labels / soft_labels.max(dim=1, keepdim=True).values.clamp_min(
            config.inverse_mse_eps
        )
    winner_indices = torch.argmin(mse, dim=1)
    gt_symlog = symlog(gt_trajectory_m)
    residual_targets = torch.zeros_like(residuals)
    batch_indices = torch.arange(logits.shape[0], device=logits.device)
    residual_targets[batch_indices, winner_indices] = gt_symlog - vocab_symlog[winner_indices]
    residual_mask = torch.zeros_like(logits, dtype=torch.bool)
    residual_mask[batch_indices, winner_indices] = True
    return TrajectoryVocabLabels(
        soft_labels=soft_labels,
        winner_indices=winner_indices,
        residual_targets=residual_targets,
        residual_mask=residual_mask,
        predicted_trajectories_m=predicted_trajectories_m,
    )


def build_agent_matching_targets(
    detection_output: DetectionDecoderOutput,
    batch: Mapping[str, Any],
    detection_config: DetectionHeadConfig,
    data_config: TrainingDataConfig,
) -> AgentMatchingTargets:
    """在物理空间执行 Agent 匈牙利匹配，并返回监督空间目标。"""

    device = detection_output.agent_class_logits.device
    class_logits = detection_output.agent_class_logits.to(dtype=torch.float32)
    pred_states = detection_output.agent_states.to(dtype=torch.float32)
    pred_future = inverse_symlog(detection_output.agent_future_trajectories.to(dtype=torch.float32))
    gt_boxes = batch["agent_boxes"].to(device=device, dtype=torch.float32)
    gt_classes = batch["agent_classes"].to(device=device, dtype=torch.long)
    gt_valid = batch["agent_valid"].to(device=device, dtype=torch.bool)
    gt_future = batch["agent_future_trajectory"].to(device=device, dtype=torch.float32)
    gt_future_valid = batch["agent_future_valid"].to(device=device, dtype=torch.bool)

    batch_size, query_count, _class_count = class_logits.shape
    none_index = len(detection_config.agent_class_names)
    class_targets = torch.full((batch_size, query_count), none_index, device=device, dtype=torch.long)
    state_targets = torch.zeros_like(pred_states)
    state_mask = torch.zeros((batch_size, query_count), device=device, dtype=torch.bool)
    mode_targets = torch.zeros((batch_size, query_count), device=device, dtype=torch.long)
    future_targets = torch.zeros_like(detection_output.agent_future_trajectories, dtype=torch.float32)
    future_mask = torch.zeros(
        (
            batch_size,
            query_count,
            detection_config.agent_future_mode_count,
            detection_config.agent_future_points,
        ),
        device=device,
        dtype=torch.bool,
    )
    matched_query_indices: list[torch.Tensor] = []
    matched_gt_indices: list[torch.Tensor] = []

    pred_xy = inverse_symlog(pred_states[..., _state_index(detection_config, "x") : _state_index(detection_config, "y") + 1])
    pred_lwh = torch.stack(
        (
            torch.expm1(pred_states[..., _state_index(detection_config, "length_log1p")]),
            torch.expm1(pred_states[..., _state_index(detection_config, "width_log1p")]),
            torch.expm1(pred_states[..., _state_index(detection_config, "height_log1p")]),
        ),
        dim=-1,
    )
    pred_yaw_vector = F.normalize(
        torch.stack(
            (
                pred_states[..., _state_index(detection_config, "sin_yaw")],
                pred_states[..., _state_index(detection_config, "cos_yaw")],
            ),
            dim=-1,
        ),
        dim=-1,
        eps=1e-6,
    )
    class_cost_source = -torch.log_softmax(class_logits, dim=-1)

    for batch_index in range(batch_size):
        valid_gt_indices = torch.nonzero(gt_valid[batch_index], as_tuple=False).flatten()
        if valid_gt_indices.numel() == 0:
            matched_query_indices.append(torch.empty(0, device=device, dtype=torch.long))
            matched_gt_indices.append(torch.empty(0, device=device, dtype=torch.long))
            continue

        gt_box = gt_boxes[batch_index, valid_gt_indices]
        gt_class = gt_classes[batch_index, valid_gt_indices].clamp(min=0, max=none_index - 1)
        gt_xy = gt_box[:, 0:2]
        gt_lwh = gt_box[:, 2:5]
        gt_yaw_vector = torch.stack((torch.sin(gt_box[:, 5]), torch.cos(gt_box[:, 5])), dim=-1)
        center_cost = torch.cdist(pred_xy[batch_index], gt_xy, p=1)
        size_cost = torch.cdist(pred_lwh[batch_index], gt_lwh, p=1)
        yaw_cost = torch.cdist(pred_yaw_vector[batch_index], gt_yaw_vector, p=1)
        class_cost = class_cost_source[batch_index][:, gt_class]
        future_cost = _agent_future_cost(
            pred_future[batch_index],
            gt_future[batch_index, valid_gt_indices],
            gt_future_valid[batch_index, valid_gt_indices],
        )
        cost_matrix = (
            data_config.agent_class_cost_weight * class_cost
            + data_config.agent_center_cost_weight * center_cost
            + data_config.agent_size_cost_weight * size_cost
            + data_config.agent_yaw_cost_weight * yaw_cost
            + data_config.agent_future_cost_weight * future_cost
        )
        query_indices, local_gt_indices = _linear_sum_assignment(cost_matrix)
        gt_indices = valid_gt_indices[local_gt_indices]
        matched_query_indices.append(query_indices)
        matched_gt_indices.append(gt_indices)

        class_targets[batch_index, query_indices] = gt_classes[batch_index, gt_indices].clamp(
            min=0,
            max=none_index - 1,
        )
        state_targets[batch_index, query_indices] = _encode_agent_state_targets(
            gt_boxes[batch_index, gt_indices],
            detection_config,
        )
        state_mask[batch_index, query_indices] = True
        winner_modes = _winner_agent_modes(
            pred_future[batch_index, query_indices],
            gt_future[batch_index, gt_indices],
            gt_future_valid[batch_index, gt_indices],
        )
        mode_targets[batch_index, query_indices] = winner_modes
        future_targets[batch_index, query_indices, winner_modes] = symlog(
            gt_future[batch_index, gt_indices]
        )
        future_mask[batch_index, query_indices, winner_modes] = gt_future_valid[
            batch_index,
            gt_indices,
        ]

    return AgentMatchingTargets(
        class_targets=class_targets,
        state_targets=state_targets,
        state_mask=state_mask,
        mode_targets=mode_targets,
        future_targets=future_targets,
        future_mask=future_mask,
        matched_query_indices=tuple(matched_query_indices),
        matched_gt_indices=tuple(matched_gt_indices),
    )


def build_map_matching_targets(
    detection_output: DetectionDecoderOutput,
    batch: Mapping[str, Any],
    detection_config: DetectionHeadConfig,
    data_config: TrainingDataConfig,
) -> MapMatchingTargets:
    """在物理空间执行 Map 匈牙利匹配，并返回监督空间目标。"""

    device = detection_output.map_class_logits.device
    class_logits = detection_output.map_class_logits.to(dtype=torch.float32)
    pred_points = inverse_symlog(detection_output.map_points.to(dtype=torch.float32))
    gt_points = batch["map_points"].to(device=device, dtype=torch.float32)
    gt_classes = batch["map_classes"].to(device=device, dtype=torch.long)
    gt_valid = batch["map_valid"].to(device=device, dtype=torch.bool)

    batch_size, query_count, _class_count = class_logits.shape
    none_index = len(detection_config.map_class_names)
    class_targets = torch.full((batch_size, query_count), none_index, device=device, dtype=torch.long)
    point_targets = torch.zeros_like(detection_output.map_points, dtype=torch.float32)
    point_mask = torch.zeros((batch_size, query_count), device=device, dtype=torch.bool)
    matched_query_indices: list[torch.Tensor] = []
    matched_gt_indices: list[torch.Tensor] = []
    class_cost_source = -torch.log_softmax(class_logits, dim=-1)
    bidirectional_class_indices = {
        detection_config.map_class_names.index(name)
        for name in data_config.map_bidirectional_class_names
        if name in detection_config.map_class_names
    }

    for batch_index in range(batch_size):
        valid_gt_indices = torch.nonzero(gt_valid[batch_index], as_tuple=False).flatten()
        if valid_gt_indices.numel() == 0:
            matched_query_indices.append(torch.empty(0, device=device, dtype=torch.long))
            matched_gt_indices.append(torch.empty(0, device=device, dtype=torch.long))
            continue

        gt_class = gt_classes[batch_index, valid_gt_indices].clamp(min=0, max=none_index - 1)
        class_cost = class_cost_source[batch_index][:, gt_class]
        point_cost, reverse_target_mask = _map_point_cost(
            pred_points[batch_index],
            gt_points[batch_index, valid_gt_indices],
            gt_class,
            bidirectional_class_indices,
        )
        cost_matrix = (
            data_config.map_class_cost_weight * class_cost
            + data_config.map_point_cost_weight * point_cost
        )
        query_indices, local_gt_indices = _linear_sum_assignment(cost_matrix)
        gt_indices = valid_gt_indices[local_gt_indices]
        matched_query_indices.append(query_indices)
        matched_gt_indices.append(gt_indices)
        class_targets[batch_index, query_indices] = gt_classes[batch_index, gt_indices].clamp(
            min=0,
            max=none_index - 1,
        )
        matched_points = gt_points[batch_index, gt_indices]
        matched_reverse_mask = reverse_target_mask[query_indices, local_gt_indices]
        if bool(matched_reverse_mask.any().item()):
            matched_points = matched_points.clone()
            matched_points[matched_reverse_mask] = torch.flip(
                matched_points[matched_reverse_mask],
                dims=(1,),
            )
        point_targets[batch_index, query_indices] = symlog(matched_points)
        point_mask[batch_index, query_indices] = True

    return MapMatchingTargets(
        class_targets=class_targets,
        point_targets=point_targets,
        point_mask=point_mask,
        matched_query_indices=tuple(matched_query_indices),
        matched_gt_indices=tuple(matched_gt_indices),
    )


def symlog(values: torch.Tensor) -> torch.Tensor:
    """计算 `sign(x) * ln(abs(x) + 1)`。"""

    values_fp32 = values.to(dtype=torch.float32)
    return torch.sign(values_fp32) * torch.log1p(torch.abs(values_fp32))


def inverse_symlog(values: torch.Tensor) -> torch.Tensor:
    """计算 Symlog 的反变换，返回物理空间 FP32 张量。"""

    values_fp32 = values.to(dtype=torch.float32)
    return torch.sign(values_fp32) * torch.expm1(torch.abs(values_fp32))


def _is_valid_sample(sample: Mapping[str, Any], config: TrainingDataConfig) -> bool:
    try:
        _require_finite_tensor(sample["images"], "images")
        if sample["images"].amin().item() < config.image_min or sample["images"].amax().item() > config.image_max:
            return False
        future_trajectory = sample["future_trajectory"]
        _require_finite_tensor(future_trajectory, "future_trajectory")
        if future_trajectory.abs().amax().item() > config.future_trajectory_abs_max_m:
            return False
        ego_motion = sample["ego_motion"]
        _require_finite_tensor(ego_motion, "ego_motion")
        if ego_motion.abs().amax().item() > config.ego_motion_abs_max:
            return False
        target_valid = sample["target_valid"]
        if target_valid.dtype != torch.bool or not bool(target_valid.any().item()):
            return False
        _require_finite_tensor(sample["target_points"], "target_points")
        if not _valid_agent_fields(sample, config):
            return False
        if not _valid_map_fields(sample, config):
            return False
    except (KeyError, TypeError, ValueError, RuntimeError):
        return False
    return True


def _valid_agent_fields(sample: Mapping[str, Any], config: TrainingDataConfig) -> bool:
    agent_boxes = sample["agent_boxes"]
    agent_valid = sample["agent_valid"]
    agent_future = sample["agent_future_trajectory"]
    _require_finite_tensor(agent_boxes, "agent_boxes")
    _require_finite_tensor(agent_future, "agent_future_trajectory")
    if agent_valid.dtype != torch.bool:
        return False
    if not bool(agent_valid.any().item()):
        return True
    valid_boxes = agent_boxes[agent_valid]
    if valid_boxes[:, 0:2].abs().amax().item() > config.agent_position_abs_max_m:
        return False
    sizes = valid_boxes[:, 2:5]
    if sizes.amin().item() < config.agent_size_min_m or sizes.amax().item() > config.agent_size_max_m:
        return False
    if valid_boxes[:, 6:10].abs().amax().item() > config.agent_motion_abs_max:
        return False
    return True


def _valid_map_fields(sample: Mapping[str, Any], config: TrainingDataConfig) -> bool:
    map_points = sample["map_points"]
    map_valid = sample["map_valid"]
    _require_finite_tensor(map_points, "map_points")
    if map_valid.dtype != torch.bool:
        return False
    if bool(map_valid.any().item()):
        if map_points[map_valid].abs().amax().item() > config.map_position_abs_max_m:
            return False
    return True


def _agent_future_cost(
    pred_future_m: torch.Tensor,
    gt_future_m: torch.Tensor,
    gt_future_valid: torch.Tensor,
) -> torch.Tensor:
    # [Q, M, K, 2] 与 [G, K, 2] -> [Q, G]，先对 mode 取最小物理空间误差。
    diff = pred_future_m[:, None, :, :, :] - gt_future_m[None, :, None, :, :]
    valid = gt_future_valid[None, :, None, :, None].to(dtype=torch.float32)
    squared = diff.square() * valid
    denominator = valid.sum(dim=(3, 4)).clamp_min(1.0)
    mode_cost = squared.sum(dim=(3, 4)) / denominator
    return mode_cost.min(dim=2).values


def _winner_agent_modes(
    pred_future_m: torch.Tensor,
    gt_future_m: torch.Tensor,
    gt_future_valid: torch.Tensor,
) -> torch.Tensor:
    diff = pred_future_m - gt_future_m[:, None, :, :]
    valid = gt_future_valid[:, None, :, None].to(dtype=torch.float32)
    squared = diff.square() * valid
    denominator = valid.sum(dim=(2, 3)).clamp_min(1.0)
    mode_cost = squared.sum(dim=(2, 3)) / denominator
    return torch.argmin(mode_cost, dim=1)


def _map_point_cost(
    pred_points_m: torch.Tensor,
    gt_points_m: torch.Tensor,
    gt_classes: torch.Tensor,
    bidirectional_class_indices: set[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    forward_cost = (pred_points_m[:, None, :, :] - gt_points_m[None, :, :, :]).abs().mean(dim=(2, 3))
    reverse_target_mask = torch.zeros_like(forward_cost, dtype=torch.bool)
    if not bidirectional_class_indices:
        return forward_cost, reverse_target_mask
    reverse_cost = (
        pred_points_m[:, None, :, :] - torch.flip(gt_points_m, dims=(1,))[None, :, :, :]
    ).abs().mean(dim=(2, 3))
    bidirectional_mask = torch.tensor(
        [int(class_index.item()) in bidirectional_class_indices for class_index in gt_classes],
        device=gt_classes.device,
        dtype=torch.bool,
    )
    reverse_target_mask = bidirectional_mask[None, :] & (reverse_cost < forward_cost)
    point_cost = torch.where(
        bidirectional_mask[None, :],
        torch.minimum(forward_cost, reverse_cost),
        forward_cost,
    )
    return point_cost, reverse_target_mask


def _encode_agent_state_targets(
    gt_boxes: torch.Tensor,
    detection_config: DetectionHeadConfig,
) -> torch.Tensor:
    targets = torch.zeros(
        (gt_boxes.shape[0], detection_config.agent_state_dim),
        device=gt_boxes.device,
        dtype=torch.float32,
    )
    values = {
        "x": symlog(gt_boxes[:, 0]),
        "y": symlog(gt_boxes[:, 1]),
        "length_log1p": torch.log1p(gt_boxes[:, 2].clamp_min(0.0)),
        "width_log1p": torch.log1p(gt_boxes[:, 3].clamp_min(0.0)),
        "height_log1p": torch.log1p(gt_boxes[:, 4].clamp_min(0.0)),
        "sin_yaw": torch.sin(gt_boxes[:, 5]),
        "cos_yaw": torch.cos(gt_boxes[:, 5]),
        "vx": symlog(gt_boxes[:, 6]),
        "vy": symlog(gt_boxes[:, 7]),
        "ax": symlog(gt_boxes[:, 8]),
        "ay": symlog(gt_boxes[:, 9]),
    }
    for field_name, value in values.items():
        targets[:, _state_index(detection_config, field_name)] = value
    return targets


def _linear_sum_assignment(cost_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise ImportError("Agent/Map 匈牙利匹配需要 scipy.optimize.linear_sum_assignment。") from exc

    row_indices, col_indices = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
    device = cost_matrix.device
    return (
        torch.as_tensor(row_indices, dtype=torch.long, device=device),
        torch.as_tensor(col_indices, dtype=torch.long, device=device),
    )


def _state_index(detection_config: DetectionHeadConfig, field_name: str) -> int:
    try:
        return detection_config.agent_state_order.index(field_name)
    except ValueError as exc:
        raise ValueError(f"agent_state_order 缺少字段 {field_name!r}。") from exc


def _require_finite_tensor(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} 必须为 torch.Tensor，实际为 {type(tensor)!r}。")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} 存在 NaN 或 Inf。")


def _validate_shape(tensor: torch.Tensor, ndim: int, name: str) -> None:
    if tensor.ndim != ndim:
        raise ValueError(f"{name} 期望 {ndim} 维，实际 shape 为 {tuple(tensor.shape)}。")


def _resolve_h5_paths(table: Mapping[str, Any], project_root: Path) -> list[Path]:
    if "h5_paths" in table:
        value = table["h5_paths"]
        if not isinstance(value, list):
            raise ValueError(f"dataset.h5_paths 必须为字符串列表，实际为 {value!r}。")
        raw_paths = [Path(item) for item in value]
    else:
        h5_dir = _resolve_project_relative_path(table, "h5_dir", project_root)
        if not h5_dir.exists():
            raise FileNotFoundError(f"dataset.h5_dir 不存在：{h5_dir}")
        raw_paths = sorted(h5_dir.glob("*.h5"))
    resolved_paths = []
    for raw_path in raw_paths:
        path = raw_path if raw_path.is_absolute() else project_root / raw_path
        resolved_path = path.resolve()
        _ensure_project_relative_path(resolved_path, project_root, "h5_paths")
        if not resolved_path.exists():
            raise FileNotFoundError(f"H5 文件不存在：{resolved_path}")
        resolved_paths.append(resolved_path)
    return resolved_paths


def _resolve_project_relative_path(
    table: Mapping[str, Any],
    key: str,
    project_root: Path,
) -> Path:
    path_text = _require_string(table, key)
    raw_path = Path(path_text)
    if raw_path.is_absolute():
        raise ValueError(f"{key} 必须为项目内相对路径，实际为 {raw_path}。")
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


def _require_bool(table: Mapping[str, Any], key: str) -> bool:
    value = table.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"配置项 {key} 必须为 bool，实际为 {value!r}。")
    return value


def _require_float(table: Mapping[str, Any], key: str) -> float:
    value = table.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"配置项 {key} 必须为数值，实际为 {value!r}。")
    return float(value)


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
