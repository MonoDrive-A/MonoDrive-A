"""MonoDrive 训练 loss 汇总。"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone import MonoDriveBackboneOutput
from train.data_processing import TrainingBatchLabels
from train.training_config import DetectionClassWeightConfig, LossWeights


_AUTO_FOCAL_GAMMA_MIN = 1.0
_AUTO_FOCAL_GAMMA_MAX = 4.0
_AUTO_FOCAL_GAMMA_BASE = 2.0
_AUTO_BACKGROUND_SCALE_MIN = 0.08
_AUTO_BACKGROUND_SCALE_MAX = 1.0


_MAP_CLASS_AUTO_SCALE_EXPONENT = 0.5


__all__ = [
    "TrainingLossOutput",
    "MonoDriveTrainingLoss",
]


class TrainingLossOutput(NamedTuple):
    """训练 loss 输出。"""

    total_loss: torch.Tensor
    components: dict[str, torch.Tensor]


class _DetectionClassLossBreakdown(NamedTuple):
    """检测分类 loss 的 none / non-none 分项。"""

    total: torch.Tensor
    non_none: torch.Tensor
    none: torch.Tensor


class MonoDriveTrainingLoss(nn.Module):
    """汇总规划、Agent 和 Map 训练 loss。

    Args:
        weights: `config/training.toml` 中读取的 loss 权重。
        detection_class_weights: 检测分类 none / non-none 策略；`auto` 为分组归一化 Focal Loss。

    Shape:
        输入模型输出沿用 `MonoDriveBackboneOutput`。
        输入训练标签沿用 `TrainingBatchLabels`。
        输出 `total_loss` 为标量 FP32 张量。
    """

    def __init__(
        self,
        weights: LossWeights,
        detection_class_weights: DetectionClassWeightConfig,
    ) -> None:
        super().__init__()
        self.weights = weights
        self.detection_class_weights = detection_class_weights

    def forward(
        self,
        model_output: MonoDriveBackboneOutput,
        labels: TrainingBatchLabels,
    ) -> TrainingLossOutput:
        """计算单个 batch 的总 loss。"""

        agent_class_breakdown = self._agent_class_ce(model_output, labels)
        map_class_breakdown = self._map_class_ce(model_output, labels)
        raw_components = {
            "trajectory_logit_soft_ce": self._trajectory_logit_soft_ce(model_output, labels),
            "trajectory_residual_mse": self._trajectory_residual_mse(model_output, labels),
            "agent_class_ce": agent_class_breakdown.total,
            "agent_state_mse": self._agent_state_mse(model_output, labels),
            "agent_mode_ce": self._agent_mode_ce(model_output, labels),
            "agent_future_mse": self._agent_future_mse(model_output, labels),
            "map_class_ce": map_class_breakdown.total,
            "map_point_mse": self._map_point_mse(model_output, labels),
        }
        weighted_components = {
            f"{name}_weighted": component * getattr(self.weights, name)
            for name, component in raw_components.items()
        }
        total_loss = sum(weighted_components.values())
        components = {
            **raw_components,
            **weighted_components,
            "agent_class_ce_non_none": agent_class_breakdown.non_none,
            "agent_class_ce_none": agent_class_breakdown.none,
            "map_class_ce_non_none": map_class_breakdown.non_none,
            "map_class_ce_none": map_class_breakdown.none,
            "total_loss": total_loss,
        }
        return TrainingLossOutput(total_loss=total_loss, components=components)

    def _trajectory_logit_soft_ce(
        self,
        model_output: MonoDriveBackboneOutput,
        labels: TrainingBatchLabels,
    ) -> torch.Tensor:
        logits = model_output.trajectory_output.logits.to(dtype=torch.float32)
        target_probabilities = labels.trajectory.soft_labels.to(
            device=logits.device,
            dtype=torch.float32,
        ).clamp_min(0.0)
        target_probabilities = target_probabilities / target_probabilities.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-12)
        log_probabilities = F.log_softmax(logits, dim=1)
        return -(target_probabilities * log_probabilities).sum(dim=1).mean()

    def _trajectory_residual_mse(
        self,
        model_output: MonoDriveBackboneOutput,
        labels: TrainingBatchLabels,
    ) -> torch.Tensor:
        residuals = model_output.trajectory_output.residuals.to(dtype=torch.float32)
        targets = labels.trajectory.residual_targets.to(device=residuals.device, dtype=torch.float32)
        mask = labels.trajectory.residual_mask.to(device=residuals.device, dtype=torch.bool)
        return _masked_mse(residuals, targets, mask, "trajectory_residual")

    def _agent_class_ce(
        self,
        model_output: MonoDriveBackboneOutput,
        labels: TrainingBatchLabels,
    ) -> _DetectionClassLossBreakdown:
        logits = model_output.detection_output.agent_class_logits.to(dtype=torch.float32)
        targets = labels.agent.class_targets.to(device=logits.device, dtype=torch.long)
        return _detection_class_cross_entropy(
            logits=logits,
            targets=targets,
            none_index=int(logits.shape[-1]) - 1,
            weight_config=self.detection_class_weights,
            non_none_weight=self.detection_class_weights.agent_non_none_weight,
            none_weight=self.detection_class_weights.agent_none_weight,
            name="agent_class_ce",
        )

    def _agent_state_mse(
        self,
        model_output: MonoDriveBackboneOutput,
        labels: TrainingBatchLabels,
    ) -> torch.Tensor:
        states = model_output.detection_output.agent_states.to(dtype=torch.float32)
        targets = labels.agent.state_targets.to(device=states.device, dtype=torch.float32)
        mask = labels.agent.state_mask.to(device=states.device, dtype=torch.bool)
        return _masked_mse(states, targets, mask, "agent_state")

    def _agent_mode_ce(
        self,
        model_output: MonoDriveBackboneOutput,
        labels: TrainingBatchLabels,
    ) -> torch.Tensor:
        logits = model_output.detection_output.agent_mode_logits.to(dtype=torch.float32)
        targets = labels.agent.mode_targets.to(device=logits.device, dtype=torch.long)
        mask = labels.agent.future_mask.to(device=logits.device, dtype=torch.bool).any(dim=(2, 3))
        return _masked_cross_entropy(logits, targets, mask)

    def _agent_future_mse(
        self,
        model_output: MonoDriveBackboneOutput,
        labels: TrainingBatchLabels,
    ) -> torch.Tensor:
        future = model_output.detection_output.agent_future_trajectories.to(dtype=torch.float32)
        targets = labels.agent.future_targets.to(device=future.device, dtype=torch.float32)
        mask = labels.agent.future_mask.to(device=future.device, dtype=torch.bool)
        return _masked_mse(future, targets, mask, "agent_future")

    def _map_class_ce(
        self,
        model_output: MonoDriveBackboneOutput,
        labels: TrainingBatchLabels,
    ) -> _DetectionClassLossBreakdown:
        logits = model_output.detection_output.map_class_logits.to(dtype=torch.float32)
        targets = labels.map.class_targets.to(device=logits.device, dtype=torch.long)
        breakdown = _detection_class_cross_entropy(
            logits=logits,
            targets=targets,
            none_index=int(logits.shape[-1]) - 1,
            weight_config=self.detection_class_weights,
            non_none_weight=self.detection_class_weights.map_non_none_weight,
            none_weight=self.detection_class_weights.map_none_weight,
            name="map_class_ce",
        )
        if self.detection_class_weights.mode != "auto":
            return breakdown
        point_count = int(model_output.detection_output.map_points.shape[-2])
        point_dim = int(model_output.detection_output.map_points.shape[-1])
        foreground_class_count = int(logits.shape[-1]) - 1
        regression_dims = point_count * point_dim
        auto_scale = (
            regression_dims / max(foreground_class_count, 1)
        ) ** _MAP_CLASS_AUTO_SCALE_EXPONENT
        return _DetectionClassLossBreakdown(
            total=breakdown.total * auto_scale,
            non_none=breakdown.non_none * auto_scale,
            none=breakdown.none * auto_scale,
        )

    def _map_point_mse(
        self,
        model_output: MonoDriveBackboneOutput,
        labels: TrainingBatchLabels,
    ) -> torch.Tensor:
        points = model_output.detection_output.map_points.to(dtype=torch.float32)
        targets = labels.map.point_targets.to(device=points.device, dtype=torch.float32)
        mask = labels.map.point_mask.to(device=points.device, dtype=torch.bool)
        return _masked_mse(points, targets, mask, "map_point")


def _masked_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if predictions.shape != targets.shape:
        raise ValueError(
            f"{name} predictions 和 targets shape 必须一致，"
            f"实际为 {tuple(predictions.shape)} 和 {tuple(targets.shape)}。"
        )
    if mask.ndim > predictions.ndim:
        raise ValueError(
            f"{name} mask 维度不能超过 predictions，"
            f"实际为 {tuple(mask.shape)} 和 {tuple(predictions.shape)}。"
        )
    expanded_mask = mask
    while expanded_mask.ndim < predictions.ndim:
        expanded_mask = expanded_mask.unsqueeze(-1)
    expanded_mask = expanded_mask.to(dtype=torch.float32)
    squared_error = (predictions - targets).square() * expanded_mask
    denominator = expanded_mask.expand_as(predictions).sum().clamp_min(1.0)
    return squared_error.sum() / denominator


def _detection_class_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    none_index: int,
    weight_config: DetectionClassWeightConfig,
    non_none_weight: float,
    none_weight: float,
    name: str,
) -> _DetectionClassLossBreakdown:
    if logits.ndim < 2:
        raise ValueError(f"{name} logits 至少需要 2 维，实际 shape 为 {tuple(logits.shape)}。")
    class_count = int(logits.shape[-1])
    if logits.shape[:-1] != targets.shape:
        raise ValueError(
            f"{name} logits 前置 shape 必须与 targets 一致，"
            f"实际为 {tuple(logits.shape)} 和 {tuple(targets.shape)}。"
        )
    if class_count <= 1:
        raise ValueError(f"{name} 至少需要 2 个分类通道，实际为 {class_count}。")
    if none_index < 0 or none_index >= class_count:
        raise ValueError(
            f"{name} none_index 必须位于 [0, {class_count})，实际为 {none_index}。"
        )

    flat_logits = logits.reshape(-1, class_count)
    flat_targets = targets.reshape(-1)
    zero_loss = flat_logits.sum() * 0.0
    if flat_targets.numel() == 0:
        return _DetectionClassLossBreakdown(total=zero_loss, non_none=zero_loss, none=zero_loss)
    target_min = int(flat_targets.amin().item())
    target_max = int(flat_targets.amax().item())
    if target_min < 0 or target_max >= class_count:
        raise ValueError(
            f"{name} targets 必须位于 [0, {class_count})，"
            f"实际最小/最大为 {target_min}/{target_max}。"
        )

    if weight_config.mode == "disabled":
        return _mean_cross_entropy_breakdown(flat_logits, flat_targets, none_index)
    if weight_config.mode == "manual":
        class_weight = _constant_detection_class_weight(
            class_count=class_count,
            none_index=none_index,
            non_none_weight=non_none_weight,
            none_weight=none_weight,
            device=targets.device,
        )
        return _weighted_cross_entropy_breakdown(
            flat_logits,
            flat_targets,
            class_weight,
            none_index,
        )
    return _group_normalized_focal_loss(
        logits=flat_logits,
        targets=flat_targets,
        none_index=none_index,
    )


def _constant_detection_class_weight(
    class_count: int,
    none_index: int,
    non_none_weight: float,
    none_weight: float,
    device: torch.device,
) -> torch.Tensor:
    class_weight = torch.full(
        (class_count,),
        float(non_none_weight),
        device=device,
        dtype=torch.float32,
    )
    class_weight[none_index] = float(none_weight)
    return class_weight


def _mean_cross_entropy_breakdown(
    logits: torch.Tensor,
    targets: torch.Tensor,
    none_index: int,
) -> _DetectionClassLossBreakdown:
    zero_loss = logits.sum() * 0.0
    none_mask = targets == none_index
    non_none_mask = ~none_mask
    non_none_loss = (
        _mean_cross_entropy(logits[non_none_mask], targets[non_none_mask])
        if bool(non_none_mask.any().item())
        else zero_loss
    )
    none_loss = (
        _mean_cross_entropy(logits[none_mask], targets[none_mask])
        if bool(none_mask.any().item())
        else zero_loss
    )
    return _DetectionClassLossBreakdown(
        total=_mean_cross_entropy(logits, targets),
        non_none=non_none_loss,
        none=none_loss,
    )


def _weighted_cross_entropy_breakdown(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weight: torch.Tensor,
    none_index: int,
) -> _DetectionClassLossBreakdown:
    zero_loss = logits.sum() * 0.0
    none_mask = targets == none_index
    non_none_mask = ~none_mask
    non_none_loss = (
        _weighted_cross_entropy(logits[non_none_mask], targets[non_none_mask], class_weight)
        if bool(non_none_mask.any().item())
        else zero_loss
    )
    none_loss = (
        _weighted_cross_entropy(logits[none_mask], targets[none_mask], class_weight)
        if bool(none_mask.any().item())
        else zero_loss
    )
    return _DetectionClassLossBreakdown(
        total=_weighted_cross_entropy(logits, targets, class_weight),
        non_none=non_none_loss,
        none=none_loss,
    )


def _group_normalized_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    none_index: int,
) -> _DetectionClassLossBreakdown:
    """匹配 / 未匹配分离的分组 Focal Loss。

    匹配 query 只在前景类上竞争（none 不参与 softmax），并额外用 objectness BCE
    压低 none logit；未匹配 query 仍监督 none。背景组均值按 ``sqrt(N_fg / N_bg)`` 自动缩放。
    """

    none_mask = targets == none_index
    non_none_mask = ~none_mask
    has_none = bool(none_mask.any().item())
    has_non_none = bool(non_none_mask.any().item())

    foreground_loss: torch.Tensor | None = None
    if has_non_none:
        foreground_logits = logits[non_none_mask, :none_index]
        foreground_targets = targets[non_none_mask]
        none_logits = logits[non_none_mask, none_index]
        foreground_gamma = _auto_focal_gamma(foreground_logits, foreground_targets)
        class_loss = _focal_cross_entropy_per_sample(
            foreground_logits,
            foreground_targets,
            foreground_gamma,
        ).mean()
        objectness_logits = torch.logsumexp(foreground_logits, dim=-1) - none_logits
        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness_logits,
            torch.ones_like(objectness_logits),
        )
        foreground_loss = class_loss + objectness_loss

    background_loss: torch.Tensor | None = None
    if has_none:
        background_logits = logits[none_mask]
        background_targets = targets[none_mask]
        background_gamma = _auto_focal_gamma(background_logits, background_targets)
        background_loss = _focal_cross_entropy_per_sample(
            background_logits,
            background_targets,
            background_gamma,
        ).mean()

    zero_loss = logits.sum() * 0.0
    if foreground_loss is not None and background_loss is not None:
        foreground_count = non_none_mask.sum().to(dtype=torch.float32)
        background_count = none_mask.sum().to(dtype=torch.float32)
        background_scale = (foreground_count / background_count).sqrt().clamp(
            min=_AUTO_BACKGROUND_SCALE_MIN,
            max=_AUTO_BACKGROUND_SCALE_MAX,
        )
        scaled_background_loss = background_scale * background_loss
        return _DetectionClassLossBreakdown(
            total=foreground_loss + scaled_background_loss,
            non_none=foreground_loss,
            none=scaled_background_loss,
        )
    if foreground_loss is not None:
        return _DetectionClassLossBreakdown(
            total=foreground_loss,
            non_none=foreground_loss,
            none=zero_loss,
        )
    if background_loss is not None:
        return _DetectionClassLossBreakdown(
            total=background_loss,
            non_none=zero_loss,
            none=background_loss,
        )
    return _DetectionClassLossBreakdown(total=zero_loss, non_none=zero_loss, none=zero_loss)


def _auto_focal_gamma(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """根据组内目标置信度自适应 focal gamma。"""

    with torch.no_grad():
        probabilities = torch.softmax(logits.detach(), dim=-1)
        target_probabilities = probabilities.gather(
            dim=1,
            index=targets.unsqueeze(1),
        ).squeeze(1)
        if target_probabilities.numel() == 0:
            return _AUTO_FOCAL_GAMMA_BASE
        mean_confidence = float(target_probabilities.mean().item())
        gamma = _AUTO_FOCAL_GAMMA_BASE + (mean_confidence - 0.5)
        return max(_AUTO_FOCAL_GAMMA_MIN, min(_AUTO_FOCAL_GAMMA_MAX, gamma))


def _focal_cross_entropy_per_sample(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    log_probabilities = F.log_softmax(logits, dim=-1)
    log_target_probability = log_probabilities.gather(
        dim=1,
        index=targets.unsqueeze(1),
    ).squeeze(1)
    target_probability = log_target_probability.exp()
    return -((1.0 - target_probability).clamp_min(0.0) ** gamma) * log_target_probability


def _mean_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(logits, targets)


def _weighted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weight: torch.Tensor,
) -> torch.Tensor:
    losses = F.cross_entropy(logits, targets, reduction="none")
    sample_weights = class_weight[targets]
    denominator = sample_weights.sum()
    if not bool((denominator > 0.0).item()):
        return logits.sum() * 0.0
    return (losses * sample_weights).sum() / denominator.clamp_min(1e-12)


def _masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if logits.shape[:-1] != targets.shape:
        raise ValueError(
            "logits 前置 shape 必须与 targets 一致，"
            f"实际为 {tuple(logits.shape)} 和 {tuple(targets.shape)}。"
        )
    if targets.shape != mask.shape:
        raise ValueError(
            "targets 与 mask shape 必须一致，"
            f"实际为 {tuple(targets.shape)} 和 {tuple(mask.shape)}。"
        )
    if not bool(mask.any().item()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], targets[mask])
