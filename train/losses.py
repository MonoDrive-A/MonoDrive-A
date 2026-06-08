"""MonoDrive 训练 loss 汇总。"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone import MonoDriveBackboneOutput
from train.data_processing import TrainingBatchLabels
from train.training_config import LossWeights


__all__ = [
    "TrainingLossOutput",
    "MonoDriveTrainingLoss",
]


class TrainingLossOutput(NamedTuple):
    """训练 loss 输出。"""

    total_loss: torch.Tensor
    components: dict[str, torch.Tensor]


class MonoDriveTrainingLoss(nn.Module):
    """汇总规划、Agent 和 Map 训练 loss。

    Args:
        weights: `config/training.toml` 中读取的 loss 权重。

    Shape:
        输入模型输出沿用 `MonoDriveBackboneOutput`。
        输入训练标签沿用 `TrainingBatchLabels`。
        输出 `total_loss` 为标量 FP32 张量。
    """

    def __init__(self, weights: LossWeights) -> None:
        super().__init__()
        self.weights = weights

    def forward(
        self,
        model_output: MonoDriveBackboneOutput,
        labels: TrainingBatchLabels,
    ) -> TrainingLossOutput:
        """计算单个 batch 的总 loss。"""

        raw_components = {
            "trajectory_logit_bce": self._trajectory_logit_bce(model_output, labels),
            "trajectory_residual_mse": self._trajectory_residual_mse(model_output, labels),
            "agent_class_ce": self._agent_class_ce(model_output, labels),
            "agent_state_mse": self._agent_state_mse(model_output, labels),
            "agent_mode_ce": self._agent_mode_ce(model_output, labels),
            "agent_future_mse": self._agent_future_mse(model_output, labels),
            "map_class_ce": self._map_class_ce(model_output, labels),
            "map_point_mse": self._map_point_mse(model_output, labels),
        }
        weighted_components = {
            f"{name}_weighted": component * getattr(self.weights, name)
            for name, component in raw_components.items()
        }
        total_loss = sum(weighted_components.values())
        components = {**raw_components, **weighted_components, "total_loss": total_loss}
        return TrainingLossOutput(total_loss=total_loss, components=components)

    def _trajectory_logit_bce(
        self,
        model_output: MonoDriveBackboneOutput,
        labels: TrainingBatchLabels,
    ) -> torch.Tensor:
        logits = model_output.trajectory_output.logits.to(dtype=torch.float32)
        soft_targets = labels.trajectory.soft_labels.to(device=logits.device, dtype=torch.float32)
        soft_targets = soft_targets.clamp(min=0.0, max=1.0)
        return F.binary_cross_entropy_with_logits(logits, soft_targets, reduction="mean")

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
    ) -> torch.Tensor:
        logits = model_output.detection_output.agent_class_logits.to(dtype=torch.float32)
        targets = labels.agent.class_targets.to(device=logits.device, dtype=torch.long)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

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
    ) -> torch.Tensor:
        logits = model_output.detection_output.map_class_logits.to(dtype=torch.float32)
        targets = labels.map.class_targets.to(device=logits.device, dtype=torch.long)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

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
