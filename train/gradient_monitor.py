"""训练梯度范数监测。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import torch
import torch.nn as nn

from train.training_config import GradientMonitorConfig


__all__ = [
    "GradientMonitorResult",
    "GradientParameterStat",
    "inspect_gradients",
]


@dataclass(frozen=True)
class GradientParameterStat:
    """单个参数的梯度统计。"""

    name: str
    norm: float
    shape: tuple[int, ...]
    reason: str


class GradientMonitorResult(NamedTuple):
    """一次梯度监测结果。"""

    total_norm: float
    max_norm: float
    min_norm: float
    parameter_count: int
    missing_gradients: tuple[GradientParameterStat, ...]
    small_gradients: tuple[GradientParameterStat, ...]
    large_gradients: tuple[GradientParameterStat, ...]
    nonfinite_gradients: tuple[GradientParameterStat, ...]

    @property
    def has_alert(self) -> bool:
        """是否存在需要记录的梯度异常。"""

        return bool(
            self.missing_gradients
            or self.small_gradients
            or self.large_gradients
            or self.nonfinite_gradients
        )


def inspect_gradients(
    model: nn.Module,
    config: GradientMonitorConfig,
) -> GradientMonitorResult:
    """统计可训练参数的梯度范数。"""

    total_norm_sq = 0.0
    max_norm = 0.0
    min_norm = math.inf
    parameter_count = 0
    missing_gradients: list[GradientParameterStat] = []
    small_gradients: list[GradientParameterStat] = []
    large_gradients: list[GradientParameterStat] = []
    nonfinite_gradients: list[GradientParameterStat] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parameter_count += 1
        if parameter.grad is None:
            if config.log_missing_gradients:
                missing_gradients.append(
                    GradientParameterStat(
                        name=name,
                        norm=0.0,
                        shape=tuple(parameter.shape),
                        reason="missing",
                    )
                )
            continue

        gradient = parameter.grad.detach().to(dtype=torch.float32)
        if not torch.isfinite(gradient).all().item():
            nonfinite_gradients.append(
                GradientParameterStat(
                    name=name,
                    norm=float("nan"),
                    shape=tuple(parameter.shape),
                    reason="nonfinite",
                )
            )
            continue

        grad_norm = float(torch.linalg.vector_norm(gradient).item())
        total_norm_sq += grad_norm * grad_norm
        max_norm = max(max_norm, grad_norm)
        min_norm = min(min_norm, grad_norm)
        if grad_norm <= config.small_grad_norm:
            small_gradients.append(
                GradientParameterStat(
                    name=name,
                    norm=grad_norm,
                    shape=tuple(parameter.shape),
                    reason="small",
                )
            )
        if grad_norm >= config.large_grad_norm:
            large_gradients.append(
                GradientParameterStat(
                    name=name,
                    norm=grad_norm,
                    shape=tuple(parameter.shape),
                    reason="large",
                )
            )

    small_gradients.sort(key=lambda item: item.norm)
    large_gradients.sort(key=lambda item: item.norm, reverse=True)
    total_norm = math.sqrt(total_norm_sq)
    if math.isinf(min_norm):
        min_norm = 0.0
    return GradientMonitorResult(
        total_norm=total_norm,
        max_norm=max_norm,
        min_norm=float(min_norm),
        parameter_count=parameter_count,
        missing_gradients=tuple(missing_gradients[: config.max_report_parameters]),
        small_gradients=tuple(small_gradients[: config.max_report_parameters]),
        large_gradients=tuple(large_gradients[: config.max_report_parameters]),
        nonfinite_gradients=tuple(nonfinite_gradients[: config.max_report_parameters]),
    )
