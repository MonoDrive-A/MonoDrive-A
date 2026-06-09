"""MonoDriveBackbone 闭环推理：轨迹词表解码与 winner 选择。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from model.backbone import MonoDriveBackbone, MonoDriveBackboneOutput


__all__ = [
    "TrajectoryDecodeResult",
    "decode_trajectories",
    "decode_winner_trajectory",
    "inverse_symlog",
]


@dataclass
class TrajectoryDecodeResult:
    """单次 backbone 前向的轨迹解码结果（batch=1）。"""

    probs: np.ndarray                 # (V,) 全词表 softmax 概率
    winner_idx: int                   # argmax(probs)
    winner_traj_phys: np.ndarray      # (K, 2) winner 物理轨迹
    top_indices: np.ndarray           # (top_k,) 可视化用 top-k 索引
    top_probs: np.ndarray             # (top_k,)
    top_trajs_phys: np.ndarray        # (top_k, K, 2) 可视化用 top-k 轨迹


def inverse_symlog(values: torch.Tensor) -> torch.Tensor:
    """把 Symlog 空间张量反变换到米制物理空间。"""
    return torch.sign(values) * torch.expm1(torch.abs(values))


def _decode_symlog_trajectories(
    vocab_symlog: torch.Tensor,
    symlog_scale: torch.Tensor,
    residuals: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """按 ``indices`` 组合词表 Symlog + 残差，返回物理轨迹 ``(N, K, 2)``。"""
    selected_vocab = vocab_symlog.index_select(0, indices)
    selected_residuals = residuals.index_select(0, indices)
    selected_symlog = selected_vocab + selected_residuals * symlog_scale
    return inverse_symlog(selected_symlog)


def decode_winner_trajectory(
    backbone_output: MonoDriveBackboneOutput,
    model: MonoDriveBackbone,
    winner_idx: int,
) -> np.ndarray:
    """解码指定词表索引的物理轨迹 ``(K, 2)``。"""
    logits = backbone_output.trajectory_output.logits[0]
    device = logits.device
    residuals = backbone_output.trajectory_output.residuals[0].to(
        device=device, dtype=torch.float32,
    )
    vocab_symlog = model.vocabulary.trajectory_vocab_symlog.to(
        device=device, dtype=torch.float32,
    )
    symlog_scale = model.vocabulary.symlog_scale.to(device=device, dtype=torch.float32)
    idx = torch.tensor([int(winner_idx)], device=device, dtype=torch.long)
    traj = _decode_symlog_trajectories(vocab_symlog, symlog_scale, residuals, idx)
    return traj[0].detach().cpu().numpy()


def decode_trajectories(
    backbone_output: MonoDriveBackboneOutput,
    model: MonoDriveBackbone,
    top_k: int = 8,
) -> TrajectoryDecodeResult:
    """从 backbone 输出解码全词表概率、winner 与 top-k 候选轨迹。"""
    logits = backbone_output.trajectory_output.logits[0].to(dtype=torch.float32)
    device = logits.device
    residuals = backbone_output.trajectory_output.residuals[0].to(
        device=device, dtype=torch.float32,
    )
    vocab_symlog = model.vocabulary.trajectory_vocab_symlog.to(
        device=device, dtype=torch.float32,
    )
    symlog_scale = model.vocabulary.symlog_scale.to(device=device, dtype=torch.float32)

    probs = torch.softmax(logits, dim=-1)
    vocab_count = int(probs.numel())
    selected_top_k = min(max(int(top_k), 1), vocab_count)
    top_probs, top_indices = torch.topk(probs, k=selected_top_k)

    top_trajs_phys = _decode_symlog_trajectories(
        vocab_symlog, symlog_scale, residuals, top_indices,
    ).detach().cpu().numpy()

    winner_idx = int(probs.argmax(dim=-1).item())
    winner_traj_phys = decode_winner_trajectory(backbone_output, model, winner_idx)

    return TrajectoryDecodeResult(
        probs=probs.detach().cpu().numpy(),
        winner_idx=winner_idx,
        winner_traj_phys=winner_traj_phys,
        top_indices=top_indices.detach().cpu().numpy(),
        top_probs=top_probs.detach().cpu().numpy(),
        top_trajs_phys=top_trajs_phys,
    )
