"""训练 checkpoint 保存与恢复。"""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer

from train.training_config import CheckpointConfig


__all__ = [
    "CheckpointLoadResult",
    "capture_rng_state",
    "find_resume_checkpoint",
    "load_checkpoint",
    "restore_rng_state",
    "save_checkpoint",
]


class CheckpointLoadResult(NamedTuple):
    """checkpoint 加载结果。"""

    path: Path
    payload: dict[str, Any]


def capture_rng_state(include_cuda: bool) -> dict[str, Any]:
    """捕获 Python、NumPy、PyTorch 和可选 CUDA 随机状态。"""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if include_cuda and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """恢复随机状态。"""

    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    config: CheckpointConfig,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler_state: dict[str, Any],
    global_step: int,
    epoch: int,
    batch_index: int,
    metrics: dict[str, float],
    rng_state: dict[str, Any] | None,
) -> Path:
    """保存一次训练 checkpoint，并更新 latest 文件。"""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / f"step_{global_step:08d}.pt"
    payload = {
        "schema_version": "monodrive_training_checkpoint_v1",
        "global_step": int(global_step),
        "epoch": int(epoch),
        "batch_index": int(batch_index),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler_state,
        "metrics": metrics,
        "rng_state": rng_state,
    }
    _atomic_torch_save(payload, checkpoint_path)
    latest_path = config.output_dir / config.latest_filename
    _atomic_torch_save(payload, latest_path)
    _prune_old_checkpoints(config)
    return checkpoint_path


def find_resume_checkpoint(config: CheckpointConfig) -> Path | None:
    """按配置查找需要恢复的 checkpoint。"""

    if config.resume_checkpoint_path is not None:
        if not config.resume_checkpoint_path.exists():
            raise FileNotFoundError(f"resume_checkpoint_path 不存在：{config.resume_checkpoint_path}")
        return config.resume_checkpoint_path
    if not config.resume_from_latest:
        return None

    latest_path = config.output_dir / config.latest_filename
    if latest_path.exists():
        return latest_path
    candidates = sorted(config.output_dir.glob("step_*.pt")) if config.output_dir.exists() else []
    return candidates[-1] if candidates else None


def load_checkpoint(path: str | Path, device: torch.device) -> CheckpointLoadResult:
    """加载 checkpoint 到指定设备。"""

    resolved_path = Path(path).resolve()
    try:
        payload = torch.load(resolved_path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(resolved_path, map_location=device)
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint payload 必须为 dict，实际为 {type(payload)!r}。")
    return CheckpointLoadResult(path=resolved_path, payload=payload)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary_path = path.with_name(f"{path.name}.tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def _prune_old_checkpoints(config: CheckpointConfig) -> None:
    if config.keep_last == 0:
        return
    candidates = sorted(config.output_dir.glob("step_*.pt"))
    stale_candidates = candidates[: max(0, len(candidates) - config.keep_last)]
    for checkpoint_path in stale_candidates:
        checkpoint_path.unlink(missing_ok=True)
