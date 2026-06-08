"""MonoDrive 训练主入口。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from model.backbone import MonoDriveBackbone, load_backbone_config
from train.checkpointing import (
    capture_rng_state,
    find_resume_checkpoint,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from train.data_processing import (
    TrainingDataConfig,
    build_training_batch_labels,
    build_training_dataset,
    load_training_data_config,
    training_collate,
)
from train.gradient_monitor import GradientMonitorResult, inspect_gradients
from train.losses import MonoDriveTrainingLoss, TrainingLossOutput
from train.training_config import OptimizationConfig, TrainingRunConfig, load_training_run_config


__all__ = [
    "TrainingSummary",
    "WarmupCosineLRScheduler",
    "run_training",
]


@dataclass(frozen=True)
class TrainingSummary:
    """训练运行摘要。"""

    global_step: int
    epoch: int
    latest_checkpoint_path: Path | None
    metrics_path: Path


class WarmupCosineLRScheduler:
    """按训练配置执行 warmup、平台期和末尾余弦退火。"""

    def __init__(self, optimizer: Optimizer, config: OptimizationConfig) -> None:
        self.optimizer = optimizer
        self.config = config
        self.current_step = -1
        self.last_lr = config.initial_lr

    def step(self, step_index: int) -> float:
        """设置当前 step 的学习率并返回该值。"""

        if step_index < 0:
            raise ValueError(f"step_index 必须为非负整数，实际为 {step_index}。")
        learning_rate = self._compute_lr(step_index)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate
        self.current_step = step_index
        self.last_lr = learning_rate
        return learning_rate

    def state_dict(self) -> dict[str, Any]:
        """返回调度器状态。"""

        return {"current_step": self.current_step, "last_lr": self.last_lr}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """恢复调度器状态。"""

        self.current_step = int(state_dict.get("current_step", -1))
        self.last_lr = float(state_dict.get("last_lr", self.config.initial_lr))

    def _compute_lr(self, step_index: int) -> float:
        if step_index < self.config.warmup_steps:
            if self.config.warmup_steps == 1:
                return self.config.peak_lr
            progress = step_index / float(self.config.warmup_steps - 1)
            return self.config.initial_lr + (self.config.peak_lr - self.config.initial_lr) * progress

        decay_start = self.config.total_steps - self.config.cosine_decay_steps
        if step_index >= decay_start:
            if self.config.cosine_decay_steps == 1:
                progress = 1.0
            else:
                progress = (step_index - decay_start) / float(self.config.cosine_decay_steps - 1)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.config.min_lr + (self.config.peak_lr - self.config.min_lr) * cosine

        return self.config.peak_lr


def run_training(
    config_path: str | Path = "config/training.toml",
    max_steps: int | None = None,
) -> TrainingSummary:
    """按配置运行训练。"""

    run_config = load_training_run_config(config_path)
    _seed_everything(run_config)
    device = _resolve_device(run_config)
    metrics_path = _prepare_metrics_path(run_config)

    backbone_config = load_backbone_config(
        run_config.backbone_config_path,
        project_root=run_config.project_root,
    )
    data_config = load_training_data_config(
        run_config.training_data_config_path,
        project_root=run_config.project_root,
    )
    dataset = build_training_dataset(data_config)
    model = MonoDriveBackbone(backbone_config).to(device)
    _ensure_dinov3_frozen(model)
    optimizer = _build_optimizer(model, run_config.optimization)
    scheduler = WarmupCosineLRScheduler(optimizer, run_config.optimization)
    criterion = MonoDriveTrainingLoss(run_config.loss_weights)

    global_step = 0
    epoch = 0
    start_batch_index = 0
    checkpoint_batch_index = 0
    latest_checkpoint_path: Path | None = None
    resume_path = find_resume_checkpoint(run_config.checkpoint)
    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path, device)
        payload = checkpoint.payload
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload.get("scheduler_state", {}))
        rng_state = payload.get("rng_state")
        if isinstance(rng_state, dict):
            restore_rng_state(rng_state)
        global_step = int(payload.get("global_step", 0))
        epoch = int(payload.get("epoch", 0))
        start_batch_index = int(payload.get("batch_index", 0))
        latest_checkpoint_path = checkpoint.path

    target_steps = run_config.optimization.total_steps
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError(f"max_steps 必须为正整数，实际为 {max_steps}。")
        target_steps = min(target_steps, max_steps)

    try:
        while global_step < target_steps:
            data_loader = _build_data_loader(dataset, run_config, epoch, device)
            epoch_finished = True
            for batch_index, batch in enumerate(data_loader):
                if batch_index < start_batch_index:
                    continue
                if global_step >= target_steps:
                    epoch_finished = False
                    break

                learning_rate = scheduler.step(global_step)
                batch = _move_batch_to_device(
                    batch,
                    device,
                    non_blocking=run_config.runtime.non_blocking,
                )
                metrics, gradient_result = _train_step(
                    model=model,
                    batch=batch,
                    data_config=data_config,
                    criterion=criterion,
                    optimizer=optimizer,
                    run_config=run_config,
                    global_step=global_step,
                    learning_rate=learning_rate,
                )
                global_step += 1
                next_batch_index = batch_index + 1
                metrics.update(
                    {
                        "global_step": float(global_step),
                        "epoch": float(epoch),
                        "batch_index": float(next_batch_index),
                    }
                )
                checkpoint_batch_index = next_batch_index
                _maybe_log_metrics(run_config, metrics_path, metrics, global_step)
                _maybe_print_metrics(metrics, gradient_result, global_step, run_config)

                if global_step % run_config.checkpoint.save_interval_steps == 0:
                    latest_checkpoint_path = _save_training_checkpoint(
                        run_config,
                        model,
                        optimizer,
                        scheduler,
                        global_step,
                        epoch,
                        next_batch_index,
                        metrics,
                        device,
                    )

                if global_step >= target_steps:
                    epoch_finished = False
                    break

            if epoch_finished:
                epoch += 1
            start_batch_index = 0
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()

    if run_config.checkpoint.save_on_exit and global_step > 0:
        latest_checkpoint_path = _save_training_checkpoint(
            run_config,
            model,
            optimizer,
            scheduler,
            global_step,
            epoch,
            checkpoint_batch_index,
            {"global_step": float(global_step)},
            device,
        )

    return TrainingSummary(
        global_step=global_step,
        epoch=epoch,
        latest_checkpoint_path=latest_checkpoint_path,
        metrics_path=metrics_path,
    )


def _train_step(
    model: MonoDriveBackbone,
    batch: Mapping[str, Any],
    data_config: TrainingDataConfig,
    criterion: MonoDriveTrainingLoss,
    optimizer: Optimizer,
    run_config: TrainingRunConfig,
    global_step: int,
    learning_rate: float,
) -> tuple[dict[str, float], GradientMonitorResult | None]:
    model.train()
    optimizer.zero_grad(set_to_none=run_config.optimization.zero_grad_set_to_none)
    model_output = model(
        batch["images"],
        batch["target_point"],
        batch["ego_motion"],
        return_layer_features=False,
    )
    labels = build_training_batch_labels(
        model_output.detection_output,
        model_output.trajectory_output,
        batch,
        data_config,
        detection_config=model.detection_config,
        trajectory_config=model.trajectory_config,
        vocabulary=model.vocabulary,
    )
    loss_output = criterion(model_output, labels)
    if not torch.isfinite(loss_output.total_loss).all().item():
        raise RuntimeError(f"训练 loss 出现 NaN 或 Inf，global_step={global_step}。")
    loss_output.total_loss.backward()

    gradient_result = None
    if run_config.gradient_monitor.enabled and (
        global_step % run_config.gradient_monitor.check_interval_steps == 0
    ):
        gradient_result = inspect_gradients(model, run_config.gradient_monitor)
        if gradient_result.nonfinite_gradients and run_config.gradient_monitor.fail_on_nonfinite:
            raise RuntimeError(f"梯度出现 NaN 或 Inf，global_step={global_step}。")

    if run_config.optimization.enable_gradient_clipping:
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            max_norm=run_config.optimization.max_grad_norm,
        )
    optimizer.step()

    metrics = _loss_metrics(loss_output)
    metrics["learning_rate"] = float(learning_rate)
    if gradient_result is not None:
        metrics.update(_gradient_metrics(gradient_result))
    return metrics, gradient_result


def _build_optimizer(model: MonoDriveBackbone, config: OptimizationConfig) -> Optimizer:
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("模型没有可训练参数。")
    return torch.optim.AdamW(
        trainable_parameters,
        lr=config.initial_lr,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )


def _build_data_loader(
    dataset: torch.utils.data.Dataset[dict[str, Any]],
    run_config: TrainingRunConfig,
    epoch: int,
    device: torch.device,
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator()
    generator.manual_seed(run_config.random.seed + epoch)
    dataloader_kwargs: dict[str, Any] = {
        "batch_size": run_config.dataloader.batch_size,
        "shuffle": run_config.dataloader.shuffle,
        "num_workers": run_config.dataloader.num_workers,
        "pin_memory": run_config.dataloader.pin_memory and device.type == "cuda",
        "drop_last": run_config.dataloader.drop_last,
        "collate_fn": training_collate,
        "generator": generator,
    }
    if run_config.dataloader.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = run_config.dataloader.persistent_workers
        dataloader_kwargs["prefetch_factor"] = run_config.dataloader.prefetch_factor
        dataloader_kwargs["worker_init_fn"] = _build_worker_init_fn(run_config, epoch)
    return DataLoader(dataset, **dataloader_kwargs)


def _build_worker_init_fn(run_config: TrainingRunConfig, epoch: int) -> Any:
    def _seed_worker(worker_id: int) -> None:
        worker_seed = (
            run_config.random.seed
            + epoch * run_config.dataloader.worker_seed_stride
            + worker_id
        )
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32))
        torch.manual_seed(worker_seed)

    return _seed_worker


def _move_batch_to_device(
    value: Any,
    device: torch.device,
    non_blocking: bool,
) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=non_blocking)
    if isinstance(value, dict):
        return {
            key: _move_batch_to_device(item, device, non_blocking)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_move_batch_to_device(item, device, non_blocking) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_batch_to_device(item, device, non_blocking) for item in value)
    return value


def _save_training_checkpoint(
    run_config: TrainingRunConfig,
    model: MonoDriveBackbone,
    optimizer: Optimizer,
    scheduler: WarmupCosineLRScheduler,
    global_step: int,
    epoch: int,
    batch_index: int,
    metrics: dict[str, float],
    device: torch.device,
) -> Path:
    rng_state = (
        capture_rng_state(include_cuda=device.type == "cuda")
        if run_config.checkpoint.save_rng_state
        else None
    )
    return save_checkpoint(
        run_config.checkpoint,
        model,
        optimizer,
        scheduler.state_dict(),
        global_step,
        epoch,
        batch_index,
        metrics,
        rng_state,
    )


def _loss_metrics(loss_output: TrainingLossOutput) -> dict[str, float]:
    return {
        f"loss/{name}": float(value.detach().to(dtype=torch.float32).cpu().item())
        for name, value in loss_output.components.items()
    }


def _gradient_metrics(result: GradientMonitorResult) -> dict[str, float]:
    return {
        "grad/total_norm": result.total_norm,
        "grad/max_norm": result.max_norm,
        "grad/min_norm": result.min_norm,
        "grad/parameter_count": float(result.parameter_count),
        "grad/missing_count": float(len(result.missing_gradients)),
        "grad/small_count": float(len(result.small_gradients)),
        "grad/large_count": float(len(result.large_gradients)),
        "grad/nonfinite_count": float(len(result.nonfinite_gradients)),
    }


def _maybe_log_metrics(
    run_config: TrainingRunConfig,
    metrics_path: Path,
    metrics: dict[str, float],
    global_step: int,
) -> None:
    if global_step % run_config.logging.log_interval_steps != 0:
        return
    with metrics_path.open("a", encoding="utf-8") as metrics_file:
        metrics_file.write(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n")


def _maybe_print_metrics(
    metrics: dict[str, float],
    gradient_result: GradientMonitorResult | None,
    global_step: int,
    run_config: TrainingRunConfig,
) -> None:
    if global_step % run_config.logging.log_interval_steps != 0:
        return
    message = (
        f"step={global_step} "
        f"loss={metrics['loss/total_loss']:.6f} "
        f"lr={metrics['learning_rate']:.8f}"
    )
    if gradient_result is not None and gradient_result.has_alert:
        message += (
            " grad_alert="
            f"missing:{len(gradient_result.missing_gradients)},"
            f"small:{len(gradient_result.small_gradients)},"
            f"large:{len(gradient_result.large_gradients)},"
            f"nonfinite:{len(gradient_result.nonfinite_gradients)}"
        )
    print(message, flush=True)


def _prepare_metrics_path(run_config: TrainingRunConfig) -> Path:
    run_config.logging.output_dir.mkdir(parents=True, exist_ok=True)
    return run_config.logging.output_dir / run_config.logging.metrics_filename


def _resolve_device(run_config: TrainingRunConfig) -> torch.device:
    configured_device = run_config.runtime.device
    if configured_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if configured_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求使用 CUDA，但当前 PyTorch 未检测到可用 CUDA。")
    return torch.device(configured_device)


def _seed_everything(run_config: TrainingRunConfig) -> None:
    seed = run_config.random.seed
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if run_config.random.deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not run_config.random.deterministic


def _ensure_dinov3_frozen(model: MonoDriveBackbone) -> None:
    trainable_dinov3 = [
        name
        for name, parameter in model.vision_embedding.dinov3.named_parameters()
        if parameter.requires_grad
    ]
    if trainable_dinov3:
        raise ValueError(
            "DINOv3 必须冻结，但以下参数仍可训练："
            f"{trainable_dinov3[:5]}。"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 MonoDrive 训练。")
    parser.add_argument(
        "--config",
        default="config/training.toml",
        help="训练主配置路径，必须位于项目目录内。",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="临时限制训练 step 数，主要用于 smoke test。",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_training(args.config, max_steps=args.max_steps)
    print(
        "训练结束："
        f"global_step={summary.global_step}, "
        f"epoch={summary.epoch}, "
        f"latest_checkpoint_path={summary.latest_checkpoint_path}, "
        f"metrics_path={summary.metrics_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
