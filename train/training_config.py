"""训练主流程配置读取与校验。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any, Mapping


__all__ = [
    "CheckpointConfig",
    "DataLoaderConfig",
    "GradientMonitorConfig",
    "LoggingConfig",
    "LossWeights",
    "OptimizationConfig",
    "RandomConfig",
    "RuntimeConfig",
    "TrainingRunConfig",
    "load_training_run_config",
]


SUPPORTED_DEVICES = {"auto", "cpu", "cuda"}
SUPPORTED_OPTIMIZERS = {"adamw"}


@dataclass(frozen=True)
class RuntimeConfig:
    """训练运行设备配置。"""

    device: str
    non_blocking: bool

    def __post_init__(self) -> None:
        if self.device not in SUPPORTED_DEVICES:
            raise ValueError(f"device 仅支持 {sorted(SUPPORTED_DEVICES)}，实际为 {self.device!r}。")


@dataclass(frozen=True)
class RandomConfig:
    """训练随机性配置。"""

    seed: int
    deterministic: bool

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError(f"seed 必须为非负整数，实际为 {self.seed}。")


@dataclass(frozen=True)
class DataLoaderConfig:
    """训练 DataLoader 配置。"""

    batch_size: int
    shuffle: bool
    num_workers: int
    pin_memory: bool
    drop_last: bool
    persistent_workers: bool
    prefetch_factor: int
    worker_seed_stride: int

    def __post_init__(self) -> None:
        _validate_positive_int(self.batch_size, "batch_size")
        if self.num_workers < 0:
            raise ValueError(f"num_workers 不能为负数，实际为 {self.num_workers}。")
        _validate_positive_int(self.prefetch_factor, "prefetch_factor")
        _validate_positive_int(self.worker_seed_stride, "worker_seed_stride")
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError("persistent_workers=True 时 num_workers 必须大于 0。")


@dataclass(frozen=True)
class OptimizationConfig:
    """优化器和学习率调度配置。"""

    optimizer: str
    initial_lr: float
    peak_lr: float
    min_lr: float
    warmup_steps: int
    total_steps: int
    cosine_decay_steps: int
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    enable_gradient_clipping: bool
    max_grad_norm: float
    zero_grad_set_to_none: bool

    def __post_init__(self) -> None:
        if self.optimizer not in SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"optimizer 仅支持 {sorted(SUPPORTED_OPTIMIZERS)}，实际为 {self.optimizer!r}。"
            )
        for field_name in ("initial_lr", "peak_lr", "min_lr", "adam_eps", "max_grad_norm"):
            value = getattr(self, field_name)
            if value <= 0.0:
                raise ValueError(f"{field_name} 必须为正数，实际为 {value}。")
        if self.initial_lr > self.peak_lr:
            raise ValueError(
                f"initial_lr 不能大于 peak_lr，实际为 {self.initial_lr} 和 {self.peak_lr}。"
            )
        if self.min_lr > self.peak_lr:
            raise ValueError(f"min_lr 不能大于 peak_lr，实际为 {self.min_lr} 和 {self.peak_lr}。")
        _validate_positive_int(self.warmup_steps, "warmup_steps")
        _validate_positive_int(self.total_steps, "total_steps")
        _validate_positive_int(self.cosine_decay_steps, "cosine_decay_steps")
        if self.warmup_steps + self.cosine_decay_steps > self.total_steps:
            raise ValueError(
                "warmup_steps + cosine_decay_steps 不能超过 total_steps，"
                f"实际为 {self.warmup_steps} + {self.cosine_decay_steps} > {self.total_steps}。"
            )
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay 不能为负数，实际为 {self.weight_decay}。")
        for field_name in ("adam_beta1", "adam_beta2"):
            value = getattr(self, field_name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{field_name} 必须位于 [0, 1)，实际为 {value}。")


@dataclass(frozen=True)
class LossWeights:
    """训练 loss 权重。"""

    trajectory_logit_bce: float
    trajectory_residual_mse: float
    agent_class_ce: float
    agent_state_mse: float
    agent_mode_ce: float
    agent_future_mse: float
    map_class_ce: float
    map_point_mse: float

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if value < 0.0:
                raise ValueError(f"{field_name} 不能为负数，实际为 {value}。")
        if all(getattr(self, field_name) == 0.0 for field_name in self.__dataclass_fields__):
            raise ValueError("loss 权重不能全部为 0。")


@dataclass(frozen=True)
class GradientMonitorConfig:
    """梯度监测配置。"""

    enabled: bool
    check_interval_steps: int
    large_grad_norm: float
    small_grad_norm: float
    max_report_parameters: int
    fail_on_nonfinite: bool
    log_missing_gradients: bool

    def __post_init__(self) -> None:
        _validate_positive_int(self.check_interval_steps, "check_interval_steps")
        _validate_positive_int(self.max_report_parameters, "max_report_parameters")
        if self.large_grad_norm <= 0.0:
            raise ValueError(f"large_grad_norm 必须为正数，实际为 {self.large_grad_norm}。")
        if self.small_grad_norm < 0.0:
            raise ValueError(f"small_grad_norm 不能为负数，实际为 {self.small_grad_norm}。")
        if self.small_grad_norm >= self.large_grad_norm:
            raise ValueError(
                "small_grad_norm 必须小于 large_grad_norm，"
                f"实际为 {self.small_grad_norm} 和 {self.large_grad_norm}。"
            )


@dataclass(frozen=True)
class CheckpointConfig:
    """checkpoint 保存和恢复配置。"""

    output_dir: Path
    save_interval_steps: int
    keep_last: int
    resume_from_latest: bool
    resume_checkpoint_path: Path | None
    latest_filename: str
    save_on_exit: bool
    save_rng_state: bool

    def __post_init__(self) -> None:
        _validate_positive_int(self.save_interval_steps, "save_interval_steps")
        if self.keep_last < 0:
            raise ValueError(f"keep_last 不能为负数，实际为 {self.keep_last}。")
        if not self.latest_filename:
            raise ValueError("latest_filename 不能为空。")


@dataclass(frozen=True)
class LoggingConfig:
    """训练日志配置。"""

    output_dir: Path
    log_interval_steps: int
    metrics_filename: str

    def __post_init__(self) -> None:
        _validate_positive_int(self.log_interval_steps, "log_interval_steps")
        if not self.metrics_filename:
            raise ValueError("metrics_filename 不能为空。")


@dataclass(frozen=True)
class TrainingRunConfig:
    """训练主流程配置。"""

    project_root: Path
    backbone_config_path: Path
    training_data_config_path: Path
    runtime: RuntimeConfig
    random: RandomConfig
    dataloader: DataLoaderConfig
    optimization: OptimizationConfig
    loss_weights: LossWeights
    gradient_monitor: GradientMonitorConfig
    checkpoint: CheckpointConfig
    logging: LoggingConfig


def load_training_run_config(
    config_path: str | Path,
    project_root: str | Path | None = None,
) -> TrainingRunConfig:
    """读取训练主配置。"""

    resolved_config_path = Path(config_path).resolve()
    resolved_project_root = (
        Path(project_root).resolve() if project_root is not None else resolved_config_path.parent.parent
    )
    _ensure_project_relative_path(resolved_config_path, resolved_project_root, "config_path")
    with resolved_config_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)

    modules_config = _require_table(raw_config, "modules")
    runtime_config = _require_table(raw_config, "runtime")
    random_config = _require_table(raw_config, "random")
    dataloader_config = _require_table(raw_config, "dataloader")
    optimization_config = _require_table(raw_config, "optimization")
    loss_weights_config = _require_table(raw_config, "loss_weights")
    gradient_monitor_config = _require_table(raw_config, "gradient_monitor")
    checkpoint_config = _require_table(raw_config, "checkpoint")
    logging_config = _require_table(raw_config, "logging")

    checkpoint_output_dir = _resolve_project_relative_path(
        checkpoint_config,
        "output_dir",
        resolved_project_root,
    )
    resume_checkpoint_path = _resolve_optional_project_relative_path(
        checkpoint_config,
        "resume_checkpoint_path",
        resolved_project_root,
    )
    logging_output_dir = _resolve_project_relative_path(
        logging_config,
        "output_dir",
        resolved_project_root,
    )

    return TrainingRunConfig(
        project_root=resolved_project_root,
        backbone_config_path=_resolve_project_relative_path(
            modules_config,
            "backbone_config_path",
            resolved_project_root,
        ),
        training_data_config_path=_resolve_project_relative_path(
            modules_config,
            "training_data_config_path",
            resolved_project_root,
        ),
        runtime=RuntimeConfig(
            device=_require_string(runtime_config, "device"),
            non_blocking=_require_bool(runtime_config, "non_blocking"),
        ),
        random=RandomConfig(
            seed=_require_int(random_config, "seed"),
            deterministic=_require_bool(random_config, "deterministic"),
        ),
        dataloader=DataLoaderConfig(
            batch_size=_require_int(dataloader_config, "batch_size"),
            shuffle=_require_bool(dataloader_config, "shuffle"),
            num_workers=_require_int(dataloader_config, "num_workers"),
            pin_memory=_require_bool(dataloader_config, "pin_memory"),
            drop_last=_require_bool(dataloader_config, "drop_last"),
            persistent_workers=_require_bool(dataloader_config, "persistent_workers"),
            prefetch_factor=_require_int(dataloader_config, "prefetch_factor"),
            worker_seed_stride=_require_int(dataloader_config, "worker_seed_stride"),
        ),
        optimization=OptimizationConfig(
            optimizer=_require_string(optimization_config, "optimizer"),
            initial_lr=_require_float(optimization_config, "initial_lr"),
            peak_lr=_require_float(optimization_config, "peak_lr"),
            min_lr=_require_float(optimization_config, "min_lr"),
            warmup_steps=_require_int(optimization_config, "warmup_steps"),
            total_steps=_require_int(optimization_config, "total_steps"),
            cosine_decay_steps=_require_int(optimization_config, "cosine_decay_steps"),
            weight_decay=_require_float(optimization_config, "weight_decay"),
            adam_beta1=_require_float(optimization_config, "adam_beta1"),
            adam_beta2=_require_float(optimization_config, "adam_beta2"),
            adam_eps=_require_float(optimization_config, "adam_eps"),
            enable_gradient_clipping=_require_bool(
                optimization_config,
                "enable_gradient_clipping",
            ),
            max_grad_norm=_require_float(optimization_config, "max_grad_norm"),
            zero_grad_set_to_none=_require_bool(
                optimization_config,
                "zero_grad_set_to_none",
            ),
        ),
        loss_weights=LossWeights(
            trajectory_logit_bce=_require_float(loss_weights_config, "trajectory_logit_bce"),
            trajectory_residual_mse=_require_float(
                loss_weights_config,
                "trajectory_residual_mse",
            ),
            agent_class_ce=_require_float(loss_weights_config, "agent_class_ce"),
            agent_state_mse=_require_float(loss_weights_config, "agent_state_mse"),
            agent_mode_ce=_require_float(loss_weights_config, "agent_mode_ce"),
            agent_future_mse=_require_float(loss_weights_config, "agent_future_mse"),
            map_class_ce=_require_float(loss_weights_config, "map_class_ce"),
            map_point_mse=_require_float(loss_weights_config, "map_point_mse"),
        ),
        gradient_monitor=GradientMonitorConfig(
            enabled=_require_bool(gradient_monitor_config, "enabled"),
            check_interval_steps=_require_int(
                gradient_monitor_config,
                "check_interval_steps",
            ),
            large_grad_norm=_require_float(gradient_monitor_config, "large_grad_norm"),
            small_grad_norm=_require_float(gradient_monitor_config, "small_grad_norm"),
            max_report_parameters=_require_int(
                gradient_monitor_config,
                "max_report_parameters",
            ),
            fail_on_nonfinite=_require_bool(gradient_monitor_config, "fail_on_nonfinite"),
            log_missing_gradients=_require_bool(
                gradient_monitor_config,
                "log_missing_gradients",
            ),
        ),
        checkpoint=CheckpointConfig(
            output_dir=checkpoint_output_dir,
            save_interval_steps=_require_int(checkpoint_config, "save_interval_steps"),
            keep_last=_require_int(checkpoint_config, "keep_last"),
            resume_from_latest=_require_bool(checkpoint_config, "resume_from_latest"),
            resume_checkpoint_path=resume_checkpoint_path,
            latest_filename=_require_string(checkpoint_config, "latest_filename"),
            save_on_exit=_require_bool(checkpoint_config, "save_on_exit"),
            save_rng_state=_require_bool(checkpoint_config, "save_rng_state"),
        ),
        logging=LoggingConfig(
            output_dir=logging_output_dir,
            log_interval_steps=_require_int(logging_config, "log_interval_steps"),
            metrics_filename=_require_string(logging_config, "metrics_filename"),
        ),
    )


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


def _resolve_optional_project_relative_path(
    table: Mapping[str, Any],
    key: str,
    project_root: Path,
) -> Path | None:
    path_text = _require_string(table, key)
    if path_text == "":
        return None
    raw_path = Path(path_text)
    if raw_path.is_absolute():
        raise ValueError(f"{key} 必须为项目内相对路径或空字符串，实际为 {raw_path}。")
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


def _validate_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} 必须为整数，实际为 {value!r}。")
    if value <= 0:
        raise ValueError(f"{field_name} 必须为正整数，实际为 {value}。")


def _require_table(raw_config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw_config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"配置缺少 [{key}] 表。")
    return value


def _require_string(table: Mapping[str, Any], key: str) -> str:
    value = table.get(key)
    if not isinstance(value, str):
        raise ValueError(f"配置项 {key} 必须为字符串，实际为 {value!r}。")
    return value


def _require_bool(table: Mapping[str, Any], key: str) -> bool:
    value = table.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"配置项 {key} 必须为 bool，实际为 {value!r}。")
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
