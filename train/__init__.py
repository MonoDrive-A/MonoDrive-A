"""MonoDrive 训练辅助模块。"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AgentMatchingTargets",
    "CheckpointConfig",
    "DataLoaderConfig",
    "GradientMonitorConfig",
    "GradientMonitorResult",
    "GradientParameterStat",
    "LoggingConfig",
    "LossWeights",
    "MapMatchingTargets",
    "MonoDriveTrainingLoss",
    "OptimizationConfig",
    "RandomConfig",
    "RuntimeConfig",
    "TrainingLossOutput",
    "TrainingBatchLabels",
    "TrainingDataConfig",
    "TrainingRunConfig",
    "TrainingSummary",
    "ValidatedTrainingDataset",
    "WarmupCosineLRScheduler",
    "build_agent_matching_targets",
    "build_map_matching_targets",
    "build_training_batch_labels",
    "build_training_dataset",
    "build_trajectory_vocab_labels",
    "capture_rng_state",
    "find_resume_checkpoint",
    "inspect_gradients",
    "load_checkpoint",
    "load_training_data_config",
    "load_training_run_config",
    "restore_rng_state",
    "run_training",
    "save_checkpoint",
    "training_collate",
]

_LAZY_EXPORTS = {
    "AgentMatchingTargets": "train.data_processing",
    "CheckpointConfig": "train.training_config",
    "DataLoaderConfig": "train.training_config",
    "GradientMonitorConfig": "train.training_config",
    "GradientMonitorResult": "train.gradient_monitor",
    "GradientParameterStat": "train.gradient_monitor",
    "LoggingConfig": "train.training_config",
    "LossWeights": "train.training_config",
    "MapMatchingTargets": "train.data_processing",
    "MonoDriveTrainingLoss": "train.losses",
    "OptimizationConfig": "train.training_config",
    "RandomConfig": "train.training_config",
    "RuntimeConfig": "train.training_config",
    "TrainingLossOutput": "train.losses",
    "TrainingBatchLabels": "train.data_processing",
    "TrainingDataConfig": "train.data_processing",
    "TrainingRunConfig": "train.training_config",
    "TrainingSummary": "train.trainer",
    "ValidatedTrainingDataset": "train.data_processing",
    "WarmupCosineLRScheduler": "train.trainer",
    "build_agent_matching_targets": "train.data_processing",
    "build_map_matching_targets": "train.data_processing",
    "build_training_batch_labels": "train.data_processing",
    "build_training_dataset": "train.data_processing",
    "build_trajectory_vocab_labels": "train.data_processing",
    "capture_rng_state": "train.checkpointing",
    "find_resume_checkpoint": "train.checkpointing",
    "inspect_gradients": "train.gradient_monitor",
    "load_checkpoint": "train.checkpointing",
    "load_training_data_config": "train.data_processing",
    "load_training_run_config": "train.training_config",
    "restore_rng_state": "train.checkpointing",
    "run_training": "train.trainer",
    "save_checkpoint": "train.checkpointing",
    "training_collate": "train.data_processing",
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(_LAZY_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
