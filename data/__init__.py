"""MonoDrive 数据处理模块。"""

from __future__ import annotations

from typing import Any

__all__ = [
    "B2DPreprocessConfig",
    "B2DH5Dataset",
    "B2DScenePreprocessor",
    "SampleWindow",
    "ScenePaths",
    "TrajectoryVocabulary",
    "TrajectoryVocabularyConfig",
    "build_sample_windows",
    "build_trajectory_vocabulary",
    "discover_b2d_scenes",
    "preprocess_b2d_dataset",
    "sample_trajectory_vocabulary",
    "save_trajectory_vocabulary",
]

_LAZY_EXPORTS = {
    "B2DH5Dataset": "data.b2d_dataset",
    "B2DPreprocessConfig": "data.b2d_preprocess",
    "B2DScenePreprocessor": "data.b2d_preprocess",
    "SampleWindow": "data.b2d_preprocess",
    "ScenePaths": "data.b2d_preprocess",
    "build_sample_windows": "data.b2d_preprocess",
    "discover_b2d_scenes": "data.b2d_preprocess",
    "preprocess_b2d_dataset": "data.b2d_preprocess",
    "TrajectoryVocabulary": "data.trajectory_vocab",
    "TrajectoryVocabularyConfig": "data.trajectory_vocab",
    "build_trajectory_vocabulary": "data.trajectory_vocab",
    "sample_trajectory_vocabulary": "data.trajectory_vocab",
    "save_trajectory_vocabulary": "data.trajectory_vocab",
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(_LAZY_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
