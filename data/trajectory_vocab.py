"""从逐场景 H5 全局采样规划轨迹词表。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


DEFAULT_TRAJECTORY_DATASET = "labels/future_trajectory"

__all__ = [
    "TrajectoryVocabulary",
    "TrajectoryVocabularyConfig",
    "build_trajectory_vocabulary",
    "load_future_trajectories",
    "sample_trajectory_vocabulary",
    "save_trajectory_vocabulary",
    "symlog",
]


@dataclass(frozen=True)
class TrajectoryVocabularyConfig:
    """轨迹词表采样配置。

    Args:
        h5_paths: 单个 H5 文件、H5 目录，或 H5 文件列表。目录模式仅读取
            当前目录下的 `*.h5`，用于跨场景全局采样。
        output_path: 可选输出 `.npz` 路径；仅 `save_trajectory_vocabulary`
            和 CLI 使用。
        num_trajectories: 词表轨迹数量，默认 256。第 0 条固定为全零静止轨迹。
        trajectory_dataset: H5 内自车未来轨迹字段路径。
        future_points: 每条轨迹的未来点数，默认 6。
        trajectory_dim: 每个轨迹点维度，默认 2，对应 ego 坐标系 XY。
        distance_batch_size: FTS 更新最小距离时的分块大小。
        symlog_scale_eps: 判断 Symlog 缩放系数是否为零的阈值。
    """

    h5_paths: str | Path | Sequence[str | Path]
    output_path: str | Path | None = None
    num_trajectories: int = 256
    trajectory_dataset: str = DEFAULT_TRAJECTORY_DATASET
    future_points: int = 6
    trajectory_dim: int = 2
    distance_batch_size: int = 65536
    symlog_scale_eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.num_trajectories < 2:
            raise ValueError(
                "num_trajectories 必须至少为 2，因为第 0 条保留给静止轨迹，"
                f"实际为 {self.num_trajectories}。"
            )
        if not self.trajectory_dataset:
            raise ValueError("trajectory_dataset 不能为空。")
        if self.future_points <= 0:
            raise ValueError(f"future_points 必须为正数，实际为 {self.future_points}。")
        if self.trajectory_dim <= 0:
            raise ValueError(f"trajectory_dim 必须为正数，实际为 {self.trajectory_dim}。")
        if self.distance_batch_size <= 0:
            raise ValueError(
                f"distance_batch_size 必须为正数，实际为 {self.distance_batch_size}。"
            )
        if self.symlog_scale_eps <= 0:
            raise ValueError(f"symlog_scale_eps 必须为正数，实际为 {self.symlog_scale_eps}。")
        if self.output_path is not None:
            object.__setattr__(self, "output_path", Path(self.output_path))


@dataclass(frozen=True)
class TrajectoryVocabulary:
    """轨迹词表及其归一化版本。

    Shape:
        `trajectory_vocab_m`: `[V, 6, 2]`，ego 坐标系米制轨迹。
        `trajectory_vocab_symlog`: `[V, 6, 2]`，Symlog 空间轨迹。
        `trajectory_vocab_normalized`: `[V, 6, 2]`，共享缩放系数归一化结果。
        `selected_source_h5_indices`: `[V]`，第 0 条静止轨迹为 `-1`。
        `selected_source_sample_indices`: `[V]`，第 0 条静止轨迹为 `-1`。
    """

    trajectory_vocab_m: np.ndarray
    trajectory_vocab_symlog: np.ndarray
    trajectory_vocab_normalized: np.ndarray
    symlog_scale: float
    selected_source_h5_indices: np.ndarray
    selected_source_sample_indices: np.ndarray
    source_h5_paths: tuple[str, ...]
    metadata: dict[str, Any]


def build_trajectory_vocabulary(config: TrajectoryVocabularyConfig) -> TrajectoryVocabulary:
    """从指定 H5 集合构造跨场景轨迹词表。

    第 0 条轨迹强制为全零静止轨迹；剩余 `num_trajectories - 1` 条从全部
    H5 的 `labels/future_trajectory` 中执行 FTS 采样。FTS 距离在物理
    ego 坐标系 meter 空间计算，采样完成后再执行 Symlog 和共享缩放归一化。
    """

    h5_paths = _resolve_h5_paths(config.h5_paths)
    if not h5_paths:
        raise FileNotFoundError(f"未找到 H5 文件：{config.h5_paths!r}")

    (
        trajectories,
        source_h5_indices,
        source_sample_indices,
        load_report,
    ) = _load_future_trajectories_with_report(
        h5_paths,
        trajectory_dataset=config.trajectory_dataset,
        future_points=config.future_points,
        trajectory_dim=config.trajectory_dim,
    )
    selected_indices = sample_trajectory_vocabulary(
        trajectories,
        num_trajectories=config.num_trajectories,
        distance_batch_size=config.distance_batch_size,
    )

    vocab_shape = (config.num_trajectories, config.future_points, config.trajectory_dim)
    trajectory_vocab_m = np.zeros(vocab_shape, dtype=np.float32)
    trajectory_vocab_m[1:] = trajectories[selected_indices]

    selected_source_h5_indices = np.full((config.num_trajectories,), -1, dtype=np.int32)
    selected_source_sample_indices = np.full((config.num_trajectories,), -1, dtype=np.int64)
    selected_source_h5_indices[1:] = source_h5_indices[selected_indices]
    selected_source_sample_indices[1:] = source_sample_indices[selected_indices]

    trajectory_vocab_symlog = symlog(trajectory_vocab_m)
    max_abs_symlog = float(np.max(np.abs(trajectory_vocab_symlog)))
    symlog_scale = max_abs_symlog if max_abs_symlog > config.symlog_scale_eps else 1.0
    trajectory_vocab_normalized = (trajectory_vocab_symlog / symlog_scale).astype(np.float32)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trajectory_dataset": config.trajectory_dataset,
        "num_trajectories": config.num_trajectories,
        "future_points": config.future_points,
        "trajectory_dim": config.trajectory_dim,
        "coordinate_system": "ego: x forward, y left, unit meter",
        "trajectory_frequency_hz": 2,
        "trajectory_seconds": 3.0,
        "sampling_algorithm": "FTS",
        "distance_metric": "mean_squared_error_in_physical_ego_meter_space",
        "static_trajectory_index": 0,
        "static_trajectory_policy": "forced_zero",
        "symlog": "sign(x) * ln(abs(x) + 1)",
        "symlog_scale": symlog_scale,
        "normalization": "shared_scale_across_all_trajectories_points_and_dimensions",
        "source_h5_paths": tuple(str(path) for path in h5_paths),
        "source_sample_count": int(load_report["total_sample_count"]),
        "valid_source_sample_count": int(load_report["valid_sample_count"]),
        "skipped_invalid_sample_count": int(load_report["skipped_invalid_sample_count"]),
        "skipped_invalid_by_h5": load_report["skipped_invalid_by_h5"],
    }
    return TrajectoryVocabulary(
        trajectory_vocab_m=trajectory_vocab_m,
        trajectory_vocab_symlog=trajectory_vocab_symlog,
        trajectory_vocab_normalized=trajectory_vocab_normalized,
        symlog_scale=symlog_scale,
        selected_source_h5_indices=selected_source_h5_indices,
        selected_source_sample_indices=selected_source_sample_indices,
        source_h5_paths=tuple(str(path) for path in h5_paths),
        metadata=metadata,
    )


def load_future_trajectories(
    h5_paths: Sequence[str | Path],
    trajectory_dataset: str = DEFAULT_TRAJECTORY_DATASET,
    future_points: int = 6,
    trajectory_dim: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取多个逐场景 H5 的自车未来轨迹字段。

    含 NaN 或 Inf 的单条轨迹会被跳过；其他合法样本继续参与全局 FTS
    采样。`source_sample_indices` 保留源 H5 内原始样本索引，因此跳过
    无效样本后仍可追溯到原文件位置。

    Returns:
        三元组 `(trajectories, source_h5_indices, source_sample_indices)`：

        - `trajectories`: `[N, 6, 2] float32`。
        - `source_h5_indices`: `[N] int32`，每条轨迹来自第几个 H5。
        - `source_sample_indices`: `[N] int64`，每条轨迹在源 H5 内的样本索引。
    """

    trajectories, source_h5_indices, source_sample_indices, _load_report = (
        _load_future_trajectories_with_report(
            h5_paths,
            trajectory_dataset=trajectory_dataset,
            future_points=future_points,
            trajectory_dim=trajectory_dim,
        )
    )
    return trajectories, source_h5_indices, source_sample_indices


def _load_future_trajectories_with_report(
    h5_paths: Sequence[str | Path],
    trajectory_dataset: str = DEFAULT_TRAJECTORY_DATASET,
    future_points: int = 6,
    trajectory_dim: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    h5py = _require_h5py()
    trajectory_arrays: list[np.ndarray] = []
    source_h5_index_arrays: list[np.ndarray] = []
    source_sample_index_arrays: list[np.ndarray] = []
    expected_shape_suffix = (future_points, trajectory_dim)
    total_sample_count = 0
    skipped_invalid_sample_count = 0
    skipped_invalid_by_h5: list[dict[str, Any]] = []

    for h5_index, h5_path_like in enumerate(h5_paths):
        h5_path = Path(h5_path_like)
        with h5py.File(h5_path, "r") as h5_file:
            if trajectory_dataset not in h5_file:
                raise KeyError(f"H5 缺少轨迹字段 {trajectory_dataset!r}：{h5_path}")
            dataset = h5_file[trajectory_dataset]
            if len(dataset.shape) != 3 or tuple(dataset.shape[1:]) != expected_shape_suffix:
                raise ValueError(
                    f"{trajectory_dataset} 期望 shape 为 [S, {future_points}, {trajectory_dim}]，"
                    f"实际为 {dataset.shape}，文件：{h5_path}"
                )
            if dataset.shape[0] <= 0:
                continue
            raw_trajectories = np.asarray(dataset, dtype=np.float32)
            total_sample_count += int(raw_trajectories.shape[0])
            finite_mask = np.isfinite(raw_trajectories).all(axis=(1, 2))
            valid_sample_indices = np.flatnonzero(finite_mask).astype(np.int64, copy=False)
            invalid_sample_indices = np.flatnonzero(~finite_mask).astype(np.int64, copy=False)
            if invalid_sample_indices.size > 0:
                skipped_count = int(invalid_sample_indices.size)
                skipped_invalid_sample_count += skipped_count
                skipped_invalid_by_h5.append(
                    {
                        "h5_path": str(h5_path),
                        "skipped_count": skipped_count,
                        "skipped_sample_indices": invalid_sample_indices.tolist(),
                    }
                )
            if valid_sample_indices.size == 0:
                continue
            trajectories = raw_trajectories[valid_sample_indices]
            trajectory_arrays.append(trajectories)
            source_h5_index_arrays.append(
                np.full((trajectories.shape[0],), h5_index, dtype=np.int32)
            )
            source_sample_index_arrays.append(valid_sample_indices)

    if not trajectory_arrays:
        raise ValueError(
            f"没有从 H5 字段 {trajectory_dataset!r} 读取到任何有效轨迹；"
            f"总样本数为 {total_sample_count}，跳过无效样本数为 {skipped_invalid_sample_count}。"
        )

    trajectories = np.concatenate(trajectory_arrays, axis=0)
    load_report = {
        "total_sample_count": int(total_sample_count),
        "valid_sample_count": int(trajectories.shape[0]),
        "skipped_invalid_sample_count": int(skipped_invalid_sample_count),
        "skipped_invalid_by_h5": skipped_invalid_by_h5,
    }
    return (
        trajectories,
        np.concatenate(source_h5_index_arrays, axis=0),
        np.concatenate(source_sample_index_arrays, axis=0),
        load_report,
    )


def sample_trajectory_vocabulary(
    trajectories: np.ndarray,
    num_trajectories: int = 256,
    distance_batch_size: int = 65536,
) -> np.ndarray:
    """使用 FTS 从轨迹全集中选择词表样本索引。

    第 0 条词表轨迹由调用方强制写成全零静止轨迹，因此本函数只返回
    `num_trajectories - 1` 个数据轨迹索引。FTS 初始化中心为全零轨迹，
    后续每一步选择到当前已选中心集合的最小 MSE 距离最大的样本。
    """

    if trajectories.ndim != 3:
        raise ValueError(f"trajectories 期望 shape 为 [N, K, D]，实际为 {trajectories.shape}。")
    if num_trajectories < 2:
        raise ValueError(f"num_trajectories 必须至少为 2，实际为 {num_trajectories}。")
    if distance_batch_size <= 0:
        raise ValueError(f"distance_batch_size 必须为正数，实际为 {distance_batch_size}。")

    sample_count = int(trajectories.shape[0])
    required_sample_count = num_trajectories - 1
    if sample_count < required_sample_count:
        raise ValueError(
            "FTS 采样样本数不足：第 0 条词表轨迹固定为静止轨迹，"
            f"还需要 {required_sample_count} 条数据轨迹，实际只有 {sample_count} 条。"
        )

    flat_trajectories = np.ascontiguousarray(trajectories.reshape(sample_count, -1), dtype=np.float32)
    feature_dim = int(flat_trajectories.shape[1])
    min_mse = np.full((sample_count,), np.inf, dtype=np.float32)
    selected_mask = np.zeros((sample_count,), dtype=np.bool_)
    selected_indices = np.empty((required_sample_count,), dtype=np.int64)

    static_center = np.zeros((feature_dim,), dtype=np.float32)
    _update_min_mse(flat_trajectories, static_center, min_mse, distance_batch_size)

    for selected_count in range(required_sample_count):
        selected_index = int(np.argmax(min_mse))
        if min_mse[selected_index] < 0 or not np.isfinite(min_mse[selected_index]):
            raise RuntimeError("FTS 采样失败：没有可选择的未采样轨迹。")
        selected_indices[selected_count] = selected_index
        selected_mask[selected_index] = True
        _update_min_mse(
            flat_trajectories,
            flat_trajectories[selected_index],
            min_mse,
            distance_batch_size,
        )
        min_mse[selected_mask] = -1.0

    return selected_indices


def save_trajectory_vocabulary(
    vocabulary: TrajectoryVocabulary,
    output_path: str | Path,
    overwrite: bool = False,
) -> Path:
    """将轨迹词表保存为 `.npz`。"""

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"输出文件已存在，若要覆盖请设置 overwrite=True：{output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_json = json.dumps(vocabulary.metadata, ensure_ascii=False, indent=2)
    np.savez_compressed(
        output_path,
        trajectory_vocab_m=vocabulary.trajectory_vocab_m,
        trajectory_vocab_symlog=vocabulary.trajectory_vocab_symlog,
        trajectory_vocab_normalized=vocabulary.trajectory_vocab_normalized,
        symlog_scale=np.asarray(vocabulary.symlog_scale, dtype=np.float32),
        selected_source_h5_indices=vocabulary.selected_source_h5_indices,
        selected_source_sample_indices=vocabulary.selected_source_sample_indices,
        source_h5_paths=np.asarray(vocabulary.source_h5_paths, dtype=np.str_),
        metadata_json=np.asarray(metadata_json, dtype=np.str_),
    )
    return output_path


def symlog(values: np.ndarray) -> np.ndarray:
    """计算 `sign(x) * ln(abs(x) + 1)`。"""

    return (np.sign(values) * np.log1p(np.abs(values))).astype(np.float32)


def _update_min_mse(
    flat_trajectories: np.ndarray,
    center: np.ndarray,
    min_mse: np.ndarray,
    batch_size: int,
) -> None:
    feature_dim = float(flat_trajectories.shape[1])
    for start_index in range(0, flat_trajectories.shape[0], batch_size):
        end_index = min(start_index + batch_size, flat_trajectories.shape[0])
        batch = flat_trajectories[start_index:end_index]
        difference = batch - center
        batch_mse = np.einsum("ij,ij->i", difference, difference, optimize=True) / feature_dim
        np.minimum(min_mse[start_index:end_index], batch_mse, out=min_mse[start_index:end_index])


def _resolve_h5_paths(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        path = Path(paths)
        if path.is_dir():
            resolved_paths = sorted(path.glob("*.h5"))
        else:
            resolved_paths = [path]
    else:
        resolved_paths = sorted(Path(path) for path in paths)

    for h5_path in resolved_paths:
        if not h5_path.exists():
            raise FileNotFoundError(f"H5 文件不存在：{h5_path}")
        if not h5_path.is_file():
            raise FileNotFoundError(f"H5 路径不是文件：{h5_path}")
        if h5_path.suffix.lower() != ".h5":
            raise ValueError(f"H5 文件扩展名必须为 .h5，实际为：{h5_path}")
    return resolved_paths


def _require_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "轨迹词表采样需要 h5py。请先在项目环境中安装 h5py，例如："
            ".\\.venv\\Scripts\\python.exe -m pip install h5py"
        ) from exc
    return h5py


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从逐场景 H5 全局 FTS 采样规划轨迹词表。")
    parser.add_argument("--h5-dir", type=Path, default=None, help="包含逐场景 H5 的目录。")
    parser.add_argument(
        "--h5-path",
        type=Path,
        action="append",
        default=None,
        help="单个 H5 文件；可重复传入多个。若传入该参数，则不使用 --h5-dir。",
    )
    parser.add_argument("--output", type=Path, required=True, help="输出 .npz 路径。")
    parser.add_argument("--num-trajectories", type=int, default=256, help="词表轨迹数量，默认 256。")
    parser.add_argument(
        "--trajectory-dataset",
        default=DEFAULT_TRAJECTORY_DATASET,
        help="H5 内自车未来轨迹字段路径。",
    )
    parser.add_argument("--future-points", type=int, default=6, help="每条轨迹未来点数。")
    parser.add_argument("--trajectory-dim", type=int, default=2, help="每个轨迹点维度。")
    parser.add_argument(
        "--distance-batch-size",
        type=int,
        default=65536,
        help="FTS 距离更新分块大小。",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在输出文件。")
    args = parser.parse_args(argv)

    if args.h5_path is None and args.h5_dir is None:
        parser.error("必须提供 --h5-dir 或至少一个 --h5-path。")
    h5_paths: Path | list[Path]
    h5_paths = args.h5_path if args.h5_path is not None else args.h5_dir

    config = TrajectoryVocabularyConfig(
        h5_paths=h5_paths,
        output_path=args.output,
        num_trajectories=args.num_trajectories,
        trajectory_dataset=args.trajectory_dataset,
        future_points=args.future_points,
        trajectory_dim=args.trajectory_dim,
        distance_batch_size=args.distance_batch_size,
    )
    vocabulary = build_trajectory_vocabulary(config)
    output_path = save_trajectory_vocabulary(vocabulary, args.output, overwrite=args.overwrite)
    print(output_path)
    skipped_invalid_sample_count = int(vocabulary.metadata["skipped_invalid_sample_count"])
    if skipped_invalid_sample_count > 0:
        print(
            "skipped_invalid_sample_count="
            f"{skipped_invalid_sample_count}, "
            f"valid_source_sample_count={vocabulary.metadata['valid_source_sample_count']}, "
            f"source_sample_count={vocabulary.metadata['source_sample_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
