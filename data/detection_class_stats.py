"""统计逐场景 H5 数据集的检测类别分布。"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


AGENT_CLASS_NAMES = {
    0: "car",
    1: "bicycle",
    2: "motorcycle",
    3: "pedestrian",
}
MAP_CLASS_NAMES = {
    0: "lane_divider",
    1: "road_edge",
    2: "crosswalk",
    3: "centerline",
}
TRAFFIC_LIGHT_CLASS_NAMES = {
    0: "red",
    1: "green",
    2: "yellow",
    3: "none",
}
STOP_SIGN_CLASS_NAMES = {
    0: "none",
    1: "present",
}


@dataclass(frozen=True)
class DetectionClassStatsConfig:
    """H5 检测类别分布统计配置。

    Args:
        h5_paths: H5 文件、目录，或文件/目录列表。目录会递归扫描 `*.h5`。
        output_json: 可选 JSON 输出路径。
        batch_size: 按样本维分块读取的大小，避免整个数据集常驻内存。
        include_invalid_traffic: 是否把无效 Traffic Light / Stop Sign 样本也计入状态分布。
    """

    h5_paths: str | Path | Sequence[str | Path]
    output_json: str | Path | None = None
    batch_size: int = 4096
    include_invalid_traffic: bool = False

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size 必须为正数，实际为 {self.batch_size}。")
        if self.output_json is not None:
            object.__setattr__(self, "output_json", Path(self.output_json))


@dataclass
class DetectionClassStats:
    """跨 H5 数据集检测类别统计结果。"""

    h5_paths: list[str]
    scene_count: int = 0
    sample_count: int = 0
    agent_slot_count: int = 0
    map_slot_count: int = 0
    valid_agent_count: int = 0
    valid_map_count: int = 0
    valid_traffic_light_count: int = 0
    valid_stop_sign_count: int = 0
    agent_counts: Counter[str] | None = None
    map_counts: Counter[str] | None = None
    traffic_light_counts: Counter[str] | None = None
    stop_sign_counts: Counter[str] | None = None
    per_scene: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        self.agent_counts = Counter() if self.agent_counts is None else self.agent_counts
        self.map_counts = Counter() if self.map_counts is None else self.map_counts
        self.traffic_light_counts = (
            Counter() if self.traffic_light_counts is None else self.traffic_light_counts
        )
        self.stop_sign_counts = Counter() if self.stop_sign_counts is None else self.stop_sign_counts
        self.per_scene = [] if self.per_scene is None else self.per_scene


def compute_detection_class_stats(config: DetectionClassStatsConfig) -> DetectionClassStats:
    """统计一个或多个 H5 的检测类别分布。

    Agent 与 Map 使用各自 `*_valid` mask 过滤 padding 查询；Traffic Light 与
    Stop Sign 默认只统计有效标签，开启 `include_invalid_traffic` 时会把无效样本
    的默认 `none` 状态也计入分布。
    """

    h5_paths = _resolve_h5_paths(config.h5_paths)
    if not h5_paths:
        raise FileNotFoundError(f"未找到 H5 文件：{config.h5_paths!r}")

    stats = DetectionClassStats(h5_paths=[str(path) for path in h5_paths])
    h5py = _require_h5py()
    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as h5_file:
            scene_stats = _compute_single_h5_stats(
                h5_file,
                h5_path,
                batch_size=config.batch_size,
                include_invalid_traffic=config.include_invalid_traffic,
            )
        _merge_scene_stats(stats, scene_stats)
    return stats


def save_detection_class_stats(stats: DetectionClassStats, output_path: str | Path) -> Path:
    """将统计结果保存为 JSON。"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _stats_to_payload(stats)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def format_detection_class_stats(stats: DetectionClassStats) -> str:
    """将统计结果格式化为命令行文本表格。"""

    lines = [
        "H5 检测类别分布统计",
        f"scenes={stats.scene_count}, samples={stats.sample_count}",
        (
            "valid_agents="
            f"{stats.valid_agent_count}/{stats.agent_slot_count}, "
            f"valid_maps={stats.valid_map_count}/{stats.map_slot_count}, "
            f"valid_traffic_lights={stats.valid_traffic_light_count}, "
            f"valid_stop_signs={stats.valid_stop_sign_count}"
        ),
        "",
        _format_counter_table("Agent", stats.agent_counts or Counter()),
        "",
        _format_counter_table("Map", stats.map_counts or Counter()),
        "",
        _format_counter_table("Traffic Light", stats.traffic_light_counts or Counter()),
        "",
        _format_counter_table("Stop Sign", stats.stop_sign_counts or Counter()),
    ]
    return "\n".join(lines)


def _compute_single_h5_stats(
    h5_file: Any,
    h5_path: Path,
    batch_size: int,
    include_invalid_traffic: bool,
) -> DetectionClassStats:
    labels = h5_file["labels"]
    sample_count = int(h5_file["samples/current_frame_id"].shape[0])
    scene_name = _decode_h5_attr(h5_file.attrs.get("scene_name", h5_path.stem))
    scene_stats = DetectionClassStats(h5_paths=[str(h5_path)], scene_count=1, sample_count=sample_count)

    _require_label(labels, "agent_classes", h5_path)
    _require_label(labels, "agent_valid", h5_path)
    _require_label(labels, "map_classes", h5_path)
    _require_label(labels, "map_valid", h5_path)
    _require_label(labels, "traffic_light_state", h5_path)
    _require_label(labels, "traffic_light_valid", h5_path)
    _require_label(labels, "stop_sign_state", h5_path)
    _require_label(labels, "stop_sign_valid", h5_path)

    for start_index in range(0, sample_count, batch_size):
        end_index = min(start_index + batch_size, sample_count)
        _accumulate_query_distribution(
            np.asarray(labels["agent_classes"][start_index:end_index]),
            np.asarray(labels["agent_valid"][start_index:end_index], dtype=np.bool_),
            AGENT_CLASS_NAMES,
            scene_stats.agent_counts,
        )
        _accumulate_query_distribution(
            np.asarray(labels["map_classes"][start_index:end_index]),
            np.asarray(labels["map_valid"][start_index:end_index], dtype=np.bool_),
            MAP_CLASS_NAMES,
            scene_stats.map_counts,
        )
        _accumulate_state_distribution(
            np.asarray(labels["traffic_light_state"][start_index:end_index]),
            np.asarray(labels["traffic_light_valid"][start_index:end_index], dtype=np.bool_),
            TRAFFIC_LIGHT_CLASS_NAMES,
            scene_stats.traffic_light_counts,
            include_invalid=include_invalid_traffic,
        )
        _accumulate_state_distribution(
            np.asarray(labels["stop_sign_state"][start_index:end_index]),
            np.asarray(labels["stop_sign_valid"][start_index:end_index], dtype=np.bool_),
            STOP_SIGN_CLASS_NAMES,
            scene_stats.stop_sign_counts,
            include_invalid=include_invalid_traffic,
        )

    agent_valid = np.asarray(labels["agent_valid"], dtype=np.bool_)
    map_valid = np.asarray(labels["map_valid"], dtype=np.bool_)
    traffic_light_valid = np.asarray(labels["traffic_light_valid"], dtype=np.bool_)
    stop_sign_valid = np.asarray(labels["stop_sign_valid"], dtype=np.bool_)
    scene_stats.agent_slot_count = int(agent_valid.size)
    scene_stats.map_slot_count = int(map_valid.size)
    scene_stats.valid_agent_count = int(agent_valid.sum())
    scene_stats.valid_map_count = int(map_valid.sum())
    scene_stats.valid_traffic_light_count = int(traffic_light_valid.sum())
    scene_stats.valid_stop_sign_count = int(stop_sign_valid.sum())
    scene_stats.per_scene.append(_scene_payload(scene_name, h5_path, scene_stats))
    return scene_stats


def _merge_scene_stats(stats: DetectionClassStats, scene_stats: DetectionClassStats) -> None:
    stats.scene_count += scene_stats.scene_count
    stats.sample_count += scene_stats.sample_count
    stats.agent_slot_count += scene_stats.agent_slot_count
    stats.map_slot_count += scene_stats.map_slot_count
    stats.valid_agent_count += scene_stats.valid_agent_count
    stats.valid_map_count += scene_stats.valid_map_count
    stats.valid_traffic_light_count += scene_stats.valid_traffic_light_count
    stats.valid_stop_sign_count += scene_stats.valid_stop_sign_count
    stats.agent_counts.update(scene_stats.agent_counts or Counter())
    stats.map_counts.update(scene_stats.map_counts or Counter())
    stats.traffic_light_counts.update(scene_stats.traffic_light_counts or Counter())
    stats.stop_sign_counts.update(scene_stats.stop_sign_counts or Counter())
    stats.per_scene.extend(scene_stats.per_scene or [])


def _accumulate_query_distribution(
    class_array: np.ndarray,
    valid_array: np.ndarray,
    class_names: Mapping[int, str],
    counter: Counter[str],
) -> None:
    if class_array.shape != valid_array.shape:
        raise ValueError(
            f"class_array 与 valid_array shape 必须一致，实际为 {class_array.shape}/{valid_array.shape}。"
        )
    _accumulate_values(class_array[valid_array], class_names, counter)


def _accumulate_state_distribution(
    state_array: np.ndarray,
    valid_array: np.ndarray,
    class_names: Mapping[int, str],
    counter: Counter[str],
    include_invalid: bool,
) -> None:
    if state_array.shape != valid_array.shape:
        raise ValueError(
            f"state_array 与 valid_array shape 必须一致，实际为 {state_array.shape}/{valid_array.shape}。"
        )
    values = state_array if include_invalid else state_array[valid_array]
    _accumulate_values(values, class_names, counter)


def _accumulate_values(
    values: np.ndarray,
    class_names: Mapping[int, str],
    counter: Counter[str],
) -> None:
    if values.size == 0:
        return
    unique_values, counts = np.unique(values.astype(np.int64, copy=False), return_counts=True)
    for class_id, count in zip(unique_values, counts, strict=True):
        counter[_class_name(int(class_id), class_names)] += int(count)


def _class_name(class_id: int, class_names: Mapping[int, str]) -> str:
    return class_names.get(class_id, f"unknown_{class_id}")


def _counter_to_payload(counter: Counter[str]) -> dict[str, Any]:
    total = int(sum(counter.values()))
    return {
        class_name: {
            "count": int(count),
            "ratio": (float(count) / total if total > 0 else 0.0),
        }
        for class_name, count in sorted(counter.items())
    }


def _scene_payload(scene_name: str, h5_path: Path, scene_stats: DetectionClassStats) -> dict[str, Any]:
    return {
        "scene_name": scene_name,
        "h5_path": str(h5_path),
        "sample_count": scene_stats.sample_count,
        "valid_agent_count": scene_stats.valid_agent_count,
        "valid_map_count": scene_stats.valid_map_count,
        "valid_traffic_light_count": scene_stats.valid_traffic_light_count,
        "valid_stop_sign_count": scene_stats.valid_stop_sign_count,
        "agent_counts": dict(scene_stats.agent_counts or Counter()),
        "map_counts": dict(scene_stats.map_counts or Counter()),
        "traffic_light_counts": dict(scene_stats.traffic_light_counts or Counter()),
        "stop_sign_counts": dict(scene_stats.stop_sign_counts or Counter()),
    }


def _stats_to_payload(stats: DetectionClassStats) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "h5_paths": stats.h5_paths,
        "scene_count": stats.scene_count,
        "sample_count": stats.sample_count,
        "agent_slot_count": stats.agent_slot_count,
        "map_slot_count": stats.map_slot_count,
        "valid_agent_count": stats.valid_agent_count,
        "valid_map_count": stats.valid_map_count,
        "valid_traffic_light_count": stats.valid_traffic_light_count,
        "valid_stop_sign_count": stats.valid_stop_sign_count,
        "agent_counts": _counter_to_payload(stats.agent_counts or Counter()),
        "map_counts": _counter_to_payload(stats.map_counts or Counter()),
        "traffic_light_counts": _counter_to_payload(stats.traffic_light_counts or Counter()),
        "stop_sign_counts": _counter_to_payload(stats.stop_sign_counts or Counter()),
        "per_scene": stats.per_scene,
    }


def _format_counter_table(title: str, counter: Counter[str]) -> str:
    total = int(sum(counter.values()))
    lines = [f"{title} distribution", f"{'class':<18} {'count':>12} {'ratio':>10}"]
    if total == 0:
        lines.append(f"{'(empty)':<18} {0:>12} {0.0:>9.2%}")
        return "\n".join(lines)
    for class_name, count in sorted(counter.items()):
        ratio = float(count) / total
        lines.append(f"{class_name:<18} {int(count):>12} {ratio:>9.2%}")
    return "\n".join(lines)


def _resolve_h5_paths(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    raw_paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    resolved_paths: list[Path] = []
    for raw_path in raw_paths:
        path = Path(raw_path)
        if path.is_dir():
            resolved_paths.extend(sorted(path.rglob("*.h5")))
        else:
            resolved_paths.append(path)

    unique_paths = sorted(dict.fromkeys(path.resolve() for path in resolved_paths))
    for h5_path in unique_paths:
        if not h5_path.exists():
            raise FileNotFoundError(f"H5 文件不存在：{h5_path}")
        if not h5_path.is_file():
            raise FileNotFoundError(f"H5 路径不是文件：{h5_path}")
        if h5_path.suffix.lower() != ".h5":
            raise ValueError(f"H5 文件扩展名必须为 .h5，实际为：{h5_path}")
    return unique_paths


def _require_label(labels: Any, dataset_name: str, h5_path: Path) -> None:
    if dataset_name not in labels:
        raise KeyError(f"H5 缺少 labels/{dataset_name}：{h5_path}")


def _decode_h5_attr(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _require_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "H5 检测类别统计需要 h5py。请先在项目环境中安装 h5py，例如："
            ".\\.venv\\Scripts\\python.exe -m pip install h5py"
        ) from exc
    return h5py


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="统计一个或多个 B2D H5 的检测类别分布。")
    parser.add_argument(
        "--h5",
        type=Path,
        action="append",
        required=True,
        help="H5 文件或目录；可重复传入。目录会递归扫描 *.h5，用于统计整个数据集。",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="可选 JSON 输出路径。")
    parser.add_argument("--batch-size", type=int, default=4096, help="按样本维分块读取大小。")
    parser.add_argument(
        "--include-invalid-traffic",
        action="store_true",
        help="把无效 Traffic Light / Stop Sign 样本也计入默认 none 状态。",
    )
    args = parser.parse_args(argv)

    config = DetectionClassStatsConfig(
        h5_paths=args.h5,
        output_json=args.output_json,
        batch_size=args.batch_size,
        include_invalid_traffic=args.include_invalid_traffic,
    )
    stats = compute_detection_class_stats(config)
    print(format_detection_class_stats(stats))
    if args.output_json is not None:
        output_path = save_detection_class_stats(stats, args.output_json)
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
