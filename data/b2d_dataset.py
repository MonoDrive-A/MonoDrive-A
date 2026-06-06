"""读取 B2D 预处理 H5 的 PyTorch Dataset。"""

from __future__ import annotations

import bisect
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


SUPPORTED_SCHEMA_VERSIONS = {"b2d_h5_v5"}


class B2DH5Dataset(Dataset[dict[str, Any]]):
    """读取由 `data.b2d_preprocess` 生成的逐场景 H5。

    Args:
        h5_paths: 单个 H5 文件、H5 目录，或 H5 文件列表。
        normalize_images: 是否将 uint8 图像除以 255 并转为浮点。
        image_dtype: 图像返回 dtype；`normalize_images=True` 时通常为 `torch.float32`。

        Shape:
        `images`: `[T, 3, H, W]`，其中 `T=8`。
        `future_trajectory`: `[6, 2]`，ego 坐标系米制轨迹。
        `agent_future_trajectory`: `[194, 6, 2]`，Agent 原点、当前 ego 坐标轴下的未来位移。
        `map_points`: `[60, 100, 2]`，当前 ego 坐标系局部 Map 元素。
        `ego_motion`: `[3]`，`[Vx, Vy, W]`。
    """

    def __init__(
        self,
        h5_paths: str | Path | list[str | Path] | tuple[str | Path, ...],
        normalize_images: bool = True,
        image_dtype: torch.dtype = torch.float32,
        random_target_point: bool = True,
    ) -> None:
        self.h5_paths = _resolve_h5_paths(h5_paths)
        if not self.h5_paths:
            raise FileNotFoundError(f"未找到 H5 文件：{h5_paths!r}")
        self.normalize_images = normalize_images
        self.image_dtype = image_dtype
        self.random_target_point = random_target_point
        self._h5py = _require_h5py()
        self._handles: dict[int, Any] = {}
        self._dataset_cache: dict[int, dict[str, Any]] = {}
        self._lengths: list[int] = []
        self._scene_names: list[str] = []

        for h5_path in self.h5_paths:
            with self._h5py.File(h5_path, "r") as h5_file:
                schema_version = _decode_h5_attr(h5_file.attrs.get("schema_version", "unknown"))
                if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
                    raise ValueError(
                        f"H5 schema_version 不受支持：{schema_version!r}，"
                        f"期望为 {sorted(SUPPORTED_SCHEMA_VERSIONS)}。请重新运行预处理生成 H5 v5。"
                    )
                sample_count = int(h5_file["samples/current_frame_id"].shape[0])
                if sample_count <= 0:
                    raise ValueError(f"H5 文件没有样本：{h5_path}")
                self._lengths.append(sample_count)
                self._scene_names.append(str(h5_file.attrs.get("scene_name", h5_path.stem)))

        self._cumulative_lengths = np.cumsum(np.asarray(self._lengths, dtype=np.int64)).tolist()

    def __len__(self) -> int:
        return int(self._cumulative_lengths[-1])

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"index 超出范围：{index}，数据集长度为 {len(self)}。")

        scene_index, local_index = self._resolve_index(index)
        datasets = self._get_datasets(scene_index)
        input_frame_indices = np.asarray(
            datasets["input_frame_indices"][local_index],
            dtype=np.int64,
        )
        image_array = np.asarray(datasets["rgb_front"][input_frame_indices])
        # [T, H, W, 3] -> [T, 3, H, W]
        image_tensor = torch.from_numpy(image_array.transpose(0, 3, 1, 2).copy())
        if self.normalize_images:
            image_tensor = image_tensor.to(dtype=self.image_dtype).div_(255.0)
        else:
            image_tensor = image_tensor.to(dtype=self.image_dtype)

        target_points = _tensor_from_h5(datasets["target_points"], local_index, torch.float32)
        target_valid = _tensor_from_h5(datasets["target_valid"], local_index, torch.bool)
        target_point, target_point_index = _sample_target_point(
            target_points,
            target_valid,
            random_target_point=self.random_target_point,
        )

        sample = {
            "images": image_tensor,
            "ego_motion": _tensor_from_h5(datasets["ego_motion"], local_index, torch.float32),
            "future_trajectory": _tensor_from_h5(
                datasets["future_trajectory"],
                local_index,
                torch.float32,
            ),
            "target_point": target_point,
            "target_points": target_points,
            "target_valid": target_valid,
            "target_point_index": target_point_index,
            "commands": _tensor_from_h5(datasets["commands"], local_index, torch.long),
            "control": _tensor_from_h5(datasets["control"], local_index, torch.float32),
            "current_pose": _tensor_from_h5(datasets["current_pose"], local_index, torch.float32),
            "agent_boxes": _tensor_from_h5(datasets["agent_boxes"], local_index, torch.float32),
            "agent_classes": _tensor_from_h5(datasets["agent_classes"], local_index, torch.long),
            "agent_valid": _tensor_from_h5(datasets["agent_valid"], local_index, torch.bool),
            "agent_future_trajectory": _tensor_from_h5(
                datasets["agent_future_trajectory"],
                local_index,
                torch.float32,
            ),
            "agent_future_valid": _tensor_from_h5(
                datasets["agent_future_valid"],
                local_index,
                torch.bool,
            ),
            "map_points": _tensor_from_h5(datasets["map_points"], local_index, torch.float32),
            "map_classes": _tensor_from_h5(datasets["map_classes"], local_index, torch.long),
            "map_valid": _tensor_from_h5(datasets["map_valid"], local_index, torch.bool),
            "traffic_light_state": _tensor_from_h5(
                datasets["traffic_light_state"],
                local_index,
                torch.long,
            ),
            "traffic_light_xy": _tensor_from_h5(
                datasets["traffic_light_xy"],
                local_index,
                torch.float32,
            ),
            "traffic_light_valid": _tensor_from_h5(
                datasets["traffic_light_valid"],
                local_index,
                torch.bool,
            ),
            "stop_sign_state": _tensor_from_h5(datasets["stop_sign_state"], local_index, torch.long),
            "stop_sign_xy": _tensor_from_h5(datasets["stop_sign_xy"], local_index, torch.float32),
            "stop_sign_valid": _tensor_from_h5(datasets["stop_sign_valid"], local_index, torch.bool),
            "input_frame_ids": _tensor_from_h5(
                datasets["input_frame_ids"],
                local_index,
                torch.long,
            ),
            "future_frame_ids": _tensor_from_h5(
                datasets["future_frame_ids"],
                local_index,
                torch.long,
            ),
            "current_frame_id": int(datasets["current_frame_id"][local_index]),
            "scene_name": self._scene_names[scene_index],
            "h5_path": str(self.h5_paths[scene_index]),
        }
        return sample

    def close(self) -> None:
        """关闭当前进程内缓存的 H5 文件句柄。"""

        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        self._dataset_cache.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_dataset_cache"] = {}
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _resolve_index(self, index: int) -> tuple[int, int]:
        scene_index = bisect.bisect_right(self._cumulative_lengths, index)
        previous_total = 0 if scene_index == 0 else self._cumulative_lengths[scene_index - 1]
        return scene_index, index - previous_total

    def _get_handle(self, scene_index: int) -> Any:
        handle = self._handles.get(scene_index)
        if handle is None:
            handle = self._h5py.File(self.h5_paths[scene_index], "r")
            self._handles[scene_index] = handle
        return handle

    def _get_datasets(self, scene_index: int) -> dict[str, Any]:
        datasets = self._dataset_cache.get(scene_index)
        if datasets is not None:
            return datasets

        h5_file = self._get_handle(scene_index)
        frames_group = h5_file["frames"]
        samples_group = h5_file["samples"]
        labels_group = h5_file["labels"]
        datasets = {
            "rgb_front": frames_group["rgb_front"],
            "current_frame_id": samples_group["current_frame_id"],
            "input_frame_indices": samples_group["input_frame_indices"],
            "input_frame_ids": samples_group["input_frame_ids"],
            "future_frame_ids": samples_group["future_frame_ids"],
            "current_pose": labels_group["current_pose"],
            "ego_motion": labels_group["ego_motion"],
            "target_points": labels_group["target_points"],
            "target_valid": labels_group["target_valid"],
            "commands": labels_group["commands"],
            "control": labels_group["control"],
            "future_trajectory": labels_group["future_trajectory"],
            "agent_boxes": labels_group["agent_boxes"],
            "agent_classes": labels_group["agent_classes"],
            "agent_valid": labels_group["agent_valid"],
            "agent_future_trajectory": labels_group["agent_future_trajectory"],
            "agent_future_valid": labels_group["agent_future_valid"],
            "map_points": labels_group["map_points"],
            "map_classes": labels_group["map_classes"],
            "map_valid": labels_group["map_valid"],
            "traffic_light_state": labels_group["traffic_light_state"],
            "traffic_light_xy": labels_group["traffic_light_xy"],
            "traffic_light_valid": labels_group["traffic_light_valid"],
            "stop_sign_state": labels_group["stop_sign_state"],
            "stop_sign_xy": labels_group["stop_sign_xy"],
            "stop_sign_valid": labels_group["stop_sign_valid"],
        }
        self._dataset_cache[scene_index] = datasets
        return datasets


def _resolve_h5_paths(paths: str | Path | list[str | Path] | tuple[str | Path, ...]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        path = Path(paths)
        if path.is_dir():
            return sorted(path.glob("*.h5"))
        return [path]
    resolved_paths = [Path(path) for path in paths]
    return sorted(resolved_paths)


def _tensor_from_h5(dataset: Any, index: int, dtype: torch.dtype) -> torch.Tensor:
    array = np.asarray(dataset[index])
    return torch.as_tensor(array, dtype=dtype)


def _decode_h5_attr(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _sample_target_point(
    target_points: torch.Tensor,
    target_valid: torch.Tensor,
    random_target_point: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_indices = torch.nonzero(target_valid, as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        return target_points[0], torch.tensor(0, dtype=torch.long)
    if random_target_point and valid_indices.numel() > 1:
        random_index = torch.randint(valid_indices.numel(), (1,), dtype=torch.long).item()
        target_index = valid_indices[random_index]
    else:
        target_index = valid_indices[0]
    return target_points[target_index], target_index.to(dtype=torch.long)


def _require_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "B2DH5Dataset 需要 h5py。请先在项目环境中安装 h5py，例如："
            ".\\.venv\\Scripts\\python.exe -m pip install h5py"
        ) from exc
    return h5py
