from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class WindowBundle:
    x: np.ndarray
    y: np.ndarray
    y_mask: np.ndarray
    target_start_idx: np.ndarray
    target_end_idx: np.ndarray
    domain_id: np.ndarray
    point_id: np.ndarray

    @property
    def size(self) -> int:
        return int(self.x.shape[0])


EMPTY_WINDOW_BUNDLE = WindowBundle(
    x=np.zeros((0, 1, 1), dtype=np.float32),
    y=np.zeros((0, 1), dtype=np.float32),
    y_mask=np.zeros((0, 1), dtype=np.float32),
    target_start_idx=np.zeros((0,), dtype=np.int64),
    target_end_idx=np.zeros((0,), dtype=np.int64),
    domain_id=np.zeros((0,), dtype=np.int64),
    point_id=np.zeros((0,), dtype=np.int64),
)


def build_windows(
    displacement: np.ndarray,
    domain_id: int,
    input_window: int,
    horizon: int,
) -> WindowBundle:
    values = np.asarray(displacement, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"`displacement` must have shape [N, T], got {values.shape}")
    num_points, total_time = values.shape
    required_observations = int(input_window) + int(horizon)
    if total_time < required_observations:
        raise ValueError(f"Sequence length {total_time} is too short for input_window={input_window}, horizon={horizon}.")

    x_list = []
    y_list = []
    mask_list = []
    target_start_list = []
    target_end_list = []
    domain_list = []
    point_list = []

    for point_index in range(num_points):
        point_values = values[point_index]
        valid_obs_idx = np.flatnonzero(np.isfinite(point_values))
        max_start = valid_obs_idx.size - required_observations + 1
        if max_start <= 0:
            continue
        observed_values = point_values[valid_obs_idx]
        for start_index in range(max_start):
            input_slice = slice(start_index, start_index + input_window)
            target_slice = slice(start_index + input_window, start_index + input_window + horizon)
            target_obs_idx = valid_obs_idx[target_slice]
            x = observed_values[input_slice].astype(np.float32)[:, None]
            y = observed_values[target_slice].astype(np.float32)
            x_list.append(x)
            y_list.append(y)
            mask_list.append(np.ones((int(horizon),), dtype=np.float32))
            target_start_list.append(int(target_obs_idx[0]))
            target_end_list.append(int(target_obs_idx[-1]))
            domain_list.append(domain_id)
            point_list.append(point_index)

    if not x_list:
        raise ValueError("No valid observation-step windows were generated.")

    return WindowBundle(
        x=np.stack(x_list).astype(np.float32),
        y=np.stack(y_list).astype(np.float32),
        y_mask=np.stack(mask_list).astype(np.float32),
        target_start_idx=np.asarray(target_start_list, dtype=np.int64),
        target_end_idx=np.asarray(target_end_list, dtype=np.int64),
        domain_id=np.asarray(domain_list, dtype=np.int64),
        point_id=np.asarray(point_list, dtype=np.int64),
    )


def slice_window_bundle(bundle: WindowBundle, indices: np.ndarray) -> WindowBundle:
    index_array = np.asarray(indices, dtype=np.int64)
    if index_array.size == 0:
        feature_dim = int(bundle.x.shape[-1])
        input_window = int(bundle.x.shape[1])
        horizon = int(bundle.y.shape[1])
        return WindowBundle(
            x=np.zeros((0, input_window, feature_dim), dtype=np.float32),
            y=np.zeros((0, horizon), dtype=np.float32),
            y_mask=np.zeros((0, horizon), dtype=np.float32),
            target_start_idx=np.zeros((0,), dtype=np.int64),
            target_end_idx=np.zeros((0,), dtype=np.int64),
            domain_id=np.zeros((0,), dtype=np.int64),
            point_id=np.zeros((0,), dtype=np.int64),
        )
    return WindowBundle(
        x=bundle.x[index_array],
        y=bundle.y[index_array],
        y_mask=bundle.y_mask[index_array],
        target_start_idx=bundle.target_start_idx[index_array],
        target_end_idx=bundle.target_end_idx[index_array],
        domain_id=bundle.domain_id[index_array],
        point_id=bundle.point_id[index_array],
    )


def concat_window_bundles(bundles: list[WindowBundle]) -> WindowBundle:
    valid = [bundle for bundle in bundles if bundle.size > 0]
    if not valid:
        return EMPTY_WINDOW_BUNDLE
    return WindowBundle(
        x=np.concatenate([bundle.x for bundle in valid], axis=0),
        y=np.concatenate([bundle.y for bundle in valid], axis=0),
        y_mask=np.concatenate([bundle.y_mask for bundle in valid], axis=0),
        target_start_idx=np.concatenate([bundle.target_start_idx for bundle in valid], axis=0),
        target_end_idx=np.concatenate([bundle.target_end_idx for bundle in valid], axis=0),
        domain_id=np.concatenate([bundle.domain_id for bundle in valid], axis=0),
        point_id=np.concatenate([bundle.point_id for bundle in valid], axis=0),
    )


def save_window_bundle(bundle: WindowBundle, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        x=bundle.x,
        y=bundle.y,
        y_mask=bundle.y_mask,
        target_start_idx=bundle.target_start_idx,
        target_end_idx=bundle.target_end_idx,
        domain_id=bundle.domain_id,
        point_id=bundle.point_id,
    )
    return target


def load_window_bundle(path: str | Path) -> WindowBundle:
    with np.load(path, allow_pickle=False) as data:
        return WindowBundle(
            x=np.asarray(data["x"], dtype=np.float32),
            y=np.asarray(data["y"], dtype=np.float32),
            y_mask=np.asarray(data["y_mask"], dtype=np.float32),
            target_start_idx=np.asarray(data["target_start_idx"], dtype=np.int64),
            target_end_idx=np.asarray(data["target_end_idx"], dtype=np.int64),
            domain_id=np.asarray(data["domain_id"], dtype=np.int64),
            point_id=np.asarray(data["point_id"], dtype=np.int64),
        )


def describe_window_bundle(bundle: WindowBundle) -> dict[str, Any]:
    return {
        "size": int(bundle.size),
        "input_window": int(bundle.x.shape[1]) if bundle.size > 0 else 0,
        "feature_dim": int(bundle.x.shape[2]) if bundle.size > 0 else 0,
        "horizon": int(bundle.y.shape[1]) if bundle.size > 0 else 0,
        "num_points": int(np.unique(bundle.point_id).size) if bundle.size > 0 else 0,
    }
