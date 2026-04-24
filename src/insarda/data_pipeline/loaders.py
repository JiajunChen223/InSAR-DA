from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

from insarda.data_pipeline.windows import WindowBundle


class WindowDataset(Dataset):
    def __init__(self, bundle: WindowBundle) -> None:
        self.x = torch.from_numpy(bundle.x)
        self.y = torch.from_numpy(bundle.y)
        self.y_mask = torch.from_numpy(bundle.y_mask)
        self.target_start_idx = torch.from_numpy(bundle.target_start_idx.astype("int64", copy=False))
        self.target_end_idx = torch.from_numpy(bundle.target_end_idx.astype("int64", copy=False))
        self.domain_id = torch.from_numpy(bundle.domain_id.astype("int64", copy=False))
        self.point_id = torch.from_numpy(bundle.point_id.astype("int64", copy=False))
        self.length = int(self.x.shape[0])

    def __len__(self) -> int:
        return self.length

    def _slice_tensors(self, index: int | slice | list[int] | torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "x": self.x[index],
            "y": self.y[index],
            "y_mask": self.y_mask[index],
            "target_start_idx": self.target_start_idx[index],
            "target_end_idx": self.target_end_idx[index],
            "domain_id": self.domain_id[index],
            "point_id": self.point_id[index],
        }

    def __getitem__(self, index: int | slice | list[int] | torch.Tensor) -> dict[str, torch.Tensor]:
        return self._slice_tensors(index)

    def __getitems__(self, indices: list[int]) -> dict[str, torch.Tensor]:
        return self._slice_tensors(indices)


@dataclass
class LoaderBundle:
    source_train: DataLoader
    source_val: DataLoader
    target_labeled: DataLoader | None
    target_unlabeled: DataLoader | None
    target_val: DataLoader
    target_test: DataLoader


def _identity_collate(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return batch


def _recommended_num_workers() -> int:
    override = os.getenv("INSARDA_NUM_WORKERS", "").strip()
    if override:
        try:
            return max(int(override), 0)
        except ValueError:
            pass
    # Default to single-process loading. These formal datasets are already
    # materialized in host memory, and long cloud sweeps proved more stable
    # without background loader workers.
    return 0


def _recommended_pin_memory() -> bool:
    override = os.getenv("INSARDA_PIN_MEMORY", "").strip().lower()
    if override:
        if override in {"1", "true", "yes", "on"}:
            return True
        if override in {"0", "false", "no", "off"}:
            return False
    # Keep the default conservative. On some Linux cloud instances, combining
    # pin_memory with long-running formal sweeps caused the pin-memory thread
    # to exit mid-evaluation.
    return False


def _build_loader(
    bundle: WindowBundle,
    batch_size: int,
    shuffle: bool,
) -> DataLoader | None:
    if bundle.size == 0:
        return None
    num_workers = _recommended_num_workers()
    pin_memory = _recommended_pin_memory()
    loader_kwargs = {
        "batch_size": max(int(batch_size), 1),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "drop_last": False,
        "collate_fn": _identity_collate,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        # On Linux, keeping workers alive across epochs reduces the large
        # per-epoch stalls that show up as GPU utilization sawtooths. The
        # runner explicitly tears loaders down after each case, so we can
        # keep workers persistent within a case without leaking them across
        # a long sweep. On Windows we keep num_workers=0 above.
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    return DataLoader(
        WindowDataset(bundle),
        **loader_kwargs,
    )


def shutdown_loader(loader: DataLoader | None) -> None:
    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    if iterator is None:
        return
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
    try:
        loader._iterator = None
    except Exception:
        pass


def shutdown_loader_bundle(bundle: LoaderBundle) -> None:
    shutdown_loader(bundle.source_train)
    shutdown_loader(bundle.source_val)
    shutdown_loader(bundle.target_labeled)
    shutdown_loader(bundle.target_unlabeled)
    shutdown_loader(bundle.target_val)
    shutdown_loader(bundle.target_test)


def build_loader_bundle(
    source_train: WindowBundle,
    source_val: WindowBundle,
    target_labeled: WindowBundle,
    target_unlabeled: WindowBundle,
    target_val: WindowBundle,
    target_test: WindowBundle,
    batch_size: int,
    eval_batch_size: int,
) -> LoaderBundle:
    source_train_loader = _build_loader(source_train, batch_size=batch_size, shuffle=True)
    source_val_loader = _build_loader(source_val, batch_size=eval_batch_size, shuffle=False)
    target_val_loader = _build_loader(target_val, batch_size=eval_batch_size, shuffle=False)
    target_test_loader = _build_loader(target_test, batch_size=eval_batch_size, shuffle=False)
    if source_train_loader is None or source_val_loader is None or target_val_loader is None or target_test_loader is None:
        raise ValueError("source_train, source_val, target_val, and target_test must all contain data.")
    return LoaderBundle(
        source_train=source_train_loader,
        source_val=source_val_loader,
        target_labeled=_build_loader(target_labeled, batch_size=batch_size, shuffle=True),
        target_unlabeled=_build_loader(target_unlabeled, batch_size=batch_size, shuffle=True),
        target_val=target_val_loader,
        target_test=target_test_loader,
    )
