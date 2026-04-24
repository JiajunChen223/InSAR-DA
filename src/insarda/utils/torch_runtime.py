from __future__ import annotations

import os
from contextlib import nullcontext

import torch

from insarda.config import TrainingConfig


def configure_torch_runtime(training_cfg: TrainingConfig, device: torch.device) -> None:
    torch.set_float32_matmul_precision("high")
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(training_cfg.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(training_cfg.allow_tf32)
    override = os.getenv("INSARDA_CUDNN_BENCHMARK", "").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        torch.backends.cudnn.benchmark = True
    elif override in {"0", "false", "no", "off"}:
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = bool(training_cfg.cudnn_benchmark)


def autocast_context(training_cfg: TrainingConfig, device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    precision = str(training_cfg.mixed_precision).strip().lower()
    if precision == "off":
        return nullcontext()
    if precision == "bf16":
        # Older GPUs such as RTX 20-series do not support CUDA bf16 autocast.
        # Fall back to fp16 so the same formal config can still run there.
        if not torch.cuda.is_bf16_supported():
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"Unsupported mixed precision mode: {training_cfg.mixed_precision}")


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    *,
    keys: tuple[str, ...] | None = None,
    non_blocking: bool = False,
) -> dict[str, torch.Tensor]:
    selected = None if keys is None else set(keys)
    return {
        key: value.to(device, non_blocking=non_blocking) if selected is None or key in selected else value
        for key, value in batch.items()
    }
