from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from insarda.config import TrainingConfig
from insarda.evaluation.metrics import regression_report
from insarda.utils.torch_runtime import autocast_context, move_batch_to_device


_PREDICTION_KEYS = ("x", "domain_id")
_METRIC_KEYS = ("x", "y", "y_mask", "domain_id")


def _overall_metrics_from_totals(
    *,
    sse: float,
    count: float,
    sum_y: float,
    sum_y2: float,
    min_y: float | None,
    max_y: float | None,
) -> dict[str, float]:
    if count <= 0.0:
        nan = float("nan")
        return {"rmse": nan, "r2": nan, "nrmse": nan}
    mse = float(sse / count)
    rmse = float(np.sqrt(mse))
    mean_y = float(sum_y / count)
    ss_tot = float(sum_y2 - (sum_y * sum_y) / count)
    r2 = float("nan") if ss_tot <= 1e-12 else float(1.0 - sse / ss_tot)
    variance = max(float(sum_y2 / count) - mean_y * mean_y, 0.0)
    scale = float(np.sqrt(variance))
    if scale <= 1e-12 and min_y is not None and max_y is not None:
        scale = float(max_y - min_y)
    nrmse = float("nan") if scale <= 1e-12 else float(rmse / scale)
    return {"rmse": rmse, "r2": r2, "nrmse": nrmse}


def _per_horizon_metrics_from_totals(
    *,
    sse: np.ndarray | None,
    count: np.ndarray | None,
    sum_y: np.ndarray | None,
    sum_y2: np.ndarray | None,
    min_y: list[float | None] | None,
    max_y: list[float | None] | None,
) -> list[dict[str, float | int]]:
    if sse is None or count is None or sum_y is None or sum_y2 is None:
        return []
    metrics: list[dict[str, float | int]] = []
    for index in range(int(sse.shape[0])):
        overall = _overall_metrics_from_totals(
            sse=float(sse[index]),
            count=float(count[index]),
            sum_y=float(sum_y[index]),
            sum_y2=float(sum_y2[index]),
            min_y=None if min_y is None else min_y[index],
            max_y=None if max_y is None else max_y[index],
        )
        metrics.append(
            {
                "horizon": int(index + 1),
                "rmse": overall["rmse"],
                "r2": overall["r2"],
                "nrmse": overall["nrmse"],
                "n": int(round(float(count[index]))),
            }
        )
    return metrics


@torch.inference_mode()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    method=None,
    training_cfg: TrainingConfig | None = None,
) -> dict:
    model.eval()
    predictions = []
    targets = []
    masks = []
    domains = []
    non_blocking = bool(device.type == "cuda")
    for batch in loader:
        moved = move_batch_to_device(batch, device, keys=_PREDICTION_KEYS, non_blocking=non_blocking)
        ctx = autocast_context(training_cfg, device) if training_cfg is not None else nullcontext()
        with ctx:
            prediction = method.predict_batch(model, moved) if method is not None else model(moved["x"])
        predictions.append(prediction.detach().float().cpu().numpy())
        targets.append(batch["y"].detach().cpu().numpy())
        masks.append(batch["y_mask"].detach().cpu().numpy())
        domains.append(batch["domain_id"].detach().cpu().numpy())
    y_pred = np.concatenate(predictions, axis=0)
    y_true = np.concatenate(targets, axis=0)
    y_mask = np.concatenate(masks, axis=0)
    domain_id = np.concatenate(domains, axis=0)
    return {
        "y_pred": y_pred,
        "y_true": y_true,
        "y_mask": y_mask,
        "domain_id": domain_id,
        "metrics": regression_report(y_true=y_true, y_pred=y_pred, domain_id=domain_id, mask=y_mask),
    }


@torch.inference_mode()
def evaluate_loader_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    method=None,
    training_cfg: TrainingConfig | None = None,
) -> dict:
    model.eval()
    sse = 0.0
    count = 0.0
    sum_y = 0.0
    sum_y2 = 0.0
    min_y: float | None = None
    max_y: float | None = None
    horizon_sse: np.ndarray | None = None
    horizon_count: np.ndarray | None = None
    horizon_sum_y: np.ndarray | None = None
    horizon_sum_y2: np.ndarray | None = None
    horizon_min: list[float | None] | None = None
    horizon_max: list[float | None] | None = None
    by_domain_totals: dict[int, dict[str, float | None]] = {}
    non_blocking = bool(device.type == "cuda")

    for batch in loader:
        moved = move_batch_to_device(batch, device, keys=_METRIC_KEYS, non_blocking=non_blocking)
        ctx = autocast_context(training_cfg, device) if training_cfg is not None else nullcontext()
        with ctx:
            prediction = method.predict_batch(model, moved) if method is not None else model(moved["x"])
        target = moved["y"].detach().float()
        mask = moved["y_mask"].detach().float()
        prediction = prediction.detach().float()
        domain_id = moved["domain_id"].detach().long()
        diff = prediction - target
        sse += float(((diff * diff) * mask).sum().detach().item())
        count += float(mask.sum().detach().item())
        sum_y += float((target * mask).sum().detach().item())
        sum_y2 += float(((target * target) * mask).sum().detach().item())
        if horizon_sse is None:
            horizon = int(target.shape[1])
            horizon_sse = np.zeros((horizon,), dtype=np.float64)
            horizon_count = np.zeros((horizon,), dtype=np.float64)
            horizon_sum_y = np.zeros((horizon,), dtype=np.float64)
            horizon_sum_y2 = np.zeros((horizon,), dtype=np.float64)
            horizon_min = [None] * horizon
            horizon_max = [None] * horizon
        horizon_sse += (((diff * diff) * mask).sum(dim=0).detach().cpu().numpy()).astype(np.float64)
        horizon_count += (mask.sum(dim=0).detach().cpu().numpy()).astype(np.float64)
        horizon_sum_y += (((target * mask).sum(dim=0)).detach().cpu().numpy()).astype(np.float64)
        horizon_sum_y2 += ((((target * target) * mask).sum(dim=0)).detach().cpu().numpy()).astype(np.float64)
        for horizon_index in range(int(target.shape[1])):
            valid_horizon = mask[:, horizon_index] > 0
            if torch.any(valid_horizon):
                valid_values = target[valid_horizon, horizon_index]
                batch_min = float(valid_values.min().detach().item())
                batch_max = float(valid_values.max().detach().item())
                horizon_min[horizon_index] = (
                    batch_min if horizon_min[horizon_index] is None else min(horizon_min[horizon_index], batch_min)
                )
                horizon_max[horizon_index] = (
                    batch_max if horizon_max[horizon_index] is None else max(horizon_max[horizon_index], batch_max)
                )
        valid = mask > 0
        if torch.any(valid):
            valid_values = target[valid]
            batch_min = float(valid_values.min().detach().item())
            batch_max = float(valid_values.max().detach().item())
            min_y = batch_min if min_y is None else min(min_y, batch_min)
            max_y = batch_max if max_y is None else max(max_y, batch_max)
        for domain_value in domain_id.unique(sorted=True).tolist():
            domain_key = int(domain_value)
            selector = domain_id == int(domain_value)
            domain_target = target[selector]
            domain_prediction = prediction[selector]
            domain_mask = mask[selector]
            domain_diff = domain_prediction - domain_target
            stats = by_domain_totals.setdefault(
                domain_key,
                {
                    "sse": 0.0,
                    "count": 0.0,
                    "sum_y": 0.0,
                    "sum_y2": 0.0,
                    "min_y": None,
                    "max_y": None,
                },
            )
            stats["sse"] = float(stats["sse"]) + float(((domain_diff * domain_diff) * domain_mask).sum().detach().item())
            stats["count"] = float(stats["count"]) + float(domain_mask.sum().detach().item())
            stats["sum_y"] = float(stats["sum_y"]) + float((domain_target * domain_mask).sum().detach().item())
            stats["sum_y2"] = float(stats["sum_y2"]) + float(
                ((domain_target * domain_target) * domain_mask).sum().detach().item()
            )
            valid_domain = domain_mask > 0
            if torch.any(valid_domain):
                valid_domain_values = domain_target[valid_domain]
                batch_domain_min = float(valid_domain_values.min().detach().item())
                batch_domain_max = float(valid_domain_values.max().detach().item())
                current_min = stats["min_y"]
                current_max = stats["max_y"]
                stats["min_y"] = (
                    batch_domain_min if current_min is None else min(float(current_min), batch_domain_min)
                )
                stats["max_y"] = (
                    batch_domain_max if current_max is None else max(float(current_max), batch_domain_max)
                )

    by_domain = {}
    for domain_key in sorted(by_domain_totals):
        stats = by_domain_totals[domain_key]
        by_domain[str(int(domain_key))] = {
            **_overall_metrics_from_totals(
                sse=float(stats["sse"]),
                count=float(stats["count"]),
                sum_y=float(stats["sum_y"]),
                sum_y2=float(stats["sum_y2"]),
                min_y=None if stats["min_y"] is None else float(stats["min_y"]),
                max_y=None if stats["max_y"] is None else float(stats["max_y"]),
            ),
            "n": int(round(float(stats["count"]))),
        }

    return {
        "metrics": {
            "overall": _overall_metrics_from_totals(
                sse=sse,
                count=count,
                sum_y=sum_y,
                sum_y2=sum_y2,
                min_y=min_y,
                max_y=max_y,
            ),
            "per_horizon": _per_horizon_metrics_from_totals(
                sse=horizon_sse,
                count=horizon_count,
                sum_y=horizon_sum_y,
                sum_y2=horizon_sum_y2,
                min_y=horizon_min,
                max_y=horizon_max,
            ),
            "by_domain": by_domain,
        }
    }


@torch.inference_mode()
def evaluate_loader_rmse(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    method=None,
    training_cfg: TrainingConfig | None = None,
) -> float:
    model.eval()
    sse = 0.0
    count = 0.0
    non_blocking = bool(device.type == "cuda")
    for batch in loader:
        moved = move_batch_to_device(batch, device, keys=_METRIC_KEYS, non_blocking=non_blocking)
        ctx = autocast_context(training_cfg, device) if training_cfg is not None else nullcontext()
        with ctx:
            prediction = method.predict_batch(model, moved) if method is not None else model(moved["x"])
        diff = prediction.detach().float() - moved["y"].detach().float()
        mask = moved["y_mask"].detach().float()
        sse += float(((diff * diff) * mask).sum().detach().item())
        count += float(mask.sum().detach().item())
    if count <= 0.0:
        return float("nan")
    return float(np.sqrt(sse / count))


def save_predictions(path: str | Path, evaluation: dict) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        y_pred=np.asarray(evaluation["y_pred"], dtype=np.float32),
        y_true=np.asarray(evaluation["y_true"], dtype=np.float32),
        y_mask=np.asarray(evaluation["y_mask"], dtype=np.float32),
        domain_id=np.asarray(evaluation["domain_id"], dtype=np.int64),
    )
    return target
