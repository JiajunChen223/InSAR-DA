from __future__ import annotations

import numpy as np


def _resolve_mask(y_true: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is None:
        return np.isfinite(y_true).astype(np.float32)
    return np.asarray(mask, dtype=np.float32)


def rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    weight = _resolve_mask(y_true, mask)
    denom = float(weight.sum())
    if denom <= 0:
        return float("nan")
    return float(np.sqrt((((y_true - y_pred) ** 2) * weight).sum() / denom))


def r2(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    weight = _resolve_mask(y_true, mask).astype(bool)
    yt = np.asarray(y_true, dtype=np.float64)[weight]
    yp = np.asarray(y_pred, dtype=np.float64)[weight]
    if yt.size < 2:
        return float("nan")
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def nrmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    weight = _resolve_mask(y_true, mask).astype(bool)
    yt = np.asarray(y_true, dtype=np.float64)[weight]
    yp = np.asarray(y_pred, dtype=np.float64)[weight]
    if yt.size == 0:
        return float("nan")
    scale = float(np.std(yt))
    if scale <= 1e-12:
        scale = float(np.max(yt) - np.min(yt))
    if scale <= 1e-12:
        return float("nan")
    return float(np.sqrt(np.mean((yt - yp) ** 2)) / scale)


def _per_horizon_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> list[dict[str, float | int]]:
    true_array = np.asarray(y_true, dtype=np.float32)
    pred_array = np.asarray(y_pred, dtype=np.float32)
    resolved_mask = np.asarray(mask, dtype=np.float32)
    if true_array.ndim == 1:
        true_array = true_array[:, None]
        pred_array = pred_array[:, None]
        resolved_mask = resolved_mask[:, None]
    reports: list[dict[str, float | int]] = []
    for horizon_index in range(true_array.shape[1]):
        horizon_mask = resolved_mask[:, horizon_index]
        reports.append(
            {
                "horizon": int(horizon_index + 1),
                "rmse": rmse(true_array[:, horizon_index], pred_array[:, horizon_index], horizon_mask),
                "r2": r2(true_array[:, horizon_index], pred_array[:, horizon_index], horizon_mask),
                "nrmse": nrmse(true_array[:, horizon_index], pred_array[:, horizon_index], horizon_mask),
                "n": int(horizon_mask.sum()),
            }
        )
    return reports


def regression_report(y_true: np.ndarray, y_pred: np.ndarray, domain_id: np.ndarray, mask: np.ndarray | None = None) -> dict:
    resolved_mask = _resolve_mask(y_true, mask)
    report = {
        "overall": {
            "rmse": rmse(y_true, y_pred, resolved_mask),
            "r2": r2(y_true, y_pred, resolved_mask),
            "nrmse": nrmse(y_true, y_pred, resolved_mask),
        },
        "per_horizon": _per_horizon_report(y_true, y_pred, resolved_mask),
        "by_domain": {},
    }
    for domain in sorted(np.unique(domain_id).tolist()):
        selector = domain_id == domain
        report["by_domain"][str(int(domain))] = {
            "rmse": rmse(y_true[selector], y_pred[selector], resolved_mask[selector]),
            "r2": r2(y_true[selector], y_pred[selector], resolved_mask[selector]),
            "nrmse": nrmse(y_true[selector], y_pred[selector], resolved_mask[selector]),
            "n": int(selector.sum()),
        }
    return report
