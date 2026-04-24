from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from insarda.config import TrainingConfig
from insarda.evaluation.evaluate import evaluate_loader_rmse
from insarda.methods.base import FormalMethod
from insarda.models.ema import EMATeacher
from insarda.utils.torch_runtime import autocast_context, configure_torch_runtime, move_batch_to_device


@dataclass
class TrainingSummary:
    best_model_state_dict: dict
    best_method_state_dict: dict
    best_epoch: int
    best_selection_rmse: float
    best_epoch_metrics: dict[str, object]
    history: list[dict[str, object]]
    budget_summary: dict[str, object]


_SOURCE_TRANSFER_KEYS = ("x", "y", "y_mask", "domain_id", "point_id", "target_start_idx", "target_end_idx")
_TARGET_LABELED_KEYS = ("x", "y", "y_mask", "domain_id", "point_id", "target_start_idx", "target_end_idx")
_TARGET_UNLABELED_KEYS = ("x", "domain_id", "point_id", "target_start_idx", "target_end_idx")


def _repeat_loader(loader: DataLoader | None):
    if loader is None:
        while True:
            yield None
    while True:
        for batch in loader:
            yield batch


def _steps_per_epoch(
    source_train: DataLoader | None,
    target_labeled: DataLoader | None,
    target_unlabeled: DataLoader | None,
) -> int:
    sizes = []
    if source_train is not None:
        sizes.append(len(source_train))
    if target_labeled is not None:
        sizes.append(len(target_labeled))
    if target_unlabeled is not None:
        sizes.append(len(target_unlabeled))
    return max(max(sizes), 1) if sizes else 1


def _epoch_loader_usage(
    method: FormalMethod,
    epoch: int,
    *,
    uses_source: bool,
    uses_target_labeled: bool,
    uses_target_unlabeled: bool,
) -> tuple[bool, bool, bool]:
    if not bool(getattr(method, "uses_source_pretraining", False)):
        return uses_source, uses_target_labeled, uses_target_unlabeled
    pretrain_epochs = max(int(getattr(method, "source_pretrain_epochs", 0)), 0)
    if epoch <= pretrain_epochs:
        return uses_source, False, False
    return (
        bool(getattr(method, "post_pretrain_uses_source", False)) and uses_source,
        bool(getattr(method, "post_pretrain_uses_target_labeled", True)) and uses_target_labeled,
        bool(getattr(method, "post_pretrain_uses_target_unlabeled", False)) and uses_target_unlabeled,
    )


def _freeze_copy(model: torch.nn.Module) -> torch.nn.Module:
    anchor = copy.deepcopy(model)
    anchor.eval()
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    return anchor


def _selection_proxy(epoch_metrics: dict[str, float]) -> float:
    proxy = 0.0
    for name in ("domain_loss", "coral_loss", "joint_alignment_loss", "spectral_alignment_loss"):
        value = epoch_metrics.get(name)
        if value is not None:
            proxy += 0.1 * float(torch.log1p(torch.tensor(max(float(value), 0.0))).item())
    for name, weight in (
        ("masked_modeling_loss", 0.04),
        ("pseudo_label_loss", 0.05),
        ("siamese_loss", 0.05),
        ("multiscale_loss", 0.04),
        ("adapter_loss", 0.04),
        ("mixup_contrastive_loss", 0.04),
        ("ranking_loss", 0.04),
        ("regression_distribution_loss", 0.04),
        ("rasc_loss", 0.05),
        ("reliability_consistency_loss", 0.05),
        ("rewarded_consistency_loss", 0.05),
        ("reward_supervision_loss", 0.03),
        ("residual_supervision_loss", 0.03),
        ("residual_magnitude_loss", 0.02),
        ("ucr_loss", 0.04),
        ("structure_ranking_loss", 0.04),
    ):
        value = epoch_metrics.get(name)
        if value is not None:
            proxy += weight * float(torch.log1p(torch.tensor(max(float(value), 0.0))).item())
    target_supervision = epoch_metrics.get("target_supervision_loss")
    if target_supervision is not None:
        proxy += 0.05 * float(torch.log1p(torch.tensor(max(float(target_supervision), 0.0))).item())
    target_consistency = epoch_metrics.get("target_consistency_loss")
    if target_consistency is not None:
        proxy += 0.05 * float(torch.log1p(torch.tensor(max(float(target_consistency), 0.0))).item())
    orthogonality = epoch_metrics.get("orthogonality_loss")
    if orthogonality is not None:
        proxy += 0.02 * float(torch.log1p(torch.tensor(max(float(orthogonality), 0.0))).item())
    safe_pseudo = epoch_metrics.get("safe_pseudo_loss")
    if safe_pseudo is not None:
        proxy += 0.02 * float(torch.log1p(torch.tensor(max(float(safe_pseudo), 0.0))).item())
    mean_safe_risk = epoch_metrics.get("mean_safe_risk")
    if mean_safe_risk is not None:
        proxy += 0.15 * max(float(mean_safe_risk), 0.0)
    anchor_fallback_rate = epoch_metrics.get("anchor_fallback_rate")
    if anchor_fallback_rate is not None:
        proxy += 0.2 * max(float(anchor_fallback_rate), 0.0)
    return float(proxy)


def _selection_smoothing_window(method: FormalMethod) -> int:
    raw = int(getattr(method, "selection_smoothing_window", 1))
    return max(raw, 1)


def _min_epochs_before_selection(method: FormalMethod) -> int:
    raw = int(getattr(method, "min_epochs_before_selection", 1))
    return max(raw, 1)


def _smoothed_selection_score(history: list[dict[str, object]], window: int) -> float:
    if not history:
        return float("inf")
    recent = history[-window:]
    values = [float(record["selection_rmse"]) for record in recent]
    return float(sum(values) / len(values))


def _build_optimizer(
    parameters: list[torch.nn.Parameter],
    training_cfg: TrainingConfig,
    device: torch.device,
):
    kwargs = {
        "lr": float(training_cfg.learning_rate),
        "weight_decay": float(training_cfg.weight_decay),
    }
    mixed_precision = str(training_cfg.mixed_precision).strip().lower()
    if device.type == "cuda" and mixed_precision == "off":
        try:
            return Adam(parameters, fused=True, **kwargs)
        except (TypeError, RuntimeError):
            pass
    return Adam(parameters, **kwargs)


def _build_grad_scaler(training_cfg: TrainingConfig, device: torch.device):
    enabled = bool(device.type == "cuda" and str(training_cfg.mixed_precision).strip().lower() == "fp16")
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except AttributeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _selection_loader(
    method: FormalMethod,
    *,
    source_val_loader: DataLoader,
    target_val_loader: DataLoader,
) -> tuple[str, DataLoader]:
    selection_name = str(getattr(method, "selection_loader_name", "source_val")).strip().lower()
    if selection_name == "target_val":
        return selection_name, target_val_loader
    return "source_val", source_val_loader


def _source_val_eval_interval(selection_loader_name: str) -> int:
    raw = os.getenv("INSARDA_SOURCE_VAL_INTERVAL", "").strip()
    if raw:
        try:
            return max(int(raw), 1)
        except ValueError:
            pass
    return 1 if selection_loader_name == "source_val" else 5


def _skip_source_val() -> bool:
    raw = os.getenv("INSARDA_SKIP_SOURCE_VAL", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _is_target_aware_method(method: FormalMethod) -> bool:
    selection_name = str(getattr(method, "selection_loader_name", "")).strip().lower()
    return bool(method.uses_target_labeled or method.uses_target_unlabeled or selection_name == "target_val")


def _target_aware_budget_mode(training_cfg: TrainingConfig) -> str:
    return str(getattr(training_cfg, "target_aware_budget_mode", "matched_target_labeled_updates")).strip().lower()


def _remaining_target_labeled_epochs(
    method: FormalMethod,
    current_epoch: int,
    *,
    total_epochs: int,
    uses_source: bool,
    uses_target_labeled: bool,
    uses_target_unlabeled: bool,
) -> int:
    remaining = 0
    for epoch in range(int(current_epoch), int(total_epochs) + 1):
        _, epoch_uses_target_labeled, _ = _epoch_loader_usage(
            method,
            epoch,
            uses_source=uses_source,
            uses_target_labeled=uses_target_labeled,
            uses_target_unlabeled=uses_target_unlabeled,
        )
        if epoch_uses_target_labeled:
            remaining += 1
    return remaining


def _matched_target_labeled_epoch_steps(
    method: FormalMethod,
    *,
    epoch: int,
    total_epochs: int,
    uses_source: bool,
    uses_target_labeled: bool,
    uses_target_unlabeled: bool,
    epoch_uses_source: bool,
    epoch_uses_target_labeled: bool,
    epoch_uses_target_unlabeled: bool,
    remaining_target_labeled_budget: int,
    target_labeled_reference_steps: int,
) -> tuple[int, int]:
    if epoch_uses_target_labeled:
        remaining_epochs = _remaining_target_labeled_epochs(
            method,
            epoch,
            total_epochs=total_epochs,
            uses_source=uses_source,
            uses_target_labeled=uses_target_labeled,
            uses_target_unlabeled=uses_target_unlabeled,
        )
        if remaining_epochs <= 0:
            return max(int(target_labeled_reference_steps), 1), 0
        planned_target_labeled_steps = max(
            (int(remaining_target_labeled_budget) + int(remaining_epochs) - 1) // int(remaining_epochs),
            1,
        )
        return planned_target_labeled_steps, planned_target_labeled_steps
    if epoch_uses_source or epoch_uses_target_unlabeled:
        return max(int(target_labeled_reference_steps), 1), 0
    return 1, 0


def train_model(
    model: torch.nn.Module,
    method: FormalMethod,
    source_train_loader: DataLoader,
    source_val_loader: DataLoader,
    target_labeled_loader: DataLoader | None,
    target_unlabeled_loader: DataLoader | None,
    target_val_loader: DataLoader,
    training_cfg: TrainingConfig,
    device: torch.device,
    log_prefix: str = "",
) -> TrainingSummary:
    configure_torch_runtime(training_cfg, device)
    raw_model = model
    raw_model.to(device)
    method.to(device)
    if bool(getattr(method, "freeze_model", False)):
        for parameter in raw_model.parameters():
            parameter.requires_grad_(False)
    trainable_params = [parameter for parameter in list(raw_model.parameters()) + list(method.parameters()) if parameter.requires_grad]
    ema_teacher = (
        EMATeacher(
            raw_model,
            decay=training_cfg.ema_decay,
            method=method if bool(getattr(method, "ema_tracks_method", False)) else None,
        )
        if method.uses_ema
        else None
    )
    optimizer = _build_optimizer(trainable_params, training_cfg, device)
    grad_scaler = _build_grad_scaler(training_cfg, device)
    best_state = None
    best_epoch = 0
    best_selection_rmse = float("inf")
    best_epoch_metrics: dict[str, object] = {}
    best_selection_score = float("inf")
    fallback_best_model_state = copy.deepcopy(raw_model.state_dict())
    fallback_best_method_state = copy.deepcopy(method.state_dict())
    fallback_best_epoch = 0
    fallback_best_selection_rmse = float("inf")
    fallback_best_epoch_metrics: dict[str, object] = {}
    stale_epochs = 0
    history: list[dict[str, object]] = []
    min_epochs_before_early_stop = max(int(getattr(method, "min_epochs_before_early_stop", 0)), 0)
    min_epochs_before_selection = _min_epochs_before_selection(method)
    selection_smoothing_window = _selection_smoothing_window(method)
    uses_source = bool(method.uses_source)
    uses_target_labeled = bool(target_labeled_loader is not None and method.uses_target_labeled)
    uses_target_unlabeled = bool(target_unlabeled_loader is not None and method.uses_target_unlabeled)
    selection_loader_name, selection_loader = _selection_loader(
        method,
        source_val_loader=source_val_loader,
        target_val_loader=target_val_loader,
    )
    is_target_aware = _is_target_aware_method(method)
    target_aware_budget_mode = _target_aware_budget_mode(training_cfg)
    matched_target_labeled_budget = bool(is_target_aware and target_aware_budget_mode == "matched_target_labeled_updates")
    if matched_target_labeled_budget and target_labeled_loader is None:
        raise ValueError("`matched_target_labeled_updates` requires a non-empty `target_labeled_loader`.")
    target_labeled_reference_steps = max(len(target_labeled_loader), 1) if target_labeled_loader is not None else 0
    target_labeled_budget_total = (
        int(training_cfg.epochs) * int(target_labeled_reference_steps) if matched_target_labeled_budget else 0
    )
    remaining_target_labeled_budget = int(target_labeled_budget_total)
    disable_early_stopping = bool(
        matched_target_labeled_budget and bool(getattr(training_cfg, "target_aware_disable_early_stopping", True))
    )
    source_val_interval = _source_val_eval_interval(selection_loader_name)
    skip_source_val = _skip_source_val()
    latest_source_val_rmse = float("nan")
    anchor_teacher = _freeze_copy(raw_model).to(device) if method.uses_anchor else None

    source_iter = _repeat_loader(source_train_loader) if uses_source else None
    non_blocking = bool(device.type == "cuda")
    optimizer_steps_completed = 0
    target_labeled_steps_completed = 0

    for epoch in range(1, int(training_cfg.epochs) + 1):
        epoch_start = time.perf_counter()
        epoch_uses_source, epoch_uses_target_labeled, epoch_uses_target_unlabeled = _epoch_loader_usage(
            method,
            epoch,
            uses_source=uses_source,
            uses_target_labeled=uses_target_labeled,
            uses_target_unlabeled=uses_target_unlabeled,
        )
        planned_target_labeled_steps = 0
        if matched_target_labeled_budget:
            steps, planned_target_labeled_steps = _matched_target_labeled_epoch_steps(
                method,
                epoch=epoch,
                total_epochs=int(training_cfg.epochs),
                uses_source=uses_source,
                uses_target_labeled=uses_target_labeled,
                uses_target_unlabeled=uses_target_unlabeled,
                epoch_uses_source=epoch_uses_source,
                epoch_uses_target_labeled=epoch_uses_target_labeled,
                epoch_uses_target_unlabeled=epoch_uses_target_unlabeled,
                remaining_target_labeled_budget=remaining_target_labeled_budget,
                target_labeled_reference_steps=target_labeled_reference_steps,
            )
        else:
            steps = _steps_per_epoch(
                source_train_loader if epoch_uses_source else None,
                target_labeled_loader if epoch_uses_target_labeled else None,
                target_unlabeled_loader if epoch_uses_target_unlabeled else None,
            )
        raw_model.train()
        method.train()
        running_loss = 0.0
        metric_sums: dict[str, float] = {}
        target_labeled_iter = _repeat_loader(target_labeled_loader) if epoch_uses_target_labeled else None
        target_unlabeled_iter = _repeat_loader(target_unlabeled_loader) if epoch_uses_target_unlabeled else None
        target_labeled_active_steps = 0
        for _ in range(steps):
            source_batch = None
            if epoch_uses_source and source_iter is not None:
                source_batch = move_batch_to_device(
                    next(source_iter),
                    device,
                    keys=_SOURCE_TRANSFER_KEYS,
                    non_blocking=non_blocking,
                )
            target_labeled_batch = None
            if epoch_uses_target_labeled and target_labeled_iter is not None:
                target_labeled_batch = next(target_labeled_iter)
                target_labeled_batch = (
                    move_batch_to_device(
                        target_labeled_batch,
                        device,
                        keys=_TARGET_LABELED_KEYS,
                        non_blocking=non_blocking,
                    )
                    if target_labeled_batch is not None
                    else None
                )
                if target_labeled_batch is not None:
                    target_labeled_active_steps += 1
            target_unlabeled_batch = None
            if epoch_uses_target_unlabeled and target_unlabeled_iter is not None:
                target_unlabeled_batch = next(target_unlabeled_iter)
                target_unlabeled_batch = (
                    move_batch_to_device(
                        target_unlabeled_batch,
                        device,
                        keys=_TARGET_UNLABELED_KEYS,
                        non_blocking=non_blocking,
                    )
                    if target_unlabeled_batch is not None
                    else None
                )

            optimizer.zero_grad(set_to_none=True)
            teacher_model = ema_teacher if ema_teacher is not None else None
            anchor_model = anchor_teacher if anchor_teacher is not None else None
            with autocast_context(training_cfg, device):
                loss_output = method.compute_loss(
                    model=raw_model,
                    source_batch=source_batch,
                    target_labeled_batch=target_labeled_batch,
                    target_unlabeled_batch=target_unlabeled_batch,
                    ema_teacher=teacher_model,
                    anchor_teacher=anchor_model,
                )
            if grad_scaler.is_enabled():
                grad_scaler.scale(loss_output.total).backward()
                grad_scaler.unscale_(optimizer)
            else:
                loss_output.total.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(training_cfg.grad_clip_norm))
            if grad_scaler.is_enabled():
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()
            if ema_teacher is not None:
                ema_teacher.update(
                    raw_model,
                    method=method if bool(getattr(method, "ema_tracks_method", False)) else None,
                )
            optimizer_steps_completed += 1
            running_loss += float(loss_output.total.detach().item())
            for name, value in loss_output.metrics.items():
                metric_sums[name] = metric_sums.get(name, 0.0) + float(value)
        target_labeled_steps_completed += int(target_labeled_active_steps)
        if matched_target_labeled_budget:
            remaining_target_labeled_budget = max(
                int(remaining_target_labeled_budget) - int(target_labeled_active_steps),
                0,
            )

        selection_model = raw_model
        if ema_teacher is not None and bool(getattr(method, "selection_uses_ema", False)):
            selection_model = ema_teacher.teacher
        method.prepare_for_evaluation(
            selection_model,
            source_loader=source_train_loader if uses_source else None,
            device=device,
            training_cfg=training_cfg,
        )

        source_val_refreshed = bool(
            (not skip_source_val)
            and (
                epoch == 1
                or epoch == int(training_cfg.epochs)
                or ((epoch - 1) % source_val_interval == 0)
                or selection_loader_name == "source_val"
            )
        )
        if source_val_refreshed:
            latest_source_val_rmse = evaluate_loader_rmse(
                selection_model,
                source_val_loader,
                device,
                method=method,
                training_cfg=training_cfg,
            )
        source_val_rmse = float(latest_source_val_rmse)
        selection_rmse = evaluate_loader_rmse(
            selection_model,
            selection_loader,
            device,
            method=method,
            training_cfg=training_cfg,
        )
        epoch_metrics = {name: float(total / max(steps, 1)) for name, total in metric_sums.items()}
        epoch_metrics["selection_proxy"] = _selection_proxy(epoch_metrics)
        epoch_metrics["selection_rmse"] = float(selection_rmse)
        epoch_metrics["target_labeled_active_steps"] = float(target_labeled_active_steps)
        epoch_metrics["target_labeled_steps_planned"] = float(planned_target_labeled_steps)
        epoch_metrics["optimizer_steps_completed"] = float(optimizer_steps_completed)
        epoch_metrics["target_labeled_steps_completed"] = float(target_labeled_steps_completed)
        epoch_metrics["target_labeled_budget_remaining"] = float(remaining_target_labeled_budget)
        epoch_seconds = float(time.perf_counter() - epoch_start)
        epoch_record = {
            "epoch": int(epoch),
            "train_loss": float(running_loss / max(steps, 1)),
            "source_val_rmse": source_val_rmse,
            "selection_loader": selection_loader_name,
            "selection_rmse": float(selection_rmse),
            "epoch_seconds": epoch_seconds,
            "source_val_refreshed": bool(source_val_refreshed),
            "budget_mode": (
                target_aware_budget_mode
                if is_target_aware
                else "source_only_source_val_lower_bound"
            ),
            "budget_basis": (
                "target_labeled_updates"
                if matched_target_labeled_budget
                else "source_loader_length"
            ),
            "planned_epoch_steps": int(steps),
            "target_labeled_steps_planned": int(planned_target_labeled_steps),
            "target_labeled_active_steps": int(target_labeled_active_steps),
            "optimizer_steps_completed": int(optimizer_steps_completed),
            "target_labeled_steps_completed": int(target_labeled_steps_completed),
            "target_labeled_budget_remaining": int(remaining_target_labeled_budget),
        }
        epoch_record.update(epoch_metrics)
        hook_updates = method.on_epoch_end(epoch=int(epoch), epoch_record=epoch_record)
        if hook_updates:
            epoch_record.update(hook_updates)
        history.append(epoch_record)
        prefix = f"{log_prefix} " if log_prefix else ""
        safe_suffix = ""
        if "safe_lambda_u" in epoch_record:
            safe_suffix = f" safe_lambda={float(epoch_record['safe_lambda_u']):.2f}"
        if "safe_mode_state" in epoch_record:
            safe_suffix += f" safe_mode={epoch_record['safe_mode_state']}"
        print(
            (
                f"{prefix}epoch {epoch:02d}/{int(training_cfg.epochs):02d} "
                f"loss={epoch_record['train_loss']:.4f} "
                f"source_val_rmse={source_val_rmse:.4f} "
                f"{selection_loader_name}_rmse={float(selection_rmse):.4f} "
                f"time={epoch_seconds:.2f}s"
                f"{safe_suffix}"
            ),
            flush=True,
        )

        if selection_rmse < fallback_best_selection_rmse:
            fallback_best_selection_rmse = selection_rmse
            fallback_best_epoch = int(epoch)
            fallback_best_model_state = copy.deepcopy(selection_model.state_dict())
            fallback_best_method_state = copy.deepcopy(method.state_dict())
            fallback_best_epoch_metrics = dict(epoch_record)

        # Early stopping is unified across all methods and depends only on
        # validation RMSE. Methods may still request a small smoothing window
        # and a minimum epoch before checkpoint selection.
        selection_score = _smoothed_selection_score(history, selection_smoothing_window)
        epoch_record["selection_score"] = float(selection_score)
        if epoch >= min_epochs_before_selection and selection_score < (best_selection_score - float(training_cfg.min_delta)):
            best_selection_score = float(selection_score)
            best_selection_rmse = selection_rmse
            best_epoch = int(epoch)
            best_state = copy.deepcopy(selection_model.state_dict())
            best_method_state = copy.deepcopy(method.state_dict())
            best_epoch_metrics = dict(epoch_record)
            best_epoch_metrics["selection_score"] = float(selection_score)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if (
                not disable_early_stopping
                and stale_epochs >= int(training_cfg.patience)
                and epoch >= min_epochs_before_early_stop
            ):
                break

    if best_state is None:
        best_state = fallback_best_model_state
        best_method_state = fallback_best_method_state
        best_epoch = int(fallback_best_epoch)
        best_selection_rmse = float(fallback_best_selection_rmse)
        best_epoch_metrics = dict(fallback_best_epoch_metrics)

    budget_summary = {
        "comparison_group": "target_aware" if is_target_aware else "source_only_lower_bound",
        "selection_loader": selection_loader_name,
        "budget_mode": target_aware_budget_mode if is_target_aware else "source_only_source_val_lower_bound",
        "budget_basis": "target_labeled_updates" if matched_target_labeled_budget else "source_loader_length",
        "target_aware_disable_early_stopping": bool(disable_early_stopping),
        "epochs_completed": int(len(history)),
        "optimizer_steps_completed": int(optimizer_steps_completed),
        "target_labeled_reference_steps_per_epoch": int(target_labeled_reference_steps),
        "target_labeled_budget_total": int(target_labeled_budget_total),
        "target_labeled_steps_completed": int(target_labeled_steps_completed),
        "target_labeled_budget_remaining": int(remaining_target_labeled_budget),
        "target_labeled_budget_exhausted": bool(
            (not matched_target_labeled_budget) or int(remaining_target_labeled_budget) == 0
        ),
    }

    return TrainingSummary(
        best_model_state_dict=best_state,
        best_method_state_dict=best_method_state,
        best_epoch=best_epoch,
        best_selection_rmse=best_selection_rmse,
        best_epoch_metrics=best_epoch_metrics,
        history=history,
        budget_summary=budget_summary,
    )
