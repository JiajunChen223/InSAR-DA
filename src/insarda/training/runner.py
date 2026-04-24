from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
import gc
import time

import torch

from insarda.config import (
    FORMAL_BACKBONES,
    FORMAL_METHODS,
    FORMAL_STUDY_TAG,
    METHOD_ALIASES,
    PROTOCOL_ALIASES,
    FormalConfig,
    RunArgs,
    build_formal_signatures,
    build_method_signature,
    ensure_formal_run_args,
    method_variant_label,
)
from insarda.data_pipeline.builder import build_case_data
from insarda.data_pipeline.loaders import build_loader_bundle, shutdown_loader_bundle
from insarda.data_pipeline.preprocess import FeatureStandardizer
from insarda.data_pipeline.splits import generate_cases, load_dataset_specs
from insarda.data_pipeline.windows import WindowBundle
from insarda.evaluation.evaluate import evaluate_loader, evaluate_loader_metrics, save_predictions
from insarda.methods import build_method
from insarda.models import ForecastModel
from insarda.training.loop import train_model
from insarda.utils.io import write_json, write_yaml


def _resolve_device(requested: str) -> torch.device:
    name = str(requested).strip().lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _format_label_rate(value: float | None) -> str:
    if value is None:
        return "default"
    return f"{float(value):g}"


def _format_label_rate_tag(value: float | None) -> str:
    if value is None:
        return "lrdefault"
    return f"lr{_format_label_rate(value).replace('.', 'p')}"


def _dedupe_preserve_order(values: list[int] | list[float] | list[str]) -> list[int] | list[float] | list[str]:
    deduped = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        deduped.append(value)
        seen.add(value)
    return deduped


def _config_snapshot(config: FormalConfig, run_args: RunArgs) -> dict:
    return {
        "paths": {
            "dataset_registry": str(config.paths.dataset_registry),
            "run_root": str(config.paths.run_root),
            "summary_root": str(config.paths.summary_root),
            "cache_root": str(config.paths.cache_root),
        },
        "data": {
            "input_window": int(config.data.input_window),
            "horizon": int(config.data.horizon),
            "split_layout": config.data.split_layout,
            "source_split_ratio": float(config.data.source_split_ratio),
            "target_adapt_ratio": float(config.data.target_adapt_ratio),
            "target_val_ratio": float(config.data.target_val_ratio),
            "target_labeled_ratio": float(config.data.target_labeled_ratio),
            "target_labeled_ratios": [float(value) for value in config.data.target_labeled_ratios],
            "target_labeled_sampling_seed_policy": str(config.data.target_labeled_sampling_seed_policy),
            "target_labeled_sampling_strategy": str(config.data.target_labeled_sampling_strategy),
            "target_labeled_strata": int(config.data.target_labeled_strata),
            "source_train_stride": int(config.data.source_train_stride),
            "source_val_stride": int(config.data.source_val_stride),
            "target_adapt_stride": int(config.data.target_adapt_stride),
            "target_test_stride": int(config.data.target_test_stride),
        },
        "model": {
            "dropout": float(config.model.dropout),
            "transformer_d_model": int(config.model.transformer_d_model),
            "transformer_nhead": int(config.model.transformer_nhead),
            "transformer_num_layers": int(config.model.transformer_num_layers),
            "transformer_ff_dim": int(config.model.transformer_ff_dim),
        },
        "training": {
            "epochs": int(config.training.epochs),
            "batch_size": int(config.training.batch_size),
            "eval_batch_size": int(config.training.eval_batch_size),
            "learning_rate": float(config.training.learning_rate),
            "weight_decay": float(config.training.weight_decay),
            "patience": int(config.training.patience),
            "min_delta": float(config.training.min_delta),
            "ema_decay": float(config.training.ema_decay),
            "grad_clip_norm": float(config.training.grad_clip_norm),
            "mixed_precision": config.training.mixed_precision,
            "allow_tf32": bool(config.training.allow_tf32),
            "cudnn_benchmark": bool(config.training.cudnn_benchmark),
            "device": config.training.device,
            "target_aware_budget_mode": str(config.training.target_aware_budget_mode),
            "target_aware_disable_early_stopping": bool(config.training.target_aware_disable_early_stopping),
        },
        "methods": {name: asdict(method) for name, method in config.methods.items()},
        "experiments": {
            "main": asdict(config.experiments.main),
        },
        "signatures": build_formal_signatures(config),
        "run": {
            "protocol": run_args.protocol,
            "case": int(run_args.case_id),
            "method": run_args.method,
            "method_variant": method_variant_label(run_args.method),
            "backbone": run_args.backbone,
            "seed": int(run_args.seed),
            "target_labeled_ratio": float(config.data.target_labeled_ratio),
            "resolved_target_labeled_sampling_seed": int(run_args.seed),
        },
    }


def _create_run_dir(run_root: Path, run_args: RunArgs) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    tag = method_variant_label(run_args.method)
    label_rate_tag = _format_label_rate_tag(run_args.target_labeled_ratio)
    run_dir = (
        run_root
        / f"{_timestamp()}_{run_args.protocol}_c{run_args.case_id}_{tag}_{run_args.backbone}_{label_rate_tag}_s{run_args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _resolve_target_labeled_ratio(config: FormalConfig, run_args: RunArgs) -> float:
    selected = float(
        config.data.target_labeled_ratio
        if run_args.target_labeled_ratio is None
        else run_args.target_labeled_ratio
    )
    allowed = {float(value) for value in config.data.target_labeled_ratios}
    if selected not in allowed:
        allowed_text = ", ".join(f"{value:.2f}" for value in sorted(allowed))
        raise ValueError(f"`target_labeled_ratio` must be one of the configured options: {allowed_text}")
    return selected


def _restandardize_bundle(bundle: WindowBundle, standardizer: FeatureStandardizer) -> WindowBundle:
    if bundle.size == 0:
        return bundle
    return WindowBundle(
        x=standardizer.transform(bundle.x),
        y=bundle.y.copy(),
        y_mask=bundle.y_mask.copy(),
        target_start_idx=bundle.target_start_idx.copy(),
        target_end_idx=bundle.target_end_idx.copy(),
        domain_id=bundle.domain_id.copy(),
        point_id=bundle.point_id.copy(),
    )


def _retarget_standardization_for_target_only(case_data):
    standardizer = FeatureStandardizer.fit(case_data.target_labeled.x)
    metadata = dict(case_data.metadata)
    metadata["feature_standardization_scope"] = "target_labeled"
    metadata["feature_standardization_reference"] = "target_only"
    return replace(
        case_data,
        source_train=_restandardize_bundle(case_data.source_train, standardizer),
        source_val=_restandardize_bundle(case_data.source_val, standardizer),
        target_labeled=_restandardize_bundle(case_data.target_labeled, standardizer),
        target_unlabeled=_restandardize_bundle(case_data.target_unlabeled, standardizer),
        target_val=_restandardize_bundle(case_data.target_val, standardizer),
        target_test=_restandardize_bundle(case_data.target_test, standardizer),
        metadata=metadata,
    )


def _requires_source_pretraining(method_name: str, config: FormalConfig, case_data) -> bool:
    probe_model = ForecastModel(
        input_dim=int(case_data.source_train.x.shape[-1]),
        horizon=int(case_data.source_train.y.shape[-1]),
        model_cfg=config.model,
        backbone="transformer",
    )
    probe_method = build_method(
        method_name,
        method_cfg=config.methods[method_name],
        feature_dim=int(probe_model.encoder.out_dim),
    )
    requires = bool(getattr(probe_method, "freeze_model", False) and not bool(getattr(probe_method, "uses_source", False)))
    del probe_method
    del probe_model
    return requires


def _run_source_pretraining(
    *,
    config: FormalConfig,
    run_args: RunArgs,
    case_data,
    loaders,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    pretrain_model = ForecastModel(
        input_dim=int(case_data.source_train.x.shape[-1]),
        horizon=int(case_data.source_train.y.shape[-1]),
        model_cfg=config.model,
        backbone=run_args.backbone,
    )
    pretrain_method = build_method(
        "source_only",
        method_cfg=config.methods["source_only"],
        feature_dim=int(pretrain_model.encoder.out_dim),
    )
    pretrain_log_prefix = (
        f"[{run_args.protocol} c{run_args.case_id} {run_args.method} "
        f"{run_args.backbone} lr={_format_label_rate(run_args.target_labeled_ratio)} s{run_args.seed} source_pretrain]"
    )
    print(f"{pretrain_log_prefix} start", flush=True)
    pretrain_summary = train_model(
        model=pretrain_model,
        method=pretrain_method,
        source_train_loader=loaders.source_train,
        source_val_loader=loaders.source_val,
        target_labeled_loader=None,
        target_unlabeled_loader=None,
        target_val_loader=loaders.source_val,
        training_cfg=config.training,
        device=device,
        log_prefix=pretrain_log_prefix,
    )
    pretrain_model.load_state_dict(pretrain_summary.best_model_state_dict)
    state_dict = {key: value.detach().cpu().clone() for key, value in pretrain_model.state_dict().items()}
    info = {
        "applied": True,
        "method": "source_only",
        "best_epoch": int(pretrain_summary.best_epoch),
        "best_selection_rmse": float(pretrain_summary.best_selection_rmse),
        "best_epoch_metrics": dict(pretrain_summary.best_epoch_metrics),
        "budget_summary": dict(pretrain_summary.budget_summary),
    }
    print(
        (
            f"{pretrain_log_prefix} done "
            f"best_epoch={int(pretrain_summary.best_epoch)} "
            f"source_val_rmse={float(pretrain_summary.best_selection_rmse):.4f}"
        ),
        flush=True,
    )
    del pretrain_method
    del pretrain_model
    return info, state_dict


def run_case(config: FormalConfig, run_args: RunArgs) -> dict:
    case_timer = time.perf_counter()
    run_args = ensure_formal_run_args(
        protocol=run_args.protocol,
        case_id=run_args.case_id,
        method=run_args.method,
        backbone=run_args.backbone,
        seed=run_args.seed,
        target_labeled_ratio=run_args.target_labeled_ratio,
    )
    effective_ratio = _resolve_target_labeled_ratio(config, run_args)
    effective_data_cfg = replace(config.data, target_labeled_ratio=effective_ratio)
    effective_config = replace(config, data=effective_data_cfg)
    run_args = replace(run_args, target_labeled_ratio=effective_ratio)
    torch.manual_seed(int(run_args.seed))
    case_data = build_case_data(
        registry_path=effective_config.paths.dataset_registry,
        protocol=run_args.protocol,
        case_id=run_args.case_id,
        data_cfg=effective_config.data,
        target_labeled_sampling_seed=int(run_args.seed),
        cache_root=effective_config.paths.cache_root,
    )
    if run_args.method == "target_only":
        case_data = _retarget_standardization_for_target_only(case_data)
    device = _resolve_device(effective_config.training.device)
    loaders = None
    model = None
    method = None
    source_pretrain_info: dict[str, object] = {"applied": False}
    try:
        loaders = build_loader_bundle(
            source_train=case_data.source_train,
            source_val=case_data.source_val,
            target_labeled=case_data.target_labeled,
            target_unlabeled=case_data.target_unlabeled,
            target_val=case_data.target_val,
            target_test=case_data.target_test,
            batch_size=effective_config.training.batch_size,
            eval_batch_size=effective_config.training.eval_batch_size,
        )
        pretrained_model_state = None
        if _requires_source_pretraining(run_args.method, effective_config, case_data):
            source_pretrain_info, pretrained_model_state = _run_source_pretraining(
                config=effective_config,
                run_args=run_args,
                case_data=case_data,
                loaders=loaders,
                device=device,
            )
        model = ForecastModel(
            input_dim=int(case_data.source_train.x.shape[-1]),
            horizon=int(case_data.source_train.y.shape[-1]),
            model_cfg=effective_config.model,
            backbone=run_args.backbone,
        )
        if pretrained_model_state is not None:
            model.load_state_dict(pretrained_model_state)
        method = build_method(
            run_args.method,
            method_cfg=effective_config.methods[run_args.method],
            feature_dim=int(model.encoder.out_dim),
        )
        signatures = build_formal_signatures(effective_config)
        log_prefix = (
            f"[{run_args.protocol} c{run_args.case_id} {run_args.method} "
            f"{run_args.backbone} lr={_format_label_rate(effective_config.data.target_labeled_ratio)} s{run_args.seed}]"
        )
        print(
            f"{log_prefix} start",
            flush=True,
        )
        summary = train_model(
            model=model,
            method=method,
            source_train_loader=loaders.source_train,
            source_val_loader=loaders.source_val,
            target_labeled_loader=loaders.target_labeled,
            target_unlabeled_loader=loaders.target_unlabeled,
            target_val_loader=loaders.target_val,
            training_cfg=effective_config.training,
            device=device,
            log_prefix=log_prefix,
        )
        model.load_state_dict(summary.best_model_state_dict)
        method.load_state_dict(summary.best_method_state_dict)
        model.to(device)
        model.eval()
        method.to(device)
        method.eval()
        method.prepare_for_evaluation(
            model,
            source_loader=loaders.source_train,
            device=device,
            training_cfg=effective_config.training,
        )

        source_val_eval = evaluate_loader_metrics(
            model,
            loaders.source_val,
            device,
            method=method,
            training_cfg=effective_config.training,
        )
        target_val_eval = evaluate_loader_metrics(
            model,
            loaders.target_val,
            device,
            method=method,
            training_cfg=effective_config.training,
        )
        target_test_eval = evaluate_loader(
            model,
            loaders.target_test,
            device,
            method=method,
            training_cfg=effective_config.training,
        )
        run_dir = _create_run_dir(effective_config.paths.run_root, run_args)
        torch.save(model.state_dict(), run_dir / "model.pt")
        torch.save(method.state_dict(), run_dir / "method.pt")
        save_predictions(run_dir / "predictions.npz", target_test_eval)
        write_json({"history": summary.history}, run_dir / "train_history.json")
        write_yaml(_config_snapshot(effective_config, run_args), run_dir / "config_snapshot.yaml")

        metrics_payload = {
            "created_at": datetime.now().isoformat(),
            "study_tag": FORMAL_STUDY_TAG,
            "data_signature": signatures["data_signature"],
            "config_signature": signatures["config_signature"],
            "method_signature": build_method_signature(
                run_args.method,
                effective_config.methods[run_args.method],
            ),
            "protocol": run_args.protocol,
            "case_id": int(run_args.case_id),
            "case_name": case_data.metadata["case_name"],
            "method": run_args.method,
            "method_variant": method_variant_label(run_args.method),
            "backbone": run_args.backbone,
            "seed": int(run_args.seed),
            "target_labeled_ratio": float(effective_config.data.target_labeled_ratio),
            "best_epoch": int(summary.best_epoch),
            "training_diagnostics": dict(summary.best_epoch_metrics),
            "training_budget": dict(summary.budget_summary),
            "source_pretrain": dict(source_pretrain_info),
            "source_val": source_val_eval["metrics"],
            "target_val": target_val_eval["metrics"],
            "target_test": target_test_eval["metrics"],
            "split_summary": case_data.metadata,
            "run_dir": str(run_dir.resolve()),
        }
        write_json(metrics_payload, run_dir / "metrics.json")
        elapsed_seconds = float(time.perf_counter() - case_timer)
        print(
            (
                f"{log_prefix} done "
                f"best_epoch={int(summary.best_epoch)} "
                f"target_test_rmse={float(target_test_eval['metrics']['overall']['rmse']):.4f} "
                f"elapsed={elapsed_seconds:.2f}s "
                f"run_dir={str(run_dir.resolve())}"
            ),
            flush=True,
        )
        return metrics_payload
    finally:
        if loaders is not None:
            shutdown_loader_bundle(loaders)
        del loaders
        del model
        del method
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_sweep(
    config: FormalConfig,
    protocol: str,
    methods: list[str] | None = None,
    backbones: list[str] | None = None,
    cases: list[int] | None = None,
    seeds: list[int] | None = None,
    label_rates: list[float] | None = None,
) -> list[dict]:
    specs = load_dataset_specs(config.paths.dataset_registry)
    available_cases = generate_cases(specs, protocol)
    selected_protocol = available_cases[0].protocol if available_cases else PROTOCOL_ALIASES.get(
        str(protocol).strip().upper(),
        str(protocol).strip().upper(),
    )
    default_grid = config.experiments.main
    selected_case_ids = (
        _dedupe_preserve_order([case.case_id for case in available_cases])
        if not cases
        else _dedupe_preserve_order([int(case) for case in cases])
    )
    selected_methods = _dedupe_preserve_order(
        [METHOD_ALIASES.get(method.lower(), method.lower()) for method in (methods or list(default_grid.methods))]
    )
    selected_backbones = _dedupe_preserve_order([str(value).lower() for value in (backbones or list(default_grid.backbones))])
    selected_seeds = _dedupe_preserve_order([int(value) for value in (seeds or list(default_grid.seeds))])
    selected_label_rates = _dedupe_preserve_order([float(value) for value in (label_rates or list(default_grid.label_rates))])

    total_runs = (
        len(selected_case_ids)
        * len(selected_methods)
        * len(selected_backbones)
        * len(selected_label_rates)
        * len(selected_seeds)
    )
    print(
        (
            f"[SWEEP] protocol={selected_protocol} "
            f"cases={len(selected_case_ids)} methods={len(selected_methods)} "
            f"backbones={len(selected_backbones)} label_rates={len(selected_label_rates)} "
            f"seeds={len(selected_seeds)} total_runs={total_runs}"
        ),
        flush=True,
    )
    records = []
    run_index = 0
    for case_id in selected_case_ids:
        for method in selected_methods:
            for backbone in selected_backbones:
                for label_rate in selected_label_rates:
                    for seed in selected_seeds:
                        run_index += 1
                        print(
                            (
                                f"[SWEEP {run_index}/{total_runs}] "
                                f"case={int(case_id)} method={method} "
                                f"backbone={backbone} lr={_format_label_rate(label_rate)} "
                                f"seed={int(seed)}"
                            ),
                            flush=True,
                        )
                        run_args = ensure_formal_run_args(
                            protocol=str(protocol),
                            case_id=int(case_id),
                            method=method,
                            backbone=backbone,
                            seed=int(seed),
                            target_labeled_ratio=float(label_rate),
                        )
                        records.append(run_case(config, run_args))
    return records
