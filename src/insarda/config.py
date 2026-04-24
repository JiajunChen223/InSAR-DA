from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from hashlib import md5
from pathlib import Path
from typing import Any

from insarda.utils.io import read_yaml


FORMAL_BASELINE_METHODS = (
    "source_only",
    "target_only",
    "supervised_fine_tuning",
    "st_joint",
    "ss_dann",
    "ss_mt",
    "ss_coral",
)
FORMAL_MAIN_METHODS = FORMAL_BASELINE_METHODS + (
    "sft_replay",
)
FORMAL_METHODS = FORMAL_MAIN_METHODS
SUPPORTED_METHODS = FORMAL_METHODS
FORMAL_PROTOCOLS = ("LODO", "IHT", "CHT")
FORMAL_MAIN_PROTOCOLS = ("LODO", "IHT", "CHT")
PROTOCOL_ALIASES = {name: name for name in FORMAL_PROTOCOLS}
PROTOCOL_PAPER_NAMES = {
    "LODO": "Leave-One-Domain-Out (LODO)",
    "IHT": "Intra-Hazard Transfer (IHT)",
    "CHT": "Cross-Hazard Transfer (CHT)",
}
METHOD_ALIASES = {
    "source_only": "source_only",
    "target_only": "target_only",
    "supervised_fine_tuning": "supervised_fine_tuning",
    "st_joint": "st_joint",
    "ss_dann": "ss_dann",
    "ss_mt": "ss_mt",
    "ss_coral": "ss_coral",
    "sft_replay": "sft_replay",
}
METHOD_PAPER_NAMES = {
    "source_only": "Source-Only",
    "target_only": "Target-Only",
    "supervised_fine_tuning": "SFT",
    "st_joint": "ST-Joint",
    "ss_dann": "SS-DANN",
    "ss_mt": "SS-MT",
    "ss_coral": "SS-CORAL",
    "sft_replay": "SFT-Replay",
}
METHOD_FORMAL_NAMES = {
    "source_only": "Source-Only",
    "target_only": "Target-Only",
    "supervised_fine_tuning": "Supervised Fine-tuning (SFT)",
    "st_joint": "Source-Target Joint Supervised Training (ST-Joint)",
    "ss_dann": "Semi-supervised Domain-Adversarial Neural Network (SS-DANN)",
    "ss_mt": "Semi-supervised Mean Teacher (SS-MT)",
    "ss_coral": "Semi-supervised CORrelation ALignment (SS-CORAL)",
    "sft_replay": "Supervised Fine-Tuning with Source Replay",
}
FORMAL_BACKBONES = ("transformer",)
SUPPORTED_BACKBONES = FORMAL_BACKBONES
FORMAL_STUDY_TAG = "formal_transformer_transfer_v1"
DATA_SIGNATURE_VERSION = "v2"

_ROOT_SECTION_KEYS = frozenset({"paths", "data", "model", "training", "methods", "experiments"})
_PATH_KEYS = frozenset({"dataset_registry", "run_root", "summary_root", "cache_root"})
_DATA_KEYS = frozenset(
    {
        "input_window",
        "horizon",
        "split_layout",
        "source_split_ratio",
        "target_adapt_ratio",
        "target_val_ratio",
        "target_labeled_ratio",
        "target_labeled_ratios",
        "target_labeled_sampling_seed_policy",
        "target_labeled_sampling_strategy",
        "target_labeled_strata",
        "source_train_stride",
        "source_val_stride",
        "target_adapt_stride",
        "target_test_stride",
    }
)
_EXPERIMENTS_KEYS = frozenset({"main"})
_EXPERIMENT_GRID_KEYS = frozenset({"protocols", "methods", "backbones", "label_rates", "seeds"})
_MODEL_KEYS = frozenset(
    {
        "dropout",
        "transformer_d_model",
        "transformer_nhead",
        "transformer_num_layers",
        "transformer_ff_dim",
    }
)
_TRAINING_KEYS = frozenset(
    {
        "epochs",
        "batch_size",
        "eval_batch_size",
        "learning_rate",
        "weight_decay",
        "patience",
        "min_delta",
        "ema_decay",
        "grad_clip_norm",
        "mixed_precision",
        "allow_tf32",
        "cudnn_benchmark",
        "device",
        "target_aware_budget_mode",
        "target_aware_disable_early_stopping",
    }
)
_METHOD_KEYS: dict[str, frozenset[str]] = {
    "source_only": frozenset({"source_supervision_weight"}),
    "target_only": frozenset({"target_supervision_weight"}),
    "supervised_fine_tuning": frozenset(
        {
            "source_supervision_weight",
            "target_supervision_weight",
            "source_pretrain_epochs",
        }
    ),
    "st_joint": frozenset({"source_supervision_weight", "target_supervision_weight"}),
    "ss_dann": frozenset({"source_supervision_weight", "target_supervision_weight", "domain_weight"}),
    "ss_mt": frozenset(
        {
            "source_supervision_weight",
            "target_supervision_weight",
            "target_consistency_weight",
            "confidence_quantile",
        }
    ),
    "ss_coral": frozenset({"source_supervision_weight", "target_supervision_weight", "domain_weight"}),
    "sft_replay": frozenset(
        {
            "source_supervision_weight",
            "target_supervision_weight",
            "source_pretrain_epochs",
            "replay_source_weight",
        }
    ),
}

@dataclass(frozen=True)
class PathsConfig:
    config_path: Path
    dataset_registry: Path
    run_root: Path
    summary_root: Path
    cache_root: Path


@dataclass(frozen=True)
class DataConfig:
    input_window: int
    horizon: int
    split_layout: str
    source_split_ratio: float
    target_adapt_ratio: float
    target_val_ratio: float
    target_labeled_ratio: float
    target_labeled_ratios: tuple[float, ...]
    target_labeled_sampling_seed_policy: str
    target_labeled_sampling_strategy: str
    target_labeled_strata: int
    source_train_stride: int
    source_val_stride: int
    target_adapt_stride: int
    target_test_stride: int


@dataclass(frozen=True)
class ModelConfig:
    dropout: float
    transformer_d_model: int = 128
    transformer_nhead: int = 4
    transformer_num_layers: int = 2
    transformer_ff_dim: int = 224


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    eval_batch_size: int
    learning_rate: float
    weight_decay: float
    patience: int
    min_delta: float
    ema_decay: float
    grad_clip_norm: float
    mixed_precision: str
    allow_tf32: bool
    cudnn_benchmark: bool
    device: str
    target_aware_budget_mode: str
    target_aware_disable_early_stopping: bool


@dataclass(frozen=True)
class ExperimentGridConfig:
    protocols: tuple[str, ...]
    methods: tuple[str, ...]
    backbones: tuple[str, ...]
    label_rates: tuple[float, ...]
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class ExperimentsConfig:
    main: ExperimentGridConfig


@dataclass(frozen=True)
class MethodConfig:
    source_supervision_weight: float = 1.0
    target_supervision_weight: float = 0.0
    domain_weight: float = 0.0
    target_consistency_weight: float = 0.0
    confidence_quantile: float = 0.2
    source_pretrain_epochs: int = 60
    replay_source_weight: float = 0.0


@dataclass(frozen=True)
class FormalConfig:
    paths: PathsConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    methods: dict[str, MethodConfig]
    experiments: ExperimentsConfig


@dataclass(frozen=True)
class RunArgs:
    protocol: str
    case_id: int
    method: str
    backbone: str
    seed: int
    target_labeled_ratio: float | None = None


def _require_mapping(raw: Any, name: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError(f"`{name}` must be a mapping.")
    return raw


def _validate_exact_keys(raw: Any, name: str, allowed: set[str] | frozenset[str] | tuple[str, ...]) -> dict[str, Any]:
    mapping = _require_mapping(raw, name)
    allowed_keys = set(allowed)
    unknown = sorted(set(mapping) - allowed_keys)
    if unknown:
        raise ValueError(f"Unsupported keys in `{name}`: {', '.join(unknown)}")
    missing = sorted(allowed_keys - set(mapping))
    if missing:
        raise ValueError(f"Missing required keys in `{name}`: {', '.join(missing)}")
    return mapping


def _resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _load_method_configs(raw: dict[str, Any]) -> dict[str, MethodConfig]:
    methods_raw = _validate_exact_keys(raw, "methods", SUPPORTED_METHODS)
    methods: dict[str, MethodConfig] = {}
    for name in SUPPORTED_METHODS:
        section = _validate_exact_keys(methods_raw.get(name), f"methods.{name}", _METHOD_KEYS[name])
        methods[name] = MethodConfig(
            source_supervision_weight=float(section.get("source_supervision_weight", 1.0)),
            target_supervision_weight=float(section.get("target_supervision_weight", 0.0)),
            domain_weight=float(section.get("domain_weight", 0.0)),
            target_consistency_weight=float(section.get("target_consistency_weight", 0.0)),
            confidence_quantile=float(section.get("confidence_quantile", 0.2)),
            source_pretrain_epochs=int(section.get("source_pretrain_epochs", 60)),
            replay_source_weight=float(section.get("replay_source_weight", 0.0)),
        )
    return methods


def _normalize_protocol_list(values: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"`{name}` must be a non-empty list.")
    normalized = tuple(dict.fromkeys(str(value).strip().upper() for value in values))
    if any(value not in FORMAL_PROTOCOLS for value in normalized):
        raise ValueError(f"`{name}` must contain only supported protocols: {', '.join(FORMAL_PROTOCOLS)}")
    return normalized


def _normalize_method_list(values: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"`{name}` must be a non-empty list.")
    normalized = tuple(dict.fromkeys(str(value).strip().lower() for value in values))
    if any(value not in SUPPORTED_METHODS for value in normalized):
        raise ValueError(f"`{name}` must contain only supported methods: {', '.join(SUPPORTED_METHODS)}")
    return normalized


def _normalize_backbone_list(values: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"`{name}` must be a non-empty list.")
    normalized = tuple(dict.fromkeys(str(value).strip().lower() for value in values))
    if any(value not in SUPPORTED_BACKBONES for value in normalized):
        raise ValueError(f"`{name}` must contain only supported backbones: {', '.join(SUPPORTED_BACKBONES)}")
    return normalized


def _normalize_label_rate_list(values: Any, *, name: str) -> tuple[float, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"`{name}` must be a non-empty list.")
    normalized = tuple(dict.fromkeys(float(value) for value in values))
    if any(not (0.0 < value < 1.0) for value in normalized):
        raise ValueError(f"`{name}` values must be in the open interval (0, 1).")
    return normalized


def _normalize_seed_list(values: Any, *, name: str) -> tuple[int, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"`{name}` must be a non-empty list.")
    return tuple(dict.fromkeys(int(value) for value in values))


def _load_experiments_config(raw: dict[str, Any], *, data_cfg: DataConfig) -> ExperimentsConfig:
    experiments_raw = _validate_exact_keys(raw, "experiments", _EXPERIMENTS_KEYS)
    main_raw = _validate_exact_keys(experiments_raw.get("main"), "experiments.main", _EXPERIMENT_GRID_KEYS)

    main = ExperimentGridConfig(
        protocols=_normalize_protocol_list(main_raw.get("protocols"), name="experiments.main.protocols"),
        methods=_normalize_method_list(main_raw.get("methods"), name="experiments.main.methods"),
        backbones=_normalize_backbone_list(main_raw.get("backbones"), name="experiments.main.backbones"),
        label_rates=_normalize_label_rate_list(main_raw.get("label_rates"), name="experiments.main.label_rates"),
        seeds=_normalize_seed_list(main_raw.get("seeds"), name="experiments.main.seeds"),
    )
    allowed_label_rates = set(float(value) for value in data_cfg.target_labeled_ratios)
    if any(rate not in allowed_label_rates for rate in main.label_rates):
        raise ValueError("`experiments.main.label_rates` must be a subset of `data.target_labeled_ratios`.")
    if set(main.backbones) != set(FORMAL_BACKBONES):
        raise ValueError(
            "`experiments.main.backbones` must match the current formal backbone set: " + ", ".join(FORMAL_BACKBONES)
        )

    return ExperimentsConfig(main=main)


def protocol_paper_name(protocol: str) -> str:
    protocol_norm = str(protocol).strip().upper()
    return PROTOCOL_PAPER_NAMES.get(protocol_norm, protocol_norm)


def method_paper_name(method: str) -> str:
    method_norm = METHOD_ALIASES.get(str(method).strip().lower(), str(method).strip().lower())
    return METHOD_PAPER_NAMES.get(method_norm, method_norm)


def method_formal_name(method: str) -> str:
    method_norm = METHOD_ALIASES.get(str(method).strip().lower(), str(method).strip().lower())
    return METHOD_FORMAL_NAMES.get(method_norm, method_norm)


def method_variant_label(method: str) -> str:
    method_norm = METHOD_ALIASES.get(str(method).strip().lower(), str(method).strip().lower())
    return method_norm


def method_variant_paper_name(method: str) -> str:
    method_norm = METHOD_ALIASES.get(str(method).strip().lower(), str(method).strip().lower())
    return method_paper_name(method_norm)


def _signature_text(payload: dict[str, Any]) -> str:
    return md5(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def _resolve_registry_path(registry_path: str | Path) -> Path:
    registry = Path(registry_path)
    if not registry.is_absolute():
        registry = registry.resolve()
    if not registry.exists():
        repo_root = Path(__file__).resolve().parents[2]
        fallback_candidates = (
            repo_root / "data" / registry.name,
            Path.cwd() / "data" / registry.name,
        )
        for candidate in fallback_candidates:
            if candidate.exists():
                registry = candidate.resolve()
                break
    return registry.resolve()


def _filesystem_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _resolve_dataset_path(registry_path: Path, item: dict[str, Any]) -> Path:
    dataset_path = Path(item["path"])
    if not dataset_path.is_absolute():
        dataset_path = (registry_path.parent / dataset_path).resolve()
    return dataset_path.resolve()


def _resolved_registry_payload(registry_path: str | Path) -> dict[str, Any]:
    registry = _resolve_registry_path(registry_path)
    raw = read_yaml(registry)
    if raw is None:
        raise ValueError(f"Empty dataset registry: {registry}")
    datasets = []
    for item in raw.get("datasets", []):
        dataset_path = _resolve_dataset_path(registry, item)
        datasets.append(
            {
                "domain_id": int(item["domain_id"]),
                "name": str(item["name"]),
                "hazard_type": str(item["hazard_type"]),
                "source_tag": str(item.get("source_tag", "")),
                "file_identity": _filesystem_identity(dataset_path),
            }
        )
    return {
        "signature_version": DATA_SIGNATURE_VERSION,
        "registry_file_identity": _filesystem_identity(registry),
        "datasets": sorted(datasets, key=lambda item: item["domain_id"]),
        "metadata": raw.get("metadata", {}),
    }


def _split_layout_metadata(split_layout: str) -> dict[str, str]:
    layout = str(split_layout).strip()
    if layout != "role_aware_source703_target523":
        raise ValueError("Only `role_aware_source703_target523` is supported.")
    return {
        "target_split_policy": "target_domain_strict_523_time_band",
        "target_labeled_sampling_scope": "target_adapt_band",
        "target_labeled_sampling_unit": "point",
        "target_val_scope": "target_domain_middle_band",
        "target_val_split_unit": "time_band",
        "target_val_point_policy": "all_points_in_time_band",
        "shift_severity_basis": "target_adapt_shift_score",
    }


def _data_sampling_signature_fields(data_section: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_labeled_sampling_seed_policy": str(
            data_section.get("target_labeled_sampling_seed_policy", "run_seed")
        ),
        "target_labeled_sampling_strategy": str(
            data_section.get("target_labeled_sampling_strategy", "deformation_gradient_stratified")
        ),
        "target_labeled_strata": int(data_section.get("target_labeled_strata", 5)),
    }


def build_data_signature_from_sections(
    dataset_registry: str | Path,
    data_section: dict[str, Any],
    *,
    window_mode: str = "observation_step_displacement_only",
) -> str:
    split_layout = str(data_section["split_layout"])
    split_metadata = _split_layout_metadata(split_layout)
    payload = {
        "registry": _resolved_registry_payload(dataset_registry),
        "data": {
            "input_window": int(data_section["input_window"]),
            "horizon": int(data_section["horizon"]),
            "split_layout": split_layout,
            "source_split_ratio": float(data_section["source_split_ratio"]),
            "target_adapt_ratio": float(data_section["target_adapt_ratio"]),
            "target_val_ratio": float(data_section["target_val_ratio"]),
            "target_labeled_ratio": float(data_section["target_labeled_ratio"]),
            "target_labeled_ratios": [
                float(value)
                for value in data_section.get("target_labeled_ratios", [data_section["target_labeled_ratio"]])
            ],
            "source_train_stride": int(data_section["source_train_stride"]),
            "source_val_stride": int(data_section["source_val_stride"]),
            "target_adapt_stride": int(data_section["target_adapt_stride"]),
            "target_test_stride": int(data_section["target_test_stride"]),
            "window_mode": str(window_mode),
            **_data_sampling_signature_fields(data_section),
            **split_metadata,
            "shift_score_space": "source_standardized_input",
            "shift_score_reference": "source_train",
        },
    }
    return _signature_text(payload)


def build_config_signature_from_sections(model_section: dict[str, Any], training_section: dict[str, Any]) -> str:
    payload = {
        "model": {
            "dropout": float(model_section["dropout"]),
            "transformer_d_model": int(model_section["transformer_d_model"]),
            "transformer_nhead": int(model_section["transformer_nhead"]),
            "transformer_num_layers": int(model_section["transformer_num_layers"]),
            "transformer_ff_dim": int(model_section["transformer_ff_dim"]),
        },
        "training": {
            "epochs": int(training_section["epochs"]),
            "batch_size": int(training_section["batch_size"]),
            "eval_batch_size": int(training_section["eval_batch_size"]),
            "learning_rate": float(training_section["learning_rate"]),
            "weight_decay": float(training_section["weight_decay"]),
            "patience": int(training_section["patience"]),
            "min_delta": float(training_section["min_delta"]),
            "ema_decay": float(training_section["ema_decay"]),
            "grad_clip_norm": float(training_section["grad_clip_norm"]),
            "mixed_precision": str(training_section["mixed_precision"]),
            "allow_tf32": bool(training_section["allow_tf32"]),
            "cudnn_benchmark": bool(training_section["cudnn_benchmark"]),
            "device": str(training_section["device"]),
            "target_aware_budget_mode": str(training_section["target_aware_budget_mode"]),
            "target_aware_disable_early_stopping": bool(training_section["target_aware_disable_early_stopping"]),
        },
    }
    return _signature_text(payload)


def build_method_signature(
    method: str,
    method_cfg: MethodConfig | dict[str, Any],
) -> str:
    allowed_keys = {field.name for field in fields(MethodConfig)}
    payload = asdict(method_cfg) if isinstance(method_cfg, MethodConfig) else dict(method_cfg)
    payload = {key: value for key, value in payload.items() if key in allowed_keys}
    payload["method"] = str(method)
    payload["method_variant"] = method_variant_label(method)
    return _signature_text(payload)


def build_formal_signatures(config: FormalConfig) -> dict[str, str]:
    return {
        "data_signature": build_data_signature_from_sections(
            config.paths.dataset_registry,
            asdict(config.data),
            window_mode="observation_step_displacement_only",
        ),
        "config_signature": build_config_signature_from_sections(asdict(config.model), asdict(config.training)),
    }


def load_formal_config(path: str | Path) -> FormalConfig:
    config_path = Path(path).resolve()
    raw = read_yaml(config_path)
    if raw is None:
        raise ValueError(f"Empty config file: {config_path}")
    raw_root = _validate_exact_keys(raw, "root", ("config",))
    config_root = _require_mapping(raw_root.get("config"), "config")
    if "extends" in config_root:
        raise ValueError("Config inheritance is not supported. Keep only the single formal config.")
    root = _validate_exact_keys(config_root, "config", _ROOT_SECTION_KEYS)

    base_dir = config_path.parent
    paths_raw = _validate_exact_keys(root.get("paths"), "paths", _PATH_KEYS)
    data_raw = _validate_exact_keys(root.get("data"), "data", _DATA_KEYS)
    model_raw = _validate_exact_keys(root.get("model"), "model", _MODEL_KEYS)
    training_raw = _validate_exact_keys(root.get("training"), "training", _TRAINING_KEYS)
    methods_raw = _validate_exact_keys(root.get("methods"), "methods", SUPPORTED_METHODS)
    experiments_raw = _validate_exact_keys(root.get("experiments"), "experiments", _EXPERIMENTS_KEYS)

    data_cfg = DataConfig(
        input_window=int(data_raw.get("input_window", 20)),
        horizon=int(data_raw.get("horizon", 5)),
        split_layout=str(data_raw.get("split_layout", "role_aware_source703_target523")).strip(),
        source_split_ratio=float(data_raw.get("source_split_ratio", 0.7)),
        target_adapt_ratio=float(data_raw.get("target_adapt_ratio", 0.5)),
        target_val_ratio=float(data_raw.get("target_val_ratio", 0.2)),
        target_labeled_ratio=float(data_raw.get("target_labeled_ratio", 0.05)),
        target_labeled_ratios=tuple(float(value) for value in data_raw.get("target_labeled_ratios", [0.05])),
        target_labeled_sampling_seed_policy=str(
            data_raw.get("target_labeled_sampling_seed_policy", "run_seed")
        ).strip().lower(),
        target_labeled_sampling_strategy=str(
            data_raw.get("target_labeled_sampling_strategy", "deformation_gradient_stratified")
        ).strip(),
        target_labeled_strata=int(data_raw.get("target_labeled_strata", 5)),
        source_train_stride=int(data_raw.get("source_train_stride", 5)),
        source_val_stride=int(data_raw.get("source_val_stride", 5)),
        target_adapt_stride=int(data_raw.get("target_adapt_stride", 5)),
        target_test_stride=int(data_raw.get("target_test_stride", 5)),
    )
    if data_cfg.input_window != 20 or data_cfg.horizon != 5:
        raise ValueError("Only the formal 20->5 forecasting setup is supported.")
    if data_cfg.split_layout != "role_aware_source703_target523":
        raise ValueError("Only `role_aware_source703_target523` is supported.")
    if not (0.0 < data_cfg.target_adapt_ratio < 1.0):
        raise ValueError("`data.target_adapt_ratio` must be in the open interval (0, 1).")
    if not (0.0 < data_cfg.target_val_ratio < 1.0):
        raise ValueError("`data.target_val_ratio` must be in the open interval (0, 1).")
    if not (float(data_cfg.target_adapt_ratio) + float(data_cfg.target_val_ratio) < 1.0):
        raise ValueError(
            "For `role_aware_source703_target523`, `target_adapt_ratio + target_val_ratio` must be < 1."
        )
    if not (0.0 < data_cfg.target_labeled_ratio < 1.0):
        raise ValueError("`data.target_labeled_ratio` must be in the open interval (0, 1).")
    if not data_cfg.target_labeled_ratios:
        raise ValueError("`data.target_labeled_ratios` must contain at least one value.")
    if any(not (0.0 < float(value) < 1.0) for value in data_cfg.target_labeled_ratios):
        raise ValueError("All values in `data.target_labeled_ratios` must be in the open interval (0, 1).")
    if data_cfg.target_labeled_sampling_seed_policy != "run_seed":
        raise ValueError(
            "Only `run_seed` is supported for `data.target_labeled_sampling_seed_policy`."
        )
    if data_cfg.target_labeled_sampling_strategy != "deformation_gradient_stratified":
        raise ValueError("Only `deformation_gradient_stratified` is supported for `data.target_labeled_sampling_strategy`.")
    if data_cfg.target_labeled_strata < 2:
        raise ValueError("`data.target_labeled_strata` must be >= 2.")
    normalized_labeled_ratios = tuple(dict.fromkeys(float(value) for value in data_cfg.target_labeled_ratios))
    data_cfg = DataConfig(
        input_window=data_cfg.input_window,
        horizon=data_cfg.horizon,
        split_layout=data_cfg.split_layout,
        source_split_ratio=data_cfg.source_split_ratio,
        target_adapt_ratio=data_cfg.target_adapt_ratio,
        target_val_ratio=data_cfg.target_val_ratio,
        target_labeled_ratio=data_cfg.target_labeled_ratio,
        target_labeled_ratios=normalized_labeled_ratios,
        target_labeled_sampling_seed_policy=data_cfg.target_labeled_sampling_seed_policy,
        target_labeled_sampling_strategy=data_cfg.target_labeled_sampling_strategy,
        target_labeled_strata=data_cfg.target_labeled_strata,
        source_train_stride=data_cfg.source_train_stride,
        source_val_stride=data_cfg.source_val_stride,
        target_adapt_stride=data_cfg.target_adapt_stride,
        target_test_stride=data_cfg.target_test_stride,
    )
    if float(data_cfg.target_labeled_ratio) not in set(data_cfg.target_labeled_ratios):
        raise ValueError("`data.target_labeled_ratio` must be one of `data.target_labeled_ratios`.")
    for field_name in ("source_train_stride", "source_val_stride", "target_adapt_stride", "target_test_stride"):
        if getattr(data_cfg, field_name) < 1:
            raise ValueError(f"`data.{field_name}` must be >= 1.")

    model_cfg = ModelConfig(
        dropout=float(model_raw.get("dropout", 0.1)),
        transformer_d_model=int(model_raw.get("transformer_d_model", 128)),
        transformer_nhead=int(model_raw.get("transformer_nhead", 4)),
        transformer_num_layers=int(model_raw.get("transformer_num_layers", 2)),
        transformer_ff_dim=int(model_raw.get("transformer_ff_dim", 224)),
    )
    if model_cfg.transformer_d_model % max(model_cfg.transformer_nhead, 1) != 0:
        raise ValueError("`transformer_d_model` must be divisible by `transformer_nhead`.")

    training_cfg = TrainingConfig(
        epochs=int(training_raw.get("epochs", 100)),
        batch_size=int(training_raw.get("batch_size", 256)),
        eval_batch_size=int(training_raw.get("eval_batch_size", training_raw.get("batch_size", 256))),
        learning_rate=float(training_raw.get("learning_rate", 1e-3)),
        weight_decay=float(training_raw.get("weight_decay", 1e-4)),
        patience=int(training_raw.get("patience", 5)),
        min_delta=float(training_raw.get("min_delta", 1e-4)),
        ema_decay=float(training_raw.get("ema_decay", 0.995)),
        grad_clip_norm=float(training_raw.get("grad_clip_norm", 1.0)),
        mixed_precision=str(training_raw.get("mixed_precision", "off")).strip().lower(),
        allow_tf32=bool(training_raw.get("allow_tf32", True)),
        cudnn_benchmark=bool(training_raw.get("cudnn_benchmark", True)),
        device=str(training_raw.get("device", "auto")).strip(),
        target_aware_budget_mode=str(
            training_raw.get("target_aware_budget_mode", "matched_target_labeled_updates")
        ).strip().lower(),
        target_aware_disable_early_stopping=bool(
            training_raw.get("target_aware_disable_early_stopping", True)
        ),
    )
    if training_cfg.mixed_precision not in {"off", "bf16", "fp16"}:
        raise ValueError("`training.mixed_precision` must be one of: off, bf16, fp16.")
    if training_cfg.target_aware_budget_mode != "matched_target_labeled_updates":
        raise ValueError("Only `matched_target_labeled_updates` is supported for `training.target_aware_budget_mode`.")

    paths_cfg = PathsConfig(
        config_path=config_path,
        dataset_registry=_resolve_path(
            base_dir,
            paths_raw.get("dataset_registry", "../data/datasets_public_true_types_obs_step_final_10k_50x50.yaml"),
        ),
        run_root=_resolve_path(base_dir, paths_raw.get("run_root", "../runtime/runs")),
        summary_root=_resolve_path(base_dir, paths_raw.get("summary_root", "../runtime/summary")),
        cache_root=_resolve_path(base_dir, paths_raw.get("cache_root", "../runtime/cache")),
    )

    methods_cfg = _load_method_configs(methods_raw)
    for method_name, method_cfg in methods_cfg.items():
        if method_cfg.source_supervision_weight < 0.0:
            raise ValueError(f"`methods.{method_name}.source_supervision_weight` must be >= 0.")
        if method_cfg.target_supervision_weight < 0.0:
            raise ValueError(f"`methods.{method_name}.target_supervision_weight` must be >= 0.")
        if method_cfg.domain_weight < 0.0:
            raise ValueError(f"`methods.{method_name}.domain_weight` must be >= 0.")
        if method_cfg.target_consistency_weight < 0.0:
            raise ValueError(f"`methods.{method_name}.target_consistency_weight` must be >= 0.")
        if not (0.0 <= method_cfg.confidence_quantile <= 1.0):
            raise ValueError(f"`methods.{method_name}.confidence_quantile` must satisfy 0 <= value <= 1.")
        if method_cfg.source_pretrain_epochs < 0:
            raise ValueError(f"`methods.{method_name}.source_pretrain_epochs` must be >= 0.")
        if method_cfg.replay_source_weight < 0.0:
            raise ValueError(f"`methods.{method_name}.replay_source_weight` must be >= 0.")

    return FormalConfig(
        paths=paths_cfg,
        data=data_cfg,
        model=model_cfg,
        training=training_cfg,
        methods=methods_cfg,
        experiments=_load_experiments_config(experiments_raw, data_cfg=data_cfg),
    )


def ensure_formal_run_args(
    protocol: str,
    case_id: int,
    method: str,
    backbone: str,
    seed: int,
    target_labeled_ratio: float | None = None,
) -> RunArgs:
    protocol_norm = str(protocol).strip().upper()
    if protocol_norm not in FORMAL_PROTOCOLS:
        raise ValueError(f"Unsupported protocol: {protocol}")
    method_norm = METHOD_ALIASES.get(str(method).strip().lower(), str(method).strip().lower())
    if method_norm not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method: {method}")
    backbone_norm = str(backbone).strip().lower()
    if backbone_norm not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported backbone: {backbone}")
    label_ratio_norm = None if target_labeled_ratio is None else float(target_labeled_ratio)
    if label_ratio_norm is not None and not (0.0 < label_ratio_norm < 1.0):
        raise ValueError("`target_labeled_ratio` must be in the open interval (0, 1).")
    return RunArgs(
        protocol=protocol_norm,
        case_id=int(case_id),
        method=method_norm,
        backbone=backbone_norm,
        seed=int(seed),
        target_labeled_ratio=label_ratio_norm,
    )
