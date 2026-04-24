from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from insarda.config import (
    FORMAL_BACKBONES,
    FORMAL_BASELINE_METHODS,
    FORMAL_MAIN_PROTOCOLS,
    FORMAL_METHODS,
    FORMAL_PROTOCOLS,
    FORMAL_STUDY_TAG,
    METHOD_ALIASES,
    PROTOCOL_ALIASES,
    SUPPORTED_METHODS,
    FormalConfig,
    build_method_signature,
    build_config_signature_from_sections,
    build_data_signature_from_sections,
    build_formal_signatures,
    method_paper_name,
    method_variant_paper_name,
    method_variant_label,
    protocol_paper_name,
)
from insarda.utils.io import read_json, read_yaml, write_json


PRIMARY_BASELINE_VARIANT = "target_only"
AUXILIARY_BASELINE_VARIANT = "source_only"


def transfer_gain(baseline_rmse: float | None, method_rmse: float | None) -> float | None:
    if baseline_rmse is None or method_rmse is None:
        return None
    baseline_value = float(baseline_rmse)
    method_value = float(method_rmse)
    if abs(baseline_value) <= 1e-12:
        return None
    return float((baseline_value - method_value) / baseline_value)


def collect_run_records(run_root: str | Path) -> list[dict[str, Any]]:
    root = Path(run_root)
    if not root.exists():
        return []
    records: list[dict[str, Any]] = []
    for metrics_file in sorted(root.rglob("metrics.json")):
        payload = read_json(metrics_file)
        payload["metrics_file"] = str(metrics_file.resolve())
        payload["run_dir"] = str(metrics_file.parent.resolve())
        recorded_method = str(payload.get("method", "")).strip().lower()
        normalized_method = METHOD_ALIASES.get(recorded_method, recorded_method)
        if normalized_method:
            payload["recorded_method"] = recorded_method
            payload["method"] = normalized_method
        payload["recorded_data_signature"] = payload.get("data_signature")
        payload["recorded_config_signature"] = payload.get("config_signature")
        payload["recorded_method_signature"] = payload.get("method_signature")
        snapshot_path = metrics_file.parent / "config_snapshot.yaml"
        if snapshot_path.exists():
            snapshot = read_yaml(snapshot_path)
            paths = snapshot.get("paths", {}) if isinstance(snapshot, dict) else {}
            data = snapshot.get("data", {}) if isinstance(snapshot, dict) else {}
            model = snapshot.get("model", {}) if isinstance(snapshot, dict) else {}
            training = snapshot.get("training", {}) if isinstance(snapshot, dict) else {}
            methods = snapshot.get("methods", {}) if isinstance(snapshot, dict) else {}
            if paths.get("dataset_registry") and data:
                try:
                    payload["resolved_data_signature"] = build_data_signature_from_sections(
                        paths["dataset_registry"],
                        data,
                        window_mode=payload.get("split_summary", {}).get(
                            "window_mode",
                            "observation_step_displacement_only",
                        ),
                    )
                except (FileNotFoundError, OSError, ValueError, TypeError, KeyError):
                    pass
            if model and training:
                try:
                    payload["resolved_config_signature"] = build_config_signature_from_sections(model, training)
                except (TypeError, ValueError, KeyError):
                    pass
            method_name = str(payload.get("recorded_method") or payload.get("method", ""))
            if method_name in methods:
                try:
                    payload["resolved_method_signature"] = build_method_signature(
                        normalized_method or method_name,
                        methods[method_name],
                    )
                except (TypeError, ValueError, KeyError):
                    pass
        payload["method_variant"] = method_variant_label(str(payload.get("method", "")))
        payload["protocol_display_name"] = protocol_paper_name(str(payload.get("protocol", "")))
        payload["method_display_name"] = method_paper_name(str(payload.get("method", "")))
        payload["method_variant_display_name"] = method_variant_paper_name(str(payload.get("method_variant", "")))
        if payload.get("target_labeled_ratio") is None:
            payload["target_labeled_ratio"] = _record_target_labeled_ratio(payload)
        records.append(payload)
    return records


def _record_pair_key(record: dict[str, Any], *, include_data_signature: bool) -> tuple[Any, ...]:
    key: tuple[Any, ...] = (
        record.get("protocol"),
        int(record.get("case_id")),
        str(record.get("backbone", "")),
        int(record.get("seed")),
        _record_target_labeled_ratio(record),
        str(record.get("config_signature", "")),
    )
    if include_data_signature:
        key += (str(record.get("data_signature", "")),)
    return key


def _record_target_labeled_ratio(record: dict[str, Any]) -> float | None:
    split_summary = record.get("split_summary", {})
    if isinstance(split_summary, dict) and split_summary.get("target_labeled_ratio") is not None:
        return float(split_summary["target_labeled_ratio"])
    if record.get("target_labeled_ratio") is not None:
        return float(record["target_labeled_ratio"])
    return None


def _variant_display_name(variant: str) -> str:
    return method_variant_paper_name(str(variant or ""))


def _observed_primary_variants(records: list[dict[str, Any]]) -> tuple[str, ...]:
    observed_set = {
        method_name
        for method_name in SUPPORTED_METHODS
        if any(str(record.get("method_variant", "")) == method_name for record in records)
    }
    ordered: list[str] = []
    for baseline_variant in (PRIMARY_BASELINE_VARIANT, AUXILIARY_BASELINE_VARIANT):
        if baseline_variant in observed_set:
            ordered.append(baseline_variant)
            observed_set.discard(baseline_variant)
    for method_name in SUPPORTED_METHODS:
        if method_name in observed_set:
            ordered.append(method_name)
            observed_set.discard(method_name)
    if ordered:
        return tuple(ordered)
    return (PRIMARY_BASELINE_VARIANT, AUXILIARY_BASELINE_VARIANT, *FORMAL_METHODS)


def _default_candidate_variants(
    records: list[dict[str, Any]],
    *,
    baseline_variant: str = PRIMARY_BASELINE_VARIANT,
) -> tuple[str, ...]:
    observed = _observed_primary_variants(records)
    return tuple(
        variant
        for variant in observed
        if variant not in FORMAL_BASELINE_METHODS and variant != baseline_variant
    )


def is_current_formal_record(
    record: dict[str, Any],
    *,
    data_signature: str | None = None,
    config_signature: str | None = None,
    method_signatures: dict[str, str] | None = None,
) -> bool:
    if str(record.get("study_tag", "")) != FORMAL_STUDY_TAG:
        return False
    if str(record.get("protocol", "")) not in FORMAL_PROTOCOLS:
        return False
    if str(record.get("method", "")) not in SUPPORTED_METHODS:
        return False
    if str(record.get("backbone", "")) not in FORMAL_BACKBONES:
        return False
    if data_signature is not None and str(record.get("data_signature", "")) != str(data_signature):
        return False
    if config_signature is not None and str(record.get("config_signature", "")) != str(config_signature):
        return False
    if method_signatures is not None:
        method_name = str(record.get("method", ""))
        expected_method_signature = method_signatures.get(method_name)
        if expected_method_signature is None:
            return False
        if str(record.get("method_signature", "")) != str(expected_method_signature):
            return False
    return True


def dedupe_latest(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        key = (
            record.get("protocol"),
            int(record.get("case_id")),
            str(record.get("method", "")),
            str(record.get("backbone", "")),
            int(record.get("seed")),
            str(record.get("data_signature", "")),
            str(record.get("config_signature", "")),
            str(record.get("method_signature", "")),
        )
        stamp = str(record.get("created_at", ""))
        current = latest.get(key)
        if current is None or stamp >= str(current.get("created_at", "")):
            latest[key] = record
    return sorted(
        latest.values(),
        key=lambda item: (
            item.get("protocol"),
            item.get("case_id"),
            item.get("method", ""),
            item.get("backbone", ""),
            item.get("seed"),
        ),
    )


def _baseline_index(
    records: list[dict[str, Any]],
    *,
    include_data_signature: bool,
    baseline_variant: str,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    index: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        if str(record.get("method")) != str(baseline_variant):
            continue
        key = _record_pair_key(record, include_data_signature=include_data_signature)
        current = index.get(key)
        if current is None or str(record.get("created_at", "")) >= str(current.get("created_at", "")):
            index[key] = record
    return index


def _resolve_paired_baseline(
    row: dict[str, Any],
    *,
    exact_index: dict[tuple[Any, ...], dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    paired = exact_index.get(_record_pair_key(row, include_data_signature=True))
    return paired, ("exact" if paired is not None else "unpaired")


def attach_transfer_gain(
    records: list[dict[str, Any]],
    source_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    baseline_records = records if source_records is None else source_records
    exact_baseline_index = _baseline_index(
        baseline_records,
        include_data_signature=True,
        baseline_variant=PRIMARY_BASELINE_VARIANT,
    )
    exact_aux_baseline_index = _baseline_index(
        baseline_records,
        include_data_signature=True,
        baseline_variant=AUXILIARY_BASELINE_VARIANT,
    )
    enriched = []
    for record in records:
        row = dict(record)
        current_rmse = row.get("target_test", {}).get("overall", {}).get("rmse")
        paired, pairing_mode = _resolve_paired_baseline(
            row,
            exact_index=exact_baseline_index,
        )
        baseline_rmse = None
        if paired is not None:
            baseline_rmse = paired.get("target_test", {}).get("overall", {}).get("rmse")
            row["paired_baseline_run_dir"] = paired.get("run_dir")
        aux_paired, aux_pairing_mode = _resolve_paired_baseline(
            row,
            exact_index=exact_aux_baseline_index,
        )
        auxiliary_baseline_rmse = None
        if aux_paired is not None:
            auxiliary_baseline_rmse = aux_paired.get("target_test", {}).get("overall", {}).get("rmse")
            row["paired_auxiliary_baseline_run_dir"] = aux_paired.get("run_dir")
        row["baseline_variant"] = PRIMARY_BASELINE_VARIANT
        row["baseline_pairing_mode"] = pairing_mode
        row["baseline_rmse"] = baseline_rmse
        row["transfer_gain"] = transfer_gain(baseline_rmse, current_rmse)
        row["auxiliary_baseline_variant"] = AUXILIARY_BASELINE_VARIANT
        row["auxiliary_baseline_pairing_mode"] = aux_pairing_mode
        row["auxiliary_baseline_rmse"] = auxiliary_baseline_rmse
        row["auxiliary_transfer_gain"] = transfer_gain(auxiliary_baseline_rmse, current_rmse)
        enriched.append(row)
    return enriched


def attach_negative_transfer(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for record in records:
        row = dict(record)
        gain = row.get("transfer_gain")
        row["negative_transfer"] = None if gain is None else bool(float(gain) < 0.0)
        auxiliary_gain = row.get("auxiliary_transfer_gain")
        row["auxiliary_negative_transfer"] = (
            None if auxiliary_gain is None else bool(float(auxiliary_gain) < 0.0)
        )
        enriched.append(row)
    return enriched


def find_unpaired_transfer_records(
    records: list[dict[str, Any]],
    *,
    baseline_variant: str = PRIMARY_BASELINE_VARIANT,
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if str(record.get("method", "")) != str(baseline_variant) and record.get("baseline_rmse") is None
    ]


def _raise_on_unpaired_transfer_records(records: list[dict[str, Any]]) -> None:
    missing_pairs = find_unpaired_transfer_records(records)
    if not missing_pairs:
        return
    preview = "; ".join(
        (
            f"{record.get('protocol')} case={int(record.get('case_id'))} "
            f"method={record.get('method')} seed={int(record.get('seed'))} "
            f"lr={_record_target_labeled_ratio(record)}"
        )
        for record in missing_pairs[:5]
    )
    raise ValueError(
        "Found "
        f"{len(missing_pairs)} non-baseline runs without an exact `{PRIMARY_BASELINE_VARIANT}` pair. "
        "Summaries now require exact alignment on protocol, case_id, backbone, seed, label_rate, "
        f"config_signature, and data_signature. Example unmatched runs: {preview}"
    )


def summarize_transfer_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    rmse_values = [float(record["target_test"]["overall"]["rmse"]) for record in records]
    r2_values = [
        float(record["target_test"]["overall"]["r2"])
        for record in records
        if record.get("target_test", {}).get("overall", {}).get("r2") is not None
    ]
    nrmse_values = [
        float(record["target_test"]["overall"]["nrmse"])
        for record in records
        if record.get("target_test", {}).get("overall", {}).get("nrmse") is not None
    ]
    transfer_values = [float(record["transfer_gain"]) for record in records if record.get("transfer_gain") is not None]
    negative_count = sum(1 for value in transfer_values if value < 0.0)
    win_count = sum(1 for value in transfer_values if value > 0.0)
    negative_severity_values = [-value for value in transfer_values if value < 0.0]
    mean_target_rmse = _mean_or_none(rmse_values)
    mean_target_r2 = _mean_or_none(r2_values)
    mean_target_nrmse = _mean_or_none(nrmse_values)
    mtg = _mean_or_none(transfer_values)
    ntr = (negative_count / len(transfer_values)) if transfer_values else None
    nts = (
        (sum(negative_severity_values) / len(negative_severity_values))
        if negative_severity_values
        else (0.0 if transfer_values else None)
    )
    std_target_rmse = _std_or_none(rmse_values)
    std_target_r2 = _std_or_none(r2_values)
    std_target_nrmse = _std_or_none(nrmse_values)
    std_transfer_gain = _std_or_none(transfer_values)
    return {
        "num_runs": len(records),
        "mean_target_rmse": mean_target_rmse,
        "std_target_rmse": std_target_rmse,
        "target_rmse_mean_std_text": _format_mean_std(mean_target_rmse, std_target_rmse),
        "mean_target_r2": mean_target_r2,
        "std_target_r2": std_target_r2,
        "target_r2_mean_std_text": _format_mean_std(mean_target_r2, std_target_r2),
        "mean_target_nrmse": mean_target_nrmse,
        "std_target_nrmse": std_target_nrmse,
        "target_nrmse_mean_std_text": _format_mean_std(mean_target_nrmse, std_target_nrmse),
        "mean_transfer_gain": mtg,
        "std_transfer_gain": std_transfer_gain,
        "transfer_gain_mean_std_text": _format_mean_std(mtg, std_transfer_gain),
        "win_rate_vs_baseline": (win_count / len(transfer_values)) if transfer_values else None,
        "negative_transfer_rate": ntr,
        "negative_transfer_severity": nts,
        "worst_case_transfer_gain": min(transfer_values) if transfer_values else None,
        "mtg": mtg,
        "ntr": ntr,
        "nts": nts,
    }


_SUMMARY_METRIC_FIELDS = (
    "mean_target_rmse",
    "mean_target_r2",
    "mean_target_nrmse",
    "mean_transfer_gain",
    "win_rate_vs_baseline",
    "negative_transfer_rate",
    "negative_transfer_severity",
    "worst_case_transfer_gain",
    "mtg",
    "ntr",
    "nts",
)


def _std_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) <= 1:
        return 0.0
    mean_value = sum(values) / len(values)
    variance = sum((float(value) - mean_value) ** 2 for value in values) / float(len(values) - 1)
    return math.sqrt(max(variance, 0.0))


def _format_mean_std(mean_value: float | None, std_value: float | None, *, digits: int = 4) -> str | None:
    if mean_value is None or std_value is None:
        return None
    return f"{float(mean_value):.{digits}f} +/- {float(std_value):.{digits}f}"


def build_group_summary_rows(records: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = tuple(record.get(field) for field in group_keys)
        grouped[key].append(record)
    rows = []
    for key, items in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        row = {field: value for field, value in zip(group_keys, key)}
        row.update(summarize_transfer_group(items))
        rows.append(row)
    return rows


def build_protocol_balanced_summary_rows(
    records: list[dict[str, Any]],
    group_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    protocol_rows = build_group_summary_rows(records, ("protocol", *group_keys))
    protocol_seed_rows = build_group_summary_rows(records, ("protocol", "seed", *group_keys))
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in protocol_rows:
        key = tuple(row.get(field) for field in group_keys)
        grouped[key].append(row)
    grouped_seed_rows: dict[tuple[Any, ...], dict[Any, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in protocol_seed_rows:
        key = tuple(row.get(field) for field in group_keys)
        grouped_seed_rows[key][row.get("seed")].append(row)

    rows: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        protocols = sorted(str(item.get("protocol")) for item in items if item.get("protocol") is not None)
        protocol_set = set(protocols)
        complete_seed_rows: list[dict[str, Any]] = []
        for seed, seed_items in sorted(grouped_seed_rows.get(key, {}).items(), key=lambda item: int(item[0])):
            seed_protocols = {str(item.get("protocol")) for item in seed_items if item.get("protocol") is not None}
            if seed_protocols != protocol_set:
                continue
            seed_row: dict[str, Any] = {"seed": int(seed)}
            for metric_name in _SUMMARY_METRIC_FIELDS:
                seed_row[metric_name] = _mean_or_none(
                    [float(item[metric_name]) for item in seed_items if item.get(metric_name) is not None]
                )
            complete_seed_rows.append(seed_row)
        row = {field: value for field, value in zip(group_keys, key)}
        row["num_runs"] = sum(int(item.get("num_runs", 0)) for item in items)
        row["num_protocols"] = len(protocols)
        row["protocols_covered"] = "|".join(protocols)
        row["protocol_weighting"] = "balanced_macro"
        row["dispersion_scope"] = "seed_balanced_macro" if complete_seed_rows else "protocol_balanced_macro_fallback"
        row["num_complete_seeds"] = len(complete_seed_rows)
        row["complete_seeds"] = "|".join(str(int(item["seed"])) for item in complete_seed_rows)
        metric_source = complete_seed_rows if complete_seed_rows else items
        for metric_name in _SUMMARY_METRIC_FIELDS:
            row[metric_name] = _mean_or_none(
                [float(item[metric_name]) for item in metric_source if item.get(metric_name) is not None]
            )
        row["std_target_rmse"] = _std_or_none(
            [float(item["mean_target_rmse"]) for item in metric_source if item.get("mean_target_rmse") is not None]
        )
        row["target_rmse_mean_std_text"] = _format_mean_std(row.get("mean_target_rmse"), row.get("std_target_rmse"))
        row["std_target_r2"] = _std_or_none(
            [float(item["mean_target_r2"]) for item in metric_source if item.get("mean_target_r2") is not None]
        )
        row["target_r2_mean_std_text"] = _format_mean_std(row.get("mean_target_r2"), row.get("std_target_r2"))
        row["std_target_nrmse"] = _std_or_none(
            [float(item["mean_target_nrmse"]) for item in metric_source if item.get("mean_target_nrmse") is not None]
        )
        row["target_nrmse_mean_std_text"] = _format_mean_std(
            row.get("mean_target_nrmse"),
            row.get("std_target_nrmse"),
        )
        row["std_transfer_gain"] = _std_or_none(
            [float(item["mean_transfer_gain"]) for item in metric_source if item.get("mean_transfer_gain") is not None]
        )
        row["transfer_gain_mean_std_text"] = _format_mean_std(
            row.get("mean_transfer_gain"),
            row.get("std_transfer_gain"),
        )
        rows.append(row)
    return rows


def build_seed_stability_rows(records: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    per_seed_rows = build_group_summary_rows(records, (*group_keys, "seed"))
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in per_seed_rows:
        key = tuple(row.get(field) for field in group_keys)
        grouped[key].append(row)

    rows: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        rmse_values = [float(item["mean_target_rmse"]) for item in items if item.get("mean_target_rmse") is not None]
        transfer_values = [float(item["mean_transfer_gain"]) for item in items if item.get("mean_transfer_gain") is not None]
        nrmse_values = [float(item["mean_target_nrmse"]) for item in items if item.get("mean_target_nrmse") is not None]
        r2_values = [float(item["mean_target_r2"]) for item in items if item.get("mean_target_r2") is not None]
        row = {field: value for field, value in zip(group_keys, key)}
        row["num_seeds"] = len(items)
        row["seeds_covered"] = "|".join(str(int(item.get("seed"))) for item in sorted(items, key=lambda value: int(value.get("seed", 0))))
        row["min_seed_runs"] = min(int(item.get("num_runs", 0)) for item in items) if items else 0
        row["max_seed_runs"] = max(int(item.get("num_runs", 0)) for item in items) if items else 0
        row["seed_mean_target_rmse"] = _mean_or_none(rmse_values)
        row["seed_std_target_rmse"] = _std_or_none(rmse_values)
        row["seed_target_rmse_range"] = (max(rmse_values) - min(rmse_values)) if rmse_values else None
        row["seed_target_rmse_mean_std_text"] = _format_mean_std(
            row.get("seed_mean_target_rmse"),
            row.get("seed_std_target_rmse"),
        )
        row["seed_mean_target_r2"] = _mean_or_none(r2_values)
        row["seed_std_target_r2"] = _std_or_none(r2_values)
        row["seed_target_r2_mean_std_text"] = _format_mean_std(
            row.get("seed_mean_target_r2"),
            row.get("seed_std_target_r2"),
        )
        row["seed_mean_target_nrmse"] = _mean_or_none(nrmse_values)
        row["seed_std_target_nrmse"] = _std_or_none(nrmse_values)
        row["seed_target_nrmse_mean_std_text"] = _format_mean_std(
            row.get("seed_mean_target_nrmse"),
            row.get("seed_std_target_nrmse"),
        )
        row["seed_mean_transfer_gain"] = _mean_or_none(transfer_values)
        row["seed_std_transfer_gain"] = _std_or_none(transfer_values)
        row["seed_transfer_gain_range"] = (max(transfer_values) - min(transfer_values)) if transfer_values else None
        row["seed_transfer_gain_mean_std_text"] = _format_mean_std(
            row.get("seed_mean_transfer_gain"),
            row.get("seed_std_transfer_gain"),
        )
        rows.append(row)
    return rows


def _diagnostic_value(record: dict[str, Any], name: str) -> float | None:
    value = record.get("training_diagnostics", {}).get(name)
    return None if value is None else float(value)


def _diagnostic_text(record: dict[str, Any], name: str) -> str | None:
    value = record.get("training_diagnostics", {}).get(name)
    return None if value is None else str(value)


def _mean_or_none(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _case_key_from_record(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("protocol"),
        int(record.get("case_id")),
        record.get("case_name"),
        _record_target_labeled_ratio(record),
        str(record.get("backbone", "")),
    )


def _case_key_from_row(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("protocol"),
        int(row.get("case_id")),
        row.get("case_name"),
        row.get("target_labeled_ratio"),
        str(row.get("backbone", "")),
    )


def exact_sign_test_p_value(wins: int, losses: int) -> float | None:
    wins = int(wins)
    losses = int(losses)
    if wins < 0 or losses < 0:
        raise ValueError("`wins` and `losses` must be non-negative.")
    paired = wins + losses
    if paired <= 0:
        return None
    smaller_tail = min(wins, losses)
    tail_probability = sum(math.comb(paired, k) for k in range(smaller_tail + 1)) / float(2**paired)
    return float(min(1.0, 2.0 * tail_probability))


def build_case_meta_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_case_key_from_record(record)].append(record)

    rows: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        protocol, case_id, case_name, target_labeled_ratio, backbone = key
        row = {
            "protocol": protocol,
            "protocol_display_name": protocol_paper_name(str(protocol)),
            "case_id": int(case_id),
            "case_name": case_name,
            "target_labeled_ratio": target_labeled_ratio,
            "backbone": backbone,
            "num_records": len(items),
        }
        for name in (
            "source_train_shift_score",
            "target_adapt_shift_score",
            "target_adapt_train_raw_shift_score",
            "target_adapt_train_shift_score",
            "target_labeled_shift_score",
            "target_unlabeled_shift_score",
            "target_val_shift_score",
            "target_test_shift_score",
        ):
            row[f"mean_{name}"] = _mean_or_none(
                [
                    float(item.get("split_summary", {}).get(name))
                    for item in items
                    if item.get("split_summary", {}).get(name) is not None
                ]
            )
        rows.append(row)
    return rows


def _annotate_shift_severity(
    rows: list[dict[str, Any]],
    case_meta_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped_meta: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in case_meta_rows:
        grouped_meta[
            (
                row.get("protocol"),
                row.get("target_labeled_ratio"),
                row.get("backbone"),
            )
        ].append(row)

    severity_index: dict[tuple[Any, ...], dict[str, Any]] = {}
    labels = ("low", "medium", "high")
    for items in grouped_meta.values():
        ordered = sorted(
            items,
            key=lambda row: (
                float(row.get("mean_target_adapt_shift_score") or float("inf")),
                int(row.get("case_id", 0)),
            ),
        )
        total = max(len(ordered), 1)
        for rank, row in enumerate(ordered, start=1):
            label_index = min(int(((rank - 1) * 3) / total), 2)
            severity_index[_case_key_from_row(row)] = {
                "shift_severity": labels[label_index],
                "shift_severity_rank": int(rank),
                "shift_severity_group_size": int(total),
                "mean_target_adapt_shift_score": row.get("mean_target_adapt_shift_score"),
                "mean_target_test_shift_score": row.get("mean_target_test_shift_score"),
            }

    annotated: list[dict[str, Any]] = []
    for row in rows:
        tagged = dict(row)
        severity = severity_index.get(_case_key_from_row(row))
        if severity is None:
            tagged["shift_severity"] = "unknown"
            tagged["shift_severity_rank"] = None
            tagged["shift_severity_group_size"] = None
        else:
            tagged.update(severity)
        annotated.append(tagged)
    return annotated


def build_shift_safety_rows(records: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = tuple(record.get(field) for field in group_keys)
        grouped[key].append(record)
    rows: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        row = {field: value for field, value in zip(group_keys, key)}
        row["protocol_display_name"] = protocol_paper_name(str(row.get("protocol", "")))
        row["method_variant_display_name"] = _variant_display_name(str(row.get("method_variant", "")))
        row["num_runs"] = len(items)
        row["mean_target_rmse"] = _mean_or_none(
            [float(item["target_test"]["overall"]["rmse"]) for item in items if item.get("target_test", {}).get("overall", {}).get("rmse") is not None]
        )
        row["mean_transfer_gain"] = _mean_or_none(
            [float(item["transfer_gain"]) for item in items if item.get("transfer_gain") is not None]
        )
        row["negative_transfer_rate"] = _mean_or_none(
            [
                float(bool(item.get("negative_transfer")))
                for item in items
                if item.get("negative_transfer") is not None
            ]
        )
        row["negative_transfer_severity"] = _mean_or_none(
            [
                -float(item["transfer_gain"])
                for item in items
                if item.get("transfer_gain") is not None and float(item["transfer_gain"]) < 0.0
            ]
        )
        transfer_values = [float(item["transfer_gain"]) for item in items if item.get("transfer_gain") is not None]
        row["worst_case_transfer_gain"] = min(transfer_values) if transfer_values else None
        for name in (
            "source_train_shift_score",
            "target_adapt_shift_score",
            "target_labeled_shift_score",
            "target_unlabeled_shift_score",
            "target_val_shift_score",
            "target_test_shift_score",
        ):
            row[f"mean_{name}"] = _mean_or_none(
                [
                    float(item.get("split_summary", {}).get(name))
                    for item in items
                    if item.get("split_summary", {}).get(name) is not None
                ]
            )
        for name in (
            "shift_gate_mean",
            "shift_score_mean",
            "guard_keep_fraction",
            "safe_lambda_applied",
            "safe_mode_code",
            "best_target_val_rmse",
            "safe_bad_epochs",
        ):
            row[f"mean_{name}"] = _mean_or_none(
                [float(item.get("training_diagnostics", {}).get(name)) for item in items if item.get("training_diagnostics", {}).get(name) is not None]
            )
        safe_states = [_diagnostic_text(item, "safe_mode_state") for item in items if _diagnostic_text(item, "safe_mode_state") is not None]
        row["fallback_rate"] = (
            sum(1 for state in safe_states if state == "fallback") / len(safe_states)
            if safe_states
            else None
        )
        row["active_rate"] = (
            sum(1 for state in safe_states if state == "active") / len(safe_states)
            if safe_states
            else None
        )
        rows.append(row)
    return rows


def build_case_summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    case_rows = build_shift_safety_rows(
        records,
        ("protocol", "case_id", "case_name", "target_labeled_ratio", "method_variant", "backbone"),
    )
    return _annotate_shift_severity(case_rows, build_case_meta_rows(records))


def build_case_level_significance_rows(
    records: list[dict[str, Any]],
    candidate_variants: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    candidate_variants = _default_candidate_variants(records) if candidate_variants is None else candidate_variants
    case_rows = build_case_summary_rows(records)
    baseline_index = {
        _case_key_from_row(row): row
        for row in case_rows
        if str(row.get("method_variant", "")) == PRIMARY_BASELINE_VARIANT
    }
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        variant = str(row.get("method_variant", ""))
        if variant not in candidate_variants:
            continue
        baseline = baseline_index.get(_case_key_from_row(row))
        candidate_rmse = row.get("mean_target_rmse")
        baseline_rmse = None if baseline is None else baseline.get("mean_target_rmse")
        if candidate_rmse is None or baseline_rmse is None:
            continue
        grouped[
            (
                row.get("protocol"),
                row.get("target_labeled_ratio"),
                row.get("backbone"),
                variant,
            )
        ].append(
            {
                "case_id": int(row.get("case_id")),
                "case_name": row.get("case_name"),
                "candidate_rmse": float(candidate_rmse),
                "baseline_rmse": float(baseline_rmse),
                "delta_rmse": float(candidate_rmse) - float(baseline_rmse),
                "candidate_transfer_gain": row.get("mean_transfer_gain"),
            }
        )

    rows: list[dict[str, Any]] = []
    for key, pairs in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        protocol, target_labeled_ratio, backbone, candidate_variant = key
        deltas = [float(item["delta_rmse"]) for item in pairs]
        transfer_values = [
            float(item["candidate_transfer_gain"])
            for item in pairs
            if item.get("candidate_transfer_gain") is not None
        ]
        wins = sum(1 for value in deltas if value < 0.0)
        losses = sum(1 for value in deltas if value > 0.0)
        ties = sum(1 for value in deltas if abs(value) <= 1e-12)
        rows.append(
            {
                "protocol": protocol,
                "protocol_display_name": protocol_paper_name(str(protocol)),
                "target_labeled_ratio": target_labeled_ratio,
                "backbone": backbone,
                "baseline_variant": PRIMARY_BASELINE_VARIANT,
                "baseline_variant_display_name": _variant_display_name(PRIMARY_BASELINE_VARIANT),
                "candidate_variant": candidate_variant,
                "candidate_variant_display_name": _variant_display_name(candidate_variant),
                "n_pairs": len(pairs),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "sign_test_p_value": exact_sign_test_p_value(wins, losses),
                "baseline_mean_target_rmse": _mean_or_none([float(item["baseline_rmse"]) for item in pairs]),
                "candidate_mean_target_rmse": _mean_or_none([float(item["candidate_rmse"]) for item in pairs]),
                "mean_delta_rmse": _mean_or_none(deltas),
                "median_delta_rmse": _median_or_none(deltas),
                "candidate_mean_transfer_gain": _mean_or_none(transfer_values),
                "candidate_case_negative_transfer_rate": (
                    sum(1 for value in transfer_values if value < 0.0) / len(transfer_values)
                    if transfer_values
                    else None
                ),
                "candidate_worst_case_transfer_gain": min(transfer_values) if transfer_values else None,
            }
        )
    return rows


def build_shift_severity_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    case_rows = build_case_summary_rows(records)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        grouped[
            (
                row.get("protocol"),
                row.get("target_labeled_ratio"),
                row.get("backbone"),
                row.get("shift_severity"),
                row.get("method_variant"),
            )
        ].append(row)

    rows: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        protocol, target_labeled_ratio, backbone, shift_severity, method_variant = key
        rmse_values = [float(item["mean_target_rmse"]) for item in items if item.get("mean_target_rmse") is not None]
        transfer_values = [float(item["mean_transfer_gain"]) for item in items if item.get("mean_transfer_gain") is not None]
        rows.append(
            {
                "protocol": protocol,
                "protocol_display_name": protocol_paper_name(str(protocol)),
                "target_labeled_ratio": target_labeled_ratio,
                "backbone": backbone,
                "shift_severity": shift_severity,
                "method_variant": method_variant,
                "method_variant_display_name": _variant_display_name(str(method_variant)),
                "num_cases": len(items),
                "mean_target_rmse": _mean_or_none(rmse_values),
                "mean_transfer_gain": _mean_or_none(transfer_values),
                "negative_transfer_rate": (
                    sum(1 for value in transfer_values if value < 0.0) / len(transfer_values)
                    if transfer_values
                    else None
                ),
                "negative_transfer_severity": _mean_or_none([-value for value in transfer_values if value < 0.0]),
                "worst_case_transfer_gain": min(transfer_values) if transfer_values else None,
                "mean_target_adapt_shift_score": _mean_or_none(
                    [float(item["mean_target_adapt_shift_score"]) for item in items if item.get("mean_target_adapt_shift_score") is not None]
                ),
                "mean_target_test_shift_score": _mean_or_none(
                    [float(item["mean_target_test_shift_score"]) for item in items if item.get("mean_target_test_shift_score") is not None]
                ),
                "mean_fallback_rate": _mean_or_none(
                    [float(item["fallback_rate"]) for item in items if item.get("fallback_rate") is not None]
                ),
                "mean_active_rate": _mean_or_none(
                    [float(item["active_rate"]) for item in items if item.get("active_rate") is not None]
                ),
            }
        )
    return rows


def build_shift_severity_comparison_rows(
    records: list[dict[str, Any]],
    candidate_variants: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    candidate_variants = _default_candidate_variants(records) if candidate_variants is None else candidate_variants
    case_rows = build_case_summary_rows(records)
    baseline_index = {
        (_case_key_from_row(row), row.get("shift_severity")): row
        for row in case_rows
        if str(row.get("method_variant", "")) == PRIMARY_BASELINE_VARIANT
    }
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        variant = str(row.get("method_variant", ""))
        if variant not in candidate_variants:
            continue
        baseline = baseline_index.get((_case_key_from_row(row), row.get("shift_severity")))
        candidate_rmse = row.get("mean_target_rmse")
        baseline_rmse = None if baseline is None else baseline.get("mean_target_rmse")
        if candidate_rmse is None or baseline_rmse is None:
            continue
        grouped[
            (
                row.get("protocol"),
                row.get("target_labeled_ratio"),
                row.get("backbone"),
                row.get("shift_severity"),
                variant,
            )
        ].append(
            {
                "candidate_rmse": float(candidate_rmse),
                "baseline_rmse": float(baseline_rmse),
                "delta_rmse": float(candidate_rmse) - float(baseline_rmse),
                "candidate_transfer_gain": row.get("mean_transfer_gain"),
            }
        )

    rows: list[dict[str, Any]] = []
    for key, pairs in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        protocol, target_labeled_ratio, backbone, shift_severity, candidate_variant = key
        deltas = [float(item["delta_rmse"]) for item in pairs]
        wins = sum(1 for value in deltas if value < 0.0)
        losses = sum(1 for value in deltas if value > 0.0)
        transfer_values = [
            float(item["candidate_transfer_gain"])
            for item in pairs
            if item.get("candidate_transfer_gain") is not None
        ]
        rows.append(
            {
                "protocol": protocol,
                "protocol_display_name": protocol_paper_name(str(protocol)),
                "target_labeled_ratio": target_labeled_ratio,
                "backbone": backbone,
                "shift_severity": shift_severity,
                "baseline_variant": PRIMARY_BASELINE_VARIANT,
                "baseline_variant_display_name": _variant_display_name(PRIMARY_BASELINE_VARIANT),
                "candidate_variant": candidate_variant,
                "candidate_variant_display_name": _variant_display_name(candidate_variant),
                "n_pairs": len(pairs),
                "wins": wins,
                "losses": losses,
                "ties": sum(1 for value in deltas if abs(value) <= 1e-12),
                "sign_test_p_value": exact_sign_test_p_value(wins, losses),
                "mean_delta_rmse": _mean_or_none(deltas),
                "median_delta_rmse": _median_or_none(deltas),
                "candidate_mean_transfer_gain": _mean_or_none(transfer_values),
                "candidate_worst_case_transfer_gain": min(transfer_values) if transfer_values else None,
            }
        )
    return rows


def build_failure_case_rows(
    records: list[dict[str, Any]],
    culprit_variants: tuple[str, ...] = ("ss_dann", "ss_mt", "ss_coral"),
    rescue_variant: str | None = None,
) -> list[dict[str, Any]]:
    if rescue_variant is None:
        candidate_variants = _default_candidate_variants(records)
        if not candidate_variants:
            return []
        rescue_variant = candidate_variants[-1]
    else:
        rescue_variant = str(rescue_variant)
    case_rows = build_case_summary_rows(records)
    baseline_index = {
        _case_key_from_row(row): row
        for row in case_rows
        if str(row.get("method_variant", "")) == PRIMARY_BASELINE_VARIANT
    }
    rescue_index = {
        _case_key_from_row(row): row
        for row in case_rows
        if str(row.get("method_variant", "")) == str(rescue_variant)
    }

    rows: list[dict[str, Any]] = []
    for row in case_rows:
        culprit_variant = str(row.get("method_variant", ""))
        culprit_gain = row.get("mean_transfer_gain")
        if culprit_variant not in culprit_variants or culprit_gain is None or float(culprit_gain) >= 0.0:
            continue
        baseline = baseline_index.get(_case_key_from_row(row))
        rescue = rescue_index.get(_case_key_from_row(row))
        culprit_rmse = row.get("mean_target_rmse")
        baseline_rmse = None if baseline is None else baseline.get("mean_target_rmse")
        rescue_rmse = None if rescue is None else rescue.get("mean_target_rmse")
        rescue_gain = None if rescue is None else rescue.get("mean_transfer_gain")
        rescue_outperformed = (
            rescue_rmse is not None and culprit_rmse is not None and float(rescue_rmse) < float(culprit_rmse)
        )
        rescue_non_negative = rescue_gain is not None and float(rescue_gain) >= 0.0
        rows.append(
            {
                "protocol": row.get("protocol"),
                "protocol_display_name": protocol_paper_name(str(row.get("protocol", ""))),
                "case_id": int(row.get("case_id")),
                "case_name": row.get("case_name"),
                "target_labeled_ratio": row.get("target_labeled_ratio"),
                "backbone": row.get("backbone"),
                "shift_severity": row.get("shift_severity"),
                "mean_target_adapt_shift_score": row.get("mean_target_adapt_shift_score"),
                "culprit_variant": culprit_variant,
                "culprit_variant_display_name": _variant_display_name(culprit_variant),
                "culprit_mean_target_rmse": culprit_rmse,
                "culprit_mean_transfer_gain": culprit_gain,
                "culprit_negative_transfer_rate": row.get("negative_transfer_rate"),
                "culprit_worst_case_transfer_gain": row.get("worst_case_transfer_gain"),
                "baseline_mean_target_rmse": baseline_rmse,
                "rescue_variant": rescue_variant,
                "rescue_variant_display_name": _variant_display_name(rescue_variant),
                "rescue_mean_target_rmse": rescue_rmse,
                "rescue_mean_transfer_gain": rescue_gain,
                "rescue_negative_transfer_rate": None if rescue is None else rescue.get("negative_transfer_rate"),
                "rescue_worst_case_transfer_gain": None if rescue is None else rescue.get("worst_case_transfer_gain"),
                "rescue_fallback_rate": None if rescue is None else rescue.get("fallback_rate"),
                "rescue_active_rate": None if rescue is None else rescue.get("active_rate"),
                "rescue_outperformed_culprit": bool(rescue_outperformed),
                "rescue_non_negative_transfer": bool(rescue_non_negative),
                "rescue_avoided_worse_result": bool(rescue_outperformed and rescue_non_negative),
                "rescue_any_fallback": bool(rescue is not None and (rescue.get("fallback_rate") or 0.0) > 0.0),
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            float(row.get("culprit_mean_transfer_gain", 0.0)),
            str(row.get("protocol", "")),
            int(row.get("case_id", 0)),
        ),
    )


def _write_results_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "protocol",
        "protocol_display_name",
        "case_id",
        "case_name",
        "method",
        "method_display_name",
        "method_variant",
        "method_variant_display_name",
        "backbone",
        "seed",
        "target_labeled_ratio",
        "target_labeled_sampling_seed",
        "target_labeled_sampling_seed_policy",
        "target_labeled_sampling_strategy",
        "target_labeled_strata",
        "target_labeled_sampling_scope",
        "target_labeled_sampling_unit",
        "target_adapt_points",
        "target_labeled_points",
        "target_unlabeled_points",
        "source_train_shift_score",
        "target_adapt_shift_score",
        "target_labeled_shift_score",
        "target_unlabeled_shift_score",
        "target_val_shift_score",
        "target_test_shift_score",
        "data_signature",
        "config_signature",
        "method_signature",
        "shift_score_space",
        "shift_score_reference",
        "shift_severity_basis",
        "source_val_rmse",
        "target_test_rmse",
        "target_test_r2",
        "target_test_nrmse",
        "baseline_variant",
        "baseline_rmse",
        "transfer_gain",
        "negative_transfer",
        "auxiliary_baseline_variant",
        "auxiliary_baseline_rmse",
        "auxiliary_transfer_gain",
        "auxiliary_negative_transfer",
        "domain_loss",
        "coral_loss",
        "joint_alignment_loss",
        "spectral_alignment_loss",
        "safe_pseudo_loss",
        "safe_accept_rate",
        "safe_threshold",
        "consistency_keep_fraction",
        "mean_anchor_gap",
        "mean_safe_risk",
        "anchor_fallback_rate",
        "shift_gate_mean",
        "shift_score_mean",
        "guard_keep_fraction",
        "safe_lambda_applied",
        "safe_mode_code",
        "safe_mode_state",
        "safe_bad_epochs",
        "best_target_val_rmse",
        "mean_transfer_score",
        "transfer_weight_entropy",
        "max_transfer_weight",
        "run_dir",
        "paired_baseline_run_dir",
        "baseline_pairing_mode",
        "paired_auxiliary_baseline_run_dir",
        "auxiliary_baseline_pairing_mode",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "protocol": record.get("protocol"),
                    "protocol_display_name": record.get("protocol_display_name", ""),
                    "case_id": int(record.get("case_id")),
                    "case_name": record.get("case_name"),
                    "method": record.get("method", ""),
                    "method_display_name": record.get("method_display_name", ""),
                    "method_variant": record.get("method_variant", ""),
                    "method_variant_display_name": record.get("method_variant_display_name", ""),
                    "backbone": record.get("backbone", ""),
                    "seed": int(record.get("seed")),
                    "target_labeled_ratio": _record_target_labeled_ratio(record),
                    "target_labeled_sampling_seed": record.get("split_summary", {}).get("target_labeled_sampling_seed"),
                    "target_labeled_sampling_seed_policy": record.get("split_summary", {}).get(
                        "target_labeled_sampling_seed_policy"
                    ),
                    "target_labeled_sampling_strategy": record.get("split_summary", {}).get(
                        "target_labeled_sampling_strategy"
                    ),
                    "target_labeled_strata": record.get("split_summary", {}).get("target_labeled_strata"),
                    "target_labeled_sampling_scope": record.get("split_summary", {}).get("target_labeled_sampling_scope"),
                    "target_labeled_sampling_unit": record.get("split_summary", {}).get("target_labeled_sampling_unit"),
                    "target_adapt_points": record.get("split_summary", {}).get("target_adapt_points"),
                    "target_labeled_points": record.get("split_summary", {}).get("target_labeled_points"),
                    "target_unlabeled_points": record.get("split_summary", {}).get("target_unlabeled_points"),
                    "source_train_shift_score": record.get("split_summary", {}).get("source_train_shift_score"),
                    "target_adapt_shift_score": record.get("split_summary", {}).get("target_adapt_shift_score"),
                    "target_labeled_shift_score": record.get("split_summary", {}).get("target_labeled_shift_score"),
                    "target_unlabeled_shift_score": record.get("split_summary", {}).get("target_unlabeled_shift_score"),
                    "target_val_shift_score": record.get("split_summary", {}).get("target_val_shift_score"),
                    "target_test_shift_score": record.get("split_summary", {}).get("target_test_shift_score"),
                    "data_signature": record.get("data_signature", ""),
                    "config_signature": record.get("config_signature", ""),
                    "method_signature": record.get("method_signature", ""),
                    "shift_score_space": record.get("split_summary", {}).get("shift_score_space"),
                    "shift_score_reference": record.get("split_summary", {}).get("shift_score_reference"),
                    "shift_severity_basis": record.get("split_summary", {}).get("shift_severity_basis"),
                    "source_val_rmse": record.get("source_val", {}).get("overall", {}).get("rmse"),
                    "target_test_rmse": record.get("target_test", {}).get("overall", {}).get("rmse"),
                    "target_test_r2": record.get("target_test", {}).get("overall", {}).get("r2"),
                    "target_test_nrmse": record.get("target_test", {}).get("overall", {}).get("nrmse"),
                    "baseline_variant": record.get("baseline_variant", PRIMARY_BASELINE_VARIANT),
                    "baseline_rmse": record.get("baseline_rmse"),
                    "transfer_gain": record.get("transfer_gain"),
                    "negative_transfer": record.get("negative_transfer"),
                    "auxiliary_baseline_variant": record.get(
                        "auxiliary_baseline_variant",
                        AUXILIARY_BASELINE_VARIANT,
                    ),
                    "auxiliary_baseline_rmse": record.get("auxiliary_baseline_rmse"),
                    "auxiliary_transfer_gain": record.get("auxiliary_transfer_gain"),
                    "auxiliary_negative_transfer": record.get("auxiliary_negative_transfer"),
                    "domain_loss": _diagnostic_value(record, "domain_loss"),
                    "coral_loss": _diagnostic_value(record, "coral_loss"),
                    "joint_alignment_loss": _diagnostic_value(record, "joint_alignment_loss"),
                    "spectral_alignment_loss": _diagnostic_value(record, "spectral_alignment_loss"),
                    "safe_pseudo_loss": _diagnostic_value(record, "safe_pseudo_loss"),
                    "safe_accept_rate": _diagnostic_value(record, "safe_accept_rate"),
                    "safe_threshold": _diagnostic_value(record, "safe_threshold"),
                    "consistency_keep_fraction": _diagnostic_value(record, "consistency_keep_fraction"),
                    "mean_anchor_gap": _diagnostic_value(record, "mean_anchor_gap"),
                    "mean_safe_risk": _diagnostic_value(record, "mean_safe_risk"),
                    "anchor_fallback_rate": _diagnostic_value(record, "anchor_fallback_rate"),
                    "shift_gate_mean": _diagnostic_value(record, "shift_gate_mean"),
                    "shift_score_mean": _diagnostic_value(record, "shift_score_mean"),
                    "guard_keep_fraction": _diagnostic_value(record, "guard_keep_fraction"),
                    "safe_lambda_applied": _diagnostic_value(record, "safe_lambda_applied"),
                    "safe_mode_code": _diagnostic_value(record, "safe_mode_code"),
                    "safe_mode_state": _diagnostic_text(record, "safe_mode_state"),
                    "safe_bad_epochs": _diagnostic_value(record, "safe_bad_epochs"),
                    "best_target_val_rmse": _diagnostic_value(record, "best_target_val_rmse"),
                    "mean_transfer_score": _diagnostic_value(record, "mean_transfer_score"),
                    "transfer_weight_entropy": _diagnostic_value(record, "transfer_weight_entropy"),
                    "max_transfer_weight": _diagnostic_value(record, "max_transfer_weight"),
                    "run_dir": record.get("run_dir"),
                    "paired_baseline_run_dir": record.get("paired_baseline_run_dir", ""),
                    "baseline_pairing_mode": record.get("baseline_pairing_mode", ""),
                    "paired_auxiliary_baseline_run_dir": record.get("paired_auxiliary_baseline_run_dir", ""),
                    "auxiliary_baseline_pairing_mode": record.get("auxiliary_baseline_pairing_mode", ""),
                }
            )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _protocol_table(records: list[dict[str, Any]], protocol: str) -> list[dict[str, Any]]:
    rows = []
    for row in build_group_summary_rows(
        [record for record in records if record.get("protocol") == protocol],
        ("target_labeled_ratio", "method_variant", "backbone"),
    ):
        rows.append(
            {
                "protocol": protocol,
                "protocol_display_name": protocol_paper_name(protocol),
                "target_labeled_ratio": row.get("target_labeled_ratio"),
                "method_variant": row.get("method_variant"),
                "method_variant_display_name": _variant_display_name(str(row.get("method_variant", ""))),
                "backbone": row.get("backbone"),
                "num_runs": row.get("num_runs"),
                "mean_target_rmse": row.get("mean_target_rmse"),
                "mean_target_r2": row.get("mean_target_r2"),
                "mean_target_nrmse": row.get("mean_target_nrmse"),
                "mean_transfer_gain": row.get("mean_transfer_gain"),
                "win_rate_vs_baseline": row.get("win_rate_vs_baseline"),
                "negative_transfer_rate": row.get("negative_transfer_rate"),
                "worst_case_transfer_gain": row.get("worst_case_transfer_gain"),
            }
        )
    return rows


def build_conclusion_gate(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_protocol_ratio_backbone: dict[str, dict[float | None, dict[str, Any]]] = {}
    tracked_variants = _observed_primary_variants(records)
    baseline_variant = PRIMARY_BASELINE_VARIANT
    candidate_variants = _default_candidate_variants(records, baseline_variant=baseline_variant)
    observed_backbones = sorted(
        {
            str(record.get("backbone"))
            for record in records
            if str(record.get("backbone")) in FORMAL_BACKBONES
        }
    )
    if not observed_backbones:
        observed_backbones = list(FORMAL_BACKBONES)
    observed_label_rates = sorted(
        {
            float(value)
            for value in (_record_target_labeled_ratio(record) for record in records)
            if value is not None
        }
    )
    if not observed_label_rates:
        observed_label_rates = [None]
    for protocol in FORMAL_MAIN_PROTOCOLS:
        by_protocol_ratio_backbone[protocol] = {}
        for label_rate in observed_label_rates:
            by_protocol_ratio_backbone[protocol][label_rate] = {}
            for backbone in observed_backbones:
                subset = [
                    record
                    for record in records
                    if record.get("protocol") == protocol
                    and record.get("backbone") == backbone
                    and _record_target_labeled_ratio(record) == label_rate
                ]
                variant_summary = {}
                for variant in tracked_variants:
                    variant_records = [record for record in subset if record.get("method_variant") == variant]
                    variant_summary[variant] = summarize_transfer_group(variant_records)
                by_protocol_ratio_backbone[protocol][label_rate][backbone] = variant_summary

    missing_pairs = find_unpaired_transfer_records(records)
    coverage_ok = all(
        by_protocol_ratio_backbone[protocol][label_rate][backbone][variant]["num_runs"] > 0
        for protocol in FORMAL_MAIN_PROTOCOLS
        for label_rate in observed_label_rates
        for backbone in observed_backbones
        for variant in tracked_variants
    )
    candidate_method_gates: dict[str, dict[str, bool]] = {}
    for variant in candidate_variants:
        candidate_method_gates[variant] = {
            "has_non_negative_mean_transfer_everywhere": all(
                by_protocol_ratio_backbone[protocol][label_rate][backbone][variant]["mean_transfer_gain"] is not None
                and by_protocol_ratio_backbone[protocol][label_rate][backbone][variant]["mean_transfer_gain"] >= 0.0
                for protocol in FORMAL_MAIN_PROTOCOLS
                for label_rate in observed_label_rates
                for backbone in observed_backbones
            ),
            "outperforms_baseline_on_mean_transfer_everywhere": all(
                by_protocol_ratio_backbone[protocol][label_rate][backbone][variant]["mean_transfer_gain"] is not None
                and by_protocol_ratio_backbone[protocol][label_rate][backbone][baseline_variant]["mean_transfer_gain"]
                is not None
                and by_protocol_ratio_backbone[protocol][label_rate][backbone][variant]["mean_transfer_gain"]
                >= by_protocol_ratio_backbone[protocol][label_rate][backbone][baseline_variant]["mean_transfer_gain"]
                for protocol in FORMAL_MAIN_PROTOCOLS
                for label_rate in observed_label_rates
                for backbone in observed_backbones
            ),
            "reduces_negative_transfer_vs_baseline_everywhere": all(
                by_protocol_ratio_backbone[protocol][label_rate][backbone][variant]["negative_transfer_rate"] is not None
                and by_protocol_ratio_backbone[protocol][label_rate][backbone][baseline_variant][
                    "negative_transfer_rate"
                ]
                is not None
                and by_protocol_ratio_backbone[protocol][label_rate][backbone][variant]["negative_transfer_rate"]
                <= by_protocol_ratio_backbone[protocol][label_rate][backbone][baseline_variant][
                    "negative_transfer_rate"
                ]
                for protocol in FORMAL_MAIN_PROTOCOLS
                for label_rate in observed_label_rates
                for backbone in observed_backbones
            ),
            "has_non_negative_worst_case_transfer_everywhere": all(
                by_protocol_ratio_backbone[protocol][label_rate][backbone][variant]["worst_case_transfer_gain"] is not None
                and by_protocol_ratio_backbone[protocol][label_rate][backbone][variant]["worst_case_transfer_gain"] >= 0.0
                for protocol in FORMAL_MAIN_PROTOCOLS
                for label_rate in observed_label_rates
                for backbone in observed_backbones
            ),
            "improves_worst_case_transfer_vs_baseline_everywhere": all(
                by_protocol_ratio_backbone[protocol][label_rate][backbone][variant]["worst_case_transfer_gain"] is not None
                and by_protocol_ratio_backbone[protocol][label_rate][backbone][baseline_variant][
                    "worst_case_transfer_gain"
                ]
                is not None
                and by_protocol_ratio_backbone[protocol][label_rate][backbone][variant]["worst_case_transfer_gain"]
                >= by_protocol_ratio_backbone[protocol][label_rate][backbone][baseline_variant][
                    "worst_case_transfer_gain"
                ]
                for protocol in FORMAL_MAIN_PROTOCOLS
                for label_rate in observed_label_rates
                for backbone in observed_backbones
            ),
        }
    return {
        "generated_at": datetime.now().isoformat(),
        "num_records": len(records),
        "missing_baseline_pairs": len(missing_pairs),
        "label_rates": observed_label_rates,
        "tracked_variants": list(tracked_variants),
        "candidate_methods": list(candidate_variants),
        "gates": {
            "all_non_baseline_methods_paired": len(missing_pairs) == 0,
            "all_main_protocol_backbone_cells_have_all_variants": coverage_ok,
            "any_candidate_has_non_negative_mean_transfer_everywhere": any(
                gate["has_non_negative_mean_transfer_everywhere"] for gate in candidate_method_gates.values()
            ),
            "any_candidate_outperforms_baseline_on_mean_transfer_everywhere": any(
                gate["outperforms_baseline_on_mean_transfer_everywhere"] for gate in candidate_method_gates.values()
            ),
            "any_candidate_reduces_negative_transfer_vs_baseline_everywhere": any(
                gate["reduces_negative_transfer_vs_baseline_everywhere"] for gate in candidate_method_gates.values()
            ),
            "any_candidate_has_non_negative_worst_case_transfer_everywhere": any(
                gate["has_non_negative_worst_case_transfer_everywhere"] for gate in candidate_method_gates.values()
            ),
            "any_candidate_improves_worst_case_transfer_vs_baseline_everywhere": any(
                gate["improves_worst_case_transfer_vs_baseline_everywhere"] for gate in candidate_method_gates.values()
            ),
        },
        "candidate_method_gates": candidate_method_gates,
        "by_protocol_ratio_backbone": by_protocol_ratio_backbone,
    }


def build_paper_tables(records: list[dict[str, Any]]) -> dict[str, Any]:
    protocol_tables = {protocol: _protocol_table(records, protocol) for protocol in FORMAL_MAIN_PROTOCOLS}
    summary_by_method_backbone = build_protocol_balanced_summary_rows(
        records,
        ("target_labeled_ratio", "method_variant", "backbone"),
    )
    summary_by_method_backbone_naive = build_group_summary_rows(
        records,
        ("target_labeled_ratio", "method_variant", "backbone"),
    )
    primary_transfer_summary = build_group_summary_rows(
        records,
        ("protocol", "target_labeled_ratio", "method_variant", "backbone"),
    )
    seed_stability_summary = build_seed_stability_rows(
        records,
        ("protocol", "target_labeled_ratio", "method_variant", "backbone"),
    )
    return {
        "generated_at": datetime.now().isoformat(),
        "protocol_tables": protocol_tables,
        "summary_by_method_backbone": summary_by_method_backbone,
        "summary_by_method_backbone_naive": summary_by_method_backbone_naive,
        "primary_transfer_summary": primary_transfer_summary,
        "seed_stability_summary": seed_stability_summary,
        "case_level_significance": build_case_level_significance_rows(records),
        "shift_severity_summary": build_shift_severity_rows(records),
        "shift_severity_comparison": build_shift_severity_comparison_rows(records),
        "failure_case_highlights": build_failure_case_rows(records)[:3],
    }


def summarize_runs(config: FormalConfig, protocol: str | None = None) -> dict[str, Any]:
    summary_root = Path(config.paths.summary_root)
    expected_data_signatures: set[str] = set()
    for label_rate in config.data.target_labeled_ratios:
        config_for_rate = replace(config, data=replace(config.data, target_labeled_ratio=float(label_rate)))
        expected_data_signatures.add(build_formal_signatures(config_for_rate)["data_signature"])
    expected_config_signature = build_formal_signatures(config)["config_signature"]
    expected_method_signatures = {
        method_name: build_method_signature(method_name, method_cfg)
        for method_name, method_cfg in config.methods.items()
    }
    records = [
        record
        for record in collect_run_records(config.paths.run_root)
        if is_current_formal_record(
            record,
            config_signature=expected_config_signature,
            method_signatures=expected_method_signatures,
        )
        and str(record.get("data_signature", "")) in expected_data_signatures
    ]
    deduped = dedupe_latest(records)
    enriched = attach_negative_transfer(attach_transfer_gain(deduped))
    if protocol is not None:
        protocol_norm = PROTOCOL_ALIASES.get(str(protocol).strip().upper(), str(protocol).strip().upper())
        enriched = [record for record in enriched if record.get("protocol") == protocol_norm]
    _raise_on_unpaired_transfer_records(enriched)

    summary_root.mkdir(parents=True, exist_ok=True)
    results_csv = summary_root / "results.csv"
    _write_results_csv(results_csv, enriched)

    protocol_tables = {}
    table_protocols = (
        FORMAL_MAIN_PROTOCOLS
        if protocol is None
        else (PROTOCOL_ALIASES.get(str(protocol).strip().upper(), str(protocol).strip().upper()),)
    )
    for protocol_name in table_protocols:
        rows = _protocol_table(enriched, protocol_name)
        table_path = summary_root / f"protocol_{protocol_name}_table.csv"
        _write_csv(
            table_path,
            rows,
            [
                "protocol",
                "protocol_display_name",
                "target_labeled_ratio",
                "method_variant",
                "method_variant_display_name",
                "backbone",
                "num_runs",
                "mean_target_rmse",
                "std_target_rmse",
                "target_rmse_mean_std_text",
                "mean_target_r2",
                "std_target_r2",
                "target_r2_mean_std_text",
                "mean_target_nrmse",
                "std_target_nrmse",
                "target_nrmse_mean_std_text",
                "mean_transfer_gain",
                "std_transfer_gain",
                "transfer_gain_mean_std_text",
                "win_rate_vs_baseline",
                "negative_transfer_rate",
                "worst_case_transfer_gain",
            ],
        )
        protocol_tables[protocol_name] = str(table_path.resolve())

    primary_transfer_rows = build_group_summary_rows(
        enriched,
        ("protocol", "target_labeled_ratio", "method_variant", "backbone"),
    )
    for row in primary_transfer_rows:
        row["protocol_display_name"] = protocol_paper_name(str(row.get("protocol", "")))
        row["method_variant_display_name"] = _variant_display_name(str(row.get("method_variant", "")))
    primary_transfer_summary_path = summary_root / "primary_transfer_summary.csv"
    _write_csv(
        primary_transfer_summary_path,
        primary_transfer_rows,
        [
            "protocol",
            "protocol_display_name",
            "target_labeled_ratio",
            "method_variant",
            "method_variant_display_name",
            "backbone",
            "num_runs",
            "mean_target_rmse",
            "std_target_rmse",
            "target_rmse_mean_std_text",
            "mean_target_r2",
            "std_target_r2",
            "target_r2_mean_std_text",
            "mean_target_nrmse",
            "std_target_nrmse",
            "target_nrmse_mean_std_text",
            "mean_transfer_gain",
            "std_transfer_gain",
            "transfer_gain_mean_std_text",
            "win_rate_vs_baseline",
            "negative_transfer_rate",
            "negative_transfer_severity",
            "worst_case_transfer_gain",
        ],
    )
    balanced_transfer_rows = build_protocol_balanced_summary_rows(
        enriched,
        ("target_labeled_ratio", "method_variant", "backbone"),
    )
    for row in balanced_transfer_rows:
        row["method_variant_display_name"] = _variant_display_name(str(row.get("method_variant", "")))
    balanced_transfer_summary_path = summary_root / "balanced_transfer_summary.csv"
    _write_csv(
        balanced_transfer_summary_path,
        balanced_transfer_rows,
        [
            "target_labeled_ratio",
            "method_variant",
            "method_variant_display_name",
            "backbone",
            "num_protocols",
            "protocols_covered",
            "protocol_weighting",
            "dispersion_scope",
            "num_complete_seeds",
            "complete_seeds",
            "num_runs",
            "mean_target_rmse",
            "std_target_rmse",
            "target_rmse_mean_std_text",
            "mean_target_r2",
            "std_target_r2",
            "target_r2_mean_std_text",
            "mean_target_nrmse",
            "std_target_nrmse",
            "target_nrmse_mean_std_text",
            "mean_transfer_gain",
            "std_transfer_gain",
            "transfer_gain_mean_std_text",
            "win_rate_vs_baseline",
            "negative_transfer_rate",
            "negative_transfer_severity",
            "worst_case_transfer_gain",
        ],
    )
    seed_stability_rows = build_seed_stability_rows(
        enriched,
        ("protocol", "target_labeled_ratio", "method_variant", "backbone"),
    )
    for row in seed_stability_rows:
        row["protocol_display_name"] = protocol_paper_name(str(row.get("protocol", "")))
        row["method_variant_display_name"] = _variant_display_name(str(row.get("method_variant", "")))
    seed_stability_summary_path = summary_root / "seed_stability_summary.csv"
    _write_csv(
        seed_stability_summary_path,
        seed_stability_rows,
        [
            "protocol",
            "protocol_display_name",
            "target_labeled_ratio",
            "method_variant",
            "method_variant_display_name",
            "backbone",
            "num_seeds",
            "seeds_covered",
            "min_seed_runs",
            "max_seed_runs",
            "seed_mean_target_rmse",
            "seed_std_target_rmse",
            "seed_target_rmse_range",
            "seed_target_rmse_mean_std_text",
            "seed_mean_target_r2",
            "seed_std_target_r2",
            "seed_target_r2_mean_std_text",
            "seed_mean_target_nrmse",
            "seed_std_target_nrmse",
            "seed_target_nrmse_mean_std_text",
            "seed_mean_transfer_gain",
            "seed_std_transfer_gain",
            "seed_transfer_gain_range",
            "seed_transfer_gain_mean_std_text",
        ],
    )
    negative_summary_path = summary_root / "negative_transfer_summary.csv"
    _write_csv(
        negative_summary_path,
        primary_transfer_rows,
        [
            "protocol",
            "protocol_display_name",
            "target_labeled_ratio",
            "method_variant",
            "method_variant_display_name",
            "backbone",
            "num_runs",
            "mean_target_rmse",
            "std_target_rmse",
            "target_rmse_mean_std_text",
            "mean_target_r2",
            "std_target_r2",
            "target_r2_mean_std_text",
            "mean_target_nrmse",
            "std_target_nrmse",
            "target_nrmse_mean_std_text",
            "mean_transfer_gain",
            "std_transfer_gain",
            "transfer_gain_mean_std_text",
            "win_rate_vs_baseline",
            "negative_transfer_rate",
            "negative_transfer_severity",
            "worst_case_transfer_gain",
        ],
    )

    shift_safety_case_rows = build_shift_safety_rows(
        enriched,
        ("protocol", "case_id", "case_name", "target_labeled_ratio", "method_variant", "backbone"),
    )
    shift_safety_case_path = summary_root / "shift_safety_case_summary.csv"
    _write_csv(
        shift_safety_case_path,
        shift_safety_case_rows,
        [
            "protocol",
            "protocol_display_name",
            "case_id",
            "case_name",
            "target_labeled_ratio",
            "method_variant",
            "method_variant_display_name",
            "backbone",
            "num_runs",
            "mean_target_rmse",
            "mean_transfer_gain",
            "mean_source_train_shift_score",
            "mean_target_adapt_shift_score",
            "mean_target_labeled_shift_score",
            "mean_target_unlabeled_shift_score",
            "mean_target_val_shift_score",
            "mean_target_test_shift_score",
            "mean_shift_gate_mean",
            "mean_shift_score_mean",
            "mean_guard_keep_fraction",
            "mean_safe_lambda_applied",
            "mean_safe_mode_code",
            "mean_safe_bad_epochs",
            "mean_best_target_val_rmse",
            "fallback_rate",
            "active_rate",
        ],
    )

    shift_safety_method_rows = build_shift_safety_rows(
        enriched,
        ("protocol", "target_labeled_ratio", "method_variant", "backbone"),
    )
    shift_safety_method_path = summary_root / "shift_safety_method_summary.csv"
    _write_csv(
        shift_safety_method_path,
        shift_safety_method_rows,
        [
            "protocol",
            "protocol_display_name",
            "target_labeled_ratio",
            "method_variant",
            "method_variant_display_name",
            "backbone",
            "num_runs",
            "mean_target_rmse",
            "mean_transfer_gain",
            "mean_source_train_shift_score",
            "mean_target_adapt_shift_score",
            "mean_target_labeled_shift_score",
            "mean_target_unlabeled_shift_score",
            "mean_target_val_shift_score",
            "mean_target_test_shift_score",
            "mean_shift_gate_mean",
            "mean_shift_score_mean",
            "mean_guard_keep_fraction",
            "mean_safe_lambda_applied",
            "mean_safe_mode_code",
            "mean_safe_bad_epochs",
            "mean_best_target_val_rmse",
            "fallback_rate",
            "active_rate",
        ],
    )

    case_level_significance_rows = build_case_level_significance_rows(enriched)
    case_level_significance_path = summary_root / "case_level_significance.csv"
    _write_csv(
        case_level_significance_path,
        case_level_significance_rows,
        [
            "protocol",
            "protocol_display_name",
            "target_labeled_ratio",
            "backbone",
            "baseline_variant",
            "baseline_variant_display_name",
            "candidate_variant",
            "candidate_variant_display_name",
            "n_pairs",
            "wins",
            "losses",
            "ties",
            "sign_test_p_value",
            "baseline_mean_target_rmse",
            "candidate_mean_target_rmse",
            "mean_delta_rmse",
            "median_delta_rmse",
            "candidate_mean_transfer_gain",
            "candidate_case_negative_transfer_rate",
            "candidate_worst_case_transfer_gain",
        ],
    )

    shift_severity_rows = build_shift_severity_rows(enriched)
    shift_severity_path = summary_root / "shift_severity_summary.csv"
    _write_csv(
        shift_severity_path,
        shift_severity_rows,
        [
            "protocol",
            "protocol_display_name",
            "target_labeled_ratio",
            "backbone",
            "shift_severity",
            "method_variant",
            "method_variant_display_name",
            "num_cases",
            "mean_target_rmse",
            "mean_transfer_gain",
            "negative_transfer_rate",
            "negative_transfer_severity",
            "worst_case_transfer_gain",
            "mean_target_adapt_shift_score",
            "mean_target_test_shift_score",
            "mean_fallback_rate",
            "mean_active_rate",
        ],
    )

    shift_severity_comparison_rows = build_shift_severity_comparison_rows(enriched)
    shift_severity_comparison_path = summary_root / "shift_severity_comparison.csv"
    _write_csv(
        shift_severity_comparison_path,
        shift_severity_comparison_rows,
        [
            "protocol",
            "protocol_display_name",
            "target_labeled_ratio",
            "backbone",
            "shift_severity",
            "baseline_variant",
            "baseline_variant_display_name",
            "candidate_variant",
            "candidate_variant_display_name",
            "n_pairs",
            "wins",
            "losses",
            "ties",
            "sign_test_p_value",
            "mean_delta_rmse",
            "median_delta_rmse",
            "candidate_mean_transfer_gain",
            "candidate_worst_case_transfer_gain",
        ],
    )

    failure_case_rows = build_failure_case_rows(enriched)
    failure_case_analysis_path = summary_root / "failure_case_analysis.csv"
    _write_csv(
        failure_case_analysis_path,
        failure_case_rows,
        [
            "protocol",
            "protocol_display_name",
            "case_id",
            "case_name",
            "target_labeled_ratio",
            "backbone",
            "shift_severity",
            "mean_target_adapt_shift_score",
            "culprit_variant",
            "culprit_variant_display_name",
            "culprit_mean_target_rmse",
            "culprit_mean_transfer_gain",
            "culprit_negative_transfer_rate",
            "culprit_worst_case_transfer_gain",
            "baseline_mean_target_rmse",
            "rescue_variant",
            "rescue_variant_display_name",
            "rescue_mean_target_rmse",
            "rescue_mean_transfer_gain",
            "rescue_negative_transfer_rate",
            "rescue_worst_case_transfer_gain",
            "rescue_fallback_rate",
            "rescue_active_rate",
            "rescue_outperformed_culprit",
            "rescue_non_negative_transfer",
            "rescue_avoided_worse_result",
            "rescue_any_fallback",
        ],
    )
    failure_case_highlights_path = summary_root / "failure_case_highlights.json"
    write_json(
        {
            "generated_at": datetime.now().isoformat(),
            "num_failure_cases": len(failure_case_rows),
            "highlights": failure_case_rows[:3],
        },
        failure_case_highlights_path,
    )

    paper_tables = build_paper_tables(enriched)
    paper_tables_path = summary_root / "paper_tables.json"
    write_json(paper_tables, paper_tables_path)

    conclusion_gate = build_conclusion_gate(enriched)
    conclusion_path = summary_root / "conclusion_gate_report.json"
    write_json(conclusion_gate, conclusion_path)
    return {
        "results_csv": str(results_csv.resolve()),
        "protocol_tables": protocol_tables,
        "primary_transfer_summary": str(primary_transfer_summary_path.resolve()),
        "balanced_transfer_summary": str(balanced_transfer_summary_path.resolve()),
        "seed_stability_summary": str(seed_stability_summary_path.resolve()),
        "negative_transfer_summary": str(negative_summary_path.resolve()),
        "shift_safety_case_summary": str(shift_safety_case_path.resolve()),
        "shift_safety_method_summary": str(shift_safety_method_path.resolve()),
        "case_level_significance": str(case_level_significance_path.resolve()),
        "shift_severity_summary": str(shift_severity_path.resolve()),
        "shift_severity_comparison": str(shift_severity_comparison_path.resolve()),
        "failure_case_analysis": str(failure_case_analysis_path.resolve()),
        "failure_case_highlights": str(failure_case_highlights_path.resolve()),
        "paper_tables": str(paper_tables_path.resolve()),
        "conclusion_gate_report": str(conclusion_path.resolve()),
        "num_records": len(enriched),
    }
