from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src").resolve()))

from insarda.config import (
    FORMAL_MAIN_METHODS,
    FORMAL_MAIN_PROTOCOLS,
    method_paper_name,
    protocol_paper_name,
)
from insarda.reporting.summarize import (
    _variant_display_name,
    _write_csv,
    _write_results_csv,
    attach_negative_transfer,
    build_case_level_significance_rows,
    build_conclusion_gate,
    build_failure_case_rows,
    build_group_summary_rows,
    build_paper_tables,
    build_protocol_balanced_summary_rows,
    build_seed_stability_rows,
    build_shift_safety_rows,
    build_shift_severity_comparison_rows,
    build_shift_severity_rows,
    collect_run_records,
    dedupe_latest,
    transfer_gain,
    _protocol_table,
)


SUMMARY_DIRS = (
    "comprehensive_summary_2016",
    "comprehensive_summary_2016_custom",
)

MANIFEST_HEADER = (
    "seed",
    "label_rate",
    "method",
    "protocol",
    "case_id",
    "run_dir",
    "relative_path",
)

CUSTOM_ALL_RUNS_HEADER = (
    "run_dir",
    "run_name",
    "created_at",
    "study_tag",
    "protocol",
    "protocol_display_name",
    "case_id",
    "case_name",
    "method",
    "method_display_name",
    "backbone",
    "seed",
    "label_rate",
    "best_epoch",
    "source_val_rmse",
    "target_val_rmse",
    "target_test_rmse",
    "data_signature",
    "config_signature",
    "method_signature",
    "target_test_h1_rmse",
    "target_test_h2_rmse",
    "target_test_h3_rmse",
    "target_test_h4_rmse",
    "target_test_h5_rmse",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild results manifests and summaries under results/.")
    parser.add_argument("--results-root", required=True)
    return parser.parse_args()


def _backup_existing(results_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = results_root / f"_backup_before_8method_refresh_{stamp}"
    backup_root.mkdir(parents=True, exist_ok=True)
    manifest_path = results_root / "unique_runs_2016_by_seed_label_rate_method" / "manifest_unique_runs_2016.csv"
    if manifest_path.exists():
        shutil.copy2(manifest_path, backup_root / manifest_path.name)
    for name in SUMMARY_DIRS:
        src = results_root / name
        if src.exists():
            shutil.copytree(src, backup_root / name)
    return backup_root


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _metrics_overall(record: dict[str, Any], split: str, metric: str) -> float | None:
    return _safe_float((((record.get(split) or {}).get("overall") or {}).get(metric)))


def _metrics_per_horizon(record: dict[str, Any], split: str) -> list[dict[str, Any]]:
    values = (record.get(split) or {}).get("per_horizon") or []
    return list(values) if isinstance(values, list) else []


def _record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    label_rate = record.get("target_labeled_ratio")
    if label_rate is None:
        label_rate = record.get("label_rate")
    return (
        str(record.get("protocol")),
        int(record.get("case_id")),
        str(record.get("backbone")),
        int(record.get("seed")),
        float(label_rate),
    )


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.stdev(values))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _sem(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(_std(values) / math.sqrt(len(values)))


def _ci95(values: list[float]) -> float:
    return float(1.96 * _sem(values))


def _write_csv_rows(path: Path, rows: list[dict[str, Any]], header: tuple[str, ...] | list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in header})


def _group_stats(rows: list[dict[str, Any]], key_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in key_fields)].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(grouped):
        bucket = grouped[key]
        row = {field: value for field, value in zip(key_fields, key)}
        target_test_values = [float(item["target_test_rmse"]) for item in bucket if item.get("target_test_rmse") is not None]
        target_val_values = [float(item["target_val_rmse"]) for item in bucket if item.get("target_val_rmse") is not None]
        best_epoch_values = [float(item["best_epoch"]) for item in bucket if item.get("best_epoch") is not None]
        row.update(
            {
                "n_x": len(target_test_values),
                "mean_target_test_rmse": _mean(target_test_values),
                "std_target_test_rmse": _std(target_test_values),
                "median_target_test_rmse": _median(target_test_values),
                "sem_target_test_rmse": _sem(target_test_values),
                "ci95_target_test_rmse": _ci95(target_test_values),
                "n_y": len(target_val_values),
                "mean_target_val_rmse": _mean(target_val_values),
                "std_target_val_rmse": _std(target_val_values),
                "median_target_val_rmse": _median(target_val_values),
                "sem_target_val_rmse": _sem(target_val_values),
                "ci95_target_val_rmse": _ci95(target_val_values),
                "n": len(best_epoch_values),
                "mean_best_epoch": _mean(best_epoch_values),
                "std_best_epoch": _std(best_epoch_values),
                "median_best_epoch": _median(best_epoch_values),
                "sem_best_epoch": _sem(best_epoch_values),
                "ci95_best_epoch": _ci95(best_epoch_values),
            }
        )
        output.append(row)
    return output


def _pair_to_target_only(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target_only_index: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if row["method"] == "target_only":
            target_only_index[_record_key(row)] = row
    paired: list[dict[str, Any]] = []
    for row in rows:
        baseline = target_only_index.get(_record_key(row))
        current = dict(row)
        current["target_only_target_test_rmse"] = baseline.get("target_test_rmse") if baseline else None
        current["target_only_target_val_rmse"] = baseline.get("target_val_rmse") if baseline else None
        delta = None
        improvement_pct = None
        if baseline is not None and row.get("target_test_rmse") is not None:
            delta = float(baseline["target_test_rmse"]) - float(row["target_test_rmse"])
            if abs(float(baseline["target_test_rmse"])) > 1e-12:
                improvement_pct = 100.0 * delta / float(baseline["target_test_rmse"])
        current["delta_rmse"] = delta
        current["improvement_pct"] = improvement_pct
        current["is_negative_transfer"] = None if delta is None else bool(delta < 0.0)
        current["is_win_vs_target_only"] = None if delta is None else bool(delta > 0.0)
        paired.append(current)
    return paired


def _attach_transfer_without_data_signature(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target_index: dict[tuple[Any, ...], dict[str, Any]] = {}
    source_index: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        key = _record_key(record)
        if str(record.get("method")) == "target_only":
            target_index[key] = record
        if str(record.get("method")) == "source_only":
            source_index[key] = record
    enriched: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        key = _record_key(record)
        current_rmse = _metrics_overall(record, "target_test", "rmse")
        target_record = target_index.get(key)
        source_record = source_index.get(key)
        target_rmse = _metrics_overall(target_record, "target_test", "rmse") if target_record else None
        source_rmse = _metrics_overall(source_record, "target_test", "rmse") if source_record else None
        row["baseline_variant"] = "target_only"
        row["baseline_pairing_mode"] = "by_protocol_case_seed_label_rate_backbone" if target_record else "unpaired"
        row["baseline_rmse"] = target_rmse
        row["paired_baseline_run_dir"] = target_record.get("run_dir") if target_record else None
        row["transfer_gain"] = transfer_gain(target_rmse, current_rmse)
        row["auxiliary_baseline_variant"] = "source_only"
        row["auxiliary_baseline_pairing_mode"] = (
            "by_protocol_case_seed_label_rate_backbone" if source_record else "unpaired"
        )
        row["auxiliary_baseline_rmse"] = source_rmse
        row["paired_auxiliary_baseline_run_dir"] = source_record.get("run_dir") if source_record else None
        row["auxiliary_transfer_gain"] = transfer_gain(source_rmse, current_rmse)
        enriched.append(row)
    return attach_negative_transfer(enriched)


def _build_manifest_rows(unique_root: Path, records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_rows: list[dict[str, Any]] = []
    all_runs_rows: list[dict[str, Any]] = []
    for record in sorted(
        records,
        key=lambda item: (
            int(item["seed"]),
            float(item["target_labeled_ratio"]),
            str(item["method"]),
            str(item["protocol"]),
            int(item["case_id"]),
        ),
    ):
        run_dir = Path(str(record["run_dir"]))
        relative_path = run_dir.relative_to(unique_root.parent)
        label_rate = float(record["target_labeled_ratio"])
        label_tag = f"{label_rate:g}".replace(".", "p")
        manifest_rows.append(
            {
                "seed": str(record["seed"]),
                "label_rate": label_tag,
                "method": record["method"],
                "protocol": record["protocol"],
                "case_id": str(record["case_id"]),
                "run_dir": run_dir.name,
                "relative_path": str(relative_path).replace("/", "\\"),
            }
        )
        per_horizon = _metrics_per_horizon(record, "target_test")
        horizon_values = [None, None, None, None, None]
        for item in per_horizon:
            horizon = int(item.get("horizon", 0))
            if 1 <= horizon <= 5:
                horizon_values[horizon - 1] = _safe_float(item.get("rmse"))
        all_runs_rows.append(
            {
                "run_dir": str(run_dir),
                "run_name": run_dir.name,
                "created_at": record.get("created_at"),
                "study_tag": record.get("study_tag"),
                "protocol": record.get("protocol"),
                "protocol_display_name": protocol_paper_name(str(record.get("protocol", ""))),
                "case_id": record.get("case_id"),
                "case_name": record.get("case_name"),
                "method": record.get("method"),
                "method_display_name": method_paper_name(str(record.get("method", ""))),
                "backbone": record.get("backbone"),
                "seed": record.get("seed"),
                "label_rate": label_rate,
                "best_epoch": record.get("best_epoch"),
                "source_val_rmse": _metrics_overall(record, "source_val", "rmse"),
                "target_val_rmse": _metrics_overall(record, "target_val", "rmse"),
                "target_test_rmse": _metrics_overall(record, "target_test", "rmse"),
                "data_signature": record.get("data_signature"),
                "config_signature": record.get("config_signature"),
                "method_signature": record.get("method_signature"),
                "target_test_h1_rmse": horizon_values[0],
                "target_test_h2_rmse": horizon_values[1],
                "target_test_h3_rmse": horizon_values[2],
                "target_test_h4_rmse": horizon_values[3],
                "target_test_h5_rmse": horizon_values[4],
            }
        )
    return manifest_rows, all_runs_rows


def _write_official_summary(summary_root: Path, enriched: list[dict[str, Any]]) -> None:
    summary_root.mkdir(parents=True, exist_ok=True)
    results_csv = summary_root / "results.csv"
    _write_results_csv(results_csv, enriched)

    protocol_tables: dict[str, str] = {}
    for protocol in FORMAL_MAIN_PROTOCOLS:
        rows = _protocol_table(enriched, protocol)
        path = summary_root / f"protocol_{protocol}_table.csv"
        _write_csv(
            path,
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
        protocol_tables[protocol] = str(path.resolve())

    primary_rows = build_group_summary_rows(
        enriched,
        ("protocol", "target_labeled_ratio", "method_variant", "backbone"),
    )
    for row in primary_rows:
        row["protocol_display_name"] = protocol_paper_name(str(row.get("protocol", "")))
        row["method_variant_display_name"] = _variant_display_name(str(row.get("method_variant", "")))
    _write_csv(
        summary_root / "primary_transfer_summary.csv",
        primary_rows,
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

    balanced_rows = build_protocol_balanced_summary_rows(
        enriched,
        ("target_labeled_ratio", "method_variant", "backbone"),
    )
    for row in balanced_rows:
        row["method_variant_display_name"] = _variant_display_name(str(row.get("method_variant", "")))
    _write_csv(
        summary_root / "balanced_transfer_summary.csv",
        balanced_rows,
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

    seed_rows = build_seed_stability_rows(
        enriched,
        ("protocol", "target_labeled_ratio", "method_variant", "backbone"),
    )
    for row in seed_rows:
        row["protocol_display_name"] = protocol_paper_name(str(row.get("protocol", "")))
        row["method_variant_display_name"] = _variant_display_name(str(row.get("method_variant", "")))
    _write_csv(
        summary_root / "seed_stability_summary.csv",
        seed_rows,
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

    _write_csv(
        summary_root / "negative_transfer_summary.csv",
        primary_rows,
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

    case_shift_rows = build_shift_safety_rows(
        enriched,
        ("protocol", "case_id", "case_name", "target_labeled_ratio", "method_variant", "backbone"),
    )
    _write_csv(
        summary_root / "shift_safety_case_summary.csv",
        case_shift_rows,
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

    method_shift_rows = build_shift_safety_rows(
        enriched,
        ("protocol", "target_labeled_ratio", "method_variant", "backbone"),
    )
    _write_csv(
        summary_root / "shift_safety_method_summary.csv",
        method_shift_rows,
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

    _write_csv(
        summary_root / "case_level_significance.csv",
        build_case_level_significance_rows(enriched),
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

    _write_csv(
        summary_root / "shift_severity_summary.csv",
        build_shift_severity_rows(enriched),
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

    _write_csv(
        summary_root / "shift_severity_comparison.csv",
        build_shift_severity_comparison_rows(enriched),
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

    failure_rows = build_failure_case_rows(enriched)
    _write_csv(
        summary_root / "failure_case_analysis.csv",
        failure_rows,
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
    with (summary_root / "failure_case_highlights.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(),
                "num_failure_cases": len(failure_rows),
                "highlights": failure_rows[:3],
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    with (summary_root / "paper_tables.json").open("w", encoding="utf-8") as handle:
        json.dump(build_paper_tables(enriched), handle, ensure_ascii=False, indent=2)
    with (summary_root / "conclusion_gate_report.json").open("w", encoding="utf-8") as handle:
        json.dump(build_conclusion_gate(enriched), handle, ensure_ascii=False, indent=2)


def _write_custom_summary(summary_root: Path, all_rows: list[dict[str, Any]]) -> None:
    summary_root.mkdir(parents=True, exist_ok=True)
    paired_rows = _pair_to_target_only(all_rows)
    _write_csv_rows(summary_root / "all_runs_manifest.csv", all_rows, CUSTOM_ALL_RUNS_HEADER)

    summary_protocol_method_label_rate = _group_stats(
        all_rows,
        ("protocol", "protocol_display_name", "label_rate", "method", "method_display_name"),
    )
    _write_csv_rows(
        summary_root / "summary_protocol_method_label_rate.csv",
        summary_protocol_method_label_rate,
        (
            "protocol",
            "protocol_display_name",
            "label_rate",
            "method",
            "method_display_name",
            "n_x",
            "mean_target_test_rmse",
            "std_target_test_rmse",
            "median_target_test_rmse",
            "sem_target_test_rmse",
            "ci95_target_test_rmse",
            "n_y",
            "mean_target_val_rmse",
            "std_target_val_rmse",
            "median_target_val_rmse",
            "sem_target_val_rmse",
            "ci95_target_val_rmse",
            "n",
            "mean_best_epoch",
            "std_best_epoch",
            "median_best_epoch",
            "sem_best_epoch",
            "ci95_best_epoch",
        ),
    )

    summary_protocol_method = _group_stats(
        all_rows,
        ("protocol", "protocol_display_name", "method", "method_display_name"),
    )
    _write_csv_rows(
        summary_root / "summary_protocol_method.csv",
        summary_protocol_method,
        (
            "protocol",
            "protocol_display_name",
            "method",
            "method_display_name",
            "n_x",
            "mean_target_test_rmse",
            "std_target_test_rmse",
            "median_target_test_rmse",
            "sem_target_test_rmse",
            "ci95_target_test_rmse",
            "n_y",
            "mean_target_val_rmse",
            "std_target_val_rmse",
            "median_target_val_rmse",
            "sem_target_val_rmse",
            "ci95_target_val_rmse",
            "n",
            "mean_best_epoch",
            "std_best_epoch",
            "median_best_epoch",
            "sem_best_epoch",
            "ci95_best_epoch",
        ),
    )

    summary_overall_method = _group_stats(all_rows, ("method", "method_display_name"))
    _write_csv_rows(
        summary_root / "summary_overall_method.csv",
        summary_overall_method,
        (
            "method",
            "method_display_name",
            "n_x",
            "mean_target_test_rmse",
            "std_target_test_rmse",
            "median_target_test_rmse",
            "sem_target_test_rmse",
            "ci95_target_test_rmse",
            "n_y",
            "mean_target_val_rmse",
            "std_target_val_rmse",
            "median_target_val_rmse",
            "sem_target_val_rmse",
            "ci95_target_val_rmse",
            "n",
            "mean_best_epoch",
            "std_best_epoch",
            "median_best_epoch",
            "sem_best_epoch",
            "ci95_best_epoch",
        ),
    )

    pairwise_rows = []
    for row in paired_rows:
        pairwise_rows.append(
            {
                **row,
            }
        )
    _write_csv_rows(
        summary_root / "pairwise_vs_target_only.csv",
        pairwise_rows,
        (
            *CUSTOM_ALL_RUNS_HEADER,
            "target_only_target_test_rmse",
            "target_only_target_val_rmse",
            "delta_rmse",
            "improvement_pct",
            "is_negative_transfer",
            "is_win_vs_target_only",
        ),
    )

    def aggregate_pairwise(group_fields: tuple[str, ...]) -> list[dict[str, Any]]:
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in paired_rows:
            grouped[tuple(row[field] for field in group_fields)].append(row)
        rows: list[dict[str, Any]] = []
        for key in sorted(grouped):
            bucket = grouped[key]
            deltas = [float(item["delta_rmse"]) for item in bucket if item.get("delta_rmse") is not None]
            improvements = [float(item["improvement_pct"]) for item in bucket if item.get("improvement_pct") is not None]
            target_test = [float(item["target_test_rmse"]) for item in bucket if item.get("target_test_rmse") is not None]
            target_only = [
                float(item["target_only_target_test_rmse"])
                for item in bucket
                if item.get("target_only_target_test_rmse") is not None
            ]
            row = {field: value for field, value in zip(group_fields, key)}
            row.update(
                {
                    "n_pairs": len(deltas),
                    "mean_target_test_rmse": _mean(target_test),
                    "mean_target_only_rmse": _mean(target_only),
                    "mean_delta_rmse": _mean(deltas),
                    "median_delta_rmse": _median(deltas),
                    "std_delta_rmse": _std(deltas),
                    "sem_delta_rmse": _sem(deltas),
                    "ci95_delta_rmse": _ci95(deltas),
                    "mean_improvement_pct": _mean(improvements),
                    "median_improvement_pct": _median(improvements),
                    "negative_transfer_rate": (
                        sum(1 for value in deltas if value < 0.0) / len(deltas) if deltas else None
                    ),
                    "win_rate_vs_target_only": (
                        sum(1 for value in deltas if value > 0.0) / len(deltas) if deltas else None
                    ),
                }
            )
            rows.append(row)
        return rows

    pairwise_plr = aggregate_pairwise(
        ("protocol", "protocol_display_name", "label_rate", "method", "method_display_name")
    )
    _write_csv_rows(
        summary_root / "pairwise_vs_target_only_summary_protocol_method_label_rate.csv",
        pairwise_plr,
        (
            "protocol",
            "protocol_display_name",
            "label_rate",
            "method",
            "method_display_name",
            "n_pairs",
            "mean_target_test_rmse",
            "mean_target_only_rmse",
            "mean_delta_rmse",
            "median_delta_rmse",
            "std_delta_rmse",
            "sem_delta_rmse",
            "ci95_delta_rmse",
            "mean_improvement_pct",
            "median_improvement_pct",
            "negative_transfer_rate",
            "win_rate_vs_target_only",
        ),
    )

    pairwise_pm = aggregate_pairwise(("protocol", "protocol_display_name", "method", "method_display_name"))
    _write_csv_rows(
        summary_root / "pairwise_vs_target_only_summary_protocol_method.csv",
        pairwise_pm,
        (
            "protocol",
            "protocol_display_name",
            "method",
            "method_display_name",
            "n_pairs",
            "mean_target_test_rmse",
            "mean_target_only_rmse",
            "mean_delta_rmse",
            "median_delta_rmse",
            "std_delta_rmse",
            "sem_delta_rmse",
            "ci95_delta_rmse",
            "mean_improvement_pct",
            "median_improvement_pct",
            "negative_transfer_rate",
            "win_rate_vs_target_only",
        ),
    )

    rank_rows: list[dict[str, Any]] = []
    grouped_ranks: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        grouped_ranks[(row["protocol"], row["case_id"], row["seed"], row["label_rate"])].append(row)
    for key in sorted(grouped_ranks):
        bucket = sorted(grouped_ranks[key], key=lambda item: float(item["target_test_rmse"]))
        for index, row in enumerate(bucket, start=1):
            rank_rows.append(
                {
                    "protocol": row["protocol"],
                    "case_id": row["case_id"],
                    "seed": row["seed"],
                    "label_rate": row["label_rate"],
                    "method": row["method"],
                    "method_display_name": row["method_display_name"],
                    "target_test_rmse": row["target_test_rmse"],
                    "rank": float(index),
                }
            )
    _write_csv_rows(
        summary_root / "pairwise_rank_table.csv",
        rank_rows,
        ("protocol", "case_id", "seed", "label_rate", "method", "method_display_name", "target_test_rmse", "rank"),
    )

    grouped_rank_summary: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rank_rows:
        grouped_rank_summary[(row["protocol"], row["label_rate"], row["method"], row["method_display_name"])].append(
            float(row["rank"])
        )
    rank_summary_rows: list[dict[str, Any]] = []
    for key in sorted(grouped_rank_summary):
        values = grouped_rank_summary[key]
        rank_summary_rows.append(
            {
                "protocol": key[0],
                "label_rate": key[1],
                "method": key[2],
                "method_display_name": key[3],
                "n_units": len(values),
                "mean_rank": _mean(values),
                "std_rank": _std(values),
                "top1_rate": sum(1 for value in values if value == 1.0) / len(values),
                "top2_rate": sum(1 for value in values if value <= 2.0) / len(values),
            }
        )
    _write_csv_rows(
        summary_root / "rank_summary_protocol_method_label_rate.csv",
        rank_summary_rows,
        ("protocol", "label_rate", "method", "method_display_name", "n_units", "mean_rank", "std_rank", "top1_rate", "top2_rate"),
    )

    horizon_rows: list[dict[str, Any]] = []
    grouped_horizons: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in all_rows:
        for horizon in range(1, 6):
            value = row.get(f"target_test_h{horizon}_rmse")
            if value is None:
                continue
            grouped_horizons[
                (row["protocol"], row["protocol_display_name"], row["label_rate"], row["method"], row["method_display_name"], horizon)
            ].append(float(value))
    for key in sorted(grouped_horizons):
        values = grouped_horizons[key]
        horizon_rows.append(
            {
                "protocol": key[0],
                "protocol_display_name": key[1],
                "label_rate": key[2],
                "method": key[3],
                "method_display_name": key[4],
                "horizon": key[5],
                "n": len(values),
                "mean_rmse": _mean(values),
                "std_rmse": _std(values),
                "sem_rmse": _sem(values),
                "ci95_rmse": _ci95(values),
            }
        )
    _write_csv_rows(
        summary_root / "horizon_summary_protocol_method_label_rate.csv",
        horizon_rows,
        (
            "protocol",
            "protocol_display_name",
            "label_rate",
            "method",
            "method_display_name",
            "horizon",
            "n",
            "mean_rmse",
            "std_rmse",
            "sem_rmse",
            "ci95_rmse",
        ),
    )

    best_rows: list[dict[str, Any]] = []
    summary_by_plr = defaultdict(list)
    for row in summary_protocol_method_label_rate:
        summary_by_plr[(row["protocol"], row["protocol_display_name"], row["label_rate"])].append(row)
    pairwise_by_plr = defaultdict(list)
    for row in pairwise_plr:
        pairwise_by_plr[(row["protocol"], row["protocol_display_name"], row["label_rate"])].append(row)
    for key in sorted(summary_by_plr):
        summary_bucket = summary_by_plr[key]
        pairwise_bucket = pairwise_by_plr[key]
        best_rmse = min(summary_bucket, key=lambda item: float(item["mean_target_test_rmse"]))
        best_gain = max(pairwise_bucket, key=lambda item: float(item["mean_delta_rmse"]))
        best_rows.append(
            {
                "protocol": key[0],
                "protocol_display_name": key[1],
                "label_rate": key[2],
                "best_rmse_method": best_rmse["method"],
                "best_rmse_method_display_name": best_rmse["method_display_name"],
                "best_mean_target_test_rmse": best_rmse["mean_target_test_rmse"],
                "best_gain_method": best_gain["method"],
                "best_gain_method_display_name": best_gain["method_display_name"],
                "best_mean_delta_rmse": best_gain["mean_delta_rmse"],
                "best_mean_improvement_pct": best_gain["mean_improvement_pct"],
            }
        )
    _write_csv_rows(
        summary_root / "best_method_by_protocol_label_rate.csv",
        best_rows,
        (
            "protocol",
            "protocol_display_name",
            "label_rate",
            "best_rmse_method",
            "best_rmse_method_display_name",
            "best_mean_target_test_rmse",
            "best_gain_method",
            "best_gain_method_display_name",
            "best_mean_delta_rmse",
            "best_mean_improvement_pct",
        ),
    )

    overview = {
        "unique_runs": len(all_rows),
        "protocols": list(FORMAL_MAIN_PROTOCOLS),
        "methods": list(FORMAL_MAIN_METHODS),
        "label_rates": [0.005, 0.01, 0.025, 0.05],
        "seeds": [42, 43, 44],
        "files": {
            "all_runs_manifest": str((summary_root / "all_runs_manifest.csv").resolve()),
            "summary_protocol_method_label_rate": str((summary_root / "summary_protocol_method_label_rate.csv").resolve()),
            "summary_protocol_method": str((summary_root / "summary_protocol_method.csv").resolve()),
            "summary_overall_method": str((summary_root / "summary_overall_method.csv").resolve()),
            "pairwise_vs_target_only": str((summary_root / "pairwise_vs_target_only.csv").resolve()),
            "pairwise_vs_target_only_summary_protocol_method_label_rate": str((summary_root / "pairwise_vs_target_only_summary_protocol_method_label_rate.csv").resolve()),
            "pairwise_vs_target_only_summary_protocol_method": str((summary_root / "pairwise_vs_target_only_summary_protocol_method.csv").resolve()),
            "pairwise_rank_table": str((summary_root / "pairwise_rank_table.csv").resolve()),
            "rank_summary_protocol_method_label_rate": str((summary_root / "rank_summary_protocol_method_label_rate.csv").resolve()),
            "horizon_summary_protocol_method_label_rate": str((summary_root / "horizon_summary_protocol_method_label_rate.csv").resolve()),
            "best_method_by_protocol_label_rate": str((summary_root / "best_method_by_protocol_label_rate.csv").resolve()),
        },
    }
    with (summary_root / "overview.json").open("w", encoding="utf-8") as handle:
        json.dump(overview, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = _parse_args()
    results_root = Path(args.results_root).resolve()
    unique_root = results_root / "unique_runs_2016_by_seed_label_rate_method"
    if not unique_root.exists():
        raise FileNotFoundError(f"Missing unique run root: {unique_root}")

    backup_root = _backup_existing(results_root)
    records = dedupe_latest(collect_run_records(unique_root))
    if len(records) != 2304:
        raise ValueError(f"Expected 2304 unique runs after adding sft_replay, found {len(records)}.")

    manifest_rows, all_rows = _build_manifest_rows(unique_root, records)
    manifest_path = unique_root / "manifest_unique_runs_2016.csv"
    _write_csv_rows(manifest_path, manifest_rows, MANIFEST_HEADER)

    official_enriched = _attach_transfer_without_data_signature(records)
    _write_official_summary(results_root / "comprehensive_summary_2016", official_enriched)
    _write_custom_summary(results_root / "comprehensive_summary_2016_custom", all_rows)

    print(
        json.dumps(
            {
                "backup_root": str(backup_root),
                "unique_root": str(unique_root),
                "manifest_path": str(manifest_path),
                "unique_runs": len(records),
                "official_summary_root": str((results_root / "comprehensive_summary_2016").resolve()),
                "custom_summary_root": str((results_root / "comprehensive_summary_2016_custom").resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
