from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from insarda.utils.io import read_yaml


@dataclass(frozen=True)
class DomainSpec:
    name: str
    domain_id: int
    hazard_type: str
    path: Path
    source_tag: str = ""


@dataclass(frozen=True)
class ExperimentCase:
    protocol: str
    case_id: int
    case_name: str
    source_domain_ids: list[int]
    target_domain_id: int
    source_hazard_type: str | None = None
    target_hazard_type: str | None = None


@dataclass(frozen=True)
class TimeSplit:
    first: np.ndarray
    second: np.ndarray


@dataclass(frozen=True)
class TimeBandSplit:
    adapt: np.ndarray
    val: np.ndarray
    test: np.ndarray
    adapt_end: int
    val_end: int


def load_dataset_specs(path: str | Path) -> list[DomainSpec]:
    registry_path = Path(path).resolve()
    raw = read_yaml(registry_path)
    if raw is None:
        raise ValueError(f"Empty dataset registry: {registry_path}")
    datasets = raw.get("datasets", [])
    specs = []
    for item in datasets:
        dataset_path = Path(item["path"])
        if not dataset_path.is_absolute():
            dataset_path = (registry_path.parent / dataset_path).resolve()
        specs.append(
            DomainSpec(
                name=str(item["name"]),
                domain_id=int(item["domain_id"]),
                hazard_type=str(item["hazard_type"]),
                path=dataset_path,
                source_tag=str(item.get("source_tag", "")),
            )
        )
    if not specs:
        raise ValueError(f"No datasets found in {registry_path}")
    return sorted(specs, key=lambda item: item.domain_id)


def _group_by_hazard(specs: Iterable[DomainSpec]) -> dict[str, list[DomainSpec]]:
    groups: dict[str, list[DomainSpec]] = {}
    for spec in specs:
        groups.setdefault(spec.hazard_type, []).append(spec)
    for key in groups:
        groups[key] = sorted(groups[key], key=lambda item: item.domain_id)
    return groups


def _hazard_tag(hazard_type: str) -> str:
    mapping = {
        "subsidence": "subs",
        "volcano": "volc",
        "landslide": "land",
    }
    return mapping.get(str(hazard_type).lower(), str(hazard_type).lower())


def _normalize_protocol(protocol: str) -> str:
    protocol_norm = str(protocol).strip().upper()
    if protocol_norm in {"LODO", "TELODO", "IHT", "CHT"}:
        return protocol_norm
    return protocol_norm


def generate_cases(specs: list[DomainSpec], protocol: str) -> list[ExperimentCase]:
    ordered = sorted(specs, key=lambda item: item.domain_id)
    protocol_norm = _normalize_protocol(protocol)
    cases: list[ExperimentCase] = []
    if protocol_norm == "LODO":
        for case_id, target in enumerate(ordered):
            source_ids = [spec.domain_id for spec in ordered if spec.domain_id != target.domain_id]
            cases.append(
                ExperimentCase(
                    protocol="LODO",
                    case_id=case_id,
                    case_name=f"LODO_target_{target.name}",
                    source_domain_ids=source_ids,
                    target_domain_id=target.domain_id,
                    source_hazard_type="mixed",
                    target_hazard_type=target.hazard_type,
                )
            )
        return cases

    if protocol_norm == "TELODO":
        for case_id, target in enumerate(ordered):
            source_ids = [
                spec.domain_id
                for spec in ordered
                if spec.domain_id != target.domain_id and spec.hazard_type != target.hazard_type
            ]
            cases.append(
                ExperimentCase(
                    protocol="TELODO",
                    case_id=case_id,
                    case_name=f"TELODO_target_{target.name}",
                    source_domain_ids=source_ids,
                    target_domain_id=target.domain_id,
                    source_hazard_type="cross_hazard_only",
                    target_hazard_type=target.hazard_type,
                )
            )
        return cases

    groups = _group_by_hazard(ordered)
    if protocol_norm == "IHT":
        case_id = 0
        for hazard in ("subsidence", "volcano", "landslide"):
            members = groups.get(hazard, [])
            for target in members:
                source_ids = [spec.domain_id for spec in members if spec.domain_id != target.domain_id]
                cases.append(
                    ExperimentCase(
                        protocol="IHT",
                        case_id=case_id,
                        case_name=f"IHT_{hazard}_target_{target.name}",
                        source_domain_ids=source_ids,
                        target_domain_id=target.domain_id,
                        source_hazard_type=hazard,
                        target_hazard_type=hazard,
                    )
                )
                case_id += 1
        return cases

    if protocol_norm == "CHT":
        case_id = 0
        for source_hazard in ("subsidence", "volcano", "landslide"):
            source_members = groups.get(source_hazard, [])
            if not source_members:
                continue
            source_ids = [spec.domain_id for spec in source_members]
            for target_hazard in ("subsidence", "volcano", "landslide"):
                if target_hazard == source_hazard:
                    continue
                for target in groups.get(target_hazard, []):
                    cases.append(
                        ExperimentCase(
                            protocol="CHT",
                            case_id=case_id,
                            case_name=f"CHT_{_hazard_tag(source_hazard)}_to_{target.name}",
                            source_domain_ids=source_ids,
                            target_domain_id=target.domain_id,
                            source_hazard_type=source_hazard,
                            target_hazard_type=target_hazard,
                        )
                    )
                    case_id += 1
        return cases

    raise ValueError(f"Unknown protocol: {protocol}")


def split_source_train_val(
    target_end_idx: np.ndarray,
    target_start_idx: np.ndarray,
    total_time_steps: int,
    ratio: float = 0.7,
) -> TimeSplit:
    split_end = int(np.floor(total_time_steps * float(ratio))) - 1
    split_end = int(np.clip(split_end, -1, max(total_time_steps - 1, 0)))
    first = np.asarray(target_end_idx) <= split_end
    second = np.asarray(target_start_idx) > split_end
    return TimeSplit(first=first, second=second)


def split_target_strict_523(
    target_end_idx: np.ndarray,
    target_start_idx: np.ndarray,
    total_time_steps: int,
    *,
    adapt_ratio: float = 0.5,
    val_ratio: float = 0.2,
) -> TimeBandSplit:
    if not (0.0 < float(adapt_ratio) < 1.0):
        raise ValueError("`adapt_ratio` must be in the open interval (0, 1).")
    if not (0.0 < float(val_ratio) < 1.0):
        raise ValueError("`val_ratio` must be in the open interval (0, 1).")
    if not (float(adapt_ratio) + float(val_ratio) < 1.0):
        raise ValueError("`adapt_ratio + val_ratio` must be < 1.")

    adapt_end = int(np.floor(total_time_steps * float(adapt_ratio))) - 1
    adapt_end = int(np.clip(adapt_end, -1, max(total_time_steps - 1, 0)))
    val_end = int(np.floor(total_time_steps * float(adapt_ratio + val_ratio))) - 1
    val_end = int(np.clip(val_end, adapt_end, max(total_time_steps - 1, 0)))

    adapt = np.asarray(target_end_idx) <= adapt_end
    val = (np.asarray(target_start_idx) > adapt_end) & (np.asarray(target_end_idx) <= val_end)
    test = np.asarray(target_start_idx) > val_end
    return TimeBandSplit(
        adapt=adapt,
        val=val,
        test=test,
        adapt_end=adapt_end,
        val_end=val_end,
    )
