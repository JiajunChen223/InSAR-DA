from __future__ import annotations

from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from typing import Any

import numpy as np

from insarda.config import DataConfig
from insarda.data_prep.io import load_npz
from insarda.data_pipeline.preprocess import FeatureStandardizer
from insarda.data_pipeline.splits import (
    DomainSpec,
    ExperimentCase,
    load_dataset_specs,
    split_source_train_val,
    split_target_strict_523,
)
from insarda.data_pipeline.windows import WindowBundle, build_windows, concat_window_bundles, load_window_bundle, save_window_bundle, slice_window_bundle


@dataclass
class DomainData:
    spec: DomainSpec
    displacement: np.ndarray
    latlon: np.ndarray


@dataclass
class CaseData:
    source_train: WindowBundle
    source_val: WindowBundle
    target_labeled: WindowBundle
    target_unlabeled: WindowBundle
    target_val: WindowBundle
    target_test: WindowBundle
    target_point_latlon: np.ndarray
    metadata: dict[str, Any]


def _load_domain_dataset(spec: DomainSpec) -> DomainData:
    bundle = load_npz(spec.path)
    return DomainData(
        spec=spec,
        displacement=np.asarray(bundle.displacement, dtype=np.float32),
        latlon=np.asarray(bundle.latlon, dtype=np.float32),
    )


def _source_file_cache_fingerprint(path: str | Path) -> str:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    text = f"{resolved}|{int(stat.st_size)}|{int(stat.st_mtime_ns)}"
    return md5(text.encode("utf-8")).hexdigest()[:16]


def _cache_key(spec: DomainSpec, data_cfg: DataConfig) -> str:
    source_fingerprint = _source_file_cache_fingerprint(spec.path)
    text = (
        f"{spec.domain_id}|{Path(spec.path).resolve()}|{source_fingerprint}|"
        f"{data_cfg.input_window}|{data_cfg.horizon}|observation_step_displacement_only_v3"
    )
    return md5(text.encode("utf-8")).hexdigest()[:12]


def window_cache_path(spec: DomainSpec, data_cfg: DataConfig, cache_root: str | Path) -> Path:
    cache_dir = Path(cache_root)
    return cache_dir / f"domain_{spec.domain_id}_{_cache_key(spec, data_cfg)}.npz"


def _load_or_build_windows(domain: DomainData, data_cfg: DataConfig, cache_root: Path | None) -> WindowBundle:
    cache_path = None
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_path = window_cache_path(domain.spec, data_cfg, cache_root)
        if cache_path.exists():
            return load_window_bundle(cache_path)
    bundle = build_windows(
        displacement=domain.displacement,
        domain_id=domain.spec.domain_id,
        input_window=data_cfg.input_window,
        horizon=data_cfg.horizon,
    )
    if cache_path is not None:
        save_window_bundle(bundle, cache_path)
    return bundle


def _select_case(specs: list[DomainSpec], protocol: str, case_id: int) -> ExperimentCase:
    from insarda.data_pipeline.splits import generate_cases

    cases = generate_cases(specs, protocol)
    for case in cases:
        if case.case_id == int(case_id):
            return case
    raise ValueError(f"Unknown case_id={case_id} for protocol={protocol}")


def _apply_standardizer(bundle: WindowBundle, standardizer: FeatureStandardizer) -> WindowBundle:
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


def _apply_stride(bundle: WindowBundle, stride: int) -> WindowBundle:
    step = int(stride)
    if bundle.size == 0 or step <= 1:
        return bundle
    indices = np.arange(0, bundle.size, step, dtype=np.int64)
    return slice_window_bundle(bundle, indices)


def _apply_stride_per_point(bundle: WindowBundle, stride: int) -> WindowBundle:
    step = int(stride)
    if bundle.size == 0 or step <= 1:
        return bundle
    indices: list[int] = []
    for point_id in np.unique(bundle.point_id.astype(np.int64, copy=False)):
        point_indices = np.flatnonzero(bundle.point_id == point_id)
        ordered = point_indices[np.argsort(bundle.target_end_idx[point_indices], kind="mergesort")]
        indices.extend(ordered[::step].tolist())
    return slice_window_bundle(bundle, np.asarray(indices, dtype=np.int64))


def _bundle_shift_score(bundle: WindowBundle) -> float:
    if bundle.size == 0:
        return 0.0
    flat = np.asarray(bundle.x, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return 0.0
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    return float(((mean**2) + ((std - 1.0) ** 2)) ** 0.5)


def _sample_labeled_points(
    bundle: WindowBundle,
    *,
    labeled_ratio: float,
    seed_text: str,
    sampling_seed: int,
    strata: int,
    point_gradient_scores: dict[int, float] | None = None,
) -> np.ndarray:
    if bundle.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if not (0.0 < float(labeled_ratio) < 1.0):
        raise ValueError("`labeled_ratio` must be in the open interval (0, 1).")
    unique_points = np.unique(bundle.point_id.astype(np.int64, copy=False))
    if unique_points.size == 0:
        return unique_points
    if unique_points.size == 1:
        return unique_points.copy()
    labeled_count = int(round(unique_points.size * float(labeled_ratio)))
    labeled_count = max(1, min(unique_points.size - 1, labeled_count))
    combined_seed_text = f"{seed_text}|sampling_seed={int(sampling_seed)}|strategy=deformation_gradient_stratified"
    combined_seed = int(md5(combined_seed_text.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(combined_seed)

    point_scores = {
        int(point_id): float((point_gradient_scores or {}).get(int(point_id), 0.0))
        for point_id in unique_points.astype(np.int64, copy=False)
    }
    ordered_points = np.asarray(
        sorted(
            (int(point_id) for point_id in unique_points.astype(np.int64, copy=False)),
            key=lambda point_id: (point_scores.get(int(point_id), 0.0), int(point_id)),
        ),
        dtype=np.int64,
    )
    strata_count = max(1, min(int(strata), int(ordered_points.size)))
    strata_groups = [np.asarray(group, dtype=np.int64) for group in np.array_split(ordered_points, strata_count) if group.size > 0]
    capacities = np.asarray([group.size for group in strata_groups], dtype=np.int64)
    allocations = np.zeros(len(strata_groups), dtype=np.int64)

    remaining_budget = int(labeled_count)
    if remaining_budget >= len(strata_groups):
        allocations += 1
        remaining_budget -= len(strata_groups)
        capacities = capacities - 1

    if remaining_budget > 0 and capacities.sum() > 0:
        expected = remaining_budget * capacities.astype(np.float64) / float(capacities.sum())
        base = np.floor(expected).astype(np.int64)
        allocations += base
        leftover = int(remaining_budget - int(base.sum()))
        if leftover > 0:
            order = np.argsort(-(expected - base))
            for index in order.tolist():
                if leftover <= 0:
                    break
                if allocations[index] >= len(strata_groups[index]):
                    continue
                allocations[index] += 1
                leftover -= 1

    selected: list[int] = []
    for group, count in zip(strata_groups, allocations.tolist(), strict=False):
        if count <= 0:
            continue
        permutation = rng.permutation(group)
        selected.extend(int(value) for value in permutation[:count])
    if len(selected) < labeled_count:
        selected_set = set(selected)
        remaining_candidates = np.asarray(
            [int(point_id) for point_id in ordered_points.tolist() if int(point_id) not in selected_set],
            dtype=np.int64,
        )
        if remaining_candidates.size > 0:
            permutation = rng.permutation(remaining_candidates)
            extra = permutation[: max(labeled_count - len(selected), 0)]
            selected.extend(int(value) for value in extra.tolist())
    return np.sort(np.asarray(selected[:labeled_count], dtype=np.int64))


def _point_gradient_scores_from_displacement(
    displacement: np.ndarray,
    *,
    candidate_points: np.ndarray,
    time_stop: int,
) -> dict[int, float]:
    scores: dict[int, float] = {}
    upper = max(int(time_stop), 1)
    for point_id in np.asarray(candidate_points, dtype=np.int64):
        series = np.asarray(displacement[int(point_id), :upper], dtype=np.float32)
        finite = series[np.isfinite(series)]
        if finite.size < 2:
            scores[int(point_id)] = 0.0
            continue
        scores[int(point_id)] = float(np.mean(np.abs(np.diff(finite))))
    return scores


def _split_target_labeled_unlabeled(
    bundle: WindowBundle,
    *,
    labeled_points: np.ndarray,
) -> tuple[WindowBundle, WindowBundle]:
    if bundle.size == 0:
        return bundle, bundle
    labeled_mask = np.isin(bundle.point_id, labeled_points.astype(np.int64, copy=False))
    labeled_indices = np.flatnonzero(labeled_mask)
    unlabeled_indices = np.flatnonzero(~labeled_mask)
    return (
        slice_window_bundle(bundle, labeled_indices),
        slice_window_bundle(bundle, unlabeled_indices),
    )
def build_case_data(
    registry_path: str | Path,
    protocol: str,
    case_id: int,
    data_cfg: DataConfig,
    target_labeled_sampling_seed: int,
    cache_root: str | Path | None = None,
) -> CaseData:
    specs = load_dataset_specs(registry_path)
    case = _select_case(specs, protocol, case_id)
    spec_by_id = {spec.domain_id: spec for spec in specs}
    source_specs = [spec_by_id[domain_id] for domain_id in case.source_domain_ids]
    target_spec = spec_by_id[case.target_domain_id]

    cache_dir = Path(cache_root) if cache_root is not None else None
    source_train_bundles = []
    source_val_bundles = []
    for spec in source_specs:
        domain = _load_domain_dataset(spec)
        windows = _load_or_build_windows(domain, data_cfg, cache_dir)
        split = split_source_train_val(
            target_end_idx=windows.target_end_idx,
            target_start_idx=windows.target_start_idx,
            total_time_steps=int(domain.displacement.shape[1]),
            ratio=data_cfg.source_split_ratio,
        )
        source_train_bundles.append(slice_window_bundle(windows, np.where(split.first)[0]))
        source_val_bundles.append(slice_window_bundle(windows, np.where(split.second)[0]))

    target_domain = _load_domain_dataset(target_spec)
    target_windows = _load_or_build_windows(target_domain, data_cfg, cache_dir)

    source_train = concat_window_bundles(source_train_bundles)
    source_val = concat_window_bundles(source_val_bundles)
    source_train = _apply_stride(source_train, data_cfg.source_train_stride)
    source_val = _apply_stride(source_val, data_cfg.source_val_stride)
    target_split = split_target_strict_523(
        target_end_idx=target_windows.target_end_idx,
        target_start_idx=target_windows.target_start_idx,
        total_time_steps=int(target_domain.displacement.shape[1]),
        adapt_ratio=data_cfg.target_adapt_ratio,
        val_ratio=data_cfg.target_val_ratio,
    )
    target_adapt = slice_window_bundle(target_windows, np.where(target_split.adapt)[0])
    target_val_pool = slice_window_bundle(target_windows, np.where(target_split.val)[0])
    target_val = target_val_pool
    target_test = slice_window_bundle(target_windows, np.where(target_split.test)[0])
    target_test = _apply_stride(target_test, data_cfg.target_test_stride)
    target_adapt_train_raw = target_adapt
    target_adapt_train = _apply_stride_per_point(target_adapt_train_raw, data_cfg.target_adapt_stride)
    candidate_points = np.unique(target_adapt.point_id.astype(np.int64, copy=False))
    point_gradient_scores = _point_gradient_scores_from_displacement(
        target_domain.displacement,
        candidate_points=candidate_points,
        time_stop=int(target_split.adapt_end) + 1,
    )
    labeled_points = _sample_labeled_points(
        target_adapt,
        labeled_ratio=data_cfg.target_labeled_ratio,
        seed_text=f"{case.protocol}|{case.case_id}|{case.target_domain_id}|{target_adapt.size}|target_labeled",
        sampling_seed=int(target_labeled_sampling_seed),
        strata=data_cfg.target_labeled_strata,
        point_gradient_scores=point_gradient_scores,
    )
    target_labeled, target_unlabeled = _split_target_labeled_unlabeled(
        target_adapt_train,
        labeled_points=labeled_points,
    )
    target_split_policy = "target_domain_strict_523_time_band"
    target_val_scope = "target_domain_middle_band"
    target_val_split_unit = "time_band"
    target_val_point_policy = "all_points_in_time_band"
    target_labeled_sampling_scope = "target_adapt_band"
    target_labeled_sampling_seed = int(target_labeled_sampling_seed)
    target_labeled_sampling_seed_policy = str(data_cfg.target_labeled_sampling_seed_policy)
    target_labeled_sampling_strategy = "deformation_gradient_stratified"
    target_labeled_strata = int(data_cfg.target_labeled_strata)
    target_adapt_end = int(target_split.adapt_end)
    target_val_end = int(target_split.val_end)

    if (
        source_train.size == 0
        or source_val.size == 0
        or target_labeled.size == 0
        or target_val.size == 0
        or target_test.size == 0
    ):
        raise ValueError("Formal split produced an empty required split.")

    standardizer = FeatureStandardizer.fit(source_train.x)
    target_adapt_shift_bundle = _apply_standardizer(target_adapt, standardizer)
    target_adapt_train_raw_shift_bundle = _apply_standardizer(target_adapt_train_raw, standardizer)
    target_adapt_train_shift_bundle = _apply_standardizer(target_adapt_train, standardizer)
    source_train = _apply_standardizer(source_train, standardizer)
    source_val = _apply_standardizer(source_val, standardizer)
    target_labeled = _apply_standardizer(target_labeled, standardizer)
    target_unlabeled = _apply_standardizer(target_unlabeled, standardizer)
    target_val = _apply_standardizer(target_val, standardizer)
    target_test = _apply_standardizer(target_test, standardizer)

    metadata = {
        "protocol": case.protocol,
        "case_id": int(case.case_id),
        "case_name": case.case_name,
        "window_mode": "observation_step_displacement_only",
        "split_layout": str(data_cfg.split_layout),
        "target_split_policy": target_split_policy,
        "source_domain_ids": list(case.source_domain_ids),
        "target_domain_id": int(case.target_domain_id),
        "source_train_windows": int(source_train.size),
        "source_val_windows": int(source_val.size),
        "target_adapt_windows": int(target_adapt.size),
        "target_adapt_train_raw_windows": int(target_adapt_train_raw.size),
        "target_adapt_train_windows": int(target_adapt_train.size),
        "target_val_pool_windows": int(target_val_pool.size),
        "target_val_windows": int(target_val.size),
        "target_adapt_ratio": float(data_cfg.target_adapt_ratio),
        "target_val_ratio": float(data_cfg.target_val_ratio),
        "target_test_ratio": float(1.0 - float(data_cfg.target_adapt_ratio) - float(data_cfg.target_val_ratio)),
        "target_adapt_end_idx": int(target_adapt_end),
        "target_val_end_idx": int(target_val_end),
        "target_labeled_windows": int(target_labeled.size),
        "target_unlabeled_windows": int(target_unlabeled.size),
        "target_test_windows": int(target_test.size),
        "target_labeled_ratio": float(data_cfg.target_labeled_ratio),
        "target_labeled_sampling_scope": target_labeled_sampling_scope,
        "target_labeled_sampling_unit": "point",
        "target_labeled_sampling_seed": int(target_labeled_sampling_seed),
        "target_labeled_sampling_seed_policy": target_labeled_sampling_seed_policy,
        "target_labeled_sampling_strategy": target_labeled_sampling_strategy,
        "target_labeled_strata": int(target_labeled_strata),
        "target_val_scope": target_val_scope,
        "target_val_split_unit": target_val_split_unit,
        "target_val_point_policy": target_val_point_policy,
        "shift_score_space": "source_standardized_input",
        "shift_score_reference": "source_train",
        "shift_severity_basis": "target_adapt_shift_score",
        "target_adapt_points": int(np.unique(target_adapt.point_id).size) if target_adapt.size > 0 else 0,
        "target_adapt_train_raw_points": int(np.unique(target_adapt_train_raw.point_id).size) if target_adapt_train_raw.size > 0 else 0,
        "target_adapt_train_points": int(np.unique(target_adapt_train.point_id).size) if target_adapt_train.size > 0 else 0,
        "target_labeled_points": int(labeled_points.size),
        "target_unlabeled_points": int(np.unique(target_unlabeled.point_id).size) if target_unlabeled.size > 0 else 0,
        "target_val_points": int(np.unique(target_val.point_id).size) if target_val.size > 0 else 0,
        "target_test_points": int(np.unique(target_test.point_id).size) if target_test.size > 0 else 0,
        "source_train_shift_score": _bundle_shift_score(source_train),
        "source_val_shift_score": _bundle_shift_score(source_val),
        "target_adapt_shift_score": _bundle_shift_score(target_adapt_shift_bundle),
        "target_adapt_train_raw_shift_score": _bundle_shift_score(target_adapt_train_raw_shift_bundle),
        "target_adapt_train_shift_score": _bundle_shift_score(target_adapt_train_shift_bundle),
        "target_labeled_shift_score": _bundle_shift_score(target_labeled),
        "target_unlabeled_shift_score": _bundle_shift_score(target_unlabeled),
        "target_val_shift_score": _bundle_shift_score(target_val),
        "target_test_shift_score": _bundle_shift_score(target_test),
        "source_train_stride": int(data_cfg.source_train_stride),
        "source_val_stride": int(data_cfg.source_val_stride),
        "target_adapt_stride": int(data_cfg.target_adapt_stride),
        "target_test_stride": int(data_cfg.target_test_stride),
    }
    return CaseData(
        source_train=source_train,
        source_val=source_val,
        target_labeled=target_labeled,
        target_unlabeled=target_unlabeled,
        target_val=target_val,
        target_test=target_test,
        target_point_latlon=np.asarray(target_domain.latlon, dtype=np.float32),
        metadata=metadata,
    )
