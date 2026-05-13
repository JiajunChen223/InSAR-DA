# InSAR-DA

InSAR-DA is the formal experiment core for transfer-risk benchmarking in
low-label MT-InSAR deformation forecasting. It fixes sampled public GeoSAR
domains, transfer tasks, label budgets, temporal splits, target-test access,
model capacity, and random seeds so that methods are compared by transfer
organization rather than by changing data construction or tuning scope.

Author: Jiajun Chen, College of Earth Sciences, Jilin University.

## Benchmark Scope

- Methods: `source_only`, `target_only`, `supervised_fine_tuning`, `st_joint`,
  `ss_dann`, `ss_mt`, `ss_coral`, `sft_replay`
- Protocols: `CHT`, `IHT`, `LODO`
- Label rates: `0.005`, `0.01`, `0.025`, `0.05`
- Seeds: `42`, `43`, `44`
- Backbone: `transformer`
- Target split: adaptation/validation/test time bands at 5:2:3
- Sampling: 10,000 points per domain on a 50 x 50 grid, sampling seed `42`

## Repository Layout

- `configs/main.yaml`: formal experiment configuration.
- `data/datasets_public_true_types_obs_step_final_10k_50x50.yaml`: dataset registry.
- `data/domains_10k_50x50/`: sampled public-domain archives used by the formal runs.
- `src/insarda/`: package source.
- `scripts/run_case.py`: run one formal case.
- `scripts/run_sweep.py`: run one protocol sweep.
- `scripts/run_official_matrix.py`: run the full formal matrix.
- `scripts/summarize.py`: summarize completed runs.

## Installation

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Python 3.10 or newer is required. The default dependency set installs NumPy,
PyYAML, and PyTorch. GPU execution is selected automatically when available.

## Quick Start

Run the release smoke tests:

```bash
python -m pytest
```

Run one case:

```bash
python scripts/run_case.py --protocol IHT --case 0 --method source_only --seed 42 --label-rate 0.005
```

Expected one-case outputs are written to a timestamped directory under
`runtime/runs/` and include `metrics.json`, `config_snapshot.yaml`,
`train_history.json`, `model.pt`, `method.pt`, and `predictions.npz`.

Run one protocol sweep:

```bash
python scripts/run_sweep.py --protocol CHT
python scripts/run_sweep.py --protocol IHT
python scripts/run_sweep.py --protocol LODO
```

Run the full formal matrix:

```bash
python scripts/run_official_matrix.py
```

Summarize completed runs:

```bash
python scripts/summarize.py
python scripts/summarize.py --protocol IHT
```

## Outputs

Runtime outputs are written under `runtime/`:

- Runs: `runtime/runs`
- Summary tables: `runtime/summary`
- Cache: `runtime/cache`

These paths are ignored by Git because trained weights, predictions, logs, and
machine-specific config snapshots are generated artifacts.

## Data

The included `.npz` files are sampled InSAR time-series archives. See
`DATASET.md` for the file format, source tags, source-data attribution,
selected point identifiers, grid metadata, task definitions, temporal split
rules, and target-label seed policy used by the benchmark.

## Reproduction Entry Points

- Main configuration: `configs/main.yaml`
- One formal case: `scripts/run_case.py`
- One protocol sweep: `scripts/run_sweep.py`
- Full formal matrix: `scripts/run_official_matrix.py`
- Summary tables: `scripts/summarize.py`

The formal matrix covers 24 transfer tasks, 8 methods, 4 target adaptation-label
rates, and 3 random seeds. Full-matrix execution is substantially longer than
the smoke tests and writes generated artifacts under `runtime/`.

## Citation

If you use this repository, cite the project using `CITATION.cff` and cite the
original INGV data archive:

```text
InSAR Working Group. (2013). InSAR ground displacement time series.
Istituto Nazionale di Geofisica e Vulcanologia (INGV).
https://doi.org/10.13127/insar/ts
```

## License

The code is released under the MIT License. The sampled data are derived from
the INGV InSAR ground displacement time-series archive, distributed under CC BY
4.0 as described in `DATASET.md`.
