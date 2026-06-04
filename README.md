# InSAR-DA

[![Python](https://img.shields.io/badge/Python-%E2%89%A53.10-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-supported-EE4C2C?logo=pytorch&logoColor=white)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-blue)](DATASET.md)

**InSAR-DA** is a formal low-label MT-InSAR transfer-risk benchmark for deformation time-series forecasting. It fixes public GeoSAR domains, transfer tasks, label budgets, temporal splits, target-test access, model capacity, and random seeds so that methods are compared by transfer organization rather than by changing data construction.

Author: **Jiajun Chen**, College of Earth Sciences, Jilin University.

## At A Glance

| Item | Setting |
| --- | --- |
| Task | Low-label domain adaptation for MT-InSAR deformation forecasting |
| Protocols | `CHT`, `IHT`, `LODO` |
| Methods | `source_only`, `target_only`, `supervised_fine_tuning`, `st_joint`, `ss_dann`, `ss_mt`, `ss_coral`, `sft_replay` |
| Label rates | `0.005`, `0.01`, `0.025`, `0.05` |
| Seeds | `42`, `43`, `44` |
| Backbone | Transformer |
| Sampling | 10,000 points per domain on a 50 x 50 grid, sampling seed `42` |
| Formal matrix | 24 transfer tasks x 8 methods x 4 label rates x 3 seeds |

## Repository Map

| Path | Purpose |
| --- | --- |
| `configs/main.yaml` | Formal experiment configuration |
| `data/datasets_public_true_types_obs_step_final_10k_50x50.yaml` | Dataset and task registry |
| `data/domains_10k_50x50/` | Sampled public-domain `.npz` archives used by formal runs |
| `src/insarda/` | Package source |
| `scripts/run_case.py` | Run one formal case |
| `scripts/run_sweep.py` | Run one protocol sweep |
| `scripts/run_official_matrix.py` | Run the full formal matrix |
| `scripts/summarize.py` | Summarize completed runs |
| `DATASET.md` | File format, provenance, license, split rules, and attribution |

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

Python 3.10 or newer is required. GPU execution is selected automatically when available.

## Quick Start

Run the smoke tests:

```bash
python -m pytest
```

Run one formal case:

```bash
python scripts/run_case.py --protocol IHT --case 0 --method source_only --seed 42 --label-rate 0.005
```

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

Runtime outputs are written under `runtime/` and ignored by Git:

| Output | Path |
| --- | --- |
| Runs | `runtime/runs/` |
| Summary tables | `runtime/summary/` |
| Cache | `runtime/cache/` |

One-case outputs include `metrics.json`, `config_snapshot.yaml`, `train_history.json`, `model.pt`, `method.pt`, and `predictions.npz`.

## Data

This repository includes compact sampled InSAR time-series `.npz` archives under `data/domains_10k_50x50/`. They are derived from the public INGV InSAR ground displacement time-series archive:

```text
InSAR Working Group. (2013). InSAR ground displacement time series.
Istituto Nazionale di Geofisica e Vulcanologia (INGV).
https://doi.org/10.13127/insar/ts
```

See [DATASET.md](DATASET.md) for source tags, file format, selected point identifiers, grid metadata, temporal split rules, target-label seed policy, license, and required attribution.

## Benchmark Boundary

The benchmark is intentionally fixed-scope. Main comparison claims should use the formal matrix settings in `configs/main.yaml` and the registry in `data/datasets_public_true_types_obs_step_final_10k_50x50.yaml`. Generated runtime artifacts are not part of the source release and should be archived separately if full reproduced outputs are published.

## Citation

If you use this repository, cite the software using [CITATION.cff](CITATION.cff) and cite the original INGV data archive listed above.

## License

The code is released under the MIT License. The sampled data are derived from the INGV InSAR archive and are distributed under CC BY 4.0 as described in [DATASET.md](DATASET.md).
