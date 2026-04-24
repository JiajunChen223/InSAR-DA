# InSAR-DA

InSAR-DA is a formal experiment core for low-supervision domain adaptation on
sampled public InSAR deformation time-series domains.

Author: Jiajun Chen, College of Earth Sciences, Jilin University.

## Scope

- Methods: `source_only`, `target_only`, `supervised_fine_tuning`, `st_joint`,
  `ss_dann`, `ss_mt`, `ss_coral`, `sft_replay`
- Protocols: `CHT`, `IHT`, `LODO`
- Label rates: `0.005`, `0.01`, `0.025`, `0.05`
- Seeds: `42`, `43`, `44`
- Backbone: `transformer`

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

Run one case:

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

Runtime outputs are written under `runtime/`:

- Runs: `runtime/runs`
- Summary tables: `runtime/summary`
- Cache: `runtime/cache`

These paths are ignored by Git because trained weights, predictions, logs, and
machine-specific config snapshots are generated artifacts.

## Data

The included `.npz` files are sampled InSAR time-series archives. See
`DATASET.md` for the file format, source tags, and the source-data attribution
fields that must be completed before a public release.

## Tests

Run the release smoke tests:

```bash
python -m pytest
```

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
