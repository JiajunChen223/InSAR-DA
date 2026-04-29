# InSAR-DA

**Low-supervision domain adaptation for InSAR deformation time series.**

InSAR-DA is a research codebase for controlled comparisons of transfer strategies across public InSAR deformation domains. It focuses on the setting where a source domain has richer supervision, a target domain has very limited labels, and the goal is robust multi-step deformation prediction under cross-hazard or cross-region shift.

The repository includes a compact public-domain experiment set, a formal configuration system, training and evaluation entry points, and summary tools for comparing domain-adaptation methods under consistent protocols.

Author: **Jiajun Chen**, College of Earth Sciences, Jilin University.

## Highlights

- Low-supervision target adaptation with label rates from `0.005` to `0.05`.
- Controlled protocol families: `CHT`, `IHT`, and `LODO`.
- Baselines and adaptation methods: `source_only`, `target_only`, `supervised_fine_tuning`, `st_joint`, `ss_dann`, `ss_mt`, `ss_coral`, and `sft_replay`.
- Transformer-based multi-step deformation forecasting for sampled InSAR time-series windows.
- Public sampled `.npz` domains with source tags, attribution, and data format documentation.
- Runtime summaries for completed formal runs and protocol-level comparisons.

## Repository Layout

```text
configs/    Formal experiment configuration
data/       Public sampled InSAR domains and dataset registry
runtime/    Ignored runtime outputs, summaries, and caches
scripts/    Case, sweep, matrix, and summary entry points
src/        InSAR-DA package source
tests/      Public-release smoke tests
```

## Installation

Create a Python environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Python 3.10 or newer is required. GPU execution is selected automatically when a CUDA device is available.

## Data

The included public sampled domains are stored under:

```text
data/domains_10k_50x50/
```

Each domain file contains 10,000 sampled points on a 50 x 50 grid layout, with deformation time series, observation indices, coordinates, and metadata. The dataset registry is:

```text
data/datasets_public_true_types_obs_step_final_10k_50x50.yaml
```

See `DATASET.md` for the file format, source tags, license, and required attribution for the INGV InSAR ground displacement time-series archive.

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

## Protocols

Protocol families:

- `CHT`: cross-hazard transfer.
- `IHT`: intra-hazard transfer.
- `LODO`: leave-one-domain-out evaluation.

Methods:

- `source_only`: train on source supervision only.
- `target_only`: train on limited target labels only.
- `supervised_fine_tuning`: source pretraining followed by target fine-tuning.
- `st_joint`: supervised joint training with source and target labels.
- `ss_dann`: semi-supervised domain-adversarial adaptation.
- `ss_mt`: semi-supervised mean-teacher adaptation.
- `ss_coral`: semi-supervised CORAL feature alignment.
- `sft_replay`: supervised fine-tuning with source replay.

The formal matrix uses label rates `0.005`, `0.01`, `0.025`, and `0.05`, seeds `42`, `43`, and `44`, and the transformer backbone configured in `configs/main.yaml`.

## Outputs

Runtime artifacts are written under `runtime/`:

```text
runtime/runs/
runtime/summary/
runtime/cache/
```

These directories are ignored by Git. They contain generated weights, predictions, logs, summaries, and machine-specific resolved configs.

## Tests

Run the smoke tests:

```bash
python -m pytest
```

The tests verify that the formal config loads, the public dataset registry resolves to bundled `.npz` files, and required release documents are present.

## Citation

If you use this repository, cite the project using `CITATION.cff` and cite the original INGV data archive:

```text
InSAR Working Group. (2013). InSAR ground displacement time series.
Istituto Nazionale di Geofisica e Vulcanologia (INGV).
https://doi.org/10.13127/insar/ts
```

## License

The code is released under the MIT License. The sampled data are derived from the INGV InSAR ground displacement time-series archive and are distributed with CC BY 4.0 attribution requirements described in `DATASET.md`.
