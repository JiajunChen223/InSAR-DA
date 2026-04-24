# Contributing

Thank you for helping improve InSAR-DA.

## Development Setup

Create an environment and install the project in editable mode:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Checks

Run the smoke tests before submitting changes:

```bash
python -m pytest
```

For experiment changes, also run at least one small formal case and summarize
the output:

```bash
python scripts/run_case.py --protocol IHT --case 0 --method source_only --seed 42 --label-rate 0.005
python scripts/summarize.py --protocol IHT
```

## Repository Hygiene

- Do not commit `runtime/` outputs, cache files, local paths, private data, or
  Python bytecode.
- Keep public data provenance in `DATASET.md` synchronized with the dataset
  registry.
- Keep changes scoped to the method, model, data-processing, or reporting area
  they affect.
- If you add a new dependency, add it to `pyproject.toml` and document why it is
  needed.
