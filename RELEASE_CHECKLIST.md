# Release Checklist

Use this checklist when preparing a new public release or paper-aligned tag.

## Required

- Confirm the MIT code license in `LICENSE`.
- Keep author and repository metadata current in `CITATION.cff`.
- Keep dataset source tags, license notes, and attribution current in `DATASET.md`.
- Run `python -m pytest`.
- Run at least one documented case command from a clean environment:

```bash
python scripts/run_case.py --protocol IHT --case 0 --method source_only --seed 42 --label-rate 0.005
```

- Confirm that generated `runtime/` outputs, checkpoints, caches, and local paths are not staged.

## Recommended

- Add a paper DOI or preprint DOI to `CITATION.cff` when available.
- Publish trained weights, predictions, and full summary artifacts through a GitHub Release, Zenodo record, or institutional archive.
- Add CI badges after GitHub Actions or another test runner is configured.
- Tag the repository version that corresponds to each submitted or published manuscript.
