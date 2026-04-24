# Open-Source Release Checklist

Use this checklist before publishing the repository.

## Required Before Public Release

- [x] Confirm the code copyright holder in `LICENSE`.
- [ ] Confirm that MIT is the intended code license, or replace `LICENSE` and
      `pyproject.toml` with the intended OSI-approved license.
- [x] Replace the author fields in `CITATION.cff`.
- [x] Replace `repository-code` and `url` in `CITATION.cff` after creating the
      public repository or archive.
- [x] Fill every DOI/URL, license, and attribution field in `DATASET.md`.
- [ ] Confirm that each included `.npz` file may be redistributed publicly.
- [x] Confirm whether COSMO-SkyMed-derived products have redistribution terms
      compatible with the chosen public release.
- [x] Remove or keep ignored any local-only `runtime/` outputs.
- [x] Run `python -m pytest`.
- [ ] Run at least one documented `scripts/run_case.py` command from a clean
      clone or fresh environment.

## Optional But Recommended

- [ ] Publish full trained weights and generated predictions as a separate
      GitHub Release, Zenodo record, or institutional archive instead of the
      source repository.
- [ ] Add a paper DOI or preprint DOI to `CITATION.cff`.
- [ ] Add badges after CI is configured.
- [ ] Configure GitHub Actions to run the smoke tests on pull requests.
