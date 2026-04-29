from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from insarda.config import load_formal_config  # noqa: E402
from insarda.data_prep.io import load_npz  # noqa: E402


def test_formal_config_loads() -> None:
    config = load_formal_config(ROOT / "configs" / "main.yaml")

    assert config.paths.dataset_registry.exists()
    assert config.experiments.main.protocols == ("CHT", "IHT", "LODO")
    assert "source_only" in config.experiments.main.methods
    assert "sft_replay" in config.experiments.main.methods


def test_dataset_registry_points_to_public_npz_files() -> None:
    registry_path = ROOT / "data" / "datasets_public_true_types_obs_step_final_10k_50x50.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))

    assert len(registry["datasets"]) == 6
    for item in registry["datasets"]:
        data_path = registry_path.parent / item["path"]
        assert data_path.exists(), data_path

        bundle = load_npz(data_path)
        assert bundle.displacement.shape[0] == 10000
        assert bundle.latlon.shape == (10000, 2)
        assert bundle.dates.ndim == 1
        assert "source_path" not in bundle.metadata


def test_required_release_documents_exist() -> None:
    for name in [
        "LICENSE",
        "README.md",
        "DATASET.md",
        "CITATION.cff",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "RELEASE_CHECKLIST.md",
        "SECURITY.md",
        ".gitignore",
    ]:
        assert (ROOT / name).exists(), name
