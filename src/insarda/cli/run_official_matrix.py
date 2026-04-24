from __future__ import annotations

import argparse
from pathlib import Path

from insarda.config import load_formal_config


def _ensure_main_config(path: str) -> None:
    if Path(path).name != "main.yaml":
        raise ValueError("Only the paper config `configs/main.yaml` is supported.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full official paper matrix.")
    parser.add_argument("--config", default="configs/main.yaml")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _ensure_main_config(args.config)
    config = load_formal_config(args.config)
    from insarda.training.runner import run_sweep

    total = 0
    for protocol in config.experiments.main.protocols:
        records = run_sweep(
            config=config,
            protocol=protocol,
            methods=list(config.experiments.main.methods),
            backbones=list(config.experiments.main.backbones),
            seeds=list(config.experiments.main.seeds),
            label_rates=list(config.experiments.main.label_rates),
        )
        total += len(records)
    print(total)


if __name__ == "__main__":
    main()
