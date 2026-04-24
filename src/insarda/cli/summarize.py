from __future__ import annotations

import argparse
from pathlib import Path

from insarda.config import load_formal_config
from insarda.reporting.summarize import summarize_runs


def _ensure_main_config(path: str) -> None:
    if Path(path).name != "main.yaml":
        raise ValueError("Only the single formal config `configs/main.yaml` is supported.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize formal InSAR-DA runs.")
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--protocol")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _ensure_main_config(args.config)
    config = load_formal_config(args.config)
    summary = summarize_runs(config=config, protocol=args.protocol)
    print(summary["results_csv"])


if __name__ == "__main__":
    main()
