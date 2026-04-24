from __future__ import annotations

import argparse
from pathlib import Path

from insarda.config import load_formal_config
from insarda.training.runner import run_sweep


def _ensure_main_config(path: str) -> None:
    if Path(path).name != "main.yaml":
        raise ValueError("Only the single formal config `configs/main.yaml` is supported.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the formal protocol sweep.")
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--case", action="append", type=int)
    parser.add_argument("--method", action="append")
    parser.add_argument("--backbone", action="append", choices=["transformer"])
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--label-rate", action="append", type=float)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _ensure_main_config(args.config)
    config = load_formal_config(args.config)
    records = run_sweep(
        config=config,
        protocol=args.protocol,
        methods=args.method,
        backbones=args.backbone,
        cases=args.case,
        seeds=args.seed,
        label_rates=args.label_rate,
    )
    print(len(records))


if __name__ == "__main__":
    main()
