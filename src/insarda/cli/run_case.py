from __future__ import annotations

import argparse
from pathlib import Path

from insarda.config import ensure_formal_run_args, load_formal_config
from insarda.training.runner import run_case


def _ensure_main_config(path: str) -> None:
    if Path(path).name != "main.yaml":
        raise ValueError("Only the single formal config `configs/main.yaml` is supported.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one formal InSAR-DA case.")
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--case", required=True, type=int)
    parser.add_argument("--method", required=True)
    parser.add_argument("--backbone", default="transformer", choices=["transformer"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--label-rate", type=float)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _ensure_main_config(args.config)
    config = load_formal_config(args.config)
    run_args = ensure_formal_run_args(
        protocol=args.protocol,
        case_id=args.case,
        method=args.method,
        backbone=args.backbone,
        seed=args.seed,
        target_labeled_ratio=args.label_rate,
    )
    metrics = run_case(config, run_args)
    print(metrics["run_dir"])


if __name__ == "__main__":
    main()
