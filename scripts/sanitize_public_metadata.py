from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(metadata)
    source_path = cleaned.pop("source_path", None)
    if source_path:
        cleaned.setdefault("source_file", Path(str(source_path)).name)
    cleaned.setdefault("public_release_sanitized", True)
    return cleaned


def sanitize_npz(path: Path, *, dry_run: bool = False) -> bool:
    with np.load(path, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}

    raw_metadata = str(payload.get("metadata_json", np.asarray("{}")).tolist())
    metadata = json.loads(raw_metadata)
    cleaned = _sanitize_metadata(metadata)
    if cleaned == metadata:
        return False

    payload["metadata_json"] = np.asarray(json.dumps(cleaned, ensure_ascii=False))
    if not dry_run:
        np.savez_compressed(path, **payload)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remove local-only paths from public .npz metadata.")
    parser.add_argument("--root", default="data/domains_10k_50x50")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root)
    changed = 0
    for path in sorted(root.rglob("*.npz")):
        if sanitize_npz(path, dry_run=args.dry_run):
            changed += 1
            print(path)
    print(f"sanitized={changed}")


if __name__ == "__main__":
    main()
