from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DataBundle:
    displacement: np.ndarray
    dates: np.ndarray
    latlon: np.ndarray
    optional: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _safe_optional_key(name: str) -> str:
    safe = "".join(char if char.isalnum() or char == "_" else "_" for char in str(name)).strip("_")
    return safe or "field"


def load_npz(path: str | Path) -> DataBundle:
    with np.load(path, allow_pickle=False) as data:
        optional: dict[str, np.ndarray] = {}
        metadata: dict[str, Any] = {}
        for key in data.files:
            if key in {"displacement_full", "displacement", "dates", "latlon"}:
                continue
            if key == "metadata_json":
                metadata = json.loads(str(data[key].tolist()))
                continue
            if key.startswith("optional__"):
                optional[key[len("optional__") :]] = np.asarray(data[key])
                continue
            optional[key] = np.asarray(data[key])
        displacement_key = "displacement_full" if "displacement_full" in data.files else "displacement"
        return DataBundle(
            displacement=np.asarray(data[displacement_key], dtype=np.float32),
            dates=np.asarray(data["dates"], dtype=np.int32),
            latlon=np.asarray(data["latlon"], dtype=np.float32),
            optional=optional,
            metadata=metadata,
        )


def save_npz(bundle: DataBundle, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "displacement_full": np.asarray(bundle.displacement, dtype=np.float32),
        "dates": np.asarray(bundle.dates, dtype=np.int32),
        "latlon": np.asarray(bundle.latlon, dtype=np.float32),
        "metadata_json": np.asarray(json.dumps(bundle.metadata, ensure_ascii=False)),
    }
    for key, value in bundle.optional.items():
        payload[f"optional__{_safe_optional_key(key)}"] = np.asarray(value)
    np.savez_compressed(out, **payload)
    return out
