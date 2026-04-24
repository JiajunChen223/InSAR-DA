from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FeatureStandardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "FeatureStandardizer":
        array = np.asarray(x, dtype=np.float32)
        if array.ndim != 3:
            raise ValueError(f"Expected x with shape [N, L, C], got {array.shape}")
        mean = array.mean(axis=(0, 1), keepdims=True)
        std = array.std(axis=(0, 1), keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float32) - self.mean) / self.std).astype(np.float32)
