"""Paired bootstrap utilities for same-study report-generation comparisons."""
from __future__ import annotations

import numpy as np


def paired_bootstrap(values_a, values_b=None, statistic=np.mean, samples: int = 2000, seed: int = 1337) -> dict[str, float]:
    a = np.asarray(values_a, dtype=float)
    if a.ndim != 1 or a.size == 0: raise ValueError("values_a must be a non-empty vector")
    rng = np.random.default_rng(seed); indices = rng.integers(0, a.size, size=(samples, a.size))
    if values_b is None:
        draws = np.asarray([statistic(a[idx]) for idx in indices])
        point = float(statistic(a)); low, high = np.percentile(draws, [2.5, 97.5])
        return {"estimate": point, "ci95_low": float(low), "ci95_high": float(high), "samples": samples, "seed": seed}
    b = np.asarray(values_b, dtype=float)
    if b.shape != a.shape: raise ValueError("paired vectors must have identical shape")
    draws = np.asarray([statistic(a[idx] - b[idx]) for idx in indices])
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"difference": float(statistic(a - b)), "ci95_low": float(low), "ci95_high": float(high), "samples": samples, "seed": seed}

