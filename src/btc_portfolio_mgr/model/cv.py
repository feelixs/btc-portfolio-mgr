"""López de Prado purged k-fold cross-validation indices.

For each test fold:
- Purge training samples whose target window overlaps the test region.
- Embargo training samples immediately after the test region whose
  feature lookback windows may reach back into the test region.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np


def purged_kfold_indices(
    n: int,
    n_folds: int,
    label_horizon_hours: int,
    embargo_hours: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) tuples for purged k-fold CV.

    Assumes the underlying dataset is already sorted by timestamp.
    """
    if n_folds <= 0:
        raise ValueError(f"n_folds must be positive, got {n_folds}")
    if label_horizon_hours < 0:
        raise ValueError(
            f"label_horizon_hours must be non-negative, got {label_horizon_hours}"
        )
    if embargo_hours < 0:
        raise ValueError(f"embargo_hours must be non-negative, got {embargo_hours}")
    fold_size = n // n_folds
    for k in range(n_folds):
        test_start = k * fold_size
        test_end = (k + 1) * fold_size if k < n_folds - 1 else n
        purge_start = max(0, test_start - label_horizon_hours)
        embargo_end = min(n, test_end + embargo_hours)
        test_idx = np.arange(test_start, test_end)
        train_idx = np.concatenate(
            [
                np.arange(0, purge_start),
                np.arange(embargo_end, n),
            ]
        )
        yield train_idx, test_idx
