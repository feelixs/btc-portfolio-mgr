from __future__ import annotations

import numpy as np
import pytest

from btc_portfolio_mgr.model.cv import purged_kfold_indices


def test_basic_fold_partition() -> None:
    # n=100, 5 folds, no purge/embargo (horizon=embargo=0).
    folds = list(
        purged_kfold_indices(n=100, n_folds=5, label_horizon_hours=0, embargo_hours=0)
    )
    assert len(folds) == 5
    # Each test fold has 20 rows
    for train_idx, test_idx in folds:
        assert test_idx.shape[0] == 20
        # train + test together cover all indices (no purge/embargo)
        assert sorted(set(train_idx.tolist()) | set(test_idx.tolist())) == list(range(100))
        # train and test are disjoint
        assert len(set(train_idx) & set(test_idx)) == 0


def test_purge_removes_samples_with_overlapping_labels() -> None:
    # n=100, 5 folds, horizon=10, embargo=0.
    # Fold 1: test_idx=[20..40). Purged from train: [10..20).
    folds = list(
        purged_kfold_indices(n=100, n_folds=5, label_horizon_hours=10, embargo_hours=0)
    )
    train_idx, test_idx = folds[1]
    assert test_idx.tolist() == list(range(20, 40))
    train_set = set(train_idx.tolist())
    # Purged region [10..20) must NOT be in train
    for i in range(10, 20):
        assert i not in train_set, f"index {i} should be purged"
    # Index 9 should still be in train
    assert 9 in train_set
    # Indices after test [40..100) should be in train (no embargo)
    for i in range(40, 100):
        assert i in train_set


def test_embargo_removes_samples_after_test() -> None:
    # n=100, 5 folds, horizon=0, embargo=10.
    folds = list(
        purged_kfold_indices(n=100, n_folds=5, label_horizon_hours=0, embargo_hours=10)
    )
    train_idx, test_idx = folds[1]
    train_set = set(train_idx.tolist())
    # Embargo [40..50) must NOT be in train
    for i in range(40, 50):
        assert i not in train_set, f"index {i} should be embargoed"
    # Index 50 should be in train
    assert 50 in train_set


def test_purge_and_embargo_combined() -> None:
    folds = list(
        purged_kfold_indices(n=100, n_folds=5, label_horizon_hours=5, embargo_hours=5)
    )
    train_idx, test_idx = folds[2]  # middle fold: test=[40..60)
    train_set = set(train_idx.tolist())
    # Purged: [35..40)
    for i in range(35, 40):
        assert i not in train_set
    # Embargoed: [60..65)
    for i in range(60, 65):
        assert i not in train_set
    # Before purge zone: in train
    assert 34 in train_set
    # After embargo zone: in train
    assert 65 in train_set


def test_first_fold_handles_negative_purge_start() -> None:
    # Fold 0: test_idx=[0..20). Purge would start at -10, clamp to 0.
    folds = list(
        purged_kfold_indices(n=100, n_folds=5, label_horizon_hours=10, embargo_hours=0)
    )
    train_idx, test_idx = folds[0]
    assert test_idx.tolist() == list(range(0, 20))
    # No negative indices in train
    assert (train_idx >= 0).all()
    assert (train_idx < 100).all()
    # Embargo end is just past test_end=20, no embargo since embargo=0
    train_set = set(train_idx.tolist())
    for i in range(20, 100):
        assert i in train_set


def test_last_fold_includes_remainder() -> None:
    # n=103, 5 folds: first 4 have size 20, last has 23.
    folds = list(
        purged_kfold_indices(n=103, n_folds=5, label_horizon_hours=0, embargo_hours=0)
    )
    sizes = [test_idx.shape[0] for _, test_idx in folds]
    assert sizes == [20, 20, 20, 20, 23]


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        list(purged_kfold_indices(n=100, n_folds=0, label_horizon_hours=0, embargo_hours=0))
    with pytest.raises(ValueError):
        list(purged_kfold_indices(n=100, n_folds=5, label_horizon_hours=-1, embargo_hours=0))
    with pytest.raises(ValueError):
        list(purged_kfold_indices(n=100, n_folds=5, label_horizon_hours=0, embargo_hours=-1))
