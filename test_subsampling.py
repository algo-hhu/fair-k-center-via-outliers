import numpy as np
import pytest
from subsampling import subsampling_fair


X = np.arange(18).reshape(9, 2)
# class 0: 6 points, class 1: 3 points  (2:1 imbalance)
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])


def test_output_size():
    X_s, y_s = subsampling_fair(X, y, num_samples=5, seed=0)
    assert X_s.shape == (5, 2)
    assert y_s.shape == (5,)


def test_rows_match():
    X_s, y_s = subsampling_fair(X, y, num_samples=5, seed=0)
    for xi, yi in zip(X_s, y_s):
        idx = np.where((X == xi).all(axis=1))[0][0]
        assert y[idx] == yi


def test_fair_weights_favour_minority():
    # With fair weights class 1 (3 points) has 2x the per-point probability
    # of class 0 (6 points).  Over many draws, class 1 should be over-represented
    # relative to its share in the original data.
    counts = np.zeros(2, dtype=int)
    for seed in range(200):
        _, y_s = subsampling_fair(X, y, num_samples=4, seed=seed)
        for label in y_s:
            counts[label] += 1
    class1_share = counts[1] / counts.sum()
    # Under uniform sampling expected share is 3/9 ≈ 0.33;
    # under fair weights it should be 0.5.
    assert class1_share > 0.40


def test_reproducible_with_seed():
    _, y1 = subsampling_fair(X, y, num_samples=5, seed=42)
    _, y2 = subsampling_fair(X, y, num_samples=5, seed=42)
    np.testing.assert_array_equal(y1, y2)


def test_different_seeds_differ():
    _, y1 = subsampling_fair(X, y, num_samples=5, seed=0)
    _, y2 = subsampling_fair(X, y, num_samples=5, seed=1)
    assert not np.array_equal(y1, y2)
