import numpy as np
import pytest

from lattice_data_tools.bootstrap import (
    BootstrapSamples,
    uncorrelated_confs_to_bts,
    parametric_gaussian_bts,
    binning,
)


def test_uncorrelated_confs_to_bts_shape():
    np.random.seed(42)
    data = np.random.normal(loc=0.0, scale=1.0, size=200)
    bts = uncorrelated_confs_to_bts(x=data, N_bts=100)
    assert bts.shape == (101,), f"Expected shape (101,), got {bts.shape}"


def test_uncorrelated_confs_to_bts_mean():
    np.random.seed(42)
    true_mean = 5.0
    sigma = 1.0
    N = 200
    data = np.random.normal(loc=true_mean, scale=sigma, size=N)
    bts = uncorrelated_confs_to_bts(x=data, N_bts=100)
    estimated_mean = bts.unbiased_mean()
    # unbiased mean should be close to true_mean within 3 sigma / sqrt(N)
    tolerance = 3.0 * sigma / np.sqrt(N)
    assert abs(estimated_mean - true_mean) < tolerance, (
        f"unbiased_mean {estimated_mean} not within {tolerance} of {true_mean}"
    )


def test_bootstrap_samples_error_blocks_numpy_mean():
    np.random.seed(42)
    data = np.random.normal(size=200)
    bts = uncorrelated_confs_to_bts(x=data, N_bts=100)
    with pytest.raises(TypeError):
        np.mean(bts)


def test_parametric_gaussian_bts():
    bts = parametric_gaussian_bts(mean=5.0, error=0.1, N_bts=500, seed=99)
    # shape should be (501,): 500 samples + 1 mean
    assert bts.shape == (501,), f"Expected shape (501,), got {bts.shape}"
    assert abs(bts.unbiased_mean() - 5.0) < 0.05, (
        f"unbiased_mean {bts.unbiased_mean()} not close to 5.0"
    )


def test_binning():
    np.random.seed(0)
    arr = np.random.normal(size=100)
    binned = binning(arr, bin_size=2)
    assert binned.shape[0] == 50, f"Expected 50 bins, got {binned.shape[0]}"


def test_bootstrap_bias():
    np.random.seed(7)
    data = np.random.normal(loc=3.0, scale=0.5, size=200)
    bts = uncorrelated_confs_to_bts(x=data, N_bts=100)
    b = bts.bias()
    assert b is not None
    assert isinstance(b, (float, np.floating, np.ndarray))
