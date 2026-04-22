import numpy as np
import pytest

from lattice_data_tools.jackknife import (
    JackknifeSamples,
    uncorrelated_confs_to_jkf,
)


def test_uncorrelated_confs_to_jkf_shape():
    np.random.seed(42)
    data = np.random.normal(loc=0.0, scale=1.0, size=200)
    jkf = uncorrelated_confs_to_jkf(x=data, N_jkf=10)
    assert jkf.shape == (10,), f"Expected shape (10,), got {jkf.shape}"


def test_jackknife_mean():
    np.random.seed(42)
    true_mean = 3.0
    data = np.random.normal(loc=true_mean, scale=1.0, size=200)
    jkf = uncorrelated_confs_to_jkf(x=data, N_jkf=10)
    estimated_mean = jkf.mean()
    assert abs(estimated_mean - true_mean) < 0.5, (
        f"Jackknife mean {estimated_mean} not close to {true_mean}"
    )


def test_jackknife_error():
    np.random.seed(42)
    data = np.random.normal(loc=0.0, scale=1.0, size=200)
    jkf = uncorrelated_confs_to_jkf(x=data, N_jkf=10)
    err = jkf.error()
    assert err > 0.0, f"Expected positive error, got {err}"


def test_jackknife_blocks_numpy_mean():
    np.random.seed(42)
    data = np.random.normal(size=200)
    jkf = uncorrelated_confs_to_jkf(x=data, N_jkf=10)
    with pytest.raises(TypeError):
        np.mean(jkf)
