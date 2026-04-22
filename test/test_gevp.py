import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

from lattice_data_tools.gevp import gevp


def _make_spd_correlator(N, T, seed=42):
    """Create a (N, N, T) random symmetric positive-definite correlator matrix."""
    np.random.seed(seed)
    C = np.zeros((N, N, T))
    for t in range(T):
        A = np.random.randn(N, N)
        # Make symmetric positive definite: A^T A + N*I
        M = A.T @ A + N * np.eye(N)
        C[:, :, t] = M
    return C


def test_gevp_eigenvalues_shape():
    N = 2
    T = 10
    C = _make_spd_correlator(N=N, T=T)
    Lam, V = gevp(C, t0=0)
    assert Lam.shape == (N, T), f"Expected Lam shape ({N},{T}), got {Lam.shape}"
    assert V.shape == (N, N, T), f"Expected V shape ({N},{N},{T}), got {V.shape}"


def test_gevp_eigenvalues_real():
    N = 2
    T = 10
    C = _make_spd_correlator(N=N, T=T)
    Lam, V = gevp(C, t0=0)
    assert np.all(np.isreal(Lam)), "Expected all eigenvalues to be real"
    assert not np.any(np.isnan(Lam)), "Unexpected NaN in eigenvalues"
