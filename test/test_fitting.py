import numpy as np
import pytest

from lattice_data_tools.fit.xyey import fit_xyey, polynomial_fit_xyey
from lattice_data_tools.fit.xiyey import fit_xiyey


def test_fit_xyey_tanh():
    np.random.seed(0)
    omega_true = 3.0
    x = np.linspace(0.1, 1.0, 50)
    y_exact = np.tanh(omega_true * x)
    noise = 0.01 * np.abs(y_exact)
    noise = np.where(noise == 0, 1e-6, noise)
    y = y_exact + np.random.normal(scale=noise)
    ey = noise

    ansatz = lambda xi, p: np.tanh(p[0] * xi)
    res = fit_xyey(ansatz=ansatz, x=x, y=y, ey=ey, guess=np.array([2.0]))
    omega_fit = res["par"][0]
    assert abs(omega_fit - omega_true) < 0.1, (
        f"Recovered omega={omega_fit} not within 0.1 of {omega_true}"
    )


def test_polynomial_fit_xyey_linear():
    np.random.seed(1)
    a0, a1 = 2.0, 3.0
    x = np.linspace(0.0, 1.0, 30)
    y = a0 + a1 * x
    ey = 1e-6 * np.ones_like(y)

    res = polynomial_fit_xyey(N_deg=1, x=x, y=y, ey=ey)
    par = res["par"]
    assert np.allclose(par, [a0, a1], atol=1e-3), (
        f"Parameters {par} not close to [{a0}, {a1}]"
    )


def test_polynomial_fit_xyey_quadratic():
    np.random.seed(2)
    a0, a1, a2 = 1.0, 2.0, 3.0
    x = np.linspace(0.0, 1.0, 40)
    y = a0 + a1 * x + a2 * x**2
    ey = 1e-6 * np.ones_like(y)

    res = polynomial_fit_xyey(N_deg=2, x=x, y=y, ey=ey)
    par = res["par"]
    assert np.allclose(par, [a0, a1, a2], atol=1e-3), (
        f"Parameters {par} not close to [{a0}, {a1}, {a2}]"
    )


def test_fit_xiyey():
    np.random.seed(3)
    a_true, b_true = 2.0, 3.0
    x0_vals = np.linspace(0.1, 1.0, 5)
    x1_vals = np.linspace(0.1, 1.0, 5)
    x0_grid, x1_grid = np.meshgrid(x0_vals, x1_vals)
    x0_flat = x0_grid.ravel()
    x1_flat = x1_grid.ravel()
    # x has shape (N_pts, 2); ansatz receives x[i,:] = row vector
    x = np.column_stack([x0_flat, x1_flat])
    y = a_true * x0_flat + b_true * x1_flat
    ey = 1e-6 * np.ones_like(y)

    # ansatz: receives a 1D array xi of length 2 and parameters p
    ansatz = lambda xi, p: p[0] * xi[0] + p[1] * xi[1]
    res = fit_xiyey(ansatz=ansatz, x=x, y=y, ey=ey, guess=np.array([1.0, 1.0]))
    par = res["par"]
    assert abs(par[0] - a_true) < 0.1, f"a={par[0]} not within 0.1 of {a_true}"
    assert abs(par[1] - b_true) < 0.1, f"b={par[1]} not within 0.1 of {b_true}"
