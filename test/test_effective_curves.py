import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

from lattice_data_tools.effective_curves import (
    get_m_eff,
    get_m_eff_log,
    fit_eff_mass,
)


def test_get_m_eff_log_constant_mass():
    m = 0.2
    T = 24
    t = np.arange(T)
    C = np.exp(-m * t)
    m_eff = get_m_eff_log(C)
    # m_eff has length T-1; all values should be close to m
    assert np.all(np.abs(m_eff - m) < 0.01), (
        f"Effective mass values not close to {m}: {m_eff}"
    )


def test_get_m_eff_strategy_invalid():
    T = 24
    t = np.arange(T)
    C = np.exp(-0.2 * t)
    with pytest.raises(ValueError):
        get_m_eff(C, strategy="invalid")


def test_fit_eff_mass_returns_float():
    m = 0.2
    T = 24
    t = np.arange(T)
    C = np.exp(-m * t)
    m_eff = get_m_eff_log(C)
    # use a stable plateau region (skip first and last points)
    m_eff_plateau = m_eff[2:10]
    dm_eff_plateau = 0.001 * np.ones_like(m_eff_plateau)
    result = fit_eff_mass(m_eff=m_eff_plateau, dm_eff=dm_eff_plateau)
    assert isinstance(result, (float, np.floating)), (
        f"Expected a float, got {type(result)}"
    )
