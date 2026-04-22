import numpy as np
import pytest

from lattice_data_tools.model_averaging.IC import (
    get_IC,
    get_weights,
    with_CDF,
)


def test_get_IC_AIC():
    n_models = 5
    ch2 = np.full(n_models, 10.0)
    n_par = np.full(n_models, 2.0)
    n_data = np.full(n_models, 10.0)

    ic = get_IC(ch2=ch2, n_par=n_par, n_data=n_data, IC="AIC")
    assert ic.shape == (n_models,), f"Expected shape ({n_models},), got {ic.shape}"
    # All models identical => all IC values equal
    assert np.allclose(ic, ic[0]), f"Expected all equal IC values, got {ic}"


def test_get_weights_sum_positive():
    n_models = 5
    ch2 = np.array([10.0, 12.0, 11.0, 9.0, 13.0])
    n_par = np.full(n_models, 2.0)
    n_data = np.full(n_models, 20.0)

    w = get_weights(ch2=ch2, n_par=n_par, n_data=n_data, IC="AIC")
    assert w.shape == (n_models,)
    assert np.all(w > 0), f"Expected all positive weights, got {w}"


def test_get_weights_AICc_vs_AIC():
    # AICc requires n_data - n_par - 1 > 0
    n_models = 4
    ch2 = np.array([8.0, 9.0, 10.0, 11.0])
    n_par = np.full(n_models, 2.0)
    n_data = np.full(n_models, 20.0)  # 20 - 2 - 1 = 17 > 0 OK

    w = get_weights(ch2=ch2, n_par=n_par, n_data=n_data, IC="AICc")
    assert w.shape == (n_models,)
    assert np.all(w > 0), f"Expected all positive AICc weights, got {w}"


def test_with_CDF_get_quantiles():
    np.random.seed(42)
    true_median = 5.0
    samples = np.random.normal(loc=true_median, scale=1.0, size=1000)

    # Build CDF from a single "model" with 1000 samples
    result = with_CDF.get_P(y=[samples], w=np.array([1.0]))
    y_vals = result["y"]
    P_vals = result["P"]

    quantiles = with_CDF.get_quantiles(y=y_vals, P=P_vals)
    median_estimate = quantiles["50%"]
    assert abs(median_estimate - true_median) < 0.2, (
        f"50% quantile {median_estimate} not close to true median {true_median}"
    )
