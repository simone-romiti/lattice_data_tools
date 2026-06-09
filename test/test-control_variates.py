
import numpy as np
import sys
sys.path.append("../../")

from lattice_data_tools.statistics_tools import control_variates


def test_control_variates():
    rng = np.random.default_rng(12345)

    n = 1_000_000

    # Control variable
    b = rng.normal(loc=2.0, scale=3.0, size=n)

    # Construct a correlated variable:
    # a = 5 + 2*b + noise
    noise_std = 4.0
    noise = rng.normal(loc=0.0, scale=noise_std, size=n)
    a = 5.0 + 2.0 * b + noise

    # Exact moments
    mean_b = 2.0
    var_b = 3.0**2
    cov_ab = 2.0 * var_b  # Cov(5 + 2b + noise, b)

    a_new = control_variates(
        a=a,
        b=b,
        mean_b=mean_b,
        var_b=var_b,
        cov_ab=cov_ab,
    )

    # ------------------------------------------------------------------
    # Mean should be preserved
    # ------------------------------------------------------------------
    np.testing.assert_allclose(
        np.mean(a_new),
        np.mean(a),
        rtol=0,
        atol=1e-2,
    )

    # ------------------------------------------------------------------
    # Variance should decrease
    # ------------------------------------------------------------------
    var_a = np.var(a)
    var_a_new = np.var(a_new)

    assert var_a_new < var_a

    # ------------------------------------------------------------------
    # Check against theoretical optimum
    #
    # Since a = 5 + 2b + noise,
    # optimal c = -Cov(a,b)/Var(b) = -2
    #
    # Therefore:
    # a_new = a - 2*(b - mean_b)
    #       = 9 + noise
    #
    # so Var(a_new) = Var(noise)
    # ------------------------------------------------------------------
    expected_var = noise_std**2

    np.testing.assert_allclose(
        var_a_new,
        expected_var,
        rtol=0.01,   # 1% tolerance
    )

    
