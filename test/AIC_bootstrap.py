# Testing the Akaike Information Criterion
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
from lattice_data_tools.bootstrap import BootstrapSamples
from lattice_data_tools.model_averaging.with_bts import ModelAverage
from lattice_data_tools.model_averaging.IC import get_mean_and_sigma2

# --- shared test fixtures ---
np.random.seed(13754)
n_models = 10
sigma_syst_exact = 1.4
sigma_stat_exact = 0.6
N_bts = 500

m_true = np.random.normal(loc=5.0, scale=sigma_syst_exact, size=n_models)
y_data = BootstrapSamples.zeros(N_bts=N_bts, shape=(n_models,))
for k in range(n_models):
    y_data[:, k] = np.random.normal(loc=m_true[k], scale=sigma_stat_exact, size=N_bts+1)

w = 1.0 / np.log(np.arange(2, n_models + 2))
res_AIC = ModelAverage.get_P(y=y_data, w=w, lam=1.0)
y    = res_AIC["y"]
w_normalized   = res_AIC["w_normalized"]
P    = res_AIC["P"]

y16 = y[np.where(P <= 0.16)[0][-1]]
y50 = y[np.where(P <= 0.50)[0][-1]]
y84 = y[np.where(P <= 0.84)[0][-1]]
y_mean, sigma2_tot = get_mean_and_sigma2(y16=y16, y50=y50, y84=y84) # assuming Gaussian


# --- tests ---

def test_result_keys():
    """get_P_from_bootstraps should return y, wP, and P."""
    assert "y" in res_AIC
    assert "w_normalized" in res_AIC
    assert "P" in res_AIC


def test_P_is_valid_cdf():
    """P should be monotonically increasing, starting near 0 and ending near 1."""
    assert np.all(np.diff(P) >= 0), "P is not monotonically increasing"
    assert P[0] < 0.1,  f"P does not start near 0, got P[0]={P[0]}"
    assert P[-1] > 0.9, f"P does not end near 1, got P[-1]={P[-1]}"


def test_w_normalized_sums_to_P():
    """Cumulative sum of w_normalized across models should equal P at each y point."""
    w_normalized_sum = np.sum(w_normalized, axis=1)
    np.testing.assert_allclose(w_normalized_sum, P, rtol=1e-5)


def test_w_normalized_shape():
    """wP should have shape (n_y_points, n_models)."""
    assert wP.shape[1] == n_models


def test_percentiles_ordered():
    """y16 <= y50 <= y84 must hold."""
    assert y16 <= y50, f"y16={y16} > y50={y50}"
    assert y50 <= y84, f"y50={y50} > y84={y84}"


def test_mean_near_true_mean():
    """AIC mean should be within 2 sigma_syst of the true mean (5.0)."""
    true_mean = 5.0
    assert abs(y_mean - true_mean) < 2 * sigma_syst_exact, \
        f"y_mean={y_mean} too far from true mean {true_mean}"


def test_sigma2_tot_positive():
    assert sigma2_tot > 0, f"sigma2_tot={sigma2_tot} is not positive"


def test_sigma2_tot_in_expected_range():
    """Total variance should be between stat-only and stat+syst combined."""
    sigma2_stat = sigma_stat_exact**2
    sigma2_syst = sigma_syst_exact**2
    sigma2_min = sigma2_stat * 0.5
    sigma2_max = (sigma2_stat + sigma2_syst) * 3.0
    assert sigma2_min < sigma2_tot < sigma2_max, \
        f"sigma2_tot={sigma2_tot} outside expected range ({sigma2_min}, {sigma2_max})"


if __name__ == "__main__":
    model_names = [f"Model-{k}" for k in range(n_models)]
    w_normalized_cumsum = np.cumsum(w_normalized, axis=1)
    n_digits = 2

    for k in range(n_models):
        perc = np.round(w_normalized[-1, k], n_digits)
        model_name = model_names[k]
        plt.plot(
            y, w_normalized_cumsum[:, k],
            linestyle="None", marker=".", alpha=0.05,
            label="{model_name}: {:10.{n_digits}f} %".format(
                100 * perc, model_name=model_name, n_digits=n_digits))

    plt.plot(y, P, color="black", label="AIC")
    plt.title("Cumulative contributions from each model")
    plt.legend()
    plt.savefig("AIC_bootstrap.pdf")

    print("Percentiles:", y16, y50, y84)
    print("y_mean:", y_mean)
    print("sigma_tot^2 =", sigma2_tot)

