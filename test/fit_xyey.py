import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../../')
from lattice_data_tools.fit.legacy.xyey import fit_xyey as old_fit_xyey
from lattice_data_tools.fit.xyey import fit_xyey

N_pts = 100
x = np.linspace(0, 1, N_pts)
omega_exact = 3.0

def ansatz(x, params):
    return np.tanh(params[0] * x)

y_exact = ansatz(x, [omega_exact])
np.random.seed(1243)
noise = np.random.normal(0, 0.01, size=x.shape)
y = y_exact + noise
ey = noise
guess = np.array([0.0])


# def test_old_and_new_agree():
#     fit1 = old_fit_xyey(ansatz=ansatz, x=x, y=y, ey=ey, guess=guess)
#     fit2 = fit_xyey(ansatz=ansatz, x=x, y=y, ey=ey, guess=guess)
#     np.testing.assert_allclose(fit1["par"], fit2["par"], rtol=1e-5)
#     np.testing.assert_allclose(fit1["ch2"], fit2["ch2"], rtol=1e-5)


def test_correlated_matches_uncorrelated():
    fit2 = fit_xyey(ansatz=ansatz, x=x, y=y, ey=ey, guess=guess)
    Cov_y_inv = np.diag(1 / ey**2)
    fit2_corr = fit_xyey(ansatz=ansatz, x=x, y=y, ey=ey, guess=guess, Cov_y_inv=Cov_y_inv)
    np.testing.assert_allclose(fit2["par"], fit2_corr["par"], rtol=1e-5)
    np.testing.assert_allclose(fit2["ch2"], fit2_corr["ch2"], rtol=1e-5)


def test_recovers_exact_omega():
    fit2 = fit_xyey(ansatz=ansatz, x=x, y=y, ey=ey, guess=guess)
    np.testing.assert_allclose(fit2["par"][0], omega_exact, atol=0.05)


if __name__ == "__main__":
    # fit1 = old_fit_xyey(ansatz=ansatz, x=x, y=y, ey=ey, guess=guess)
    fit2 = fit_xyey(ansatz=ansatz, x=x, y=y, ey=ey, guess=guess)
    Cov_y_inv = np.diag(1 / ey**2)
    fit2_corr = fit_xyey(ansatz=ansatz, x=x, y=y, ey=ey, guess=guess, Cov_y_inv=Cov_y_inv)

    print("Compare the 2 fit routines (should give the same result)")
    # for fit in [fit1, fit2, fit2_corr]:
    for fit in [fit2, fit2_corr]:
        print("---")
        for k in ["par", "ch2"]:
            print(k, fit[k])
    print("==============")
    print("Exact results:")
    print("omega_exact (par):", omega_exact)

    plt.plot(x, y)
    plt.plot(x, y_exact)
    plt.savefig("./fit_xyey.pdf")

