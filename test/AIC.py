# Testing the Akaike Information Criterion

import numpy as np
# from scipy.stats import norm
import sys
import matplotlib.pyplot as plt

sys.path.append('../../')

import lattice_data_tools.model_averaging.AIC as AIC

np.random.seed(13754)

n_models = 10000
sigma_syst_exact = 1.4
sigma2_syst_exact = sigma_syst_exact**2

m_true = np.random.normal(loc=5.0, scale=sigma_syst_exact, size=n_models)

sigma_stat_exact = 0.6
sigma2_stat_exact = sigma_stat_exact**2

ch2 = []
n_par = []
n_data = []
m = []
sigma = []
for i in range(n_models):
    N_pts = 1000
    x = np.random.normal(loc=m_true[i], scale=sigma_stat_exact, size=N_pts)
    ch2.append(24)
    n_par.append(3)
    n_data.append(24)
    m.append(np.mean(x))
    sigma.append(np.std(x))
####

ch2    = np.array(ch2)
n_par  = np.array(n_par)
n_data = np.array(n_data)
m      = np.array(m)
sigma  = np.array(sigma)

w = AIC.get_weights(ch2=ch2, n_par=n_par, n_data=n_data)

ymin = -20.0 # np.min(m)
ymax = 20.0 # np.max(m)
eps = 1e-3
y = np.arange(ymin, ymax, eps)


y16, y50, y84 = AIC.get_y16_y50_y84(w=w, m=m, sigma=sigma, lam=1.0, ymin=ymin, ymax=ymax, eps=eps)
y_mean, sigma2_tot = AIC.get_mean_and_sigma2(y16=y16, y50=y50, y84=y84)

y16_l2, y50_l2, y84_l2 = AIC.get_y16_y50_y84(w=w, m=m, sigma=sigma, lam=2.0, ymin=ymin, ymax=ymax, eps=eps)
y_mean_l2, sigma2_tot_l2 = AIC.get_mean_and_sigma2(y16=y16_l2, y50=y50_l2, y84=y84_l2)

print("Percentiles:",  y16, y50, y84)
print("y_mean: ", y_mean)
print("sigma_tot^2 = ", sigma2_tot)

sigma2_stat = sigma2_tot_l2 - sigma2_tot
sigma2_syst = sigma2_tot - sigma2_stat
sigma2_syst_alternative = (sigma2_tot_l2 - 2*sigma2_tot)/(1-2)

print("       exact    AIC")
print("syst: ", sigma2_syst_exact, sigma2_syst)
print("syst_alternative: ", sigma2_syst_exact, sigma2_syst_alternative)
print("stat: ", sigma2_stat_exact, sigma2_stat)

Pi = AIC.get_Pi(y=y, w=w, m=m, sigma=sigma, lam=1.0)
for i in range(n_models):
    # plt.plot(y[:-1], np.diff(Pi[i]), linestyle="--", alpha=0.1)
    pass
####
P = AIC.get_P(y=y, w=w, m=m, sigma=sigma, lam=1.0)
plt.plot(y[:-1], np.diff(P), label="tot")

plt.vlines([y16, y50, y84], ymin=0.0, ymax=np.max(np.diff(P)), color="red", label="Percentiles: 16%, 50%, 84%")

plt.legend()
# plt.show()
print("Saving the plot on AIC.pdf")
plt.savefig("AIC.pdf")

