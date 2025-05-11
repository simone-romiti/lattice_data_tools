# Testing the Akaike Information Criterion

import numpy as np
# from scipy.stats import norm
import sys
import matplotlib.pyplot as plt

sys.path.append('../../')

import lattice_data_tools.model_averaging.AIC as AIC

np.random.seed(13754)

n_models = 10
sigma_syst_exact = 1.4
sigma2_syst_exact = sigma_syst_exact**2
model_names = [str(f"Model-{k}") for k in range(n_models)]

m_true = np.random.normal(loc=5.0, scale=sigma_syst_exact, size=n_models)

sigma_stat_exact = 0.6
sigma2_stat_exact = sigma_stat_exact**2

N_bts = 500
y = np.zeros(shape=(N_bts, n_models))
for k in range(n_models):
    y[:,k] = np.random.normal(loc=m_true[k], scale=sigma_stat_exact, size=N_bts)
####
w = 1.0/np.log(np.arange(2, n_models+2))

res_AIC = AIC.get_P_from_bootstraps(y=y, w=w, lam=1.0)

y = res_AIC["y"]
wP = res_AIC["wP"]
P = res_AIC["P"]

wP_cumsum = np.cumsum(wP, axis=1)

n_digits = 2

for k in range(n_models):
    perc = np.round(wP[-1,k], n_digits)
    model_name = model_names[k]
    plt.plot(
        y, wP_cumsum[:,k], 
        linestyle="None", marker=".", alpha=0.05, 
        label="{model_name}: {:10.{n_digits}f} %".format(100*perc, model_name=model_name, n_digits=n_digits))
#---
plt.plot(y, P, color="black", label="AIC")

plt.title("Cumulative contributions from each model")
plt.legend()
plt.savefig("AIC_bootstrap.pdf")

# plt.show()

# percentiles
y16 = y[np.where(P <= 0.16)[0][-1]]
y50 = y[np.where(P <= 0.50)[0][-1]]
y84 = y[np.where(P <= 0.84)[0][-1]]

y_mean, sigma2_tot = AIC.get_mean_and_sigma2(y16=y16, y50=y50, y84=y84)

# y16_l2, y50_l2, y84_l2 = AIC.get_y16_y50_y84(w=w, m=m, sigma=sigma, lam=2.0, ymin=ymin, ymax=ymax, eps=eps)
# y_mean_l2, sigma2_tot_l2 = AIC.get_mean_and_sigma2(y16=y16_l2, y50=y50_l2, y84=y84_l2)

print("Percentiles:",  y16, y50, y84)
print("y_mean: ", y_mean)
print("sigma_tot^2 = ", sigma2_tot)

# sigma2_stat = sigma2_tot_l2 - sigma2_tot
# sigma2_syst = sigma2_tot - sigma2_stat
# sigma2_syst_alternative = (sigma2_tot_l2 - 2*sigma2_tot)/(1-2)

# print("       exact    AIC")
# print("syst: ", sigma2_syst_exact, sigma2_syst)
# print("syst_alternative: ", sigma2_syst_exact, sigma2_syst_alternative)
# print("stat: ", sigma2_stat_exact, sigma2_stat)
