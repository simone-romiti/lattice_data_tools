# Testing the Akaike Information Criterion

import numpy as np
# from scipy.stats import norm

import model_averaging.AIC as AIC
import matplotlib.pyplot as plt


n_models = 100
sigma_syst_exact = 1.0
sigma2_syst_exact = sigma_syst_exact**2

m_true = np.random.normal(loc=0.0, scale=sigma_syst_exact, size=n_models)

sigma_stat_exact = 0.5
sigma2_stat_exact = sigma_stat_exact**2

ch2 = []
n_par = []
n_data = []
m = []
sigma = []
for i in range(n_models):
    N_pts = 500
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

ymin = -50.0 # np.min(m)
ymax = 50.0 # np.max(m)
eps = 1e-3
y = np.arange(ymin, ymax, eps)

Pi = AIC.get_Pi(y=y, w=w, m=m, sigma=sigma, lam=1.0)
for i in range(n_models):
    plt.plot(y[:-1], np.diff(Pi[i]), label=str(i))
####
P = AIC.get_P(y=y, w=w, m=m, sigma=sigma, lam=1.0)
plt.plot(y[:-1], np.diff(P), label="tot")
####
plt.legend()
# print(P.shape)
# print(np.ediff1d(P).shape)
# plt.show()
# quit()

sigma2_tot = AIC.get_sigma2_tot(w=w, m=m, sigma=sigma, lam=1.0, ymin=ymin, ymax=ymax, eps = 1e-3)
sigma2_tot_l2 = AIC.get_sigma2_tot(w=w, m=m, sigma=sigma, lam=2.0, ymin=ymin, ymax=ymax, eps = 1e-3)

print("sigma_tot^2 = ", sigma2_tot)

sigma2_stat = sigma2_tot_l2 - sigma2_tot
sigma2_syst = sigma2_tot - sigma2_stat

print("       exact      AIC")
print("syst: ", sigma2_syst_exact, sigma2_syst)
print("stat: ", sigma2_stat_exact, sigma2_stat)


