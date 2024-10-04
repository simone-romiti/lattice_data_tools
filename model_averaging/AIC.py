"""
Model averaging of lattice determinations using the Akaike Information Criterion (AIC)

Reference: Section 21 of https://arxiv.org/pdf/2002.12347

"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def get_weights(ch2: np.ndarray, n_par: np.ndarray, n_data: np.ndarray):
    """ 
    Returns the 1d array of weights as in eq. 161 of https://arxiv.org/pdf/2002.12347
    The inputs are the array of models (each with a \chi^2) the number of fit parameters and number of points
    """
    A = ch2 + 2*n_par - n_data
    w_i = np.exp(-A/2)
    w_i /= np.sum(w_i)
    return w_i
#---

def get_Pi(y: np.ndarray, w: np.ndarray, m: np.ndarray, sigma: np.ndarray, lam: np.float64):
    """ 
    Returns the Cumulative Density Functions (CDF) P_i(y, lambda) of Eq. 162 of https://arxiv.org/pdf/2002.12347
    for all the values of the array "y"
    """
    N_tot = y.shape[0]
    n_models = w.shape[0]
    Pi = np.zeros(shape=(n_models, N_tot))
    sigma_scaled = np.sqrt(lam) * sigma
    for i in range(n_models):
        Pi[i,:] = w[i] * norm.cdf(y, loc=m[i], scale=sigma_scaled[i])
        Pi[i,:] /= Pi[i,-1]
    #---
    return Pi
#---

def get_P(y: np.ndarray, w: np.ndarray, m: np.ndarray, sigma: np.ndarray, lam: np.float64):
    """ 
    Returns the sum of the Cumulative Density Functions (CDF) P_i(y, lambda) of Eq. 162 of https://arxiv.org/pdf/2002.12347
    for all the values of the array "y"
    """
    Pi = get_Pi(y=y, w=w, m=m, sigma=sigma, lam=lam)
    P = np.sum(Pi, axis=0)
    P_normalized = P/P[-1] # np.sum(P)
    return P_normalized
#---


def get_y16_y50_y84(w: np.ndarray, m: np.ndarray, sigma: np.ndarray, lam: np.float64, ymin: np.float64, ymax: np.float64, eps: np.float64):
    """ Returns percentile values y16, y50 (the median) and y84 """
    y = np.arange(ymin, ymax, eps)
    P = get_P(y=y, w=w, m=m, sigma=sigma, lam=lam)
    y16 = y[np.where(P <= 0.16)[0][-1]]
    y50 = y[np.where(P <= 0.50)[0][-1]]
    y84 = y[np.where(P <= 0.84)[0][-1]]
    return (y16, y50, y84)
#---

def get_mean_and_sigma2(y16, y50, y84):
    """ Returns mean and variance from the percentiles """
    y_mean = y50
    sigma2_tot = ((y84-y16)/2.0)**2.0
    return (y_mean, sigma2_tot)
#---

def get_sigma2_contributions(w: np.ndarray, m: np.ndarray, sigma: np.ndarray, lam: np.float64, ymin: np.float64, ymax: np.float64, eps: np.float64):
    y16, y50, y84 = get_y16_y50_y84(w=w, m=m, sigma=sigma, lam=1.0, ymin=ymin, ymax=ymax, eps=eps)
    y_mean, sigma2_tot = get_mean_and_sigma2(y16=y16, y50=y50, y84=y84)

    y16_l2, y50_l2, y84_l2 = get_y16_y50_y84(w=w, m=m, sigma=sigma, lam=2.0, ymin=ymin, ymax=ymax, eps=eps)
    y_mean_l2, sigma2_tot_l2 = get_mean_and_sigma2(y16=y16_l2, y50=y50_l2, y84=y84_l2)
    sigma2_stat = sigma2_tot_l2 - sigma2_tot
    sigma2_syst = sigma2_tot - sigma2_stat
    return {"stat": sigma2_stat, "syst": sigma2_syst}
#---

