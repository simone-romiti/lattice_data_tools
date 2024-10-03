"""
Model averaging of lattice determinations using the Akaike Information Criterion (AIC)

Reference: Section 21 of https://arxiv.org/pdf/2002.12347

"""

import numpy as np
from scipy.stats import norm

def get_weights(ch2: np.ndarray, n_par: np.ndarray, n_data: np.ndarray):
    """ 
    Returns the 1d array of weights as in eq. 161 of https://arxiv.org/pdf/2002.12347
    The inputs are the array of models (each with a \chi^2) the number of fit parameters and number of points
    """
    A = ch2 + 2*n_par - n_data
    w_i = np.exp(-A/2)
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


def get_sigma2_tot(w: np.ndarray, m: np.ndarray, sigma: np.ndarray, lam: np.float64, ymin: np.float64, ymax: np.float64, eps: np.float64):
    """ 
    Returns sigma_tot as in eq. 164 of https://arxiv.org/pdf/2002.12347
    
    The y values are in the range [-R, R], with resolution epsilon
    """
    y = np.arange(ymin, ymax, eps)
    P = get_P(y=y, w=w, m=m, sigma=sigma, lam=lam)
    y16, y84 = np.percentile(P, [16.0, 84.0])
    sigma2_tot = ((y84-y16)/2.0)**2.0
    return sigma2_tot
#---
