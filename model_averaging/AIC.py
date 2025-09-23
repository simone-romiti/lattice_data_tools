"""
Model averaging of lattice determinations using the Akaike Information Criterion (AIC)

Main reference: https://arxiv.org/pdf/2208.14983. See Eqs. (65),(64),(62)
    Used in: https://arxiv.org/pdf/2411.08852, eq. 17
    ACHTUNG: Typo in https://arxiv.org/pdf/2002.12347, eq. 161: factor 2 in front of n_data is missing

"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def get_weights(ch2: np.ndarray, n_par: np.ndarray, n_data: np.ndarray):
    """ 
    Returns the 1d array of weights as in eq. 17 of https://arxiv.org/pdf/2411.08852
    The inputs are the array of models (each with a \\chi^2) the number of fit parameters and number of points
    """
    A = ch2 + 2.0*n_par - 2.0*n_data
    w_i = np.exp(-A/2)
    w_i /= np.sum(w_i)
    return w_i
#---

def get_Pi(y: np.ndarray, m: np.ndarray, sigma: np.ndarray, lam: np.float64):
    """ 
    Returns the Cumulative Density Functions (CDF) P_i(y, lambda) of Eq. 162 of https://arxiv.org/pdf/2002.12347
    for all the values of the array "y".
    """
    N_tot = y.shape[0]
    n_models = m.shape[0]
    Pi = np.zeros(shape=(n_models, N_tot))
    sigma_scaled = np.sqrt(lam) * sigma
    for i in range(n_models):
        Pi[i,:] = norm.cdf(y, loc=m[i], scale=sigma_scaled[i])
        # Pi[i,:] /= Pi[i,-1]
    #---
    return Pi
#---

def get_P(y: np.ndarray, w: np.ndarray, m: np.ndarray, sigma: np.ndarray, lam: np.float64):
    """ 
    Returns the sum of the Cumulative Density Functions (CDF) P_i(y, lambda) of Eq. 162 of https://arxiv.org/pdf/2002.12347
    for all the values of the array "y". These values are the domain of values that we can get from each model, 
    i.e. the m[i] should fall into the interval of the y values.

    Remark: We numerically evaluate the CDF at some values of y (specified by the array), 
    and estimate the percentiles from those. Thus, the resolution in the y[i] should be much larger that the sigma[i]
    """
    Pi = get_Pi(y=y, m=m, sigma=sigma, lam=lam)
    P = np.matmul(w, Pi)/np.sum(w)
    return P
    # P_normalized = P/P[-1] # np.sum(P)
    # return P_normalized
#---

def get_P_from_bootstraps(y: np.ndarray, w: np.ndarray, lam: np.float64):
    """CDFs from boostrap samples 
    
    We build the CDF numerically, by counting how many occurrencies of y we have before each y_0.
    This is done as follows:
    
    Step 0 (optional): we rescale the variance of the bootstrap samples
    Step 1: We consider all the y values, i.e. from each bootstrap and each each model, all together
    Step 2: We build the CDFs of each model, scaled by its weight (normalized to 1).
            This is done by building, for each model k, a fictitious histogram count (with binning 1). 
            Only for the values of "y" of the k-th model, we set the count to the weight w_k.
    Step 3: The cumulative sums of these fictitious histogram gives the CDF 
            in terms of the whole array of values coming from all the models.

    Arguments:
        y: array of size (N_bts, n_models)
        lam: scaling parameter for the variance --> sqrt(lambda) for the uncertainty
        
    Returns:
        dictionary with the results of this procedure
    """
    N_bts, n_models = y.shape
    # producing bootstraps with same mean but rescaled uncertainty
    y_rescaled = np.copy(y)
    if lam != 1.0:
        y_avg = np.average(y, axis=0)
        sqrt_lam = np.sqrt(lam)
        y_rescaled = np.array([y_avg[i] + sqrt_lam*(y[i,:]-y_avg[i]) for i in range(N_bts)])
    #---
    # sorting the Pi contributions according to the values of y
    y = np.unique(y_rescaled.flatten()) ## sorted array of unique elements
    N_pts = y.shape[0] ## total number of points
    # weighted p.d.f. with binning 1
    w_pi = np.zeros(shape=(N_pts, n_models))
    sum_wk = np.sum(w)
    for k in range(n_models):
        yk_idx = np.where(np.isin(y, y_rescaled[:,k]))[0]
        # the weights are normalized to 1
        w_pi[yk_idx, k] = w[k] / sum_wk
    #---
    # each model contributes with N_bts points
    wP = np.cumsum(w_pi, axis=0)/N_bts 
    P = np.sum(wP, axis=1) # cumulating the models' contributions
    return {"y": y, "wP": wP, "P": P}
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

def get_sigma2_contributions(
    w: np.ndarray, m: np.ndarray, sigma: np.ndarray, 
    ymin: np.float64, ymax: np.float64, eps: np.float64, 
    lambda1 = 1.0, lambda2 = 2.0):
    """
    Statistical and systematic variances using the AIC model averaging
    
    w: weights
    m: means of the models
    sigma: statistical uncertainties of the models
    ymin, ymax, eps: parameters defining the interval for numerically reproduding the CDF finely enough

    """
    y16, y50, y84 = get_y16_y50_y84(w=w, m=m, sigma=sigma, lam=lambda1, ymin=ymin, ymax=ymax, eps=eps)
    y_mean, sigma2_tot = get_mean_and_sigma2(y16=y16, y50=y50, y84=y84)

    y16_l2, y50_l2, y84_l2 = get_y16_y50_y84(w=w, m=m, sigma=sigma, lam=lambda2, ymin=ymin, ymax=ymax, eps=eps)
    y_mean_l2, sigma2_tot_l2 = get_mean_and_sigma2(y16=y16_l2, y50=y50_l2, y84=y84_l2)
    sigma2_stat = (sigma2_tot_l2 - sigma2_tot)/(lambda2-lambda1)
    sigma2_syst = (lambda2*sigma2_tot - lambda1*sigma2_tot_l2)/(lambda2-lambda1)
    return {"mean": y_mean, "stat": sigma2_stat, "syst": sigma2_syst}
#---

