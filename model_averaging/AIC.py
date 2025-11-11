"""
Model averaging of lattice determinations using the Akaike Information Criterion (AIC)

Main reference: https://arxiv.org/pdf/2208.14983. See Eqs. (65),(64),(62)
    Used in: https://arxiv.org/pdf/2411.08852, eq. 17
    ACHTUNG: Typo in https://arxiv.org/pdf/2002.12347, eq. 161: factor 2 in front of n_data is missing

"""

import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d
# import matplotlib.pyplot as plt
from typing import List

from lattice_data_tools.bootstrap import BootstrapSamples

def get_weights(ch2: np.ndarray, n_par: np.ndarray, n_data: np.ndarray):
    """ 
    Returns the 1d array of weights as in eq. 17 of https://arxiv.org/pdf/2411.08852
    The inputs are the array of models (each with a \\chi^2) the number of fit parameters and number of points
    """
    A = ch2 + 2.0*n_par - 2.0*n_data
    w_i = np.exp(-A/2)
    # w_i /= np.sum(w_i)
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

def get_P_from_bootstraps(y: BootstrapSamples, w: np.ndarray, lam: np.float64):
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
    # N_bts, n_models = y.shape
    N_bts = y.N_bts()
    y_rescaled = BootstrapSamples(np.copy(y))
    # producing bootstraps with same mean but rescaled uncertainty
    if lam != 1.0:
        y_avg = y.unbiased_mean()
        sqrt_lam = np.sqrt(lam)
        y_rescaled = BootstrapSamples.from_lambda(N_bts=N_bts, fun = lambda i : y_avg[i] + sqrt_lam*(y[i,:]-y_avg[i]))
    #---
    y, idx_y = np.unique(y_rescaled.flatten(), return_index=True)
    w_unique = BootstrapSamples.from_lambda(N_bts=N_bts, fun=lambda i: w).flatten()[idx_y]
    P = np.cumsum(w_unique)/np.sum(w_unique)
    return {"y": y, "P": P}
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
    ymin, ymax, eps: parameters defining the interval for numerically reproducing the CDF finely enough

    """
    y16, y50, y84 = get_y16_y50_y84(w=w, m=m, sigma=sigma, lam=lambda1, ymin=ymin, ymax=ymax, eps=eps)
    y_mean, sigma2_tot = get_mean_and_sigma2(y16=y16, y50=y50, y84=y84)

    y16_l2, y50_l2, y84_l2 = get_y16_y50_y84(w=w, m=m, sigma=sigma, lam=lambda2, ymin=ymin, ymax=ymax, eps=eps)
    y_mean_l2, sigma2_tot_l2 = get_mean_and_sigma2(y16=y16_l2, y50=y50_l2, y84=y84_l2)
    sigma2_stat = (sigma2_tot_l2 - sigma2_tot)/(lambda2-lambda1)
    sigma2_syst = (lambda2*sigma2_tot - lambda1*sigma2_tot_l2)/(lambda2-lambda1)
    return {"mean": y_mean, "stat": sigma2_stat, "syst": sigma2_syst}
#---



class with_CDF:
    @staticmethod
    def get_rescaled_y(y: np.ndarray, P: np.ndarray, lam: float):
        y_half = y[P <= 0.50][-1]
        y_rescaled = np.copy(y_half + np.sqrt(lam)*(y-y_half))
        return y_rescaled
    #---
    @staticmethod
    def get_P(y: List[np.ndarray], w: np.ndarray):
        """CDFs from list of models
        
        We build the CDF numerically, by counting how many occurrencies of y we have before each y_0.
        This is done as follows:
        
        Step 1: We consider all the y values
        Step 2: We build the CDFs of each model, scaled by its weight (normalized to 1).
                This is done by building, for each model k, a fictitious histogram count (with binning 1). 
                Only for the values of "y" of the k-th model, we set the count to the weight w_k.
        Step 3: The cumulative sums of these fictitious histogram gives the CDF 
                in terms of the whole array of values coming from all the models.

        Arguments:
            y: list of size (n_models). Each item is an array of values of "y", whose distribution determine their CDF. The array can have different sizes.
            w: array of length n_models
            
        Returns:
            dictionary with the results of this procedure
        """
        assert (len(y) == w.shape[0])
        n_models = len(y)
        y_flat, idx_y = np.unique(np.concatenate([yi.ravel() for yi in y]), return_index=True)
        w_unique = np.concatenate([np.full(shape=y[i].shape, fill_value=w[i]) for i in range(n_models)]).flatten()[idx_y]
        P = np.cumsum(w_unique)
        if not (w_unique==0.0).all():
            P /= np.sum(w_unique)
        #---
        return {"y": y_flat, "P": P}
    #---
    @staticmethod
    def get_quantiles(y: np.ndarray, P: np.ndarray, quantiles = [16, 50, 84]):
        """ 
        Dictionary of quantiles corresponding to 
        16%, 50%, 84% of probabilities
        """
        # copying the arrays to work on them
        P_work = np.copy(P)
        y_work = np.copy(y)
        # adding left and right padding to avoid discretization errors
        P_work = np.concatenate(([0.0], P_work, [1.0]))
        y_work = np.concatenate(([y_work[0]], np.copy(y), [y_work[-1]]))
        # defining quantiles and finding the interpolated values of "y"
        targets = np.array([q / 100 for q in quantiles], dtype=float)
        values = np.interp(targets, P_work, y_work)
        return {f"{int(q)}%": float(v) for q, v in zip(quantiles, values)}
    #---
    @staticmethod
    def get_contributions(y1: np.ndarray, y2: np.ndarray, P1: np.ndarray, P2: np.ndarray, lam1: float, lam2: float):
        """ 
        Contributions to the total error.
        
        !!! ACHTUNG !!!
        The result may be inconsistent if one chooses the wrong values of lam1 and lam2.
        In fact, the procedure works if (lam2*sigma^2_tot_l1 - lam1*sigma2_tot_l2) > 0. 
        The user is expected to use the results of this function to update the value of lam2 in order to get all positive variances.
        
        These are obtained by using 2 different AICs, where each time we rescaled the statistical uncertainty with lam1 and lam2 respectively.
        NOTE: by "statistical" we mean the uncertainty of the individual models before the AIC. The latter can be the total error coming from a sub-model averaging
        
        This function returns:
          - mean: 50% quantile of the total distribution
          - sigma^2_stat: variance due to the statistical fluctuations
          - sigma^2 syst: variance due to the systematics
        """
        Q1 = with_CDF.get_quantiles(y=y1, P=P1)
        y1_mean, sigma2_tot_l1 = get_mean_and_sigma2(y16=Q1["16%"], y50=Q1["50%"], y84=Q1["84%"])
        Q2 = with_CDF.get_quantiles(y=y2, P=P2)
        y2_mean, sigma2_tot_l2 = get_mean_and_sigma2(y16=Q2["16%"], y50=Q2["50%"], y84=Q2["84%"])
        # sigma2_stat = (sigma2_tot_l2 - sigma2_tot_l1)/(lam2-lam1)
        # sigma2_syst = sigma2_tot_l1 - lam1*sigma2_stat
        sigma2_stat = (((Q2["84%"]-Q2["16%"]) - (Q1["84%"]-Q1["16%"]))/2)**2
        sigma2_syst = (np.sqrt(sigma2_tot_l1) - np.sqrt(sigma2_stat))**2
        return {"mean": y1_mean, "tot_lam1": sigma2_tot_l1, "tot_lam2": sigma2_tot_l2, "stat": sigma2_stat, "syst": sigma2_syst}
    @staticmethod
    def variance_from_CDF(y, P, n_samples=10_000):
        """
        Estimate the variance of a variable from its CDF (y, P) by inverse transform sampling.

        Parameters
        ----------
        y : np.ndarray
            Sorted variable values.
        P : np.ndarray
            Corresponding CDF values (monotonic increasing).
        n_samples : int, optional
            Number of uniform samples to draw from [0, 1].

        Returns
        -------
        float
            Estimated variance.
        """
        y = np.asarray(y)
        P = np.asarray(P)

        # Ensure CDF starts at 0 and ends at 1 (pad if necessary)
        if P[0] > 0:
            P = np.concatenate(([0.0], P))
            y = np.concatenate(([y[0]], y))
        if P[-1] < 1:
            P = np.concatenate((P, [1.0]))
            y = np.concatenate((y, [y[-1]]))

        # Sample uniformly in [0, 1]
        u = np.random.rand(n_samples)

        # Inverse CDF interpolation
        y_sampled = np.interp(u, P, y)

        # Return sample variance (unbiased)
        return np.var(y_sampled, ddof=1)
