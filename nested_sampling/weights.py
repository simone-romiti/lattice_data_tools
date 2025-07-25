""" Likelihood weights routines for the analysis of Nested Sampling data """

import numpy as np
import scipy


def get_log_wL_unnormalized(log_w: np.ndarray, log_L: np.ndarray):
    """ \\log(w_i*L_i), where w_i are the phase space weights, and L_i the values of the Likelihood """
    log_wL = (log_w+log_L)
    return log_wL
#---

def get_log_wL_normalized(log_wL: np.ndarray):
    """ 
    Total weights w_i*L_i, where 1 = Z = \\sum_i w_i*L_i
    NOTES:     
    - The normalization using Z (see implementation) is crucial.
        If not used, exp(log(w_i*L_i)) is typically compatible with 0 numerically.
    - The partition function is Z = \\sum_i w_i * L_i
        As a function of the phase space volume X_i, the w_i and L_i are:
        - very small in almost all the range
        - mildly overlapping 
        Therefore, if we first exponentiate their logs and sum them naively, we get 0 on all the interval.  
        Solution: we work with logarithms all the time.

    """
    logZ_unnormalized = scipy.special.logsumexp(log_wL)
    log_wL_normalized = log_wL - logZ_unnormalized
    return log_wL_normalized
#---


def get_log_Z_curve(log_wL_normalized: np.ndarray):
    """
    The partition function is normalized to 1.
    The total weights w_i*L_i are the discrete version of the probability density function.
    The cumulative sum of the total weights gives use the cumulative density function. This function returns the log of that curve.
    """
    n_iter = log_wL_normalized.shape[0]
    s = log_wL_normalized[0]
    logZ = [s]
    for i in range(n_iter-1):
        s = np.logaddexp(s, log_wL_normalized[i+1])
        logZ.append(s)
    #---
    res = np.array(logZ)
    return res
#---


def get_log_rho(log_X: np.ndarray, log_S: np.ndarray):
    """ 
    Logarithm \\log(\\rho(S)), where \\rho(S) is the the density of states: \\rho(S) = dS/dX
    The derivative is computed in terms of the derivatives of \\log(S) with respect to \\log(X)
    NOTE: The action plays the analogous role of the energy in statistical mechanics
    """
    log_der = np.log(np.gradient(log_X, log_S))
    log_rho = log_X - log_S + log_der
    return log_rho
#---


def get_idx_pruned_interval(log_wL_normalized: np.ndarray, eps_wL: float):
    """ 
    
    At fixed \\beta, we have the total weights for all values of the phase space variable "X".
    In general however, only a small subset of values of the action "S" really contributes to the integral,
    thus we can prune the interval such that wL > eps_wL (a custom threshold).

    This function retunrs the indices corresponding to the pruning.
    They should be used to prune all the other variables used in the analysis at that \\beta,
    such as the observables arrays.    
    
    """
    wL = np.exp(log_wL_normalized)
    ## finding the indices of the pruned interval where the "important" configurations lie
    pruned_interval = (wL>eps_wL)
    return  np.where(pruned_interval)
#---


def get_average_observable(log_wL_normalized: np.ndarray, obs: np.ndarray):
    """ 
    Average value of an observable O, where:
    
    - (w*L)_i are the total weights (normalized to 1) at each value of the phase space variable X_i
    - \\log(O_i) are the logs of the observable computed at those points
    
    """
    wL = np.exp(log_wL_normalized)
    O_avg = np.sum(wL*obs)
    return O_avg
#---


