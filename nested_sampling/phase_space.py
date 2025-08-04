""" Phase space volume routines for the analysis of Nested Sampling data """

import numpy as np
from typing import Literal


def get_log_t(n_live: int, n_iter: int, sampling_t: bool) -> np.ndarray:
    """
    Get the log of the compression factor for Nested Sampling data.
    
    Parameters:
    n_live (int): Number of live points (walkers).
    n_iter (int): Number of iterations.
    sampling_t (bool): If True, sample t from a Beta distribution; otherwise, use the naive approach.
    
    Returns:
    np.ndarray: Array of log compression factors.
    """
    if sampling_t:
        t_arr = np.random.power(n_live, size=n_iter)
        log_t = np.log(t_arr)
        return log_t
    else:
        log_t_value = (-1.0/n_live)
        log_t = np.full(shape=(n_iter), fill_value=log_t_value)
        return log_t
    #---
#---

def get_log_X(log_t: np.ndarray) -> np.ndarray:
    """ logarithms of the phase space volumes """
    return  np.cumsum(log_t)
#---

def get_log_w(
    log_t: np.ndarray, 
    log_X: np.ndarray, 
    strategy: Literal["fwd", "symm"]
) -> np.ndarray:
    """ weights w_i (see eq. 14 of https://arxiv.org/pdf/2205.15570)
    
    strategy:
        - fwd: forward difference: w_i = X_{i+1} - X_{i}
        - symm: symmetric difference: w_i = (1/2)*(X_{i-1} - X_{i+1})
    
    """
    t = np.exp(log_t)
    if strategy == "fwd":
        ## log of forward difference X[i] - X[i+1]
        log_w = log_X  + np.log(1.0 - t)
        return log_w
    elif strategy == "symm":
        # log of symmetric difference (X[i-1] - X[i+1])/2
        log_w = -np.log(2) + log_X - log_t + np.log(1.0 - t**2)
        return log_w
    else:
        raise ValueError(f"Illegal strategy for computing log(w): {strategy}")
    #---
#---



