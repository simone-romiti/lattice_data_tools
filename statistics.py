"""
General routines for statistical analysis
"""

import numpy as np

def sample_from_CDF(y, P, n_samples=10_000):
    """
    Sample points y_sampled from the numerical histogram of the interpolated cumulative distribution P(y).
    
    This is achieved by
    - sampling uniformly u \\in [0,1] (value of P)
    - finding the corresponding value of y such that u=P(y)
    - the values of y and P in input are used to do the interpolation

    Parameters
    ----------
    y : np.ndarray
        Values for which P(y) is known (e.g. from cumulative histogram).
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
    return y_sampled
#---

def variance_from_CDF(y, P, n_samples=10_000):
    """ unbiased estimator of sample variance from CDF histogram """
    y_sampled = sample_from_CDF(y=y, P=P, n_samples=n_samples)
    return np.var(y_sampled, ddof=1) # Return sample variance (unbiased)


def covariance_to_correlation(C: np.ndarray):
    """ covariance matrix --> correlation matrix """
    v = np.diag(C) # diagonal matrix of variances
    sigma_inv = np.sqrt(1.0/v) # inverse sigma_i
    S_inv = np.diag(sigma_inv)
    rho = S_inv @ C @ S_inv
    return rho

def correlation_to_covariance(rho, errors):
    """ correlation matrix --> covariance """
    S = np.diag(errors)
    C = S @ rho @ S
    return C


def rooting(A: np.ndarray, decimals: float = 15):
    """ Rooting the matrix A (assumed to be symmetric) """
    assert(len(A.shape) == 2 and A.shape[0] == A.shape[1]) # check that it is a square matrix
    assert(np.all((A - A.T).round(decimals=decimals) == 0.0)) # check that symmetric matrix
    AAT = (A @ (A.T)) # semi-positive definite by construction, numerically safe
    lam4, M = np.linalg.eigh(AAT) #  diagonalization
    lam4 = np.maximum(lam4, 10**(-decimals)) # numerical stabilizer: lam4 are analytically >=0
    D = np.diag(np.sqrt(lam4)) # apply the rooting
    A_rooted = M @ D @ M.T # sqrt(rho @ rho.T)
    return A_rooted

