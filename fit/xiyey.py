"""Fit of a 1d function of n variables """


import numpy as np
from lattice_data_tools.fit.trajectory import fit_trajectory

def fit_xiyey(
    ansatz, 
    x: np.ndarray, y: np.ndarray, ey: np.ndarray, 
    guess: np.ndarray, 
    method = "BFGS",
    Cov_y_inv = None
    ):
    """Fit of y=f(\\vec{x}), there f: R^n \\to R^1

    Args:
        ansatz : ansatz taking a float and returning a float
        x (np.ndarray): 1d array of x values
        y (np.ndarray): 1d array of y values
        ey (np.ndarray):1d array of errors on the values
        guess (np.ndarray): 1d array fo guesses for the ansatz_
        maxiter (int, optional): _description_. Defaults to 10000.
        method (str, optional): _description_. Defaults to "BFGS".
        Cov_y_inv: estimate of inverse covariance matrix of y
    """
    def ansatz_casted(x, p):
        return np.array([ansatz(x, p)])
    #---
    ex = np.zeros(shape=x.shape)
    yp = np.array([y]).transpose()
    eyp = np.array([ey]).transpose()
    C_inv_full = Cov_y_inv
    if not (Cov_y_inv is None):
        Nx = x.flatten().shape[0]
        Ny = y.flatten().shape[0]
        assert(Cov_y_inv.shape == (Ny, Ny))
        C_inv_full = 0.0*np.eye(N=(Nx+Ny)) # ficticious covariance matrix to pass to the general routine
        C_inv_full[Nx:,Nx:] = Cov_y_inv
    #---
    res = fit_trajectory(
        ansatz=ansatz_casted, 
        x=x, ex=ex, y=yp, ey=eyp, 
        guess=guess, 
        method=method,
        Cov_inv=C_inv_full)
    return res
#---
