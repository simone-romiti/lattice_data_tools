"""Fit of a 1d function of 1 variable """


import numpy as np
from lattice_data_tools.fit.trajectory import fit_trajectory

def fit_xyey(
    ansatz, 
    x: np.ndarray, y: np.ndarray, ey: np.ndarray, 
    guess: np.ndarray, 
    method = "BFGS",
    Cov_y_inv = None
    ):
    """Fit of y=f(x), there f: \\mathbb{R}^1 \\to \\mathbb{R}^1

    Args:
        ansatz : ansatz taking a float and returning a float
        x (np.ndarray): 1d array of x values
        y (np.ndarray): 1d array of y values
        ey (np.ndarray):1d array of errors on the values
        guess (np.ndarray): 1d array fo guesses for the ansatz_
        maxiter (int, optional): _description_. Defaults to 10000.
        method (str, optional): _description_. Defaults to "BFGS".
        Cov_y_inv: if != None is an estimate of the inverse of the covariance matrix of the y points (e.g. from the bootstrap samples)
    """
    def ansatz_casted(x, p):
        return np.array([ansatz(x[0], p)])
    #---
    xp = np.array([x]).transpose()
    ex = np.zeros(shape=xp.shape)
    yp = np.array([y]).transpose()
    eyp = np.array([ey]).transpose()
    C_inv_full = Cov_y_inv
    if not (Cov_y_inv is None):
        Nx = xp.flatten().shape[0]
        Ny = yp.flatten().shape[0]
        assert(Cov_y_inv.shape == (Ny, Ny))
        C_inv_full = 0.0*np.eye(N=(Nx+Ny)) # ficticious covariance matrix to pass to the general routine
        C_inv_full[Nx:,Nx:] = Cov_y_inv
    #---
    res = fit_trajectory(
        ansatz=ansatz_casted, 
        x=xp, ex=ex, y=yp, ey=eyp, 
        guess=guess, 
        method=method,
        Cov_inv=C_inv_full)
    return res
#---
