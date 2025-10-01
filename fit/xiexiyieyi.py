"""Fit of a 1d function of n variables with errors on the "x" too """

import numpy as np
from lattice_data_tools.fit.trajectory import fit_trajectory

def fit_xiexiyieyi(
    ansatz, 
    x: np.ndarray, ex: np.ndarray, y: np.ndarray, ey: np.ndarray, 
    guess: np.ndarray, 
    maxiter = 10000, method = "BFGS",
    Cov_inv = None
    ):
    """Fit of y=f(\vec{x}), there f: R^n \to R^1
    NOTE: This function is just an alias for fit_trajectory

    Args:
        ansatz : ansatz taking a float and returning a float
        x (np.ndarray): 1d array of x values
        y (np.ndarray): 1d array of y values
        ey (np.ndarray):1d array of errors on the values
        guess (np.ndarray): 1d array fo guesses for the ansatz_
        maxiter (int, optional): _description_. Defaults to 10000.
        method (str, optional): _description_. Defaults to "BFGS".
    """
    res = fit_trajectory(
        ansatz=ansatz, 
        x=x, ex=ex, y=y, ey=ey, 
        guess=guess, 
        maxiter=maxiter, method=method,
        Cov_inv=Cov_inv)
    return res
#---
