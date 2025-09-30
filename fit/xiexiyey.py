"""Fit of a 1d function of n variables with errors on the "x" too """

import numpy as np
from lattice_data_tools.fit.trajectory import fit_trajectory

def fit_xiexiyey(
    ansatz, 
    x: np.ndarray, ex: np.ndarray, y: np.ndarray, ey: np.ndarray, 
    guess: np.ndarray, 
    method = "BFGS",
    Cov_estimate = None,
    ):
    """Fit of y=f(\vec{x}), there f: R^n \to R^1

    Args:
        ansatz : ansatz taking a float and returning a float
        x (np.ndarray): 1d array of x values
        y (np.ndarray): 1d array of y values
        ey (np.ndarray):1d array of errors on the values
        guess (np.ndarray): 1d array fo guesses for the ansatz_
        maxiter (int, optional): _description_. Defaults to 10000.
        method (str, optional): _description_. Defaults to "BFGS".
    """
    def ansatz_casted(x, p):
        return np.array([ansatz(x, p)])
    #---
    yp = np.array([y]).transpose()
    eyp = np.array([ey]).transpose()
    res = fit_trajectory(
        ansatz=ansatz_casted, 
        x=x, ex=ex, y=yp, ey=eyp, 
        guess=guess, 
        method=method,
        Cov_estimate=Cov_estimate)
    return res
#---
