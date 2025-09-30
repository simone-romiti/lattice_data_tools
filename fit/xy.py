"""Fit of a 1d function of 1 variable without errors on data points """


import numpy as np
from lattice_data_tools.fit.xyey import fit_xyey

def fit_xy(
    ansatz, 
    x: np.ndarray, y: np.ndarray, 
    guess: np.ndarray, 
    method = "BFGS",
    Cov_estimate = None
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
    """
    N_pts = y.shape[0]
    ey = np.full(shape=(N_pts), fill_value=1.0)
    res = fit_xyey(
        ansatz=ansatz, 
        x=x, y=y, ey=ey, 
        guess=guess, 
        method=method,
        Cov_estimate=Cov_estimate)
    return res
#---
