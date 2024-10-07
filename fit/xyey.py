"""Fit of a 1d function of 1 variable """


import numpy as np
from lattice_data_tools.fit.trajectory import fit_trajectory

def fit_xyey(
    ansatz, 
    x: np.ndarray, y: np.ndarray, ey: np.ndarray, 
    guess: np.ndarray, 
    maxiter = 10000, method = "BFGS"
    ):
    """Fit of y=f(x), there f: \mathbb{R}^1 \to \mathbb{R}^1

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
        return np.array([ansatz(x[0], p)])
    #---
    xp = x=np.array([x]).transpose()
    ex = np.zeros(shape=xp.shape)
    yp = np.array([y]).transpose()
    eyp = np.array([ey]).transpose()
    res = fit_trajectory(
        ansatz=ansatz_casted, 
        x=xp, ex=ex, y=yp, ey=eyp, 
        guess=guess, 
        maxiter=maxiter, method=method)
    return res
#---
