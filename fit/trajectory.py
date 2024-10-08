"""Routines for fitting a function $f: \mathbb{R}^n \to \mathbb{R}^m$

This file contains functions for fitting multi-dimensional functions 
of many variables. This is done using a the maximum likelyhood principle 
and a "trajectory method":
the function $\vec{y} = \vec{f}(\vec{x})$ is thought as 
a trajectory $\vec{z}(t)$ in a n+m dimensional space: 

$$z_i = x_i \, , \, i=1,...,n$$
$$z_i = y_i \, , \, i=n+1,...,m$$

The fit minimizes the distance of this trajectory 
from the theoretical expectation.
"""

import numpy as np
import scipy.optimize as opt

def fit_trajectory(
    ansatz, 
    x: np.ndarray, ex: np.ndarray, 
    y: np.ndarray, ey: np.ndarray, 
    guess: np.ndarray, 
    method = "BFGS"
    ):
    """Fit a function f: \mathbb{R}^n \to \mathbb{R}^m with the trajectory method

    Args:
        ansatz : function f(x, p): x=1d array, p=1d array, returns 1d array
        x (np.ndarray): x values: array of size (N_pts, n)
        ex (np.ndarray): errors on x values: array of size (N_pts, n)
        y (np.ndarray): y values: array of size (N_pts, m)
        ey (np.ndarray): errors on y values: array of size (N_pts, m)
        guess (np.ndarray): guesses for the ansatz of the fit
        maxiter (int, optional): _description_. Defaults to 10000.
        method (str, optional): _description_. Defaults to "BFGS".

    Returns:
        dict: Dictionary with the information about the fit
    """
    assert (x.shape[0] == y.shape[0]) ## same number of points
    N_pts = x.shape[0] # number of points
    ix_with_err = tuple(np.argwhere(ex > 0).T) # indices of points with error
    x_fit = np.copy(x[ix_with_err])
    ex_fit = ex[ix_with_err]
    iy_with_err = np.where(ey > 0) # indices of points with error
    y_fit = y[iy_with_err]
    ey_fit = ey[iy_with_err]

    N_par = len(guess) # number of parameters of the fit ansatz
    N_dof = N_pts - N_par # number of degrees of freedom

    # chi square residual function
    def ch2(p_all):
        p_ansatz = p_all[0:N_par] ## parameters of the fit only
        p_x = p_all[N_par:] # .reshape(x_fit.shape)
        x[ix_with_err] = p_x ## replacing x_i (with errors) with fit parameters
        y_th = np.array([ansatz(x[i,:], p_ansatz) for i in range(N_pts)])[iy_with_err] # theoretical values
        ch2_x = np.sum(((x_fit - p_x)/ex_fit)**2)
        ch2_y = np.sum(((y_fit - y_th)/ey_fit)**2)
        ch2_res = ch2_x + ch2_y
        return ch2_res
    #---
    guess = np.concatenate((guess, np.copy(x[ix_with_err].flatten())))
    mini = opt.minimize(fun = ch2, x0 = guess, method = method)
    ch2_value = ch2(mini.x)
    
    res = dict({})

    res["ansatz"] = ansatz
    res["N_par"] = N_par
    res["par"] = mini.x
    res["ch2"] = ch2_value
    res["dof"] = N_dof ## degrees of freedom
    res["ch2_dof"] = ch2_value / N_dof

    return(res)
#---
