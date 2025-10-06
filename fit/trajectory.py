"""Routines for fitting a function $f: \\mathbb{R}^n \\to \\mathbb{R}^m$

This file contains functions for fitting multi-dimensional functions 
of many variables. This is done using the maximum likelyhood principle 
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
    method = "BFGS",
    Cov_inv = None
    ):
    """
    Fit a function f: \\mathbb{R}^n \\to \\mathbb{R}^m with the trajectory method

    Args:
        ansatz : function f(x, p): x=1d array, p=1d array, returns 1d array
        x (np.ndarray): x values: array of size (N_pts, n)
        ex (np.ndarray): errors on x values: array of size (N_pts, n)
        y (np.ndarray): y values: array of size (N_pts, m)
        ey (np.ndarray): errors on y values: array of size (N_pts, m)
        guess (np.ndarray): guesses for the ansatz of the fit
        maxiter (int, optional): _description_. Defaults to 10000.
        method (str, optional): _description_. Defaults to "BFGS".
        Cov_inv: inverse covariance matrix estimate (e.g. from a fit on the bootstrap mean)

    Returns:
        dict: Dictionary with the information about the fit
    """
    assert (x.shape[0] == y.shape[0]) ## same number of points
    N_pts = x.shape[0] # number of points
    Nx = x.flatten().shape[0]
    ix_with_err = (ex > 0) # indices of points with error
    x_fit = np.copy(x[ix_with_err])
    ex_fit = ex[ix_with_err]
    iy_with_err = np.where(ey > 0) # indices of points with error
    Ny = y.flatten().shape[0]
    y_fit = y[iy_with_err]
    ey_fit = ey[iy_with_err]

    N_par = len(guess) # number of parameters of the fit ansatz
    N_dof = N_pts - N_par # number of degrees of freedom

    # chi square residual function
    def ch2_uncorr(p_all):
        p_ansatz = p_all[0:N_par] # parameters of the fit only
        p_x = p_all[N_par:] 
        ch2_x = np.sum(((x_fit - p_x)/ex_fit)**2)
        y_th = np.array([ansatz(x[i,:], p_ansatz) for i in range(N_pts)])[iy_with_err] # theoretical values
        # print(".", (y_fit - y_th).flatten()[0:4])
        ch2_y = np.sum(((y_fit - y_th)/ey_fit)**2)
        ch2_res = ch2_x + ch2_y
        return ch2_res
    if Cov_inv is None:
        ch2 = lambda p: ch2_uncorr(p)
    else:
        assert(len(Cov_inv.shape) == 2)
        assert(Cov_inv.shape[0] == Cov_inv.shape[1])
        assert(Cov_inv.shape[0] == Nx+Ny)
        X_th = np.copy(x)
        Y_th = np.copy(y)
        def ch2(p_all):
            p_ansatz = p_all[0:N_par] # parameters of the fit only
            p_x = p_all[N_par:]
            X_th[ix_with_err] = p_x 
            dx = (x - X_th).T
            Y_th[iy_with_err] = np.array([ansatz(X_th[i,:], p_ansatz) for i in range(N_pts)])[iy_with_err]
            dy = (y - Y_th).T
            z = np.concatenate((dx.flatten(), dy.flatten()))
            ch2_res = np.sum(z.T @ Cov_inv @ z)
            # print(",", dy.flatten()[0:4])
            # ch2_uncorr_value = ch2_uncorr(p_all=p_all)
            return ch2_res
    #-------
    guess = np.concatenate((guess, np.copy(x[ix_with_err].flatten())))
    mini = opt.minimize(fun = ch2, x0 = guess, method = method)
    par = mini.x
    ch2_value = ch2(par)
    
    res = dict({})

    res["ansatz"] = ansatz
    res["N_par"] = N_par
    res["N_pts"] = N_pts
    res["par"] = par
    res["ch2"] = ch2_value
    res["N_dof"] = N_dof ## degrees of freedom
    ch2_dof = float("nan")
    if N_dof > 0:
        ch2_dof = ch2_value / N_dof
    #---
    res["ch2_dof"] = ch2_dof
    return(res)
#---
