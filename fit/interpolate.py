""" Interpolation routines """

import numpy as np
from scipy.optimize import root

from lattice_data_tools.fit.xyey import fit_xyey

def interpolate_x0(y0, ansatz, par, x0_guess):
    g = lambda z: ansatz(z, par) - y0
    # Find the root of g(x)
    result = root(g, np.average(x0_guess))
    if result.success:
        x_0 = result.x[0]
        return x_0
    else:
        raise ValueError("Root not found. Try a different initial_guess.")
#-------

def xyey(
        ansatz, guess: np.ndarray, 
        y0: np.float64, 
        x: np.ndarray, y: np.ndarray, ey: np.ndarray, 
        maxiter = 10000, method = "BFGS"):
    """ Interpolate y(x) to x0 """
    mini = fit_xyey(ansatz, x=x, y=y, ey=ey, guess=guess, maxiter = maxiter, method = method)
    par_fit = mini["par"]
    return interpolate_x0(y0=y0, ansatz=ansatz, par=par_fit, x0_guess=np.average(x))
#---

