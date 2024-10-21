""" Interpolation routines """

# from math import fabs
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

from lattice_data_tools.fit.xyey import fit_xyey

def interpolate_x0(y0, ansatz, par, x0_guess):
    """Finds the value x0 such that ansatz(x0, par)==y0 """
    g = lambda z: y0 - ansatz(z, par) 
    # Find the root of g(x)
    result = root(g, x0_guess, method='lm', tol=1e-15)
    if result.success:
        x_0 = result.x[0]
        return x_0
    else:
        raise ValueError("Root not found. Try a different initial_guess.")
#-------

def x0_from_xyey(
        ansatz, guess: np.ndarray, 
        y0: np.float64, 
        x: np.ndarray, y: np.ndarray, ey: np.ndarray, 
        maxiter = 10000, method = "BFGS"):
    """ Interpolate y(x) to x0 """
    mini = fit_xyey(ansatz, x=x, y=y, ey=ey, guess=guess, method = method)
    par_fit = mini["par"]
    x0 = interpolate_x0(y0=y0, ansatz=ansatz, par=par_fit, x0_guess=np.average(x))
    ansatz_th = lambda z: ansatz(z, par_fit)
    res = {"x0": x0, "theory": ansatz_th, "par_fit": par_fit}
    return res
#---

