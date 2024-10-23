"""Routines for the combination of different fermionic regularizations (e.g. twisted mass and Osterwalder Seyler)

The idea is to exploit the fact that in the continuum limit,
different lattice fermionic regularizations must agree
"""

import numpy as np
from typing import Callable
import scipy.optimize as opt

def minimize_discretization_lambda_factor(
    a: np.ndarray, y1: np.ndarray, y2: np.ndarray, 
    ansatz: Callable, guess=np.ndarray, method="BFGS") -> dict:
    """Minimizing the discretization effects by finding the optima parameter \lambda

    The observables y1 and y2 are the same in the continuum, as y(lambda) = lambda*y1 + (1-lambda)*y2 for any lambda

    Steps:
    - we impose lambda=ansatz(a) for each lattice spacing "a"
    - we minimize numerically the slope of y(lambda)

    NOTE: 
    The ansatz should be chosen such that ansatz(a=0) is any finite value (e.g. 1/2 --> average of y1 and y2). 
    In the continuum this gives the correct result. 
    
    Remark: 
    In principle one can also minimize assuming the same lambda for each lattice spacing,
    but this is too restrictive. What is sufficient is that lambda(a=0)=0. 
    This means that we need to minimize at the level of the parameters of the lambda(a) ansatz.

    a: array of lattice spacings
    y1: array of values of the 1st observable (i.e. for each lattice spacing)
    y2: array of values of the 2nd observable (i.e. for each lattice spacing)
    ansatz: ansatz function for lambda(a)
    """
    assert (a.shape==y1.shape) and (y1.shape==y2.shape)
    diff_a = np.diff(a)
    def avg_slope_sqr(p):
        lam = np.array([ansatz(ai, p) for ai in a])
        y_lambda = lam*y1 + (1.0-lam)*y2
        # print(lam, lam*y1, lam*y2)
        diff_y = np.diff(y_lambda)
        slope_avg = np.average((diff_y/diff_a)**2)
        return slope_avg
    #---        
    mini = opt.minimize(fun=avg_slope_sqr, x0=guess, method=method)
    par_fit = mini.x
    lam_fit = np.array([ansatz(ai, par_fit) for ai in a])
    y_lambda_fit = lam_fit*y1 + (1.0-lam_fit)*y2
    print(y_lambda_fit)
    res = {"par": par_fit, "lambda": lam_fit, "residue": avg_slope_sqr(par_fit), "y": y_lambda_fit}
    return res
#---


