"""Routines for the combination of different lattice regularizations (e.g. twisted mass and Osterwalder Seyler fermions)

The idea is to exploit the fact that in the continuum limit,
different lattice regularizations must agree,
because they differ only by lattice artifacts
"""

import numpy as np
from typing import Callable
import scipy.optimize as opt

class lambda_method:
    """Minimization routines for minimizing the discretization effects with the "lambda method"
    
    If two observables y1 and y2 are the same in the continuum, 
    this quantity converges to the same limit for any \lambda:
    y(lambda) = lambda*y1 + (1-lambda)*y2 for any lambda
    
    This class defines routines for finding the optimal parameter \lambda 
    such that discretization effects on y(lambda) are minimal.
    
    """
    def __init__(self, a: np.ndarray):
        """Setting the lattice spacing values

        Args:
            a (np.ndarray): 1d array of lattice spacings
        """
        self.a = a
    #---
    def y1y2_lambda_fixed(
        self, 
        y1: np.ndarray, y2: np.ndarray, 
        method="BFGS") -> dict:
        """Finding the optimal lambda: the same for each value of the lattice spacing

        Args:
            y1 (np.ndarray): array of values of the 1st observable (i.e. for each lattice spacing)
            y2 (np.ndarray): array of values of the 2nd observable (i.e. for each lattice spacing)
            method (str, optional): minimization routine. Defaults to "BFGS".

        Returns:
            dict: dictionary of results
        """
        assert ((self.a).shape==y1.shape) and (y1.shape==y2.shape)
        diff_a = np.diff(self.a)
        def avg_slope_sqr(lam):
            y_lambda = lam*y1 + (1.0-lam)*y2
            # print(lam, lam*y1, lam*y2)
            diff_y = np.diff(y_lambda)
            slope_avg = np.average((diff_y/diff_a)**2)
            return slope_avg
        #---        
        mini = opt.minimize(fun=avg_slope_sqr, x0=0.5, method=method)
        lam_fit = mini.x[0]
        y_lambda_fit = lam_fit*y1 + (1.0-lam_fit)*y2
        res = {"lambda": lam_fit, "residue": avg_slope_sqr(lam_fit), "y": y_lambda_fit}
        return res
    #---
    def y1y2_lambda_variable(
        self,
        y1: np.ndarray, y2: np.ndarray, 
        ansatz: Callable, guess=np.ndarray, method="BFGS") -> dict:
        """Finding the optimal lambda: it can depend on the lattice spacing

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

        Args:
            y1 (np.ndarray): array of values of the 1st observable (i.e. for each lattice spacing)
            y2 (np.ndarray): array of values of the 2nd observable (i.e. for each lattice spacing)
            method (str, optional): minimization routine. Defaults to "BFGS".

        Returns:
            dict: dictionary of results
        """
        a = self.a
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
#---
