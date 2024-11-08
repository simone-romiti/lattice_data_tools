"""Routines for the combination of different lattice regularizations (e.g. twisted mass and Osterwalder Seyler fermions)

The idea is to exploit the fact that in the continuum limit,
different lattice regularizations must agree,
because they differ only by lattice artifacts
"""

import numpy as np
from typing import Callable
import scipy.optimize as opt
from typing import Callable, List

import matplotlib.pyplot as plt

class lambda_method:
    """Minimization routines for minimizing the discretization effects with the "lambda method"
    
    If n observables y1(a) and y2(a) are the same in the continuum, 
    this quantity converges to the same limit for any \lambda(a) such that 
    $\lim_{a\to 0} \lambda(a) = \lambda_0 < \infty$:
    
    $$y_\lambda(a) = \sum_{i=1}^{n} \lambda_i(a) y_i(a) \, , \, \sum_{i} \lambda_i(a)=1$$
    
    This class defines routines for finding the optimal parameter \lambda(a) 
    such that discretization effects on y_\lambda are minimal. 
    This is implemented by minimizing the average of the squares at each point.
    See the documentation of the for more details.

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
            diff_y = np.diff(y_lambda)
            slope_avg = np.average((diff_y/diff_a)**2)
            return slope_avg
        #---        
        mini = opt.minimize(fun=avg_slope_sqr, x0=guess, method=method)
        par_fit = mini.x
        lam_fit = np.array([ansatz(ai, par_fit) for ai in a])
        y_lambda_fit = lam_fit*y1 + (1.0-lam_fit)*y2
        res = {"par": par_fit, "lambda": lam_fit, "residue": avg_slope_sqr(par_fit), "y": y_lambda_fit}
        return res
    #---
    def y1y2_lambda_auto(
        self,
        y1: np.ndarray, y2: np.ndarray,
        fk: List[Callable],
        method="Nelder-Mead") -> dict:
        def ansatz_lambda(ai, p):
            N_par = p.shape[0]
            res = p[0]
            for k in range(1, N_par):
                res += (p[k]/2)*fk[k-1](ai)
            #---
            return res
        #---
        n_guess = 1+len(fk) # number of guesses
        guess = np.zeros(shape=(n_guess)) # array of guesses
        f0 = lambda ai: 1
        fk_values = np.array([fk_fun(self.a) for fk_fun in fk]) # arrays of values of f_k(a)
        lambda_km1 = y2 # lambda=0
        y_bar = (y1+y2)/2 # average of the 2 curves
        N_pts = (self.a).shape[0]
        y_lambda_km1 = np.zeros(shape=(N_pts))
        for k in range(n_guess):
            if k==0:
                ck = np.average(np.diff(-y2)/np.diff(y1-y2))
            else:
                lambda_0 = guess[0]
                ck = np.average(np.diff(-y_lambda_km1)/np.diff(fk_values[k-1]*y_bar))
            #---
            guess[k] = ck
            res_k = self.y1y2_lambda_variable(y1=y1, y2=y2, ansatz=ansatz_lambda, guess=guess[0:(k+1)], method=method)
            par_fit = res_k["par"]
            lambda_km1 = res_k["lambda"]
            y_lambda_km1 = lambda_km1*y1 + (1.0-lambda_km1)*y2
            guess[0:(k+1)] = par_fit
        #---
        res = res_k
        res["guess"] = guess
        return res
    #---
#---
