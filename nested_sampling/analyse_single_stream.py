""" 
Set of routines for the analysis of a single stream of data produced with the nested sampling algorithm.

Assumptions: 
  - We work in a "ndims" dimensional spacetime lattice
  - The action is given by beta*S. This is the case e.g. for pure gauge SU(N) theories

"""

import math
import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from numba import jit
import time

#@jit(cache=True, nopython=True)
def log_sum_with_logaddexp(log_wL):
    """ Finds  logZ = log( \sum_i (wL)_i ) from the log( (wL)_i )
    
    -----------
    Explanation:
    
    The partition function is Z = \sum_i w_i * L_i
    As a function of the phase space volume X_i, the w_i and L_i are:
      - very small in almost all the range
      - mildly overlapping 
    Therefore, if we sum them naively we get 0 on all the interval.  
        Solution: we work with logarithms all the time.

    Args:
        log_wL (np.ndarray): log(w[i]*L[i])

    Returns:
        float: logarithm of the sum
    """
    logZ = logsumexp(log_wL)
    return logZ
####

class MotherAnalysis:
    """ Mother class for the types of analyses """
    def __init__(self, n_live, n_iter, L, T, ndims, S):
        self.n_live = n_live ## number of live points for each iterations
        self.n_iter = n_iter ## number of iterations
        self.L = L ## spatial extent of the lattice
        self.T = T ## time extent of the lattice
        self.ndims = ndims ## number of spacetime dimensions
        self.V_tot = (L**(ndims-1))*T ## total volume in lattice units
        self.S = S ## ordered (descending) values of the action
    ####
    def get_log_w(self, strategy="fwd"):
        """ weights w_i (see eq. 14 of https://arxiv.org/pdf/2205.15570)
        
        strategy:
            - fwd: forward difference: w_i = X_{i+1} - X_{i}
            - symm: symmetric difference: w_i = (1/2)*(X_{i-1} - X_{i+1})
        
        """
        logX_arr = self.get_logX()
        t = self.get_compression_factor()
        if strategy == "fwd":
            ## log of forward difference X[i] - X[i+1]
            log_w = logX_arr  + np.log(1.0 - t)
            return log_w
        elif strategy == "symm":
            logt = self.get_log_compression_factor()
            # print(t, t**2, t**(-2))
            log_w = -np.log(2) + logX_arr - logt + np.log(1.0 - t**2)
            return log_w
        else:
            raise ValueError("Illegal strategy for computing log(w): {strategy}")
        ####
    ####
    def get_logX(self):
        """ logarithms of the phase space volumes """
        logt_arr = self.get_logt()
        return  np.cumsum(logt_arr)
    ####
    def get_log_wL_normalized(self, log_wL):
        """ 
        Total weights w_i*L_i, where 1 = Z = \sum_i w_i*L_i
        NOTE: The normalization using Z (see implementation) is crucial.
        If not used, exp(log(w_i*L_i)) is typically compatible with 0 numerically.
        """
        logZ = log_sum_with_logaddexp(log_wL)
        log_wL = log_wL - logZ
        return log_wL
    ####
    def get_log_Z_curve(self, log_wL_normalized):
        """
        The partition function is normalized to 1.
        The total weights w_i*L_i are the discrete version of the probability density function.
        The cumulative sum of the total weights gives use the cumulative density function. This function returns the log of that curve.
        """
        s = log_wL_normalized[0]
        logZ = [s]
        for i in range(self.n_iter-1):
            s = np.logaddexp(s, log_wL_normalized[i+1])
            logZ.append(s)
        ####
        return np.array(logZ)
    ####
    def get_log_rho(self):
        """ 
        Logarithm of the density of states \\rho(S) = dS/dX
        NOTE: The action plays the analogous role of the energy in statistical mechanics
        """
        log_S = np.log(self.S)
        logX = self.get_logX()
        log_der = np.log(np.gradient(logX, log_S))
        log_rho = logX - log_S + log_der
        return log_rho
    ####
    def run_analysis_beta_ref(self, beta_ref, wi_strategy="fwd", eps_wL=1e-7):
        """ Behavior at a given refernce value of beta """
        ## phase space volumes
        logX = self.get_logX()

        ## computing the (normalized) total weights
        log_w = self.get_log_w(strategy=wi_strategy)
        logL = -beta_ref*self.S
        log_wL = (log_w+logL)

        log_wL = self.get_log_wL_normalized(log_wL=log_wL)
        wL = np.exp(log_wL)

        ## finding the indices of the pruned interval where the "important" configurations lie
        pruned_interval = (wL>eps_wL)
        idx_prn = np.where(pruned_interval)
        ## Cumulants of partition function
        log_Z_curve = self.get_log_Z_curve(log_wL_normalized=log_wL)
        # Z = np.exp(log_Z)
                
        res = dict({
            "S": self.S,
            "logX": logX,
            "log_w": log_w,
            "logL": logL,
            "log_wL": log_wL,
            "idx_pruned_interval": idx_prn,
            "log_Z_curve": log_Z_curve 
        }) 
        
        return res
    ####
    def get_average_plaquette(self, beta_range):
        """ 
        The average plaquette is:  <P> = \int dX L(X) * P(X)
        
        where:
            P = 1 - (S/V/ndims_fact)
            ndims_fact = d(d-1)/2
        """
        ## Plaquette expectation value
        n_arr = beta_range.shape[0]
        P_avg = []
        dims_fact = self.ndims*(self.ndims-1)/2
        for i in range(n_arr):
            beta = beta_range[i]
            log_w = self.get_log_w(strategy="fwd")
            logL = -beta*self.S
            log_wL = self.get_log_wL_normalized(log_wL=(log_w+logL))
            P = 1.0 - self.S/self.V_tot/dims_fact
            wL = np.exp(log_wL)
            P_avg_val = np.sum(wL*P)
            P_avg.append(P_avg_val)
            ##
        ####
        P_avg = np.array(P_avg)
        return P_avg
    ####
    def get_average_P2(self, beta_range):
        """ 
        The average plaquette^2 is: 
        
        <P> = \sum_i wL[i]*(P[i]^2) 
        
        where P(S) = 1 - (S/V/ndims_fact)
        """
        ## density of states
        # log_rho = self.get_log_rho()

        ## Plaquette expectation value
        n_arr = beta_range.shape[0]
        P2_avg = []
        dims_fact = self.ndims*(self.ndims-1)/2
        for i in range(n_arr):
            beta = beta_range[i]
            log_w = self.get_log_w(strategy="fwd")
            logL = -beta*self.S
            log_wL = self.get_log_wL_normalized(log_wL=(log_w+logL))
            wL = np.exp(log_wL)
            P_i = 1.0 - (self.S/self.V_tot/dims_fact)
            P2_i = P_i**2  
            P2_avg_val = np.sum(wL*P2_i)
            P2_avg.append(P2_avg_val)
            # log_rhoL = log_rho - beta*self.S
            # log_rhoL -= log_sum_with_logaddexp(log_rho - beta*self.S) ## normalizing
            # log_S_avg = log_sum_with_logaddexp(log_rhoL + np.log(self.S/self.V_tot))
            # P_avg.append(1.0 - np.exp(log_S_avg))
        ####
        P2_avg = np.array(P2_avg)
        return P2_avg
    ####
####


class naive_analysis(MotherAnalysis):
    """ 
    Analysis of nested sampling data using the naive compression factor:
    see eq. 12 of https://arxiv.org/pdf/2205.15570
    """
    def __init__(self, n_live, n_iter, L, T, ndims, S):
        super().__init__(
            n_live=n_live, n_iter=n_iter, 
            L=L, T=T, ndims=ndims, 
            S=S
            )
    ####
    def get_log_compression_factor(self):
        return (-1/self.n_live)
    ####
    def get_compression_factor(self):
        return np.exp(self.get_log_compression_factor())
    ####
    def get_logt(self):
        """ array of logs of the compression factors. In the naive approach they are all the same """
        cf = self.get_log_compression_factor()
        logt = np.array(self.n_iter*[cf])
        return logt
    ####
####

class bts_analysis(MotherAnalysis):
    """ 
    
    !!! --------------------------------------- !!!
    !!! WARNING: I have never tested this class !!!
    !!! --------------------------------------- !!!
    
    Analysis of the nested sampling data using the bootstrap sampling. 
    The compression factor is estimated sampling from the beta distribution
    see eq. 10 of https://arxiv.org/pdf/2205.15570
    """
    def __init__(self, n_live, n_iter, L, T, ndims, S):
        super().__init__(
            n_live=n_live, n_iter=n_iter, 
            L=L, T=T, ndims=ndims, 
            S=S
            )
    ####
    def get_t(self):
        return np.random.power(self.n_live, size=self.n_iter)
    ####
    def get_logt(self):
        return np.log(self.get_t())
    ####
    def bts_phase_space_volumes(self):
        t = self.get_compression_factor()
        return np.cumsum(t,axis=1)
    ####
    def get_log_w(self, strategy="fwd"):
        """ weights w_i (see eq. 14 of https://arxiv.org/pdf/2205.15570)
        
        strategy:
            - fwd: forward difference: w_i = X_{i+1} - X_{i}
            - symm: symmetric difference: w_i = (1/2)*(X_{i-1} - X_{i+1})
        
        """
        logX_arr = self.get_logX()
        t = self.get_t()
        tp = np.append(t[1:], np.random.power(self.n_live-1, size=1))
        logt = np.log(t)
        if strategy == "fwd":
            ## forward difference X[i+1] - X[i]
            log_w = logX_arr  + np.log(1.0 - t)
            return log_w
        elif strategy == "symm":
            log_w = -np.log(2) + logX_arr - logt + np.log(1.0 - t*tp)
            return log_w
        else:
            raise ValueError("Illegal strategy for computing log(w): {strategy}")
        ####
    ####
####


