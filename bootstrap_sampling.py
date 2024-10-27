## resampling techniques from gauge configurations

import numpy as np

from lattice_data_tools import uwerr

class Boot(np.ndarray):
    """Class for bootstrap samples array

    The central limit theorem implies that:
    1. The mean over the N_bts samples of the arithmetic averages over the K points converges to the mean value
    2. The N_bts means over the K points are gaussianly distributed. 
       The Standard Error on the Mean is estimated as from those. 
       This doesn't require to know the original distribution.
    """
    def __new__(cls, input_array):
        # Convert input to an instance of our subclassed array
        obj = np.asarray(input_array).view(cls)
        assert(len(obj.shape) >= 2)
        cls.N_bts = obj.shape[0] # number of bootstrap samples
        cls.K = obj.shape[1] # number of points of each bootstrap sample 
        return obj
    #---
    def mean(self):
        """Estimator of the mean through the bootstrap samples"""
        return np.ndarray.mean(np.ndarray.mean(self, axis=1), axis=0)
    #---
    def average(self):
        return self.mean()
    #---
    def std(self):
        """Estimator of the standard error on the mean using the bootstrap samples"""
        avgs = np.ndarray.mean(self, axis=1)
        return np.ndarray.std(avgs, axis=0, ddof=1)
    def var(self):
        """Estimator of (variance/N) using the bootstrap samples"""
        return self.std()**2
#-------

def parametric_gaussian(mu, sigma, N_bts, S, seed=12345):
    return Boot([np.random.normal(loc=mu, scale=sigma, size=S) for i in range(N_bts)])
#---

def uncorrelated_confs_to_bts(x, N_bts, S=2, seed=12345):
    """Bootstrap samples from array of data
    
    generates N_bts bootstrap samples from N configurations,
    It assumes x.shape[0] == N.
    
    Steps:
    - Draw S points with replacements among the x values. This set of values is a bootstrap sample.
    - Repeat N_bts times.

    Args:
        x (np.ndarray): time series. Bootstrapping is done on 1st index
        N_bts (int): Number of bootstraps
        S (int): Number of points of each sample. The default is S=2
        block_size (int): size of the block
    
    Returns:
        np.ndarray: Bootstrap samples
    """
    N = x.shape[0]
    np.random.seed(seed=seed)
    res = Boot([x[np.random.randint(0, high=N, size=S, dtype=int)] for i in range(N_bts)])
    return res
#---

def correlated_confs_to_bts(Cg: np.ndarray, N_bts: int, seed=12345, output_file=None) -> np.ndarray:
    """Bootstrap samples from array of correlated configurations

    - The configurations are sampled every tau_int, (integrated autocorrelation time)
    - The bootstrap samples are drawn from the uncorrelated configurations

    Args:
        C (np.ndarray): 1-dimensional "correlator" (observable computed for each configuration)
        N_bts (int): Number of bootstraps

    Returns:
        np.ndarray: Bootstrap samples
    """
    Ng = Cg.shape[0] ## total number of configurations
    tauint = int(uwerr.uwerr_primary(Cg, output_file=output_file)["tauint"]) ## integrated autocorrelation time
    if tauint == 0:
        tauint = 1 ## uncorrelated data
    #---
    Cg_uncorr = Cg[0:Ng:tauint] ## uncorrelated values
    return uncorrelated_confs_to_bts(x=Cg_uncorr, N_bts=N_bts, seed=seed)
#---

