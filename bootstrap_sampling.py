## resampling techniques from gauge configurations

import numpy as np

from lattice_data_tools import uwerr





def parametric_gaussian(mu, sigma, N_bts, S=None, seed=12345):
    """Parametric boostraps

    Args:
        mu (_type_): _description_
        sigma (_type_): _description_
        N_bts (_type_): _description_
        S (_type_, optional): _description_. Defaults to None.
        seed (int, optional): _description_. Defaults to 12345.

    Returns:
        _type_: _description_
    """
    np.random.seed(seed)
    if S==None:
        S=N_bts
    #---
    return np.array([np.mean(np.random.normal(loc=mu, scale=sigma, size=S)) for i in range(N_bts)])
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

