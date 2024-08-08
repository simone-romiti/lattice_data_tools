

import numpy as np

import lattice_data_tools.gm2.HVP.kernel
from lattice_data_tools.constants import alpha_EM

def get_amu_precomp_K(ti: np.ndarray, Vi: np.ndarray, K: np.ndarray, Z_ren: float, strategy="trapezoidal"):
    """ 
    a_\mu as in eq. 1 of https://inspirehep.net/literature/2615948
    
    ti : array of times, e.g. [1,2,3,...]
    Vi : values of the (bare) correlator at each ti (already including the charge factors)
    Ki: precomputed value of the kernel at each ti
    Z_ren: renormalization constant of the vector current: Z_A for tm and Z_V for OS
    
    NOTE: All observables should be in lattice units
    """
    t2 = ti**2
    integrand = t2*K*Vi
    res = 0.0
    if strategy == "trapezoidal":
        res = np.trapz(integrand)
    elif strategy == "rectangles":
        res = np.sum(integrand)
    else:
        raise ValueError("Illegal integration strategy: {strategy}".format(strategy=strategy))
    ####
    res *= ( 2.0 * (alpha_EM**2) * (Z_ren**2) )
    return res
####


def get_amu(m: float, ti: np.ndarray, Vi: np.ndarray, N_int: int, strategy="trapezoidal"):
    """ a_\mu from get_amu_precomp_K(), but computing the Kernel on the fly for each "t" """
    K = np.array([kernel.K(mt=m*t, N=N_int) for t in ti])
    return get_amu_precomp_K(ti=ti, Vi=Vi, K=K, strategy=strategy)
####