import numpy as np
import lattice_data_tools.gm2.HVP.kernel
from lattice_data_tools.constants import alpha_EM
import yaml
from constants_charm import alpha_EM
from scipy.integrate import simpson

def get_amu(ens_name,ti: np.ndarray, Vi: np.ndarray, K: np.ndarray, Z_ren: float, strategy: str = "trapezoidal", mode: str = "SD") -> float:
    """ 
    Calculate a_\mu using different methods (SD, W, LD, or full) as in https://inspirehep.net/literature/2615948.

    Args:
        ens_name (str): The ensemble name.
        ti (np.ndarray): Array of times, e.g., [1,2,3,...].
        Vi (np.ndarray): Values of the (bare) correlator at each ti (including charge factors).
        K (np.ndarray): Precomputed value of the kernel at each ti.
        Z_ren (float): Renormalization constant of the vector current: Z_A for TM and Z_V for OS.
        strategy (str): Integration strategy (e.g., "trapezoidal").
        mode (str): Calculation mode. Options are:
                    "SD" for short-distance contribution,
                    "W" for window contribution,
                    "LD" for long-distance contribution,
                    "full" for using the precomputed kernel.

    Returns:
        float: The value of a_\mu for the specified mode.
    """

    # Load ensemble information from the YAML file
    with open("charm-ens_info.yaml", 'r') as f:
        ens_info = yaml.safe_load(f)
    
    # Extract lattice spacing (a_fm) and calculate various constants
    a_fm = ens_info[ens_name]["a_fm"]
    t_0 = 0.4 / a_fm
    t_1 = 1 / a_fm
    Delta_const = 0.15 / a_fm
    t2 = ti**2
    
    # Determine the integration weights based on the mode
    if mode == "SD":
        Theta_SD = 1 - 1 / (1 + np.exp(-2 * (ti - t_0) / Delta_const))
        integrand = t2 * K * Theta_SD * Vi
    
    elif mode == "W":
        Theta_W = 1 / (1 + np.exp(-2 * (ti - t_0) / Delta_const)) - 1 / (1 + np.exp(-2 * (ti - t_1) / Delta_const))
        integrand = t2 * K * Theta_W * Vi
    
    elif mode == "LD":
        Theta_LD = 1 / (1 + np.exp(-2 * (ti - t_1) / Delta_const))
        integrand = t2 * K * Theta_LD * Vi
    
    elif mode == "full":
        integrand = t2 * K * Vi
    
    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose from 'SD', 'W', 'LD', or 'full'.")

    # Perform integration based on the chosen strategy
    res = 0.0
    if strategy == "trapezoidal":
        res = np.trapz(integrand, x=ti)
    elif strategy == "rectangles":
        res = np.sum(integrand)
    elif strategy == "simpson":
        res = simpson(integrand, x=ti)
    else:
        raise ValueError(f"Illegal integration strategy: {strategy}")

    # Multiply by the appropriate factors
    res *= (2.0 * (alpha_EM**2) * (Z_ren**2))
    
    return res

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
