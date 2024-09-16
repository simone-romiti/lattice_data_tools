""" Bounding methods for a_\mu, for a single bootstrap sample """

import numpy as np

from lattice_data_tools.gm2.HVP.amu import get_amu_precomp_K

def ZeroTail(V: np.ndarray, K: np.ndarray, Z_ren: float, t0: int, strategy="trapezoidal") -> float:
    """ 
    Calculation of a_\mu as \int_{0}^{t_0} t^2 K(t) Z^2 V(t)
    In other word, V(t) is replaced with 0 from t=t0 onwards
    NOTE: all quantities should be in lattice units

    Args:
        K (np.ndarray): values of the Kernel: K(0), K(1), ..., K(T_max)
        V (np.ndarray): values of the correlator: V(0), V(1), ..., V(T_max)
        NOTE: 
            the correlator should be "bare" (i.e. not renormalized),
            and should already include the charge factors

        Z_ren: renormalization constant for the vector current: Z_A for tm and Z_V for OS
        t0 (int): value (included) of t after which we set V=0
        strategy (str): integration strategy

    Returns:
        float: value of the integral
    """
    T_ext = V.shape[0]
    ti = np.arange(0, T_ext)
    return get_amu_precomp_K(ti=ti[1:t0], Vi=V[1:t0], K=K[1:t0], Z_ren=Z_ren, strategy=strategy)
####


def MesonPair(M_meson: float, L: float, V: np.ndarray, K: np.ndarray, Z_ren: float, t0: int, strategy="trapezoidal") -> float:
    """ 
    Calculation of a_\mu as \int_{0}^{t_0} t^2 K(t) Z^2 V(t) + \int_{t0}^{T_max} A e^{-E_{\pi\pi} t}
    In other words, we replace the tail of the correlator with the 2 pions state with minimum momentum on the lattice: p_min=(2\pi)/L
    NOTE: this is necessary because 2 pions at rest can't have total angular momentum 1, as the vector current in the correlator V(t).

    Args:
        M_meson (float): meson mass, e.g. pion mass
        L (float): volume
        V (np.ndarray): bare vector-vector correlator
        K (np.ndarray): precomputed values of the Kernel K(t)
        Z_ren (float): renormalization constant of the vector current
        t0 (int): separation point of the 2 contributions to the integral
        strategy (str, optional): numerical integration strategy. Defaults to "trapezoidal".

    Returns:
        float: value of the integral
    """
    T_ext = V.shape[0]
    ti = np.arange(0, T_ext)
    res1 = get_amu_precomp_K(ti=ti[1:t0], Vi=V[1:t0], K=K[1:t0], Z_ren=Z_ren, strategy=strategy)
    p_min = (2*np.pi/L) ## minumum momentum !=0 on the lattice
    E_2pi = 2*np.sqrt(M_meson**2 + p_min**2)
    A0 = V[t0]*np.exp(E_2pi*t0)
    V_2pi_tail = np.array([A0*np.exp(-E_2pi*t) for t in ti[t0:T_ext]]) 
    res2 = get_amu_precomp_K(ti=ti[t0:T_ext], Vi=V_2pi_tail, K=K[t0:T_ext], Z_ren=Z_ren, strategy=strategy)
    res = (res1+res2)
    return res
####

def M_eff_t0_Tail(M_eff: np.ndarray, V: np.ndarray, K: np.ndarray, Z_ren: float, t0: int, strategy="trapezoidal") -> float:
    """ 
    Calculation of a_\mu as \int_{0}^{t_0} t^2 K(t) Z^2 V(t) + \int_{t0}^{T_max} A e^{- M_eff(t0)* t}
    In other words, we replace the tail of the correlator with the state with energy equal to the effective matt at t=t0
    
    The constant "A" is chosen such that the tail matches the correlator at t0, viz. A = C(t0)*e^{+M_eff(t0)*t}

    Args:
        Meff (np.ndarray): effective mass curve of the correlator V(t)
        V (np.ndarray): bare vector-vector correlator V(t)
        K (np.ndarray): precomputed values of the Kernel K(t)
        Z_ren (float): renormalization constant of the vector current
        t0 (int): separation point of the 2 contributions to the integral
        strategy (str, optional): numerical integration strategy. Defaults to "trapezoidal".

    Returns:
        float: value of the integral
    """
    T_ext = V.shape[0]
    ti = np.arange(0, T_ext)
    res = get_amu_precomp_K(ti=ti[1:t0], Vi=V[1:t0], K=K[1:t0], Z_ren=Z_ren, strategy=strategy)
    if t0 < T_ext-1:
        ## NOTE: for t0=T_ext-1 the effective mass may be not defined
        E_tail = M_eff[t0]
        A0 = V[t0]*np.exp(E_tail*t0)
        V_tail = np.array([A0*np.exp(-E_tail*t) for t in ti[t0:T_ext]]) 
        res += get_amu_precomp_K(ti=ti[t0:T_ext], Vi=V_tail, K=K[t0:T_ext], Z_ren=Z_ren, strategy=strategy)
    ####
    return res
####

