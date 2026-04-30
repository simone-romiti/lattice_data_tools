""" Implementation of the QEd kernel for the HVP contribution to the muon's g-2 """

import numpy as np
import typing
import scipy
from numba import njit


def integral_01(f, N: int, method : typing.Literal["trapezoid", "rectangles", "Simpson"] = "trapezoid"):
    """ \\int_0^1 dy f(y) """
    eps = 1.0 / float(N)
    y = np.array([i*eps for i in range(N)]) + (eps/2)
    if method=="trapezoid":
        return eps*np.trapezoid(f(y))
    elif method=="rectangles":
        return eps*np.sum(f(y))
    elif method=="Simpson":
        return scipy.integrate.simpson(f(y), x=y)
#---


def K(mt: float, N: int):
    """ 
    kernel K(m*t) as in eq. 1 of https://inspirehep.net/literature/2615948 
    
    mt:  m*t (NOTE: dimensionless)
    
    """
    def f(y):
        """ integrand of eq. 2 of https://inspirehep.net/literature/2615948 """
        w = (mt / 2.0) * y / np.sqrt(1 - y)
        j0 = np.sin(w) / w
        return (1.0 - y) * (1.0 - j0**2)
    #---
    I = 2.0 * integral_01(f, N)
    return I
#---

def dK_da(a: float, m: float, mt: float, N: int):
    """ 
    Derivative of the kernel K(m*t) as in eq. 1 of https://inspirehep.net/literature/2615948
    with respect to the lattice spacing "a". See K() above for an implementation of the kernel itself.

    a: lattice spacing (dimensionful)
    m: lepton mass (dimensionful, must be in the same dimension of a^{-1})
    mt:  m*t (NOTE: dimensionless)
    
    """
    t = mt/m # time (dimensionful)
    t_over_a = t/a # time in lattice units (no fluctuations, it is exact)
    def df(y):
        """
        Derivative of the integrand of eq. 2 of https://inspirehep.net/literature/2615948
        with respect to the lattice spacing.

        NOTE: \\partial_a w = t_over_a*m*y/2/sqrt(1-y) because there is no dependence
        """
        w = (mt / 2.0) * y / np.sqrt(1 - y)
        j0 = np.sin(w) / w
        j0_prime = np.cos(w)/w - np.sin(w)/(w**2)
        dw_da = m*t_over_a*y/(2.0 * np.sqrt(1-y))
        return (1.0 - y) * (-2.0*j0*j0_prime*dw_da)
    #---
    I = 2.0 * integral_01(df, N)
    return I
#---

