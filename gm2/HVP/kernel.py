""" Implementation of the QEd kernel for the HVP contribution to the muon's g-2 """

import numpy as np

def integral_01(f, N: int):
    """ \int_0^1 dy f(y) """
    eps = 1.0 / float(N)
    y = np.array([i*eps for i in range(N)]) + (eps/2)
    return eps*np.trapz(f(y))
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