""" Routines to deal with action values, tailored for the Nested Sampling data"""

import numpy as np

def get_n_plaq(T: int, L: int, ndims: int):
    dims_fact = (ndims*(ndims-1))/2 # number of plaquettes U_{\mu \nu} of the lattice
    V_tot = (L**(ndims-1))*T # spacetime volume in lattice units
    n_plaq = dims_fact*V_tot
    return n_plaq
#---

class PureGauge:
    """ Routines for the pure gauge theory """
    @staticmethod
    def plaquette_to_action(beta: float, P: np.ndarray, T: int, L: int, ndims: int):
        n_plaq = get_n_plaq(T=T, L=L, ndims=ndims)
        S = beta*n_plaq*(1 - P)
        return S
    #---
    @staticmethod
    def action_to_plaquette(beta: float, S: np.ndarray, T: int, L: int, ndims: int):
        n_plaq = get_n_plaq(T=T, L=L, ndims=ndims)
        P = 1 - (S/beta/n_plaq)
        return P
    #---
#---

