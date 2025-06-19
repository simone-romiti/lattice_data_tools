""" 
Gounaris-Sakurai (GS) model of Finite Volume Effects (FVEs) 

This file contains an implementation of the GS model as used in https://inspirehep.net/literature/2103903

"""


import numpy as np
from typing import Callable
from scipy.optimize import brentq

def get_k(omega: float, MP: float):
    k = np.sqrt((omega**2/4) - MP**2)
    return k
#---

def get_omega(k: float, MP: float):
    omega = 2.0*np.sqrt(k**2 + MP**2)
    return k
#---
    
class Luscher_2Pions:
    """ Luscher's formalism for 2-pions state on the lattice """
    def __init__(self, MP: float, Nx: int, N_lev: int, mi_max: int):
        self.MP = MP # pion mass (in lattice units)
        self.N_lev = N_lev # number of energy levels
        self.Nx = Nx  # L/a (volume in lattice units)
        self.mi_max = mi_max # maximum values of \vec{m}_i (see eq. F3 of https://arxiv.org/pdf/2206.15084)
    #---
    def tan_phi(self, z):
        # Generate all integer 3-vectors with each component in [-mi_max, mi_max]
        grid = np.arange(-self.mi_max, self.mi_max + 1)
        mesh = np.stack(np.meshgrid(grid, grid, grid, indexing='ij'), -1).reshape(-1, 3)
        Z3_vectors = mesh[np.max(np.abs(mesh), axis=1) < self.mi_max]
        m2_arr = np.linalg.norm(Z3_vectors, axis=1)**2 # list of |\vec{m}|^2
        num = -2*(np.pi**2)*z
        den = np.sum(1.0/(m2_arr - z**2))
        return (num/den)
    #---
    def find_omega_n(self, delta_11: Callable[[float], float], eps: float):
        """
        Find the momentum k_n for each energy level n using the phase shift function delta_11,
        as in eq F2 of https://arxiv.org/pdf/2206.15084. 
        NOTE: we bring \delta_11 to the right hand side and take the tangent of both sides.

        Parameters
        ----------
        delta_11 : callable
            A lambda function that takes a float (momentum) and returns the phase shift delta_11(k).
        eps : float
            The step size used to search for the roots of the equation.
            

        Returns
        -------
        kn_list : list
            List of momentum values \omega_n for each energy level.
        """
        omega_n_list = np.zeros(shape=(self.N_lev))
        omega_min = 0.0
        sign_left, sign_right = -1, -1
        for n in range(self.N_lev):
            def f(omega):
                k = get_k(omega=omega, MP=self.MP)
                z = k * self.Nx / (2 * np.pi)
                lhs = self.tan_phi(z)
                rhs = np.tan(n * np.pi - delta_11(k))
                return (lhs - rhs)
            #---
            sign_left = np.sign(f(omega_min))
            sign_right = sign_left
            omega_i = omega_min 
            while sign_left != sign_right:
                omega_i += eps
                sign_right = np.sign(f(omega_i))
            #---
            omega_root = brentq(f, omega_min, omega_i)
            omega_n[n] = omega_root
            omega_min = omega_i
        #---
        return np.array(omega_n)
#-----------        


class GS_model:
    """ 
    Gounaris Sakurai model (see eq. F6 of https://arxiv.org/pdf/2206.15084)
    
    MP --> pseudoscalar meson mass (pion)
    MV --> resonance mass (rho meson)
    
    """
    def __init__(self, MP: float, MV: float, g_VPP: float):
        self.MP = MP  # Pion mass (in lattice units)
        self.MV = MV  # Rho mass (in lattice units)
        self.g_VPP = g_VPP  # Rho-pion-pion coupling
    #---
    def h(self, omega: float):
        g = self.g_VPP
        A = g**2/(6*np.pi)
        k = get_k(omega=omega, MP=self.MP)
        B = (k**3/omega)*(2.0/np.pi)
        C = np.log((omega + 2.0*k)/(2*self.MP))
        return A*B*C
    #---
    def hprime(self, omega: float):
        g = self.g_VPP
        A = g**2/(6*np.pi)
        k = get_k(omega=omega, MP=self.MP)
        B = k**2/(np.pi*omega)
        C = 1 + (1 + 2.0*(self.MP**2)/(omega**2))*(omega/k)*np.log((omega+2.0*k)/(2.0*self.MP))
        return A*B*C
    #---
    def Gamma_VPP(self, omega: float):
        g = self.g_VPP
        A = (g**2/(6.0*np.pi))
        k = get_k(omega=omega, MP=self.MP)
        B = k**3/(omega**2)
        return A*B
    #---
    def A_PP(self, omega: float):
        one = self.h(self.MV)
        two = (omega**2 - self.MV**2)*self.hprime(self.MV)/(2*self.MV)
        three = -self.h(omega)
        four = 1j*omega*self.Gamma_VPP(omega)
        return (one+two+three+four)
    #---    
    def F_P(self, omega: float):
        MV2 = self.MV**2
        num = MV2 - self.A_PP(0)
        den = MV2 - omega**2 - self.A_PP(omega)
        return (num/den)
    #---
    def cot_delta_11(self, k: float):
        MP, MV = self.MP, self.MV
        omega = get_omega(k=k, MP=MP)
        num = MV**2 - omega**2 - self.h(MV) - (omega**2 - MV**2)*self.hprime(MV)/(2.0*MV) + self.h(omega)
        den = omega*self.Gamma_VPP(omega)
        return (num/den)
    def delta_11(self, k: float):
        return np.arctan(1.0/self.cot_delta_11(k))


if __name__ == "__main__":
    MP = 0.140 # pion mass
    MV = 0.775 # rho mass
    g_VPP =  5.95 # r_{\rho\pi\pi}
    L2P = Luscher_2Pions(MP=MP, Nx=4, N_lev=5, mi_max=10)
    GS1 = GS_model(MP=MP, MV=MV, g_VPP=g_VPP)
    delta_11 = lambda k: GS1.delta_11(k)
    omega_n = L2P.find_omega_n(delta_11 = delta_11, eps=1e-10)
    print("Roots:")
    print(omega_n)
    print("Done!")