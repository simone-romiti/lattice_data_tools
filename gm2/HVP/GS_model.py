""" 
Gounaris-Sakurai (GS) model of Finite Volume Effects (FVEs) 

This file contains an implementation of the GS model as used in https://inspirehep.net/literature/2103903

!!! ACHTUNG !!! 
The original paper has a typo in the definition of the function h'(omega) (eq. 20 of https://arxiv.org/pdf/1808.00887).
A factor of "k" is missing in the equation, but the implementation below includes it.

"""


import numpy as np
from typing import Callable
from scipy.optimize import brentq

def get_k(omega: np.float64, MP: np.float64):
    omega2 = omega**2
    MP2 = MP**2
    k = np.sqrt((omega2/4.0) - MP2)
    return k
#---

def get_omega(k: np.float64, MP: np.float64):
    k2 = k**2
    MP2 = MP**2
    omega = 2.0*np.sqrt(k2 + MP2)
    return omega
#---

class Z3_vectors:
    """ 
    Utilities to dead with Z^3 vectors, 
    as in the calculation of \phi(z) [see eq. 13 of https://arxiv.org/pdf/1808.00887]
    """
    def __init__(self, mi_max: int):
        self.mi_max = mi_max # maximum values of \vec{m}_i (see eq. F3 of https://arxiv.org/pdf/2206.15084)
        grid = np.arange(-self.mi_max, self.mi_max + 1) # numbers from -mi_max to mi_max (included)
        self.Z3_vectors = np.stack(np.meshgrid(grid, grid, grid, indexing='ij'), -1).reshape(-1, 3)
        self.m2_arr = np.linalg.norm(self.Z3_vectors, axis=1)**2 # list of |\vec{m}|^2
        self.unique_m2, self.m2_multiplicities = np.unique(self.m2_arr, return_counts=True) # unique |m|^2 and multiplicities
    #---
    def m2_info(self):
        return {"m2": self.unique_m2, "nu": self.m2_multiplicities}
    #---
#---    
class Luscher_2Pions:
    """ Luscher's formalism for 2-pions state on the lattice """
    def __init__(self, MP: np.float64, L: int, N_lev: int, Z3_obj: Z3_vectors):
        self.MP = MP # pion mass (in lattice units)
        self.N_lev = N_lev # number of energy levels
        self.L = L  # volume
        self.Z3_obj = Z3_obj
        m2_info = self.Z3_obj.m2_info() # |\vec{m}|^2 values and their multiplicities
        self.m2_vals, self.nu_m2 = m2_info["m2"], m2_info["nu"]
    #---
    def tan_phi(self, z):
        # Generate all integer 3-vectors with each component in [-mi_max, mi_max]
        num = -2.0*(np.pi**2)*z
        nth_terms = self.nu_m2 * (1.0/(self.m2_vals - z**2))
        den = np.sum(nth_terms)
        return (num/den)
    #---
    def phi(self, z):
        return np.arctan(self.tan_phi(z=z))
    #---
    def omega_n_residue_function(self, delta_11: Callable[[float], float], k: np.float64):
        """ The zeroes of this function give the energy levels of the PP states """
        z = k * self.L / (2.0 * np.pi)
        lhs = self.tan_phi(z)
        rhs = -np.tan(delta_11(k))
        res = (lhs-rhs)
        return res
    #---        
    def find_omega_n(self, delta_11: Callable[[float], float], eps: np.float64):
        """
        Find the momentum k_n for each energy level n using the phase shift function delta_11,
        as in eq F2 of https://arxiv.org/pdf/2206.15084. 
        NOTE: we bring \delta_11 to the right hand side and take the tangent of both sides.

        Parameters
        ----------
        delta_11 : callable
            A lambda function that takes a float (momentum) and returns the phase shift delta_11(k).
        eps : np.float64
            The step size used to search for the roots of the equation.

        Returns
        -------
        kn_list : list
            List of momentum values \omega_n for each energy level.
        """
        omega_n = np.zeros(shape=(self.N_lev))
        omega_min = 2.0*self.MP + eps # at 2*MP delta_11 diverges
        sign_left, sign_right = -1, -1
        for n in range(self.N_lev):
            f = lambda omega: self.omega_n_residue_function(delta_11=delta_11, k=get_k(omega=omega, MP=self.MP))
            #---
            sign_left = np.sign(f(omega_min))
            sign_right = sign_left
            omega_i = omega_min 
            while sign_left == sign_right:
                omega_i += eps
                sign_right = np.sign(f(omega_i))
            #---
            omega_root = brentq(f, omega_min, omega_i)
            omega_n[n] = omega_root
            omega_min = omega_i
        #---
        return np.array(omega_n)
    #---
    def get_nuA2_n(
        self, 
        n: int, omega_arr: int, 
        FP2: Callable[[float], float], delta_11: Callable[[float], float],
        eps_k: np.float64
        ):
        """ \nu_n * |A_n|^2 as in eq. 15 of https://arxiv.org/pdf/1808.00887 """
        omega_n = omega_arr[n]
        k_n = get_k(omega=omega_arr[n], MP=self.MP)
        A = (2.0*(k_n**5))/(3.0*np.pi*(omega_n**2))
        B = FP2(omega_n) # |F(\omega)|^2
        der_delta_11 = (delta_11(k_n+eps_k) - delta_11(k_n)) / eps_k
        q_n = k_n * (self.L/(2.0*np.pi))
        eps_q = eps_k * (self.L/(2.0*np.pi))
        der_phi = (self.phi(q_n+eps_q) - self.phi(q_n)) / eps_q
        C = 1.0/(k_n*der_delta_11 + q_n*der_phi)
        return A*B*C
    #---
    def get_V_PP(
        self, 
        times: np.ndarray, 
        FP2: Callable[[float], float], delta_11: Callable[[float], float],
        omega_n: np.ndarray,
        eps_der: np.float64
        ):
        """ 
        Compute the two-pion vector correlator V_{\pi\pi}(t) as defined in equation 14 of 
        https://arxiv.org/pdf/1808.00887, using the provided form factor and phase shift functions.

            times (np.ndarray): Array of time values at which to evaluate the correlator.
            FP2 (Callable[[float], float]): Function returning the squared pion form factor F_\pi^2(s) for a given energy squared s.
            delta_11 (Callable[[float], float]): Function returning the isospin-1, angular momentum-1 \pi\pi scattering phase shift δ_11(s) for a given energy squared s.
            eps_roots (float): Tolerance for finding the roots of the quantization condition (energy levels).
            eps_der (float, optional): Tolerance for numerical derivatives used in the calculation. Defaults to 1e-12.

        Returns:
            np.ndarray: The computed V_{\pi\pi}(t) correlator evaluated at the input time values.
        """
        nuA2_n = np.array([self.get_nuA2_n(n=n, omega_arr=omega_n, FP2=FP2, delta_11=delta_11, eps_k=eps_der) for n in range(self.N_lev)])
        exp_fact = np.array([np.exp(-omega_n*t_i) for t_i in times]).transpose()
        V_PP = np.sum(nuA2_n[:,np.newaxis] * exp_fact, axis=0)
        return V_PP
#-----------        


class GS_model:
    """ 
    Gounaris Sakurai model (see eq. F6 of https://arxiv.org/pdf/2206.15084)
    
    MP --> pseudoscalar meson mass (pion)
    MV --> resonance mass (rho meson)
    
    """
    def __init__(self, MP: np.float64, MV: np.float64, g_VPP: np.float64):
        self.MP = MP  # Pion mass (in lattice units)
        self.MV = MV  # Rho mass (in lattice units)
        self.g_VPP = g_VPP  # Rho-pion-pion coupling
    #---
    def arg_log(self, omega: np.float64):
        """ Argument of the logarithm in the h function """
        k = get_k(omega=omega, MP=self.MP)
        return (omega + 2.0*k)/(2.0*self.MP)
    #---
    def h(self, omega: np.float64):
        g = self.g_VPP
        A = g**2/(6.0*np.pi)
        k = get_k(omega=omega, MP=self.MP)
        B = (k**3/omega)*(2.0/np.pi)
        C = np.log(self.arg_log(omega=omega))
        return A*B*C
    #---
    def hprime(self, omega: np.float64):
        """
        Eq. 20 of https://arxiv.org/pdf/1808.00887.

        !!! ACHTUNG !!! 
        The original paper has a typo in the definition of the function h'(omega) 
        A factor of "k" is missing in the equation, but the implementation below includes it.
        """
        g = self.g_VPP
        A = g**2/(6.0*np.pi)
        k = get_k(omega=omega, MP=self.MP)
        # splitting the sum in 2 terms in order to avoid divergences at k=0
        B = k/np.pi # factor of k is missing in the paper, but it is a mistake
        C1 = k/omega
        C2 = (1.0 + 2.0*(self.MP**2)/(omega**2))*np.log(self.arg_log(omega=omega))
        return A*B*(C1 + C2)
    #---
    def Gamma_VPP(self, omega: np.float64):
        # assert (omega != 0) # Gamma_VPP(2*M_P) diverges
        g = self.g_VPP
        A = (g**2/(6.0*np.pi))
        k = get_k(omega=omega, MP=self.MP)
        B = k**3/(omega**2)
        return A*B
    #---
    def A_PP_zero(self):
        one = self.h(self.MV)
        two = (-self.MV/2.0)*self.hprime(self.MV)
        three = (self.g_VPP**2/(6.0*np.pi))*(self.MP**2.0/np.pi)
        return (one+two+three)
    #---            
    def A_PP(self, omega: np.float64):
        one = self.h(self.MV)
        two = (omega**2 - self.MV**2)*self.hprime(self.MV)/(2.0*self.MV)
        three = -self.h(omega=omega)
        four = 1j*omega*self.Gamma_VPP(omega=omega)
        return (one+two+three+four)
    #--- 
    def F_P(self, omega: np.float64):
        MV2 = self.MV**2
        num = MV2 - self.A_PP_zero()
        den = MV2 - omega**2 - self.A_PP(omega)
        return (num/den)
    #---
    def cot_delta_11(self, k: np.float64):
        MP, MV = self.MP, self.MV
        omega = get_omega(k=k, MP=MP)
        num = MV**2 - omega**2 - self.h(MV) - (omega**2 - MV**2)*self.hprime(MV)/(2.0*MV) + self.h(omega)
        den = omega*self.Gamma_VPP(omega)
        return (num/den)
    #---
    def delta_11(self, k: np.float64):
        delta = np.arctan(1.0/self.cot_delta_11(k))
        delta += np.pi*(delta < 0)
        return delta


def get_V_PP_GSmodel(
    times: np.ndarray,
    MP: np.float64, MV: np.float64, g_VPP: np.float64, L: np.float64, 
    N_lev: int, Z3_obj: Z3_vectors, 
    eps_roots: np.float64, 
    eps_der: np.float64
    ):
    """ Representation of the Vector-Vector correlator using the Luscher's formalism for PP states in a finite volume """
    PP_mod = Luscher_2Pions(MP=MP, L=L, N_lev=N_lev, Z3_obj=Z3_obj)
    GS_mod = GS_model(MP=MP, MV=MV, g_VPP=g_VPP)
    delta_11 = lambda k: GS_mod.delta_11(k)
    F_squared = lambda omega: np.abs(GS_mod.F_P(omega))**2
    omega_n = PP_mod.find_omega_n(delta_11=delta_11, eps=eps_roots)
    V_PP = PP_mod.get_V_PP(times=times, FP2=F_squared, delta_11=delta_11, omega_n=omega_n, eps_der=eps_der)
    res = {"omega_n": omega_n, "V_PP": V_PP}
    return res
#---    
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt    
    Z3_obj = Z3_vectors(mi_max=100)
    MP_MeV = 0.135 # pion mass
    MV_MeV = 0.775 # rho mass
    g_VPP =  5.5 # g_{\rho\pi\pi}
    # hbarc_MeV_fm = 197.3269631
    # a_fm = 0.082
    # a_MeV_inv = a_fm / hbarc_MeV_fm
    # Nx = 24
    # L_MeV_inv = a_fm * Nx
    # aMP = a_MeV_inv*MP_MeV
    # aMV = a_MeV_inv*MV_MeV
    # Plot delta_11 (in degrees) as a function of omega in the interval [0.4, 1.2]
    GS_mod = GS_model(MP=MP_MeV, MV=MV_MeV, g_VPP=g_VPP)
    omega_vals = np.linspace(0.4, 1.2, 200)
    k_vals = np.array([get_k(omega=omega, MP=MP_MeV) for omega in omega_vals])
    delta_vals_deg = np.array([GS_mod.delta_11(k=k) for k in k_vals])*(180/np.pi)  # Convert radians to degrees
    print("Calling GS_mod methods with omega_vals or k_vals:")
    print("  arg_log:", GS_mod.arg_log(omega_vals)[:2])
    print("  h:", GS_mod.h(omega_vals)[:2])
    print("  hprime:", GS_mod.hprime(omega_vals)[:2])
    print("  Gamma_VPP:", GS_mod.Gamma_VPP(omega_vals)[:2])
    print("  A_PP:", GS_mod.A_PP(omega_vals)[:2])
    print("  F_P:", GS_mod.F_P(omega_vals)[:2])
    print("A_PP_zero:", GS_mod.A_PP_zero())
    print("  cot_delta_11:", GS_mod.cot_delta_11(k_vals)[:2])
    print("  delta_11:", GS_mod.delta_11(k_vals)[:2])    
    F2_vals = np.array([np.abs(GS_mod.F_P(omega=omega))**2 for omega in omega_vals])
    plt.figure()
    plt.plot(omega_vals, delta_vals_deg, label=r'$\delta_{11}$ from GS model')
    plt.plot(omega_vals, F2_vals)
    plt.legend()
    plt.grid(True)
    plt.show()
    # # Example: print the residue function for a range of omega values
    # times = np.arange(3, 23, 1)
    # for N_lev in range(0, 5):
    #     # in lattice units
    #     res = get_V_PP_GSmodel(
    #         times=times,
    #         MP=aMP,
    #         MV=aMV,
    #         g_VPP=g_VPP,
    #         L=Nx,
    #         N_lev=N_lev,
    #         Z3_obj=Z3_obj,
    #         eps_roots=1e-3,
    #         eps_der=1e-14
    #         )
    #     V_PP = res["V_PP"]
    #     plt.plot(times, V_PP, label=f"N_lev={N_lev}")
    #     # plt.yscale('log')
    # #---
    # plt.xlim([2, 22.1])
    # plt.ylim([1e-7, 0.0015])
    # plt.yscale("log")
    # plt.tick_params(direction='in')
    # plt.legend()
    # plt.show()
