""" Finite Volume formalism for the PP in a finite volume."""

import numpy as np
from typing import Callable
from scipy.optimize import brentq

from lattice_data_tools.gm2.HVP.Z_function import Z_00_Calculator
from lattice_data_tools.gm2.HVP.Gounaris_Sakurai_model import GS_model, get_k

class PP_model:
    """ Luscher's formalism for 2-pions state on the lattice """
    def __init__(self, MP: np.float64, L: int, N_lev: int, Z_00_obj: Z_00_Calculator):
        self.MP = MP # pion mass (in lattice units)
        self.N_lev = N_lev # number of energy levels
        self.L = L  # volume
        self.Z_00_obj = Z_00_obj
    #---
    def omega_n_residue_function(self, delta_11: Callable[[np.float64], np.float64], k: np.float64, n: int):
        """ The zeroes of this function give the energy levels of the PP states """
        q = k * self.L / (2.0 * np.pi)
        lhs =  delta_11(k) + (self.Z_00_obj).phi(q=q)
        rhs = n*np.pi
        res = (lhs-rhs)
        return res
    #---        
    def find_omega_n(self, delta_11: Callable[[np.float64], np.float64], eps: np.float64):
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
            f = lambda omega: self.omega_n_residue_function(delta_11=delta_11, k=get_k(omega=omega, MP=self.MP), n=n+1)
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
        FP2: Callable[[np.float64], np.float64], delta_11: Callable[[np.float64], np.float64],
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
        der_phi = (self.Z_00_obj.phi(q_n+eps_q) - self.Z_00_obj.phi(q_n)) / eps_q
        C = 1.0/(k_n*der_delta_11 + q_n*der_phi)
        return A*B*C
    #---
    def get_V_PP(
        self, 
        times: np.ndarray, 
        FP2: Callable[[np.float64], np.float64], delta_11: Callable[[np.float64], np.float64],
        omega_n: np.ndarray,
        eps_der: np.float64
        ):
        """ 
        Compute the two-pion vector correlator V_{\pi\pi}(t) as defined in equation 14 of 
        https://arxiv.org/pdf/1808.00887, using the provided form factor and phase shift functions.

            times (np.ndarray): Array of time values at which to evaluate the correlator.
            FP2 (Callable[[np.float64], np.float64]): Function returning the squared pion form factor F_\pi^2(s) for a given energy squared s.
            delta_11 (Callable[[np.float64], np.float64]): Function returning the isospin-1, angular momentum-1 \pi\pi scattering phase shift δ_11(s) for a given energy squared s.
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


def get_V_PP_GSmodel(
    times: np.ndarray,
    MP: np.float64, MV: np.float64, g_VPP: np.float64, L: np.float64, 
    N_lev: int, Z_00_obj: Z_00_Calculator, 
    eps_roots: np.float64, 
    eps_der: np.float64
    ):
    """ Representation of the Vector-Vector correlator using the Luscher's formalism for PP states in a finite volume """
    PP_mod = PP_model(MP=MP, L=L, N_lev=N_lev, Z_00_obj=Z_00_obj)
    N_gauss = 100  # number of Gauss-Legendre points
    Lambda = 1.0
    Lambda_Z3 = 10 # cutoff for |n| in Z_00

    q2_max = (N_lev+1)**2
    Z_00_obj = Z_00_Calculator(Lambda_Z3=Lambda_Z3, Lambda=Lambda, N_gauss=N_gauss, q2_max=q2_max)
    GS_mod = GS_model(MP=MP, MV=MV, g_VPP=g_VPP)
    delta_11 = lambda k: GS_mod.delta_11(k)
    F_squared = lambda omega: np.abs(GS_mod.F_P(omega))**2
    omega_n = PP_mod.find_omega_n(delta_11=delta_11, eps=eps_roots)
    V_PP = PP_mod.get_V_PP(times=times, FP2=F_squared, delta_11=delta_11, omega_n=omega_n, eps_der=eps_der)
    res = {"omega_n": omega_n, "V_PP": V_PP}
    return res
#---
    
