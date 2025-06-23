""" Finite Volume formalism for the PP in a finite volume."""


import numpy as np
from typing import Callable
from scipy.optimize import brentq
import scipy

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

class Z_00:
    """ 
    Z_00(q2) as in eq. C8 of https://doi.org/10.1016/0550-3213(91)90366-6,
    with a cutoff on the norm of the 3-vector \vec{n}.    
    """
    def __init__(self, lambda_Z3_reg: float, Lambda_Z3: float):
        self.Lambda_Z3 = Lambda_Z3 # maximum values of |\vec{n}|
        self.Lambda_Z3_reg = lambda_Z3_reg # maximum values of |\vec{n}| for the regularized expression
        grid = np.arange(-self.Lambda_Z3, self.Lambda_Z3 + 1) # numbers from -Lambda_Z3 to Lambda_Z3 (included)
        Z3_mesh = np.stack(np.meshgrid(grid, grid, grid, indexing='ij'), -1).reshape(-1, 3)        
        norms = np.linalg.norm(Z3_mesh, axis=1) # array of all the norms
        # Select only vectors with norm less than Lambda_Z3
        subset = (norms < self.Lambda_Z3)
        self.Z3_arr = Z3_mesh[subset,:]
        self.m2_arr = norms[subset] # array of |\vec{m}|^2
        self.unique_m2, self.m2_multiplicities = np.unique(self.m2_arr, return_counts=True) # unique |m|^2 and multiplicities
        # Select only vectors with norm less than Lambda_Z3_reg
        subset_reg = (norms < self.Lambda_Z3_reg) # subset for the regularized expression
        self.Z3_arr_reg = Z3_mesh[subset_reg,:]
        self.m2_arr_reg = norms[subset_reg] # array of |\vec{m}|^2
        self.unique_m2_reg, self.m2_multiplicities_reg = np.unique(self.m2_arr_reg, return_counts=True) # unique |m|^2 and multiplicities
    #---
    def m2_info(self):
        return {"m2": self.unique_m2, "nu": self.m2_multiplicities}
    #---
    def curly_Y_lm(self, l: int, m:int, r: np.float64, theta: np.float64, phi: np.float64):
        """ \mathcal{Y}_{lm}(r, \theta, \phi) as in eq. 3.14 of https://doi.org/10.1016/0550-3213(91)90366-6 """
        res = (r**l) * scipy.special.sph_harm_y(m, l, phi, theta)
        return res
    #---
    def K_00_small_t(self, t: np.float64, r: np.ndarray, Z3_arr: np.ndarray):
        """ 
        Kernel as in eq. C1 of https://doi.org/10.1016/0550-3213(91)90366-6 
        NOTE: this is the expression that should be used at small t (t < 1)

        r is a 3-dimensional vector in R^3,
        """
        rn = np.dot(Z3_arr, r)
        n2 = np.linalg.norm(Z3_arr, axis=1)**2.0
        tn2 = t*n2
        nth_terms = np.exp(1j*rn - tn2)
        res = np.sum(nth_terms)/((2.0*np.pi)**3.0)
        return np.real(res) # sum is real
    #---

    def K_00_large_t(self, t: np.float64, r: np.ndarray, Z3_arr: np.ndarray):
        """
        Kernel as in eq. C2 of https://doi.org/10.1016/0550-3213(91)90366-6

        r is a 3-dimensional vector in R^3,
        Z3_arr is an array of 3-dimensional vectors in Z^3.
        NOTE: this is the expression that should be used at large t (t > 1)
        """
        diff = r - 2.0 * np.pi * Z3_arr  # shape (N, 3)
        args = -np.linalg.norm(diff, axis=1)**2/(4.0 * t)
        exp_terms = np.exp(args)
        res = np.sum(exp_terms) * ((4.0 * np.pi * t)**(3.0/2.0))
        return res
    #---
    def K_00_reg(self, t: np.float64, r: np.ndarray):
        """ 
        Kernel as in eq. C3 of https://doi.org/10.1016/0550-3213(91)90366-6 
        """
        if t < 1.0:
            bare = self.K_00_small_t(t=t, r=r, Z3_arr=self.Z3_arr)
        else:
            bare = self.K_00_large_t(t=t, r=r, Z3_arr=self.Z3_arr)
        #--- 
        res = bare - self.K_00_small_t(t=t, r=r, Z3_arr=self.Z3_arr_reg)
        return res
    #---
    def get_Z_00_reg(self, q2: np.float64):
        """ Z_00(q2) as in Appendix A of https://arxiv.org/pdf/1306.2532 """
        A = self.curly_Y_lm(l=0, m=0, r=1.0, theta=0.0, phi=0.0)  # Y_00(1, 0, 0) = 1/sqrt(4*pi)
        denominators = (self.unique_m2_reg - q2)
        nth_terms = self.m2_multiplicities_reg * (1.0/denominators)
        sum_Z3 = A*np.sum(nth_terms)
        r0 = np.zeros(shape=(3))
        def integrand(t: np.float64):
            one = np.exp(+t*q2) * self.K_00_reg(t=t, r=r0) 
            two = - 1.0/(((4.0*np.pi)**2)*(t**(3.0/2.0)))
            return one + two
        #---
        integral = scipy.integrate.quad(integrand, 0, np.inf)
        res = sum_Z3 + integral[0]
        return res
    #---


# class Z3_vectors:
#     """ 
#     Utilities to dead with Z^3 vectors, 
#     as in the calculation of \phi(q) [see eq. 13 of https://arxiv.org/pdf/1808.00887]
#     """
#     def __init__(self, Lambda_Z3: int, lam: int):
#         """
#         Initialize the Z^3 vectors with a maximum value of the norm.
#         Lambda_Z3: (\Lambda) is the maximum norm overall
#         lam: (\lambda) is the maximum norm for the regularized expressions
#         """
#         self.Lambda_Z3 = Lambda_Z3 # maximum values of \vec{m}_i (see eq. F3 of https://arxiv.org/pdf/2206.15084)
#         grid = np.arange(-self.Lambda_Z3, self.Lambda_Z3 + 1) # numbers from -Lambda_Z3 to Lambda_Z3 (included)
#         Z3_mesh = np.stack(np.meshgrid(grid, grid, grid, indexing='ij'), -1).reshape(-1, 3)        
#         # Select only vectors with norm less than Lambda_Z3
#         norms = np.linalg.norm(Z3_mesh, axis=1) # array of all the norms
#         subset = (norms < self.Lambda_Z3)
#         subset_reg = (norms < lam) # subset for the regularized expression
#         self.Z3_arr = self.Z3_arr[subset,:]
#         self.Z3_arr_reg = self.Z3_arr[subset_reg,:]
#         self.m2_arr = norms[subset] # array of |\vec{m}|^2
#         self.m2_arr_reg = norms[subset_reg] # array of |\vec{m}|^2
#         self.unique_m2, self.m2_multiplicities = np.unique(self.m2_arr, return_counts=True) # unique |m|^2 and multiplicities
#     #---
#     def m2_info(self):
#         return {"m2": self.unique_m2, "nu": self.m2_multiplicities}
#     #---
# #---    

class Luscher_2Pions:
    """ Luscher's formalism for 2-pions state on the lattice """
    def __init__(self, MP: np.float64, L: int, N_lev: int, Z_00_obj: Z_00):
        self.MP = MP # pion mass (in lattice units)
        self.N_lev = N_lev # number of energy levels
        self.L = L  # volume
        self.Z_00_obj = Z_00_obj
        m2_info = self.Z_00_obj.m2_info() # |\vec{m}|^2 values and their multiplicities
        self.m2_vals, self.nu_m2 = m2_info["m2"], m2_info["nu"]
    #---
    def Z_00_bf(self, q2: np.float64):
        """ Z_00(q2) as in Appendix A of https://arxiv.org/pdf/1306.2532 """
        A = 1.0/np.sqrt(4.0*np.pi)
        denominators = (self.m2_vals - q2)
        nth_terms = self.nu_m2 * (1.0/denominators)
        sum_Z3 = np.sum(nth_terms)
        res = A*sum_Z3
        return res
    #---
    def Z_00_reg(self, q2: np.float64):
        """ Z_00(q2) as in Appendix C of https://doi.org/10.1016/0550-3213(91)90366-6 """
        res = self.Z_00_obj.get_Z_00_reg(q2=q2)
        return res
    #---
    def tan_phi(self, q: np.float64):
        # Generate all integer 3-vectors with each component in [-Lambda_Z3, Lambda_Z3]
        num = -(np.pi**(3.0/2.0)) * q
        den = self.Z_00(q2=q**2)
        res = num / den
        return res
    #---
    def phi(self, q: np.float64):
        phi_val = np.arctan(self.tan_phi(q=q))
        phi_val += (int(q))*np.pi
        return phi_val
    #---
    def omega_n_residue_function(self, delta_11: Callable[[np.float64], np.float64], k: np.float64):
        """ The zeroes of this function give the energy levels of the PP states """
        z = k * self.L / (2.0 * np.pi)
        lhs = self.tan_phi(z)
        rhs = -np.tan(delta_11(k))
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
        der_phi = (self.phi(q_n+eps_q) - self.phi(q_n)) / eps_q
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
        MV2, omega2 = MV**2, omega**2
        num = MV2 - omega2 - self.h(MV) - (omega2 - MV2)*self.hprime(MV)/(2.0*MV) + self.h(omega)
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
    N_lev: int, Z_00_obj: Z_00, 
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



def Z00_indep(q2, cutoff=10):
    """
    Compute the Lüscher Zeta function Z_{00}(1, q^2) with a cutoff on |n|.

    Parameters:
    q2 : np.float64
        Argument of the function (q^2).
    cutoff : int
        Limit on the sum: |n_i| <= cutoff.

    Returns:
    float
        Approximate value of Z_{00}(1, q^2)
    """
    result = 0.0
    for nx in range(-cutoff, cutoff+1):
        for ny in range(-cutoff, cutoff+1):
            for nz in range(-cutoff, cutoff+1):
                n2 = nx**2 + ny**2 + nz**2
                if n2 == q2:
                    continue  # Avoid singularity
                result += 1.0 / (n2 - q2)
    
    return result / np.sqrt(4 * np.pi)

    
if __name__ == "__main__":
    import matplotlib.pyplot as plt    
    Z_00_obj = Z_00(Lambda_Z3=100, lambda_Z3_reg=10)
    MP_MeV = 320 # pion mass
    MV_MeV = 2.77*MP_MeV # rho mass
    g_VPP =  5.22 # g_{\rho\pi\pi}
    hbarc_MeV_fm = 197.3269631
    a_fm = 0.089
    a_MeV_inv = a_fm / hbarc_MeV_fm
    Nx = 24
    L_MeV_inv = a_fm * Nx
    aMP = a_MeV_inv*MP_MeV
    aMV = a_MeV_inv*MV_MeV
    print("aMP:", aMP)
    print("aMV:", aMV)
    # Print q_n and phi'(q_n) for the given points
    q2_values = np.linspace(0.001, 10.1, 100)
    luscher = Luscher_2Pions(MP=aMP, L=Nx, N_lev=1, Z_00_obj=Z_00_obj)
    y1 = np.array([luscher.Z_00_bf(q2=q2) for q2 in q2_values ])
    # y2 = np.array([luscher.Z_00_reg(q2=q2) for q2 in q2_values ])
    # y2 = np.pi * q2_values
    for i, y in enumerate(y1):
        print(q2_values[i], f"{y:15.4f}")
    # plt.plot(q2_values, y1, label='phi(q)', marker='o', markersize=3, linewidth=1.5)
    # plt.plot(q2_values, y2, label='$\\pi * q^2$')
    # plt.legend()
    # plt.show()
    # q_n_values = [0.6877, 1.2007, 1.6208, 1.8806, 2.0620, 2.3532, 2.6826, 2.8789]
    # print(f"{'n':>2} {'q_n':>8} {'phi_prime(q_n)':>15}")
    # for idx, q_n in enumerate(q_n_values, 1):
    #     eps_q = 1e-12
    #     # Use Luscher_2Pions for phi derivative
    #     luscher = Luscher_2Pions(MP=aMP, L=Nx, N_lev=1, Z3_obj=Z3_obj)
    #     phi_prime = (luscher.phi(q_n + eps_q) - luscher.phi(q_n)) / eps_q
    #     print(f"{idx:2d} {q_n:8.4f} {phi_prime:15.4f}")
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
    # plt.xlim([2, 22])
    # plt.ylim([1e-7, 0.0015])
    # plt.yscale("log")
    # plt.gca().set_aspect(5.0)
    # plt.tick_params(direction='in', which='both', top=True, right=True)
    # plt.legend()
    # plt.savefig("V_PP_GSmodel.pdf", dpi=600, bbox_inches='tight')
    # plt.show()
