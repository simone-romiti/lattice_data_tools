""" 
Calculation of the Luscher's Zeta function Z_00(1, q2)
and of phi(q) = arctan(tan_phi(q))

References: 
- https://doi.org/10.1016/0550-3213(91)90366-6 (analytic contninuation of the Luscher's Zeta function)
- https://doi.org/10.1016/0550-3213(91)90584-K (tabulated values of phi(q))
"""

import math
import numpy as np
from typing import Callable
from scipy.optimize import brentq
import scipy

class Z_00_Calculator:
    """ 
    Z_00(q2) as in eq. C8 of https://doi.org/10.1016/0550-3213(91)90366-6,
    with a cutoff on the norm of the 3-vector \vec{n}.    
    
    NOTE: 
    There are 2 cutoffs, a small one for the regularization of K_00, 
    and a large one that is the unavoidable numerical truncation of the sum over \vec{n} in the Z_00(q2) expression.

    Additional reference: https://arxiv.org/pdf/hep-lat/0404001, appendix A
    """
    def __init__(self, lambda_Z3_small: float, Lambda_Z3: float):
        self.Lambda_Z3 = Lambda_Z3 # maximum values of |\vec{n}|
        self.lambda_Z3_small = lambda_Z3_small # maximum values of |\vec{n}| for the regularized expression
        grid = np.arange(-self.Lambda_Z3, self.Lambda_Z3 + 1) # numbers from -Lambda_Z3 to Lambda_Z3 (included)
        Z3_mesh = np.stack(np.meshgrid(grid, grid, grid, indexing='ij'), -1).reshape(-1, 3)        
        norms = np.linalg.norm(Z3_mesh, axis=1) # array of all the norms
        n2_all = norms**2 # array of all the norms squared
        # Select only vectors with norm less than Lambda_Z3
        subset = (norms < self.Lambda_Z3)
        self.Z3_arr = Z3_mesh[subset,:]
        self.m2_arr = n2_all[subset] # array of |\vec{m}|^2
        self.unique_n2, self.n2_multiplicities = np.unique(self.m2_arr, return_counts=True) # unique |m|^2 and multiplicities
        # vectors with \lambda <= |\vec{n}| < \Lambda_Z3_small
        subset_reg = (norms >= self.lambda_Z3_small) & (norms < self.Lambda_Z3) 
        self.Z3_arr_reg = Z3_mesh[subset_reg,:]
        self.m2_arr_reg = n2_all[subset_reg] # array of |\vec{m}|^2
        self.unique_n2_reg, self.n2_multiplicities_reg = np.unique(self.m2_arr_reg, return_counts=True) # unique |m|^2 and multiplicities
        # vectors with |\vec{n}| < \lambda
        subset_small = (norms < self.lambda_Z3_small)
        self.Z3_arr_small = Z3_mesh[subset_small,:]
        self.m2_arr_small = n2_all[subset_small] # array of |\vec{m}|^2
        self.unique_n2_small, self.n2_multiplicities_small = np.unique(self.m2_arr_small, return_counts=True) # unique |m|^2 and multiplicities
    #---
    def curly_Y_lm(self, l: int, m:int, r: np.float64, theta: np.float64, phi: np.float64):
        """ \mathcal{Y}_{lm}(r, \theta, \phi) as in eq. 3.14 of https://doi.org/10.1016/0550-3213(91)90366-6 """
        res = (r**l) * scipy.special.sph_harm_y(m, l, phi, theta)
        return res
    #---
    # def exp_tq2_K_00_r0_large_t_Lambda(self, t: np.float64, q2: np.float64):
    #     """ 
    #     Kernel as in eq. C1 of https://doi.org/10.1016/0550-3213(91)90366-6,
    #     for \vec{r}=0
    #     NOTE: this is the expression that should be used at small t (t < 1)
    #     """
    #     tn2 = t*self.unique_n2
    #     nth_terms = np.exp(-tn2 + t*q2)
    #     res = np.sum(self.n2_multiplicities*nth_terms)/((2.0*np.pi)**3.0)
    #     return res
    # #---
    def exp_tq2_K_00_r0_large_t_lambda(self, t: np.float64, q2: np.float64):
        """ Same as self.K_00_r0_small_t_Lambda, but truncated at the small cutoff "self.lambda_Z3_small"  """
        tn2 = t*self.unique_n2_small
        nth_terms = np.exp(-tn2 + t*q2)
        res = np.sum(self.n2_multiplicities_small*nth_terms)/((2.0*np.pi)**3.0)
        return res
    #---
    def exp_tq2_K_00_r0_small_t(self, t: np.float64, q2: np.float64):
        """
        Kernel as in eq. C2 of https://doi.org/10.1016/0550-3213(91)90366-6
        for \vec{r}=0        

        NOTE: this is the expression that should be used at large t (t > 1)
        """
        args_exp = t*q2 - (np.pi**2) * self.unique_n2 / t 
        exp_terms = self.n2_multiplicities * np.exp(args_exp)
        A = (4.0 * np.pi* t)**(-3.0/2.0)
        res = A * np.sum(exp_terms) 
        return res
    #---
    def exp_tq2_K_00_r0_reg(self, t: np.float64, q2: np.float64):
        """ 
        Kernel as in eq. C3 of https://doi.org/10.1016/0550-3213(91)90366-6 
        for \vec{r}=0
        """
        if t <= 1.0:
            bare = self.exp_tq2_K_00_r0_small_t(t=t, q2=q2)
            res = bare - self.exp_tq2_K_00_r0_large_t_lambda(t=t, q2=q2)
        else:
            tn2 = t*self.unique_n2_reg
            nth_terms = np.exp(-tn2 + t*q2)
            res = np.sum(self.n2_multiplicities_reg*nth_terms)/((2.0*np.pi)**3.0)
        #--- 
        return res
    #---
    def integrand_Z_00(self, t: np.float64, q2: np.float64):
        """ Integrand of the 2nd term contributing to Z_00(1, q2)"""
        one = self.exp_tq2_K_00_r0_reg(t=t, q2=q2)
        two = -1.0/(((4.0*np.pi)**2)*(t**(3.0/2.0)))
        res = one + two
        return res
    #---
    def Z_00_fast_convergence(self, q2: np.float64):
        """ Z_00(q2) as in Appendix A of https://arxiv.org/pdf/1306.2532 """
        A = self.curly_Y_lm(l=0, m=0, r=1.0, theta=0.0, phi=0.0).real  # Y_00(1, 0, 0) = 1/sqrt(4*pi)
        denominators = (self.unique_n2_small - q2)
        # print(f"Denominators: {denominators}")  # Debugging line
        nth_terms = self.n2_multiplicities_small * (1.0/denominators)
        sum_Z3 = A*np.sum(nth_terms)
        def integrand(t: np.float64):
            """ Integrand of the 2nd term contributing to Z_00(1, q2) """
            res = self.integrand_Z_00(t=t, q2=q2)
            return res
        #---
        integral = scipy.integrate.quad(integrand, 0.0, np.inf)
        res = sum_Z3 + ((2.0*np.pi)**3.0)*integral[0]
        return res
    #---
    def Z_00_brute_force(self, q2: np.float64):
        """ Z_00(q2) as in Appendix A of https://arxiv.org/pdf/1306.2532 """
        A = 1.0/np.sqrt(4.0*np.pi)
        denominators = (self.unique_n2 - q2)
        nth_terms = self.n2_multiplicities * (1.0/denominators)
        sum_Z3 = np.sum(nth_terms)
        res = A*sum_Z3
        return res
    #---

def tan_phi(q: np.float64, Z_00_obj: Z_00_Calculator):
    """ \tan{\phi(q)} as in eq. A.3 of https://doi.org/10.1016/0550-3213(91)90366-6"""
    q2 = q**2
    num = (- np.pi**(3.0/2.0)) * q
    den = Z_00_obj.Z_00_fast_convergence(q2=q2) 
    res = (num/den)
    return res
#---

def phi(q: np.float64, Z_00_obj: Z_00_Calculator):
    q2 = q**2
    res = math.floor(q2)*np.pi + np.arctan( (- np.pi**(3.0/2.0)) * q / Z_00_obj.Z_00_fast_convergence(q2=q2))
    return res

if __name__ == "__main__":
    import matplotlib.pyplot as plt    
    Z_00_obj = Z_00_Calculator(Lambda_Z3=100, lambda_Z3_small=5)
    n_values = [1, 2, 3, 4, 5, 6, 7, 8]
    q_values = [0.6877, 1.2007, 1.6208, 1.8806, 2.0620, 2.3532, 2.6826, 2.8789]
    eps = 1e-2  # small perturbation for numerical derivative
    print(f"{'n':>2} {'q':>8} {'phi(q)':>20}")
    for n, q in zip(n_values, q_values):
        phi_prime = (phi(q=q+eps, Z_00_obj=Z_00_obj)-phi(q=q, Z_00_obj=Z_00_obj)) / eps
        print(f"{n:2d} {q:8.4f} {phi_prime:20.4f}")
    # Lambda_values = [60]
    # q2_values = np.arange(0.0, 10.0, 0.01)
    # results = []

    # for Lambda in Lambda_values:
    #     Z_00_obj = Z_00_Calculator(Lambda_Z3=Lambda, lambda_Z3_small=50)
    #     Z_00_vals = []
    #     for q2 in q2_values:
    #         q = np.sqrt(q2)
    #         value = math.floor(q2)*np.pi + np.arctan( (- np.pi**(3.0/2.0)) * q / Z_00_obj.Z_00_fast_convergence(q2=q2))
    #         Z_00_vals.append(value)
    #     results.append(Z_00_vals)

    # import matplotlib.pyplot as plt
    # for i, Lambda in enumerate(Lambda_values):
    #     plt.plot(q2_values, results[i], label=f"Lambda_Z3={Lambda}")
    # plt.xlabel("$q^2$")
    # plt.ylabel(r"$\phi(q)$")
    # plt.title(r"$\phi(q)$ vs $q^2$ for different $\Lambda_{Z3}$")
    # plt.legend()
    # plt.show()