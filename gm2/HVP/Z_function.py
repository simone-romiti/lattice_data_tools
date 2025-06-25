""" 
Implementation of the Luescher's Zeta function, 
as in eq. (7) of https://arxiv.org/pdf/1707.05817

NOTE: For Z_00 in the Center of Mass frame: \vec{s}=(0,0,0), l=m=0, gamma=1.


Additional references:
- https://doi.org/10.1016/0550-3213(91)90366-6 (analytic continuation of the Luscher's Zeta function)
- https://doi.org/10.1016/0550-3213(91)90584-K (tabulated values of phi(q))

"""

import numpy as np
import scipy.special
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
    def __init__(self, Lambda_Z3: float, Lambda: float):
        self.Lambda = Lambda
        self.Lambda_Z3 = Lambda_Z3 # maximum values of |\vec{n}|
        grid = np.arange(-self.Lambda_Z3, self.Lambda_Z3 + 1) # numbers from -Lambda_Z3 to Lambda_Z3 (included)
        Z3_mesh = np.stack(np.meshgrid(grid, grid, grid, indexing='ij'), -1).reshape(-1, 3)        
        norms = np.linalg.norm(Z3_mesh, axis=1) # array of all the norms
        n2_all = norms**2 # array of all the norms squared
        # Select only vectors with norm less than Lambda_Z3
        subset = (norms < self.Lambda_Z3)
        self.Z3_arr = Z3_mesh[subset,:]
        self.m2_arr = n2_all[subset] # array of |\vec{m}|^2
        unique_n2, n2_multiplicities = np.unique(self.m2_arr, return_counts=True) # unique |m|^2 and multiplicities
        idx_sort = np.argsort(unique_n2)
        self.unique_n2 = unique_n2[idx_sort] # unique |m|^2 sorted
        self.n2_multiplicities = n2_multiplicities[idx_sort]
    #---
    def curly_Y_lm(self, l: int, m:int, r: np.float64, theta: np.float64, phi: np.float64):
        """ \mathcal{Y}_{lm}(r, \theta, \phi) as in eq. 3.14 of https://doi.org/10.1016/0550-3213(91)90366-6 """
        res = (r**l) * scipy.special.sph_harm_y(m, l, phi, theta)
        return res
    #---
    def F0(self, x: float) -> float:
        """ F_0(x) as in eq. (11) of https://arxiv.org/pdf/1707.05817 """
        if x >= 0:
            sx = np.sqrt(x)
            D = scipy.special.dawsn(sx)
            return np.exp(x)*(2.0*sx*D - 1)
        else:
            return -np.exp(x)  - (np.sqrt(-np.pi*x))* scipy.special.erf(np.sqrt(-x))

    def get_Z_00(self, u2: float, N_gauss: int) -> float:
        # NOTE: for Z_00, Y_lm does not depend on the summed vector
        A1 = self.curly_Y_lm(l=0,m=0,r=np.array([0,0,0]), theta=0.0, phi=0.0).real
        n2_minus_u2 = self.unique_n2 - u2
        nth_terms = np.exp(-self.Lambda*n2_minus_u2)/n2_minus_u2
        one = A1*np.sum(self.n2_multiplicities*nth_terms)
        #
        two = (np.pi/np.sqrt(self.Lambda))*self.F0(self.Lambda*u2)
        #
        A3 = (1.0/np.sqrt(self.Lambda))*(np.pi**(3.0/2.0))*A1
        def integrand_0_1(t: float):
            """ integrand for the interval [0, 1] """
            C =  t**(-3.0/2.0) * np.exp(self.Lambda * t * u2)
            # summing over n2 != 0
            sum_Z3 = np.sum(self.n2_multiplicities[1:] * np.exp(- (np.pi**2)*self.unique_n2[1:]/(t*self.Lambda)))
            return C * sum_Z3
        #---
        def integrand_m1_1(tprime: float):
            """ integrand for the interval [-1, 1], obtained by mapping t --> t' = (2*t-1) """
            J = 1.0/2.0 # Jacobian of the transformation t' = (2*t-1)
            return J*integrand_0_1(t=(tprime+1)/2.0) 
        # Use Gauss-Legendre quadrature with 50 points
        nodes, weights = np.polynomial.legendre.leggauss(N_gauss)
        # Transform nodes from [-1, 1] to [0, 1]
        tot_weights = np.array([weights * integrand_m1_1(tprime_i) for tprime_i in nodes])
        I3 = np.sum(tot_weights)
        three = A3*I3
        res = one+two+three
        return res
    #---
#---



if __name__ == "__main__":
    import matplotlib.pyplot as plt    
    u2 = 0.5  # example value for u^2
    N_gauss = 50  # number of Gauss-Legendre points

    Lambda_values = np.linspace(0.1, 10.0, 30)
    Lambda_Z3_values = np.arange(2, 20, 1)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Vary Lambda, fix Lambda_Z3
    Lambda_Z3_fixed = 6
    Z00_vs_Lambda = []
    for Lambda in Lambda_values:
        calc = Z_00_Calculator(Lambda_Z3=Lambda_Z3_fixed, Lambda=Lambda)
        Z00_vs_Lambda.append(calc.get_Z_00(u2, N_gauss))
    axs[0].plot(Lambda_values, Z00_vs_Lambda, marker='o')
    axs[0].set_xlabel(r'$\Lambda$')
    axs[0].set_ylabel(r'$Z_{00}$')
    axs[0].set_title(r'$Z_{00}$ vs $\Lambda$ (fixed $\Lambda_{Z3}=%d$)' % Lambda_Z3_fixed)

    # Vary Lambda_Z3, fix Lambda
    Lambda_fixed = 1.0
    Z00_vs_LambdaZ3 = []
    for Lambda_Z3 in Lambda_Z3_values:
        calc = Z_00_Calculator(Lambda_Z3=Lambda_Z3, Lambda=Lambda_fixed)
        Z00_vs_LambdaZ3.append(calc.get_Z_00(u2, N_gauss))
    axs[1].plot(Lambda_Z3_values, Z00_vs_LambdaZ3, marker='o')
    axs[1].set_xlabel(r'$\Lambda_{Z3}$')
    axs[1].set_ylabel(r'$Z_{00}$')
    axs[1].set_title(r'$Z_{00}$ vs $\Lambda_{Z3}$ (fixed $\Lambda=%.1f$)' % Lambda_fixed)

    plt.tight_layout()
    plt.show()