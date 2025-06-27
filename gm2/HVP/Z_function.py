""" 
Implementation of the Luescher's Zeta function, 
as in eq. (7) of https://arxiv.org/pdf/1707.05817.

NOTE: For Z_00 in the Center of Mass frame: \vec{s}=(0,0,0) [see eq. (3)], l=m=0, gamma=1.


Additional references:

- https://doi.org/10.1016/0550-3213(91)90366-6 (analytic continuation of the Luscher's Zeta function)
- https://doi.org/10.1016/0550-3213(91)90584-K (tabulated values of phi(q))
- https://arxiv.org/abs/1202.2145, see appendix
- https://arxiv.org/pdf/1107.5023: eq. (5) and Eq.(2)+Tab.II for the tabulated energy levels
- https://arxiv.org/pdf/1011.5288. Tab.V for some tabulated values


- `python2` implementation: https://github.com/knippsch/LueschersZetaFunction
    - For `python3` calls, use  
        ```
        -np.arctan(np.pi**1.5*np.sqrt(q2) * 1./np.real(Z(q2, gamma=1., d=np.zeros(3, dtype=int), m_split=1., precision=1e-8)))/(np.pi*q2)
        ```
- Morningstar's implementation: https://github.com/cjmorningstar10/TwoHadronsInBox
    - For a `python` wrapper, see this repo as a template 
      (note that the actual repository needed is a fork of the above `TwoHadronsBox`):
      https://github.com/ebatz/pythib
    
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
    def __init__(self, Lambda_Z3: float, Lambda: float, N_gauss: int):
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
        self.N_gauss = N_gauss # number of Gauss-Legendre points for the integral
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
            res = np.exp(x)*(2.0*sx*D - 1)
        else:
            res = -np.exp(x)  - (np.sqrt(-np.pi*x))* scipy.special.erf(np.sqrt(-x))
        #---
        return res
    #---
    def find_Z_00_zeros(self, u2_min: float, u2_max: float, tol: float = 1e-14):
        """
        Find all zeros of Z_00(u2) in the range [u2_min, u2_max].
        Returns a list of u2 values where Z_00(u2) = 0.
        """
        # Find zeros between integer values in [u2_min, u2_max]
        zeros = []
        int_min = int(np.floor(u2_min))
        int_max = int(np.floor(u2_max))
        for n in range(int_min, int_max):
            u2_guess = int_min+n+0.5
            root = scipy.optimize.newton(self.get_Z_00, x0=u2_guess, tol=tol)
            zeros.append(root)
        #---
        self.zeros_Z_00 = np.array(zeros)
        self.zeros_Z_00 = zeros
    #---
    def get_Z_00(self, u2: float) -> float:
        # NOTE: for Z_00, Y_lm does not depend on the summed vector
        A1 = (1.0/np.sqrt(4.0*np.pi)) # self.curly_Y_lm(l=0,m=0,r=np.array([0,0,0]), theta=0.0, phi=0.0).real
        n2_minus_u2 = self.unique_n2 - u2
        nth_terms = np.exp(- self.Lambda*n2_minus_u2)/n2_minus_u2
        one = A1*np.sum(self.n2_multiplicities*nth_terms)
        #
        two = (np.pi/np.sqrt(self.Lambda))*self.F0(self.Lambda*u2)
        #
        A3 = (1.0/np.sqrt(self.Lambda))*(np.pi**(3.0/2.0))*A1
        def integrand_0_1(t: float):
            """ integrand for the interval [0, 1] """
            C =  t**(-3.0/2.0)
            # summing over n2 != 0
            sum_Z3 = np.sum(self.n2_multiplicities[1:] * np.exp(self.Lambda * t * u2 - (np.pi**2)*self.unique_n2[1:]/(t*self.Lambda)))
            return C * sum_Z3
        #---
        # Gauss-Legendre quadrature points and weights for [0,1]
        x, w = np.polynomial.legendre.leggauss(self.N_gauss)
        # Transform from [-1,1] to [0,1]
        t = (1.0/2.0) * (x + 1)
        wt = (1.0/2.0) * w
        I3 = np.sum([wt_i * integrand_0_1(t_i) for t_i, wt_i in zip(t, wt)])
        three = A3*I3
        #  returning the sum of the three terms
        res = one+two+three
        return res
    #--- 
    def get_Z_00_brute_force(self, u2: float):
        """ 
        This is the brute-force calculation of Z_00.
        It should be used only to show that it actually does not converge,
        and we need the get_Z_00() method implemented in this class 
        """
        denominators = (self.unique_n2 - u2)
        res = np.sum(self.n2_multiplicities / denominators)
        return res
    #---
    def tan_phi(self, q: float) -> float:
        num = -np.pi**(3.0/2.0) * q
        den = Z_00_obj.get_Z_00(u2=q**2)
        return num/den
    #---
    def phi(self, q: float) -> float:
        res = np.arctan(self.tan_phi(q=q))
        nu = np.sum(np.array(Z_00_obj.zeros_Z_00) < q**2)
        res += np.pi * nu
        return res
#---


if __name__ == "__main__":
    import matplotlib.pyplot as plt    
    N_gauss = 100  # number of Gauss-Legendre points
    Lambda = 1.0
    Lambda_Z3 = 5 # cutoff for |n| in Z_00

    Z_00_obj = Z_00_Calculator(Lambda_Z3=Lambda_Z3, Lambda=Lambda, N_gauss=N_gauss)
    Z_00_obj.find_Z_00_zeros(u2_min=0.0, u2_max=10.0)

    q2_vals = np.arange(0.1, 9.0, 0.1)
    print(f"{'q^2':>10} {'phi(q)/pi/q^2':>20}")
    for q2 in q2_vals:
        q = np.sqrt(q2)
        phi_val = Z_00_obj.phi(q)
        print(f"{q2:10.4f} {phi_val/np.pi/q2:20.8f}")
