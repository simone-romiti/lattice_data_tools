""" 
Gounaris-Sakurai (GS) model of Finite Volume Effects (FVEs) 

This file contains an implementation of the GS model as used in https://inspirehep.net/literature/2103903

!!! ACHTUNG !!! 
The original paper has a typo in the definition of the function h'(omega) (eq. 20 of https://arxiv.org/pdf/1808.00887).
A factor of "k" is missing in the equation, but the implementation below includes it.

"""


import numpy as np

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
    #---
#---

# physical values taken from https://arxiv.org/pdf/2206.15084
physical_parameters = {"M_rho_MeV": 775, "g_rho_pi_pi": 5.95}


if __name__ == "__main__":
    import matplotlib.pyplot as plt    
    MP_GeV = 0.135 # pion mass
    MV_GeV = 0.775 # rho mass
    g_VPP =  5.5 # g_{\rho\pi\pi}
    GS_mod = GS_model(MP=MP_GeV, MV=MV_GeV, g_VPP=g_VPP)
    omega_vals = np.linspace(0.4, 1.2, 200)
    k_vals = np.array([get_k(omega=omega, MP=MP_GeV) for omega in omega_vals])
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
    plt.plot(omega_vals, delta_vals_deg, label='$\\delta_{11}$ from GS model')
    plt.plot(omega_vals, F2_vals)
    plt.legend()
    plt.grid(True)
    plt.title('$\\delta_{11}$ and $F_P^2$ from GS model, as in Fig. 3 of https://arxiv.org/pdf/1808.00887')
    plt.show()
