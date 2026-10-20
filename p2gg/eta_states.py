"""
Routines for the contructions and improvement of 2-point functions for the $\\eta$ states,
namely $\\eta_8$,$\\eta_0$ (for the $SU(3)_f$ symmetric limit),
and for the $\\eta,\\eta'$ once the mixing has been computed (e.g. with the GEVP)
"""

import numpy as np
import typing

def topological_improvement(C: np.ndarray, deg: typing.Literal[1,2]):
    """
    Applying either:
    - Eq. (19) of https://arxiv.org/pdf/1710.07986
    - Eq. (30) of https://arxiv.org/pdf/2503.09895

    C : np.array of shape (...,T_ext)
    deg: degree of derivative (only 1 and 2 supported for now)
    
    NOTEs:
    - the topological contribution it removes is present only for the disconnected.
    - the **full** correlator should be 'improved' using this function. Otherwise the relative amplitudes of the connected and disconnected are inconsistent with each other.
    
    """
    T_ext = C.shape[-1] # number of times
    T_new = T_ext-deg
    C_tilde = np.zeros(shape=C.shape[:-1]+(T_new,))
    for t in range(T_new):
        if deg==1:
            C_tilde[...,t] = C[...,t] - C[...,t+1] # 1st derivative
        else:
            C_tilde[...,t] = C[...,t] - 2.0*C[...,t+1] + C[...,t+2] # 2nd derivative
    #-------
    return C_tilde


class FlavorEigenstates:
    """
    Construction of the 2-point correlation functions (connected+disconnected diagrams) for the $SU(3)_f$ flavor eigenstates $\\eta_8,\\eta_0$ (mass eigenstates in the limit ${m_u=m_d=m_s}$).

    NOTE: We assume the mass degeneracy of the 2 components of each doublet in the lattice QCD simulation. By symmetry arguments, the only Wick contractions that need to be computed are the ones of the `up` components of each quark doublet. Here we call them `u` and `s` (which is the `$s_+$ component of the doublet).

    The correlators are built from the following Wick contractions (the projection at $\\vec{p}=\\vec{0}$ by $\\sum_{\\vec{x}}$ is understood):

    $$
    T_f = \\braket{ P_f(x) P^\\dagger_f(0)}^{\\rm conn.} = \\Gamma_{ab}S_{bc}(x,0)\\Gamma_{cd}S_{da}(0,x)
    $$

    $$
    B_{fg} = L_f(x)L_g(0) = - \braket{P_f(x) P^\\dagger_f(y)}^{\text{disc.}}
    $$

    where $L_f$ is the loop:
    $$
    L_f = - \\braket{  \\bar\\psi_a\\Gamma_{ab}\\psi_b } = -\\Gamma_{ab}S_{ba}(x,x).
    $$
    
    """
    @staticmethod
    def build_eta8(T_u, T_s, B_uu, B_us, B_su, B_ss):
        """
        The interpolator for the $\\eta_8$ is:
        $$\\eta_8 = (\\bar{u} \\gamma_5 u + \\bar{d} \\gamma_5 d - 2 \\bar{s} \\gamma_5 s )/\\sqrt{6}$$
        """
        C_eta8_conn = (1/6)*2*(T_u + 2*T_s)
        C_eta8_disc = -(1/3)*(B_uu - np.sqrt(2)*B_us - np.sqrt(2)*B_su + 2*B_ss) # (-1) sign coming from $P^\\dagger=-P$
        C_eta8 = (C_eta8_conn + C_eta8_disc)
        return C_eta8
    #---
    @staticmethod
    def build_eta0(T_u, T_s, B_uu, B_us, B_su, B_ss):
        """
        The interpolator for the $\\eta_0$ is:
        $$\\eta_8 = (\\bar{u} \\gamma_5 u + \\bar{d} \\gamma_5 d + \\bar{s} \\gamma_5 s )/\\sqrt{3}$$
        """
        C_eta8_conn = (1/3)*(2.0*T_u + T_s)
        C_eta8_disc = -(1/3)*(2*B_uu - np.sqrt(2)*B_us - np.sqrt(2)*B_su + B_ss) # (-1) sign coming from $P^\\dagger=-P$
        C_eta8 = (C_eta8_conn + C_eta8_disc)
        return C_eta8
