""" 
Windows contribution to $a_\\mu$.

References:

- https://arxiv.org/abs/2206.15084 (eqs. 18, 19, 20, 21)
- https://arxiv.org/pdf/2411.08852 (eqs. 8, 9, 10, 11)
- https://arxiv.org/pdf/1801.07224 (RBC/UKQCD set the convention)

"""

import numpy as np
from typing import Literal

import lattice_data_tools.gm2.HVP.amu as amu

# windows parameters. See eq. 21 of https://arxiv.org/abs/2206.15084
t0_fm, t1_fm, Delta_fm = 0.4, 1.0, 0.15 

class TimeModulator:
    @staticmethod
    def SD(t_fm):
        """ Short distance """
        Theta_SD = 1.0 - 1.0 / (1.0 + np.exp(-2.0 * (t_fm - t0_fm) / Delta_fm))
        return Theta_SD
    #---
    @staticmethod
    def W(t_fm: np.float128):
        """ Window """
        Theta_W = 1.0 / (1.0 + np.exp(-2 * (t_fm - t0_fm) / Delta_fm)) - 1.0 / (1.0 + np.exp(-2 * (t_fm - t1_fm) / Delta_fm))
        return Theta_W
    #---
    @staticmethod
    def LD(t_fm: np.float128):
        """ Long Distance """
        Theta_LD = 1.0 / (1.0 + np.exp(-2 * (t_fm - t1_fm) / Delta_fm))
        return Theta_LD
    #---
    @staticmethod
    def full(t_fm: np.float128):
        """ full contrbution, no modulation. It is equivalent to the sum of the 3 windows: SD, W, LD """
        return 1.0
    #---
#---
TimeModulator_dict = {"SD": TimeModulator.SD, "W": TimeModulator.W, "LD": TimeModulator.LD, "full": TimeModulator.full}

class a_mu:
    @staticmethod
    def with_precomputed_K(
        window: Literal["SD", "W", "LD", "full"], 
        a_fm: np.float128, t_lat: np.ndarray, 
        Vi: np.ndarray, K: np.ndarray, Z_ren: float, 
        strategy: str
        ):
        t_fm = a_fm*t_lat # times in [fm]
        Theta = TimeModulator_dict[window](t_fm=t_fm)
        K_modulated = K*Theta # time modulation of the kernel
        res = amu.get_amu_precomp_K(ti=t_lat, Vi=Vi, K=K_modulated, Z_ren=Z_ren, strategy=strategy)
        return res
    #---
#---

