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
#---
TimeModulator_dict = {"SD": TimeModulator.SD, "W": TimeModulator.W, "LD": TimeModulator.LD}

class a_mu:
    @staticmethod
    def with_precomputed_K(
        window: Literal["SD", "W", "LD"], 
        a_fm: np.float128, ti_lat: np.ndarray, 
        Vi: np.ndarray, K: np.ndarray, Z_ren: float, 
        strategy: str
        ):
        ti_fm = a_fm*ti_lat # times in [fm]
        Theta = TimeModulator_dict[window](ti_fm=ti_fm)
        K_modulated = K*Theta # time modulation of the kernel
        res = amu.get_amu_precomp_K(ti=ti_lat, Vi=Vi, K=K_modulated, Z_ren=Z_ren, strategy=strategy)
        return res
    #---
#---

