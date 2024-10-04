""" 
Tuning the masses of the light strange and charm 

NOTE: These routines are meant to run on a single sample (bootstrap, jackknife, etc.)
"""

import numpy as np

from lattice_data_tools.fit.interpolate import xyey as interpolate_xyey

def ml_phys_from_Mpi(
        ml: np.ndarray, 
        Mpi: np.ndarray, dMpi: np.ndarray,
        Mpi_phys: np.float64,
        ansatz, guess: np.ndarray):
    """ Interpolating to m_l^{phys} from the values of M_\pi """
    ml_phys = interpolate_xyey(ansatz=ansatz, guess=guess, y0=Mpi_phys, x=ml, y=Mpi, ey=dMpi)
    return ml_phys
#---

def ms_phys_from_MK(
        ml: np.ndarray, ms: np.ndarray, 
        ml_phys: np.float64, 
        MK: np.ndarray, dMK: np.ndarray,
        MK_phys: np.float64):
    """ 
    Interpolating to m_s^{phys} from the values of M_K 
    
    MK and dMK are arrays of dimension (N_ml, N_ms)
    """
    pass
#---

def mc_phys_from_MDs(
          ml: np.ndarray, ms: np.ndarray, mc: np.ndarray, 
          ms_phys: np.float64, mc_phys: np.float64, 
          MDs: np.ndarray, dMDs: np.ndarray,
          MDs_phys: np.float64):
    """ 
    Interpolating to m_c^{phys} from the values of M_Ds 
    
    MD and dMDs are arrays of dimension (N_ml, N_ms, N_mc)
    """
    pass
#---


def m_lsc_phys_from_Mpi_MK_MDs(
        ml: np.ndarray, ms: np.ndarray, mc: np.ndarray,
        Mpi: np.ndarray, dMpi: np.ndarray, Mpi_phys: np.float64, 
        MK: np.ndarray, dMK: np.ndarray, MK_phys: np.float64,
        MDs: np.ndarray, dMDs: np.ndarray, MDs_phys: np.float64):
    """ Interpolating to the physical point of m_l, m_s, m_c """
    pass
#---