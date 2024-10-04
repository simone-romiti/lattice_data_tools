""" 
Tuning the masses of the light strange and charm 

The routines determine the bootstrap samples
"""

import numpy as np

from lattice_data_tools.fit.xyey import fit_xyey
from lattice_data_tools.fit.interpolate import x0_from_xyey as x0_from_xyey

def bts_ml_phys_from_Mpi(
        ml: np.ndarray, 
        Mpi: np.ndarray,
        Mpi_phys: np.float64,
        ansatz, guess: np.ndarray,
        maxiter = 10000, method = "BFGS"):
    """ 
    Interpolating to m_l^{phys} from the values of M_\pi 
    
    NOTE: The 1st dimension of M_pi and dM_pi is the bootstrap index
    """
    N_bts = Mpi.shape[0] ## number of bootstraps
    dMpi = np.std(Mpi, axis=0)
    ml_phys = np.array([x0_from_xyey(
        ansatz=ansatz, guess=guess, 
        y0=Mpi_phys, x=ml, y=Mpi[i,:], ey=dMpi, 
        maxiter=maxiter, method=method)["x0"] for i in range(N_bts)])
    return ml_phys
#---

def bts_ms_phys_from_MK(
        ml: np.ndarray, ms: np.ndarray, ml_phys: np.ndarray, 
        MK: np.ndarray, MK_phys: np.float64,
        ansatz_ml, guess_ml: np.ndarray,
        ansatz_ms, guess_ms: np.ndarray,
        maxiter = 10000, method = "BFGS"):
    """ 
    Interpolating to m_s^{phys} from the values of M_K 
    
    MK is an array of dimension (N_bts, N_ml, N_ms)
    ml_phys is the array of N_bts samples found previously (e.g. with M_\pi)
    """
    N_bts = MK.shape[0] # number of bootstraps
    N_ms = MK.shape[2] # number of strange quark masses
    dMK = np.std(MK, axis=0)
    MK_ml_phys = np.zeros(shape=(N_bts, N_ms))
    ## interpolating to m_l^{phys} at fixed m_s
    for i in range(N_bts):
        for k in range(N_ms):
            mini = fit_xyey(
                ansatz=ansatz_ml, 
                x=ml, y=MK[i,:,k], ey=dMK[:,k], 
                guess=guess_ml, 
                maxiter = maxiter, method = method)
            par_fit = mini["par"]
            MK_ml_phys[i,k] = ansatz_ml(ml_phys, par_fit)
    #-------
    dMK_ml_phys = np.std(MK_ml_phys, axis=0)
    ms_phys = np.array([x0_from_xyey(
            ansatz=ansatz_ms, guess=guess_ms, 
            y0=MK_phys, x=ml, y=MK_ml_phys[i,:], ey=dMK_ml_phys, 
            maxiter=maxiter, method=method) for i in range(N_bts)])
    return ms_phys
#---

def bts_mc_phys_from_MDs(
        ms: np.ndarray, mc: np.ndarray, ms_phys: np.ndarray, 
        MDs: np.ndarray, MDs_phys: np.float64,
        ansatz_ms, guess_ms: np.ndarray,
        ansatz_mc, guess_mc: np.ndarray,
        maxiter = 10000, method = "BFGS"):
    """ 
    Interpolating to m_s^{phys} from the values of M_K 
    
    MDs is an array of dimension (N_bts, N_ml, N_ms)
    ml_phys is the array of N_bts samples found previously (e.g. with M_\pi)
    """
    N_bts = MDs.shape[0] # number of bootstraps
    N_mc = MDs.shape[2] # number of strange quark masses
    dMDs = np.std(MDs, axis=0)
    MDs_ms_phys = np.zeros(shape=(N_bts, N_mc))
    ## interpolating to m_l^{phys} at fixed m_s
    for i in range(N_bts):
        for k in range(N_mc):
            mini = fit_xyey(
                ansatz=ansatz_ms, 
                x=ms, y=MDs[i,:,k], ey=dMDs[:,k], 
                guess=guess_ms, 
                maxiter = maxiter, method = method)
            par_fit = mini["par"]
            MDs_ms_phys[i,k] = ansatz_ms(ms_phys, par_fit)
    #-------
    dMDs_ms_phys = np.std(MDs_ms_phys, axis=0)
    mc_phys = np.array([x0_from_xyey(
            ansatz=ansatz_mc, guess=guess_mc, 
            y0=MDs_phys, x=ms, y=MDs_ms_phys[i,:], ey=dMDs_ms_phys, 
            maxiter=maxiter, method=method) for i in range(N_bts)])
    return mc_phys
#---

def bts_m_lsc_phys_from_Mpi_MK_MDs(
        ml: np.ndarray, ms: np.ndarray, mc: np.ndarray,
        Mpi: np.ndarray, Mpi_phys: np.float64, 
        MK: np.ndarray, MK_phys: np.float64,
        MDs: np.ndarray, MDs_phys: np.float64,
        ansatz_dict: dict, guess_dict: dict,
        maxiter = 10000, method = "BFGS"):
    """ Interpolating to the physical point of m_l, m_s, m_c """
    ml_phys = bts_ml_phys_from_Mpi(
        ml=ml, Mpi=Mpi, Mpi_phys=Mpi_phys, 
        ansatz=ansatz_dict["Mpi"]["ml"], guess=guess_dict["Mpi"]["ml"], 
        maxiter=maxiter, method=method)
    ms_phys = bts_ms_phys_from_MK(
        ms=ms, ml_phys=ml_phys, 
        MK=MK, MK_phys=MK_phys,
        ansatz_ml=ansatz_dict["MK"]["ml"], guess_ml=guess_dict["MK"]["ml"],
        ansatz_ms=ansatz_dict["MK"]["ms"], guess_ms=guess_dict["MK"]["ms"],
        maxiter = maxiter, method = method)
    mc_phys = bts_mc_phys_from_MDs(
        mc=mc, ms_phys=ms_phys, 
        MDs=MDs, MDs_phys=MDs_phys,
        ansatz_ms=ansatz_dict["MDs"]["ms"], guess_ms=guess_dict["MDs"]["ms"],
        ansatz_mc=ansatz_dict["MDs"]["mc"], guess_mc=guess_dict["MDs"]["mc"],
        maxiter = maxiter, method = method)
    res = {"l": ml_phys, "s": ms_phys, "c": mc_phys}
    return res
#---