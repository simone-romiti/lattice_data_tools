print("""
      -------------------------------------------
      Continuum limit: different fit combinations
      -------------------------------------------
      """
      )

import numpy as np
from typing import Literal
import os
import scipy.optimize as opt
import itertools
import matplotlib.pyplot as plt

from lattice_data_tools.bootstrap import BootstrapSamples
from lattice_data_tools.io import with_dill, with_yaml
from lattice_data_tools.dictionaries import NestedDict
from lattice_data_tools.fit.xiexiyey import fit_xiexiyey

ens_info = with_yaml.load("ensembles.yaml")

aux_dir = ens_info["data"]["auxiliary"]
N_bts = ens_info["N_bts"]
fermion_types = ens_info["fermion_types"]
windows_keys = [f"{window}_window" for window in ens_info["windows"]]

a_fm_dict = with_dill.load(f'{aux_dir}/a_fm_Lref.pkl') # lattice spacing bootstraps
a_mu = with_dill.load(f'{aux_dir}/a_mu-pp_UV_corrected.pkl')

e_short = a_mu["ens_list"]
n_ens_tot = len(e_short)
a_fm_bts = BootstrapSamples([a_fm_dict[e] for e in e_short]).T # bootstraps of lattice spacing

# ansaetze in a^2
# IMPORTANT: all of them should be defined such that p[0] is the continuum value
ansaetze = {
        "constant": {"n_par": 1, "lambda": lambda a, p: (0.0*a) + p[0]}, # 0.0*a just gives the right shape if passing an array
        "linear": {"n_par": 2, "lambda": lambda a, p: p[0] + p[1] * (a**2)},
        "quadratic": {"n_par": 3, "lambda": lambda a, p: p[0] + p[1] * (a**2) + p[2] * (a**2)**2},
        "a^2log(a)": {"n_par": 3, "lambda": lambda a, p: p[0] + p[1] * (a**2) + p[2] * (a**2) * np.log(a + (a==0).astype(int))}, # regularized log, returns 0 for a==0
    }
    
   
valid_ansaetze = list(ansaetze.keys())

def get_ch2_fun(
    a_fm: dict,
    a_mu: dict, err_a_mu: dict, 
    fit_type: Literal["combined", "individual"], 
    tm_ansatz: Literal[valid_ansaetze], OS_ansatz: Literal[valid_ansaetze],
    Cov_inv = None
    ):
    n_par_tm = ansaetze[tm_ansatz]["n_par"]
    n_par_OS = ansaetze[OS_ansatz]["n_par"]
    ansatz_tm = ansaetze[tm_ansatz]["lambda"]
    ansatz_OS = ansaetze[OS_ansatz]["lambda"]
    if Cov_inv is None:
        """ uncorrelated fit"""
        ch2_tm = lambda p: np.sum( ((a_mu["tm"] - ansatz_tm(a_fm["tm"], p))/err_a_mu["tm"])**2 )
        ch2_OS = lambda p: np.sum( ((a_mu["OS"] - ansatz_OS(a_fm["OS"], p))/err_a_mu["OS"])**2 )
        if fit_type == "individual":
            ch2 = lambda p: ch2_tm(p[0:n_par_tm]) + ch2_OS(p[n_par_tm:])
        elif fit_type == "combined":
            """ forcing the continuum limit value, p[0], to be the same for both """
            ch2 = lambda p: ch2_tm(np.array([p[0], *p[1:n_par_tm]])) + ch2_OS(np.array([p[0], *p[n_par_tm:]]))
        #---
    else:
        def ch2_corr(p_tm, p_OS):
            delta = np.concatenate((a_mu["tm"] - ansatz_tm(a_fm["tm"], p_tm), a_mu["OS"] - ansatz_OS(a_fm["OS"], p_OS)))
            res = delta @ Cov_inv @ delta
            return res
        #---
        if fit_type == "individual":
            ch2 = lambda p: ch2_corr(p_tm=p[0:n_par_tm], p_OS=p[n_par_tm:])
        elif fit_type == "combined":
            """ forcing the continuum limit value, p[0], to be the same for both """
            ch2 = lambda p: ch2_corr(p_tm=np.array([p[0], *p[1:n_par_tm]]), p_OS=np.array([p[0], *p[n_par_tm:]]))
    #-------
    return ch2
#---

plots_fld="./plots/continuum_limit/fits/"
os.makedirs(plots_fld, exist_ok=True)

# We consider different combinations of numbers of ensembles for tm and OS
n_ens_combinations_tm_OS = [
    # ((2,3), (2,3)), 
    # ((2,3),(1,2,3)), ((2,3),(0,1,2,3)), # (2,2), (2,3), (2,4)
    # ((1,2,3),(1,2,3)), ((1,2,3),(0,1,2,3)), # (3,3), (3,4)
    # ((0,1,2,3),(1,2,3)), # (4,3)
    ((0,1,2,3),(0,1,2,3)) # , (4,4)
    # ((1,2,3),(2,3)), ((1,2,3),(1,2,3)), # (3,x), x=2,3: excluding cB ensemble
    # ((0,2,3),(2,3)), ((0,2,3),(0,2,3)), # (3,x), x=2,3: excluding cC ensemble
    # ((0,1,2),(1,2)), ((0,1,2),(0,1,2))  # (3,x), x=2,3: excluding cD ensemble
    ]

for window_key in windows_keys:
    print(f"{window_key}")
    a_mu_cont_fit = NestedDict()
    a_mu_cont_fit["ansaetze"] = ansaetze

    sensible_fit_pairs = ens_info["continuum_limit"]["fit"]["tm_OS"][window_key]
    a_mu_cont_fit["fit_combinations"] = [[tm_ansatz, OS_ansatz] for tm_ansatz, OS_ansatz in sensible_fit_pairs]

    fit_types = ens_info["continuum_limit"]["fit"]["fit_types"]

    for tm_idx, OS_idx in n_ens_combinations_tm_OS:
        n_ens_tm, n_ens_OS = len(tm_idx), len(OS_idx) # number of points considered for the 2 regularizations
        ens_considered_tm = [e_short[i] for i in tm_idx] # tm ensembles
        a_fm_tm = BootstrapSamples([a_fm_bts[:, i] for i in tm_idx]).T # lattice spacing bootstraps for the tm ensembles
        ens_considered_OS = [e_short[i] for i in OS_idx] # finest n_ens_OS ensembles
        a_fm_OS = BootstrapSamples([a_fm_bts[:, i] for i in OS_idx]).T # lattice spacing bootstraps for the tm ensembles
        n_ens_min = min(n_ens_tm, n_ens_OS) # minimum number of ensembles per type of fermion (tm or OS)
        n_pts_tot = n_ens_tm+n_ens_OS # total number of points
        n_ens_key = f"tm_{"".join(ens_considered_tm)}-OS_{"".join(ens_considered_OS)}"
        print(f" tm: {ens_considered_tm}, OS: {ens_considered_OS} | key: {n_ens_key}")
        a_mu_cont_fit[n_ens_key]["a_fm"]["tm"] = a_fm_tm
        a_mu_cont_fit[n_ens_key]["a_fm"]["OS"] = a_fm_OS
        a_fm_tm_dense = np.linspace(0.0, np.max(a_fm_tm), 100) # dense values for fit predictions
        a_mu_cont_fit[n_ens_key]["a_fm_dense"]["tm"] = a_fm_tm_dense
        a_fm_OS_dense = np.linspace(0.0, np.max(a_fm_OS), 100) # dense values for fit predictions
        a_mu_cont_fit[n_ens_key]["a_fm_dense"]["OS"] = a_fm_OS_dense
        #------------
        t_thr_keys = list(a_mu[ens_considered_tm[0]]["tm"][window_key].keys()) # thr values from a reference ensemble
        for t_thr_key in t_thr_keys:
            print(f"  {t_thr_key}")
            t_cut_strategies = list(a_mu[ens_considered_tm[0]]["tm"][window_key][t_thr_key].keys())
            for t_cut_strategy in t_cut_strategies:
                print(f"   {t_cut_strategy}")
                dt_plateau_strategies = list(a_mu[ens_considered_tm[0]]["tm"][window_key][t_thr_key][t_cut_strategy].keys())
                for dt_plateau_key in dt_plateau_strategies:
                    print(f"     {dt_plateau_key}")
                    a_mu_tm = BootstrapSamples([a_mu[e]["tm"][window_key][t_thr_key][t_cut_strategy][dt_plateau_key] for e in ens_considered_tm]).T
                    a_mu_OS = BootstrapSamples([a_mu[e]["OS"][window_key][t_thr_key][t_cut_strategy][dt_plateau_key] for e in ens_considered_OS]).T
                    diff_a_mu_tm = BootstrapSamples(np.diff(a_mu_tm, axis=1)/np.diff(a_mu_tm, axis=1))
                    diff_a_mu_OS = BootstrapSamples(np.diff(a_mu_OS, axis=1)/np.diff(a_mu_OS, axis=1))
                    # a_mu_cont_fit[n_ens_key][t_thr_key][t_cut_strategy][dt_plateau_key]["a_mu"]["tm"] = a_mu_tm
                    # a_mu_cont_fit[n_ens_key][t_thr_key][t_cut_strategy][dt_plateau_key]["a_mu"]["OS"] = a_mu_OS
                    # a_mu_concatenated = BootstrapSamples(np.concatenate((a_mu_tm, a_mu_OS), axis=1))
                    
                    ansatz_type = "constant"
                    n_par_OS = ansaetze[ansatz_type]["n_par"]
                    guess = np.zeros(shape=(n_par_OS))
                    guess[0] = np.mean(diff_a_mu_OS.mean())
                    ex = a_fm_OS.error()[:-1]
                    ey = diff_a_mu_OS.error()
                    diff_a_mu_OS_fit = BootstrapSamples(np.zeros_like(diff_a_mu_OS))
                    ansatz = ansaetze[ansatz_type]["lambda"]
                    for i in range(N_bts):
                        x = np.array([a_fm_OS[i,:-1]]).T
                        y = diff_a_mu_OS[i,:]
                        OS_fit_res = fit_xiexiyey(
                            ansatz = ansatz, 
                            x=x, ex=ex, 
                            y=y, ey=ey,
                            guess=guess, method = "Nelder-Mead")
                        fit_par = OS_fit_res["par"]
                        diff_a_mu_OS_fit[i,:] = np.array([ansatz(a, fit_par) for a in x[0,:]])

                    a_mu_OS_corrected = np.copy(a_mu_OS)
                    a_mu_OS_corrected[:,0:3] -= np.cumsum(a_fm_OS[:,0:3]*diff_a_mu_OS_fit, axis=1)
                    a_mu_OS_corrected = BootstrapSamples(a_mu_OS_corrected)
                    plt.errorbar(x=a_fm_tm.mean(), y=a_mu_tm.mean(), yerr=a_mu_tm.error(), label="tm-bare")
                    plt.errorbar(x=a_fm_OS.mean(), y=a_mu_OS.mean(), yerr=a_mu_OS.error(), label="OS-bare")
                    plt.errorbar(x=a_fm_OS.mean(), y=a_mu_OS_corrected.mean(), yerr=a_mu_OS_corrected.error(), label="OS-corrected")
                    plt.legend()
                    plt.show()
                        


