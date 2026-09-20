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


from lattice_data_tools.bootstrap import BootstrapSamples
from lattice_data_tools.io import with_dill, with_yaml
from lattice_data_tools.dictionaries import NestedDict

ens_info = with_yaml.load("ensembles.yaml")

aux_dir = ens_info["data"]["auxiliary"]
plots_dir = ens_info["data"]["plots"]
N_bts = ens_info["N_bts"]
fermion_types = ens_info["fermion_types"]
windows_keys = [f"{window}_window" for window in ens_info["windows"]]


gamma2 = 0.42 # from Husung model: https://arxiv.org/abs/2501.17036
lambda0 = (0.3/0.1973269804) # Lambda_QCD in fm
def Husung(a, p): 
    return p[0] + p[1] * (a**2) * (np.log(1.0/(a*lambda0 + (a==0).astype(int)))**(-gamma2 + (a==0).astype(int)))


f_pi_list = ens_info["data"]["f_pi_MeV"]
for f_pi in f_pi_list:
    print(f"f_pi:{f_pi}")
    fpi_suffix = f"fpi_{f_pi}_MeV"
    fpi_dir = f"{aux_dir}/{fpi_suffix}/"


    a_fm_dict = with_dill.load(f'{fpi_dir}/a_fm_Lref.pkl') # lattice spacing bootstraps
    a_mu = with_dill.load(f'{fpi_dir}/a_mu-pp_UV_corrected.pkl')

    e_short = a_mu["ens_list"]
    n_ens_tot = len(e_short)
    a_fm_bts = BootstrapSamples([a_fm_dict[e] for e in e_short]).T # bootstraps of lattice spacing

    # ansaetze in a^2
    # IMPORTANT: all of them should be defined such that p[0] is the continuum value
    ansaetze = {
            "constant": {"n_par": 1, "lambda": lambda a, p: (0.0*a) + p[0]}, # 0.0*a just gives the right shape if passing an array
            "linear": {"n_par": 2, "lambda": lambda a, p: p[0] + p[1] * (a**2)},
            "quadratic": {"n_par": 3, "lambda": lambda a, p: p[0] + p[1] * (a**2) + p[2] * (a**2)**2},
            # "a^2log(a)": {"n_par": 2, "lambda": lambda a, p: p[0] + p[1] * (a**2) * np.log(a + (a==0).astype(int))}, # regularized log, returns 0 for a==0
            # "Husung": {"n_par": 2, "lambda": lambda a, p: p[0] + p[1] * (a**2) * (np.log(a*lambda0 + (a==0).astype(int))**(-gamma2 + (a==0).astype(int)))}, # regularized log, returns 0 for a==0
            "Husung": {"n_par": 2, "lambda": lambda a, p: p[0] + p[1] * (a**2) * (np.log(1.0/(a*lambda0 + (a==0).astype(int)))**(-gamma2 + (a==0).astype(int)))}, # regularized log, returns 0 for a==0
            "linear+Husung": {"n_par": 3, "lambda": lambda a, p: p[0] + p[1] * (a**2) + p[2] * (a**2) * (np.log(1.0/(a*lambda0 + (a==0).astype(int)))**(-gamma2 + (a==0).astype(int)))}, # regularized log, returns 0 for a==0
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

    plots_fld = f"./{plots_dir}/{fpi_suffix}/continuum_limit/fits/"
    os.makedirs(plots_fld, exist_ok=True)

    # We consider different combinations of numbers of ensembles for tm and OS
    n_ens_combinations_tm_OS = [
        # ((2,3), (2,3)), 
        # ((2,3),(1,2,3)), 
        # ((2,3),(0,1,2,3)), # (2,2), (2,3), (2,4)
        # ((1,2,3),(1,2,3)), ((1,2,3),(0,1,2,3)), # (3,3), (3,4)
        # ((0,1,2,3),(1,2,3)), 
        # ((0,1,2,3),(0,1,2,3)), # (4,3), (4,4)
        # ((1,2,3),(1,2,3)), # (3,3)
        ((0,1,2,3),(0,1,2,3)) #, # (4,4)
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
                        # print(e_short)
                        # print((9/10) * a_mu_tm.error())
                        # print((9/10) * a_mu_OS.error())
                        # quit()
                        a_mu_cont_fit[n_ens_key][t_thr_key][t_cut_strategy][dt_plateau_key]["a_mu"]["tm"] = a_mu_tm
                        a_mu_cont_fit[n_ens_key][t_thr_key][t_cut_strategy][dt_plateau_key]["a_mu"]["OS"] = a_mu_OS
                        a_mu_concatenated = BootstrapSamples(np.concatenate((a_mu_tm, a_mu_OS), axis=1))

                        # for correlated_fit in [True, False]:
                        for correlated_fit in [True]:
                            correlation_flag = "correlated_fit" if correlated_fit else "uncorrelated_fit"
                            print(f"    {correlation_flag}")
                            if correlated_fit:
                                # NOTE: Different ensembles have no correlation among each other. 
                                # The bootstrap samples are just the same in number. 
                                # Thus, any correlation we get from here is artificial. 
                                # Below we impose 0 correlation for the entries mixing different ensembles
                                 Cov_full = a_mu_concatenated.covariance_matrix()
                                 Cov = np.zeros_like(Cov_full)
                                 for i in range(n_pts_tot):
                                     Cov[i,i] = Cov_full[i,i]
                                     if i < n_ens_min:
                                         j = i+n_ens_min
                                         Cov[i,j] = Cov_full[i,j]
                                         Cov[j,i] = Cov_full[j,i]
                                 #---
                                 # Use SVD-based pseudo-inverse for stability
                                 try:
                                     Cov_inv = np.linalg.inv(Cov)
                                 except np.linalg.LinAlgError:
                                     Cov_inv = np.linalg.pinv(Cov)

                            else:
                                Cov_inv = None
                            #---
                            for fit_type in fit_types:
                                print(f"      Fit type: {fit_type}")
                                # number of points: lattice spacings \\times fermion types (tm, OS)
                                for tm_ansatz, OS_ansatz in sensible_fit_pairs:
                                    print(f"      tm-OS ansaetze: {tm_ansatz}-{OS_ansatz}")
                                    n_par_tm = ansaetze[tm_ansatz]["n_par"]
                                    n_par_OS = ansaetze[OS_ansatz]["n_par"]
                                    n_par = n_par_tm+n_par_OS
                                    if fit_type == "combined":
                                        n_par -= 1 # p[0] is in common
                                    #---
                                    guess = np.zeros(shape=(n_par))
                                    guess[0] = np.average(a_mu_tm.mean())
                                    if fit_type == "individual":
                                        guess[n_par_tm] = np.average(a_mu_OS.mean())
                                    else:
                                        if tm_ansatz != "constant":
                                            guess[1] = (a_mu_OS.mean()[-1]-a_mu_OS.mean()[0])/(a_fm_tm.mean()[-1]- a_fm_tm.mean()[0])
                                        if OS_ansatz != "constant":
                                            idx_OS = (n_par+1 if fit_type=="individual" else 1)
                                            guess[idx_OS] = (a_mu_OS.mean()[-1]-a_mu_OS.mean()[0])/(a_fm_OS.mean()[-1]- a_fm_OS.mean()[0])
                                    #---
                                    err_a_mu_tm = a_mu_tm.error()
                                    err_a_mu_OS = a_mu_OS.error()
                                    a_mu_fit_tm, a_mu_fit_OS = BootstrapSamples.zeros(N_bts=N_bts), BootstrapSamples.zeros(N_bts=N_bts) # continuum extrapolation
                                    ch2_values = BootstrapSamples.zeros(N_bts=N_bts)
                                    par_tm = BootstrapSamples.zeros(N_bts=N_bts, shape=(n_par_tm))
                                    par_OS = BootstrapSamples.zeros(N_bts=N_bts, shape=(n_par_OS))
                                    def fit_bts_by_bts(i: int) -> dict:
                                        ch2_fun = get_ch2_fun(
                                            a_fm={"tm": a_fm_tm[i,:], "OS": a_fm_OS[i,:]},
                                            a_mu={"tm": a_mu_tm[i,:], "OS": a_mu_OS[i,:]}, 
                                            err_a_mu={"tm": err_a_mu_tm, "OS": err_a_mu_OS},
                                            fit_type=fit_type, tm_ansatz=tm_ansatz, OS_ansatz=OS_ansatz,
                                            Cov_inv=Cov_inv
                                            )
                                        mini = opt.minimize(fun = ch2_fun, x0 = guess, method = "Nelder-Mead") # Nelder-Mead seems to be good
                                        par = mini.x
                                        return {
                                            "par_tm": par[0:n_par_tm],
                                            "par_OS": par[n_par_tm:] if fit_type=="individual" else np.array([par[0], *par[n_par_tm:]]),
                                            "a_mu_fit_tm": par[0],
                                            "a_mu_fit_OS": par[n_par_tm] if fit_type=="individual" else par[0],
                                            "ch2_values":  ch2_fun(par)
                                            }
                                    #---
                                    fit_results = BootstrapSamples.from_lambda(N_bts=N_bts, fun=fit_bts_by_bts, parallel=True)[:]
                                    par_tm = BootstrapSamples([fr["par_tm"] for fr in fit_results])
                                    par_OS = BootstrapSamples([fr["par_OS"] for fr in fit_results])
                                    a_mu_fit_tm = BootstrapSamples([fr["a_mu_fit_tm"] for fr in fit_results])
                                    a_mu_fit_OS = BootstrapSamples([fr["a_mu_fit_OS"] for fr in fit_results])
                                    ch2_values = BootstrapSamples([fr["ch2_values"] for fr in fit_results])
                                    # saving values in the nested dictionary
                                    a_mu_cont_fit_results = a_mu_cont_fit[n_ens_key][t_thr_key][t_cut_strategy][dt_plateau_key][correlation_flag][f"{fit_type}_fit"][f"tm:{tm_ansatz}"][f"OS:{OS_ansatz}"]
                                    a_mu_cont_fit_results["par_tm"] = par_tm
                                    a_mu_cont_fit_results["par_OS"] = par_OS
                                    a_mu_cont_fit_results["a_mu"]["tm"] = a_mu_fit_tm
                                    a_mu_cont_fit_results["a_mu"]["OS"] = a_mu_fit_OS
                                    a_mu_cont_fit_results["minimization"]["ch2_values"] = ch2_values
                                    a_mu_cont_fit_results["minimization"]["n_par"] = n_par 
                                    a_mu_cont_fit_results["minimization"]["n_pts"]["tm"] = n_ens_tm 
                                    a_mu_cont_fit_results["minimization"]["n_pts"]["OS"] = n_ens_OS 
                                    a_mu_cont_fit_results["minimization"]["n_pts"]["tot"] = n_pts_tot
                                    n_dof = n_pts_tot-n_par 
                                    a_mu_cont_fit_results["minimization"]["n_dof"] = n_dof
                                    # results for later plotting
                                    a_mu_cont_fit_results["a_mu_tm_dense"] = BootstrapSamples.from_lambda(N_bts=N_bts, fun=lambda i: ansaetze[tm_ansatz]["lambda"](a_fm_tm_dense, par_tm[i,:]), parallel=True)
                                    a_mu_cont_fit_results["a_mu_OS_dense"] = BootstrapSamples.from_lambda(N_bts=N_bts, fun=lambda i: ansaetze[OS_ansatz]["lambda"](a_fm_OS_dense, par_OS[i,:]), parallel=True)
        #---------------------------
        pkl_file = f"{fpi_dir}/{window_key}-a_mu-cont_lim_fit.pkl"
        print("--> Output:", pkl_file)
        with_dill.dump(a_mu_cont_fit, pkl_file)

#---

