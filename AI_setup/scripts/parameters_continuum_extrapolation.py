print("""
      ------------------------------------------------------------
      Checking if parameters for cont. extrap. are well determined
      ------------------------------------------------------------
      """
      )


import os
import numpy as np

from lattice_data_tools.io import with_dill, with_yaml
from lattice_data_tools.bootstrap import BootstrapSamples

ens_info = with_yaml.load("ensembles.yaml")


aux_dir = ens_info["data"]["auxiliary"]
plots_dir = ens_info["data"]["plots"]
# N_bts = ens_info["N_bts"]
fermion_types = ens_info["fermion_types"]
windows = ens_info["windows"]


fit_types = ens_info["continuum_limit"]["fit"]["fit_types"]

windows = ens_info["windows"]


f_pi_list = ens_info["data"]["f_pi_MeV"]
for f_pi in f_pi_list:
    print(f"f_pi:{f_pi}")
    fpi_suffix = f"fpi_{f_pi}_MeV"
    fpi_dir = f"{aux_dir}/{fpi_suffix}/"

    for window in windows:
        window_key = f"{window}_window"
        print(f" {window_key}")
        plots_fld=f"./{plots_dir}/{fpi_suffix}/continuum_limit/fits/"
        os.makedirs(plots_fld, exist_ok=True)
        summary_file = open(f"{plots_fld}/{window_key}-fit_parameters.csv", "w")
        summary_file.write(f"points|correlated_extr|t_thr|t_cut|dt_plat|fit_type|tm_ansatz|OS_ansatz|n_pts|n_par|dof|det_rho|chi2_dof|c0_tm|err_c0_tm|c1_tm|err_c1_tm|c2_tm|err_c2_tm|c0_OS|err_c0_OS|c1_OS|err_c1_OS|c2_OS|err_c2_OS\n")
        a_mu_cont_fit = with_dill.load(f"{fpi_dir}/{window_key}-a_mu-cont_lim_fit.pkl")
        fit_pairs = a_mu_cont_fit["fit_combinations"]
        n_ens_keys = [k for k in a_mu_cont_fit.keys() if (("tm" in k) and ("OS" in k))]
        for n_ens_key in n_ens_keys:
            a_fm_tm = a_mu_cont_fit[n_ens_key]["a_fm"]["tm"]
            a_fm_OS = a_mu_cont_fit[n_ens_key]["a_fm"]["OS"]
            t_thr_keys = [k for k in a_mu_cont_fit[n_ens_key].keys() if "t_thr" in k]
            for t_thr_key in t_thr_keys:
                print(f"  {t_thr_key}")
                t_cut_strategies = list(a_mu_cont_fit[n_ens_key][t_thr_key].keys())
                for t_cut_strategy in t_cut_strategies:
                    print(f"   {t_cut_strategy}")
                    dt_plateau_strategies = list(a_mu_cont_fit[n_ens_key][t_thr_key][t_cut_strategy].keys())
                    for dt_plateau_key in dt_plateau_strategies:
                        print(f"     {dt_plateau_key}")
                        a_mu_tm  = a_mu_cont_fit[n_ens_key][t_thr_key][t_cut_strategy][dt_plateau_key]["a_mu"]["tm"]
                        a_mu_OS  = a_mu_cont_fit[n_ens_key][t_thr_key][t_cut_strategy][dt_plateau_key]["a_mu"]["OS"]
                        correlation_flags = [k for k in a_mu_cont_fit[n_ens_key][t_thr_key][t_cut_strategy][dt_plateau_key].keys() if "corr" in k]
                        for correlation_flag in correlation_flags:
                            print(f"    {correlation_flag}")
                            for fit_type in fit_types:
                                print(f"     Fit type: {fit_type}")
                                for tm_ansatz, OS_ansatz in fit_pairs:
                                    print(f"      Ansatze: tm:{tm_ansatz}, OS:{OS_ansatz}")
                                    a_mu_cont_fit_results = a_mu_cont_fit[f"{n_ens_key}"][t_thr_key][t_cut_strategy][dt_plateau_key][correlation_flag][f"{fit_type}_fit"][f"tm:{tm_ansatz}"][f"OS:{OS_ansatz}"]
                                    par_tm = a_mu_cont_fit_results["par_tm"] 
                                    par_OS = a_mu_cont_fit_results["par_OS"]

                                    # determinant of correlation matrix among parameters
                                    all_par = BootstrapSamples(np.concatenate((par_tm, par_OS[:,int(fit_type == "combined"):]), axis=1))
                                    rho_par = all_par.correlation_matrix() 
                                    det_rho_par = np.linalg.det(rho_par)

                                    ch2_values = a_mu_cont_fit_results["minimization"]["ch2_values"] 
                                    n_par  = a_mu_cont_fit_results["minimization"]["n_par"] 
                                    n_ens_tm  = a_mu_cont_fit_results["minimization"]["n_pts"]["tm"] 
                                    n_ens_OS  = a_mu_cont_fit_results["minimization"]["n_pts"]["OS"] 
                                    n_pts_tot = a_mu_cont_fit_results["minimization"]["n_pts"]["tot"] 
                                    n_dof = n_pts_tot-n_par 
                                    n_dof = a_mu_cont_fit_results["minimization"]["n_dof"]
                                    summary_file.write(f"{n_ens_key.split("points-")[-1]}|{correlation_flag}|{t_thr_key}|{t_cut_strategy}|{dt_plateau_key}|{fit_type}_fit|tm:{tm_ansatz}|OS:{OS_ansatz}|{n_pts_tot}|{n_par}|{n_dof}|{det_rho_par}")
                                    ch2_dof = ch2_values.unbiased_mean()/n_dof
                                    ch2_dof_str  = "NaN" if n_dof==0 else ch2_dof
                                    # oom_ch2_dof = np.floor(np.log10(np.abs(ch2_dof))).astype(int) # order of magnitude
                                    # ch2_dof_to_print = f"{np.round(ch2_dof*10.0**(-oom_ch2_dof) , decimals=decimals)}" if n_dof !=0 else "NaN"
                                    summary_file.write(f"|{ch2_dof_str}")
                                    par_tm_mean = par_tm.mean()
                                    # oom_tm = np.floor(np.log10(np.abs(par_tm_mean))).astype(int) # order of magnitude
                                    # par_tm_mean = np.round(par_tm_mean * 10.0**(-oom_tm), decimals=decimals)
                                    par_tm_error = par_tm.error()
                                    n_par_tm = par_tm_mean.shape[0]
                                    for i in range(3):
                                        if i < n_par_tm:
                                            summary_file.write(f"|{par_tm_mean[i]}|{par_tm_error[i]}")
                                        else:
                                            summary_file.write(f"|NaN|NaN")
                                    #---
                                    par_OS_mean = par_OS.mean()
                                    # oom_OS = np.floor(np.log10(np.abs(par_OS_mean))).astype(int) # order of magnitude
                                    # par_OS_mean = np.round(par_OS_mean * 10.0**(-oom_OS), decimals=decimals)
                                    par_OS_error = par_OS.error()
                                    n_par_OS = par_OS_mean.shape[0]
                                    for i in range(3):
                                        if i < n_par_OS:
                                            summary_file.write(f"|{par_OS_mean[i]}|{par_OS_error[i]}")
                                        else:
                                            summary_file.write(f"|NaN|NaN")
                                    #---

                                    summary_file.write("\n")
#---

