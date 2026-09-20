print("""
      --------------------------------------------
      Bounding method: fitting different boundings
      --------------------------------------------
      """
      )

import numpy as np
import matplotlib.pyplot as plt


from lattice_data_tools.bootstrap import BootstrapSamples
from lattice_data_tools.dictionaries import NestedDict
from lattice_data_tools.io import with_dill, with_yaml
from lattice_data_tools.fit.xyey import fit_xyey

# Loading input files

ens_info = with_yaml.load("ensembles.yaml")

ens_names = ens_info["data"]["ens_list"]
aux_dir = ens_info["data"]["auxiliary"]
N_bts = ens_info["N_bts"]

windows = ens_info["windows"]
fermion_types = ens_info["fermion_types"]

N_int = ens_info["N_int_QED_kernel"]

b0, b1 = "2pions", "MV_tail" # boundings_fit

f_pi_list = ens_info["data"]["f_pi_MeV"]
for f_pi in f_pi_list:
    print(f"f_pi:{f_pi}")
    fpi_suffix = f"fpi_{f_pi}_MeV"
    fpi_dir = f"{aux_dir}/{fpi_suffix}/"

    a_fm_dict = with_dill.load(f'{fpi_dir}/a_fm.pkl') # lattice spacing bootstraps

    a_mu_eff_dict = {b: with_dill.load(f'{fpi_dir}/a_mu-{b}.pkl') for b in ["ZeroTail", "2pions", "MV_tail"]}
    boundings_criterion = with_dill.load(f"{fpi_dir}/boundings_criterion.pkl")

    a_mu_bounding_t_thr = NestedDict()
    for ens_name in ens_names:
        print("Ensemble:", ens_name)
        a_fm = a_fm_dict[ens_name]
        a_fm_mean = a_fm.mean()
        T = ens_info[ens_name]["T"]
        T_half = int(T/2)
        L = ens_info[ens_name]["L"] 
        ti = np.arange(0, T_half+1)
        dt_fm = ens_info["bounding"]["dt_fm"]
        for fermion_type in fermion_types:
            print(" Fermions:", fermion_type)

            # print("# Loading the correlator, and including charge factors")
            for corr in ["sim", "pp"]:
                if corr == "sim" and ens_name[0:2] != "cB":
                    continue
                #---
                print(f"  Correction: {corr}")
                windows = list(a_mu_eff_dict[b0][ens_name][fermion_type][corr].keys())
                for window in windows:
                    print("   Window:", window)
                    a_mu_upper = a_mu_eff_dict[b0][ens_name][fermion_type][corr][window]
                    t_thr_keys = a_mu_eff_dict[b1][ens_name][fermion_type][corr].keys()
                    for t_thr_key in t_thr_keys:
                        print(f"    {t_thr_key}")
                        a_mu_lower = a_mu_eff_dict[b1][ens_name][fermion_type][corr][t_thr_key][window]
                        t_cut_strategies = list(boundings_criterion[ens_name][fermion_type][corr][window][t_thr_key].keys())
                        for t_cut_strategy in t_cut_strategies:
                            print(f"     {t_cut_strategy}")
                            dt_plateau_keys = list(boundings_criterion[ens_name][fermion_type][corr][window][t_thr_key][t_cut_strategy].keys())
                            for dt_plateau_key in dt_plateau_keys:
                                print(f"      {dt_plateau_key}")
                                a_mu_dict_target = a_mu_bounding_t_thr[ens_name][fermion_type][corr][window][t_thr_key][t_cut_strategy][dt_plateau_key]
                                if window in ["full_window", "LD_window"]:
                                    # print(f"      # Fitting {window}: plateau region determined by t_cut and dt_plateau")
                                    t_cut = boundings_criterion[ens_name][fermion_type][corr][window][t_thr_key][t_cut_strategy][dt_plateau_key]["t_cut_fit"]
                                    t_end = boundings_criterion[ens_name][fermion_type][corr][window][t_thr_key][t_cut_strategy][dt_plateau_key]["t_end_fit"]
                                    # fitting to a constant in the compatibility region
                                    ti_fit = np.concatenate((ti[t_cut:t_end], ti[t_cut:t_end]))
                                    a_mu_points = BootstrapSamples.from_lambda(N_bts=N_bts, fun=lambda i: np.concatenate((a_mu_upper[i, t_cut:t_end], a_mu_lower[i, t_cut:t_end])))
                                    # Cov_inv = np.linalg.inv(a_mu_points.covariance_matrix())
                                    ansatz = lambda x, p: p[0]
                                    a_mu_guess = [np.mean(a_mu_points.mean())]
                                    a_mu_err_arr = a_mu_points.error()
                                    bts_lambda = lambda i: fit_xyey(ansatz, ti_fit, a_mu_points[i, :], a_mu_err_arr, guess=a_mu_guess, method="Nelder-Mead", Cov_y_inv=None)
                                    a_mu_fit_res = BootstrapSamples.bts_list_from_lambda(N_bts=N_bts, fun=bts_lambda, parallel=True)
                                    a_mu_dict_target["a_mu_fit"] = BootstrapSamples.from_lambda(N_bts=N_bts, fun=lambda i: a_mu_fit_res[i]["par"][0])
                                    a_mu_dict_target["ch2"] = BootstrapSamples.from_lambda(N_bts=N_bts, fun=lambda i: a_mu_fit_res[i]["ch2"])
                                    a_mu_dict_target["ch2_dof"] = BootstrapSamples.from_lambda(N_bts=N_bts, fun=lambda i: a_mu_fit_res[i]["ch2_dof"])
                                    a_mu_dict_target["N_pts"] = a_mu_fit_res[0]["N_pts"] # I can read it from one of the bootstraps
                                    a_mu_dict_target["N_par"] = a_mu_fit_res[0]["N_par"] # I can read it from one of the bootstraps
                                else:
                                    # print(f"      # Fitting {window}: simply take the last point of the upper bound, the error is stable over time ")
                                    a_mu_dict_target["a_mu_fit"] = BootstrapSamples(a_mu_upper[:,-1])
                                    a_mu_dict_target["ch2"] = BootstrapSamples.zeros(N_bts=N_bts, shape=())
                                    a_mu_dict_target["ch2_dof"] = BootstrapSamples.zeros(N_bts=N_bts, shape=())
                                    a_mu_dict_target["N_pts"] = 0 # no fit, just taking the last value
                                    a_mu_dict_target["N_par"] = 0 # no fit, just taking the last value
    #-------------------------------
    with_dill.dump(a_mu_bounding_t_thr, f'{fpi_dir}/a_mu_bounding.pkl') # Save to pickle
#---

