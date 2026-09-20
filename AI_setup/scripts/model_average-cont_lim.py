print("""
      -------------------------------------
      Model average of the continuum limits
      -------------------------------------
      """
      )


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Helvetica"
# })

import os

from lattice_data_tools.bootstrap import BootstrapSamples
import lattice_data_tools.model_averaging.IC as IC
from lattice_data_tools.model_averaging.with_bts import ModelAverage
from lattice_data_tools.io import with_dill, with_yaml
from lattice_data_tools.dictionaries import NestedDict
from lattice_data_tools.plotting.with_matplotlib.IC import FromBootstraps as IC_plot_from_bts


ens_info = with_yaml.load("ensembles.yaml")

aux_dir = ens_info["data"]["auxiliary"]
plots_dir = ens_info["data"]["plots"]
N_bts = ens_info["N_bts"]
fermion_types = ens_info["fermion_types"]
windows = ens_info["windows"]

fit_types = ens_info["continuum_limit"]["fit"]["fit_types"]

N_max = 8 # maximum number of points in cont. lim extrapolations

strategies = ["preferred", "conservative", "aggressive", "moderate", "ToV"]

windows = ens_info["windows"]


f_pi_list = ens_info["data"]["f_pi_MeV"]
for f_pi in f_pi_list:
    print(f"f_pi:{f_pi}")
    fpi_suffix = f"fpi_{f_pi}_MeV"
    fpi_dir = f"{aux_dir}/{fpi_suffix}/"

    res_IC_averaging_dict = NestedDict() # results of the IC averagings
    for window in windows:
        window_key = f"{window}_window"
        print(f"{window_key}")
        a_mu_cont_fit = with_dill.load(f"{fpi_dir}/{window_key}-a_mu-cont_lim_fit.pkl")
        n_ens_keys = [k for k in a_mu_cont_fit.keys() if (("tm" in k) and ("OS" in k))]
        fit_combinations = a_mu_cont_fit["fit_combinations"]
        window = window_key.split("_")[0]
        t_thr_keys = [k for k in a_mu_cont_fit[n_ens_keys[0]].keys() if "t_thr" in k]
        t_cut_strategies = list(a_mu_cont_fit[n_ens_keys[0]][t_thr_keys[0]].keys())
        dt_plateau_strategies = list(a_mu_cont_fit[n_ens_keys[0]][t_thr_keys[0]][t_cut_strategies[0]].keys())
        correlation_flags = [k for k in a_mu_cont_fit[n_ens_keys[0]][t_thr_keys[0]][t_cut_strategies[0]][dt_plateau_strategies[0]].keys() if "corr" in k]
        for correlation_flag in correlation_flags:
            print(f" Correlation: {correlation_flag}")
            for strategy in strategies:
                print(f"  Analysis strategy: {strategy}")
                plots_fld=f"./{plots_dir}/{fpi_suffix}/continuum_limit/model_average/{strategy}_strategy/{correlation_flag}/{window}_window/"
                os.makedirs(plots_fld, exist_ok=True)
                # calculating the error budget of each effect
                Y = NestedDict()
                n_models = 0
                models_info_file = open(f"{plots_fld}/models_info.txt", "w")
                for n_ens_key in n_ens_keys:
                    print(f"   Number of ensembles: {n_ens_key}")
                    tm_key, OS_key = n_ens_key.split("-")
                    a_fm_tm = a_mu_cont_fit[n_ens_key]["a_fm"]["tm"]
                    a_fm_OS = a_mu_cont_fit[n_ens_key]["a_fm"]["OS"]
                    n_tm = a_fm_tm.shape[1]
                    n_OS = a_fm_OS.shape[1]
                    for t_thr_key in t_thr_keys:
                        print(f"   {t_thr_key}")
                        for t_cut_strategy in t_cut_strategies:
                            print(f"    {t_cut_strategy}")
                            for dt_plateau_key in dt_plateau_strategies:
                                print(f"     {dt_plateau_key}")
                                for fit_type in fit_types:
                                    if fit_type != "combined":
                                        continue # we want to consider only combined fits: we know that tm and OS coincide in the continuum
                                    print(f"     Fit type: {fit_type}")
                                    for tm_ansatz, OS_ansatz in fit_combinations:
                                        print(f"      Ansatze: tm:{tm_ansatz}, OS:{OS_ansatz}")

                                        # if (strategy in ["aggressive", "preferred"]) and (window in ["full", "LD"]):
                                        #     b1 = (tm_ansatz == "constant") # and OS_ansatz == "linear")                                        
                                        #     b2 = (n_ens_key == "tm_BCDE-OS_BCDE") or (n_ens_key=="tm_CDE-OS_CDE" and OS_ansatz=="linear")
                                        #     if strategy == "aggressive":
                                        #         b2 = (n_ens_key in ["tm_BCDE-OS_BCDE"] and tm_ansatz=="constant" and OS_ansatz=="linear")
                                        #     elif strategy == "preferred":
                                        #         if ((n_ens_key in ["tm_CDE-OS_BCDE", "tm_CDE-OS_CDE"]) and (tm_ansatz=="linear" and OS_ansatz=="linear")):
                                        #             continue

                                        #     if not (b1 and b2):
                                        #     # if not (b2):
                                        #         continue
                                        # #---

                                        extrapolation_key  = f"{fit_type},tm:{tm_ansatz},OS:{OS_ansatz}"
                                        DictFitPairs = a_mu_cont_fit[n_ens_key][t_thr_key][t_cut_strategy][dt_plateau_key][correlation_flag][f"{fit_type}_fit"][f"tm:{tm_ansatz}"][f"OS:{OS_ansatz}"]
                                        a_mu_tm = DictFitPairs["a_mu"]["tm"]
                                        a_mu_OS = DictFitPairs["a_mu"]["OS"]
                                        ch2_mean = DictFitPairs["minimization"]["ch2_values"].unbiased_mean()
                                        n_par = DictFitPairs["minimization"]["n_par"]

                                        par_tm = DictFitPairs["par_tm"]
                                        par_OS = DictFitPairs["par_OS"]
                                        # computing the effective number of parameters
                                        all_par = BootstrapSamples(np.concatenate((par_tm, par_OS[:,int(fit_type == "combined"):]), axis=1))
                                        all_par_abs_mean = np.abs(all_par.mean())
                                        all_par_err = all_par.error()

                                        # if (not (("E" in tm_key) and ("E" in OS_key))):
                                        #     continue # excluding fits where the "E" ensemble is included for the tm fermions but not for the OS

                                        # if (OS_key=="OS_DE"):
                                        #     continue # excluding fits with only 2 points for OS fermions
                                        
                                        # considering only fits that have well-determined parameters
                                        if (strategy=="moderate"):
                                            if ( np.any(all_par_err/all_par_abs_mean > 3.0)):
                                                continue

                                        # considering only fits that have well-determined parameters
                                        if (strategy=="aggressive"):
                                            if ( np.any(all_par_err/all_par_abs_mean > 0.5) or (tm_ansatz != "constant" or OS_ansatz!="linear") or (not (t_thr_key=="t_thr_fm=1.8" and t_cut_strategy=="aggressive_t_cut" and dt_plateau_key=="dt=0.25[fm]"))):
                                                continue

                                        # considering only fits that have well-determined parameters
                                        if (strategy=="preferred"):
                                            # if ( np.any(all_par_err/all_par_abs_mean > 1.0) or ((n_ens_key in ["tm_CDE-OS_BCDE", "tm_CDE-OS_CDE"]) and (tm_ansatz=="linear" and OS_ansatz=="linear"))):
                                            if ( np.any(all_par_err/all_par_abs_mean > 1.0)):
                                                continue

                                        if (strategy=="ToV"):
                                            ansaetze = [tm_ansatz, OS_ansatz] 
                                            b_ToV = (tm_ansatz == "constant" and OS_ansatz not in ["linear", "Husung"])
                                            if b_ToV:
                                                continue

                                        # if strategy in ["conservative", "moderate", "preferred"]:
                                        # # if strategy in ["conservative", "aggressive", "moderate"]:
                                        #     if strategy == "conservative":
                                        #         rel_err_threshold = 1.0
                                        #     elif strategy == "moderate":
                                        #         rel_err_threshold = 0.5
                                        #     elif strategy == "preferred":
                                        #         rel_err_threshold = 1.0
                                        #     #---
                                        #     if np.any(all_par_err/all_par_abs_mean > rel_err_threshold):
                                        #         """ relative uncertainty higher than a threshold """
                                        #         continue
                                        # #-------

                                        n_pts = DictFitPairs["minimization"]["n_pts"]["tot"]
                                        rel_err = np.abs(all_par_err / np.where(all_par_abs_mean == 0, np.nan, all_par_abs_mean))
                                        rel_err_pct = [f"{val:.1f}%" if not np.isnan(val) else "nan" for val in np.round(100.0 * rel_err, 2)]
                                        models_info_file.write(
                                            f"{extrapolation_key}, {n_ens_key}, {t_thr_key}, {t_cut_strategy}, {dt_plateau_key}, "
                                            f"ch2/dof: {ch2_mean/n_par:.2f}, n_par: {n_par}, n_pts: {n_pts}, "
                                            f"param_pct_err: {rel_err_pct}\n"
                                        )

                                        Y[n_ens_key][t_thr_key][t_cut_strategy][dt_plateau_key][extrapolation_key]["y"] = (a_mu_tm+a_mu_OS)/2.0
                                        Y[n_ens_key][t_thr_key][t_cut_strategy][dt_plateau_key][extrapolation_key]["ch2"] = ch2_mean
                                        Y[n_ens_key][t_thr_key][t_cut_strategy][dt_plateau_key][extrapolation_key]["n_par"] = n_par
                                        Y[n_ens_key][t_thr_key][t_cut_strategy][dt_plateau_key][extrapolation_key]["n_data"] = n_pts
                                        n_models += 1
                #-----------------------
                models_info_file.close()
                syst_names = ["n_ens", "t_thr", "t_cut", "dt_plateau", "extrapolation"]
                for IC_name in IC.valid_IC_list:
                    print(f"  IC: {IC_name}")
                    EBT = ModelAverage.error_budget_table(Y=Y, syst_names=syst_names, IC="AIC", Nmax=N_max)
                    for syst_key in EBT.keys():
                        y = EBT[syst_key]["y"]
                        P = EBT[syst_key]["P"]
                        IPR = EBT[syst_key]["IPR"]
                        sigma2_tot = EBT[syst_key]["sigma2_tot"]
                        sigma2_stat = EBT[syst_key]["sigma2_stat"]
                        sigma2_syst = EBT[syst_key]["sigma2_syst"]
                        title_str = f"CDF of systematic effect from: {syst_key}, IPR={IPR}"
                        title_str += f"\n$\\sigma^2_{{tot}}={np.round(1e+10*np.sqrt(sigma2_tot),4)}$, $\\sigma^2_{{stat}}={np.round(1e+10*np.sqrt(sigma2_stat),4)}$, $\\sigma^2_{{syst}}={np.round(1e+10*np.sqrt(sigma2_syst),4)}$"
                        fig, ax = IC_plot_from_bts.plot_cdf(y=y, P=P, title=title_str)
                        fig.set_figwidth(16)
                        fig.set_figwidth(9)            
                        plt.tight_layout()
                        subplots_fld = f"{plots_fld}/systematics/"
                        os.makedirs(subplots_fld, exist_ok=True)
                        plt.savefig(f"{subplots_fld}/{strategy}-{syst_key}.svg")
                        plt.close()
                    #---
                    sigma2_syst_arr = np.array([EBT["syst_tot"][f"sigma2_{sigma2_name}"] for sigma2_name in ["tot", "stat", "syst"] ] + [EBT[syst_key]["sigma2_syst"] for syst_key in syst_names])
                    y, P = EBT["syst_tot"]["y"], EBT["syst_tot"]["P"]
                    # y50 = IC.with_CDF.get_quantiles(y=y, P=P)["50%"]
                    a_mu_mean = EBT["syst_tot"]["mean"]
                    title_str = f"Model Averaging | {window} window | {n_models} models | IPR = {np.round(IPR,3)} | IC: {IC_name} \n"
                    title_str += f"$a_\\mu \\times 10^{{10}}={np.round(1e+10*a_mu_mean ,4)} \\pm {np.round(1e+10*np.sqrt(EBT["syst_tot"]["sigma2_tot"]),4)}$"
                    title_str += f" ($\\approx {np.round(100*np.sqrt(EBT["syst_tot"]["sigma2_tot"])/a_mu_mean,2)} \\%$)"
                    fig, ax = IC_plot_from_bts.plot_cdf(y=y, P=P, title=title_str)
                    fig.set_figwidth(16)
                    fig.set_figwidth(9)
                    # Add a new axes to the *same* figure
                    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
                    ax.set_position(gs[0].get_position(fig))  # position CDF plot in left grid cell
                    ax.set_subplotspec(gs[0])                 # associate it with gridspec

                    ax_table = fig.add_subplot(gs[1])
                    ax_table.axis('off')

                    # Table
                    df = pd.DataFrame({
                        "Error budget": ["tot", "stat", "syst_tot"]+syst_names,
                        "$\\Delta a_\\mu \\cdot 10^{10}$": [
                            f"{val:0.4f}" for val in np.round(1e+10 * np.sqrt(sigma2_syst_arr), 4)
                        ]
                    })
                    table = ax_table.table(cellText=df.values, colLabels=df.columns, loc='center')
                    table.auto_set_font_size(False)
                    table.set_fontsize(12)
                    table.scale(1.0, 2.0)

                    plt.tight_layout()
                    plt.tight_layout()
                    plt.savefig(f"{plots_fld}/{strategy}-{IC_name}.svg", format='svg')
                    plt.close()
                    res_IC_averaging_dict[window][correlation_flag][strategy][IC_name] = EBT
    #---------------
    pkl_file = f"{fpi_dir}/model_average_continuum_limit.pkl"
    print("--> Output:", pkl_file)
    with_dill.dump(res_IC_averaging_dict, pkl_file)
#---

