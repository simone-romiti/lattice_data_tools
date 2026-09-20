print("""
      ----------------------------------------
      Plotting: continuum limit extrapolations
      ----------------------------------------
      """
      )

import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
plt.rcParams.update({
    # "text.usetex": True,
    # "font.family": "Helvetica",
    "font.size": 22
})



# from lattice_data_tools.bootstrap import BootstrapSamples
from lattice_data_tools.io import with_dill, with_yaml
# from lattice_data_tools.dictionaries import NestedDict

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
        a_mu_cont_fit = with_dill.load(f"{fpi_dir}/{window_key}-a_mu-cont_lim_fit.pkl")
        fit_pairs = a_mu_cont_fit["fit_combinations"]
        n_ens_keys = [k for k in a_mu_cont_fit.keys() if (("tm" in k) and ("OS" in k))]
        for n_ens_key in n_ens_keys:
            print(f"  {n_ens_key}")
            a_fm_tm = a_mu_cont_fit[n_ens_key]["a_fm"]["tm"]
            a_fm_OS = a_mu_cont_fit[n_ens_key]["a_fm"]["OS"]
            t_thr_keys = [k for k in a_mu_cont_fit[n_ens_key].keys() if "t_thr" in k]
            for t_thr_key in t_thr_keys:
                print(f"   {t_thr_key}")
                t_cut_strategies = list(a_mu_cont_fit[n_ens_key][t_thr_key].keys())
                for t_cut_strategy in t_cut_strategies:
                    print(f"    {t_cut_strategy}")
                    dt_plateau_strategies = list(a_mu_cont_fit[n_ens_key][t_thr_key][t_cut_strategy].keys())
                    for dt_plateau_key in dt_plateau_strategies:
                        print(f"     {dt_plateau_key}")
                        a_mu_tm  = a_mu_cont_fit[n_ens_key][t_thr_key][t_cut_strategy][dt_plateau_key]["a_mu"]["tm"]
                        a_mu_OS  = a_mu_cont_fit[n_ens_key][t_thr_key][t_cut_strategy][dt_plateau_key]["a_mu"]["OS"]
                        if a_mu_tm.shape[1] == 4 and a_mu_OS.shape[1] == 4:
                            summary_dict = {
                                "a_fm": a_fm_tm.unbiased_mean(), "da_fm": a_fm_tm.error(),
                                "a_mu-tm": a_mu_tm.unbiased_mean(), "da_mu-tm": a_mu_tm.error(),
                                "a_mu-OS": a_mu_OS.unbiased_mean(), "da_mu-OS": a_mu_OS.error(),
                                "(9/10)*a_mu-tm": a_mu_tm.unbiased_mean(), "(9/10) * da_mu-tm": a_mu_tm.error(),
                                "(9/10)*a_mu-OS": a_mu_OS.unbiased_mean(), "(9/10) * da_mu-OS": a_mu_OS.error()
                                }
                            out_fld = f"./{plots_dir}/{fpi_suffix}/continuum_limit/points/"
                            os.makedirs(out_fld, exist_ok=True)
                            pd.DataFrame(summary_dict).to_csv(f"{out_fld}/{t_thr_key}-{t_cut_strategy}-{dt_plateau_key}.csv")
                        #---
                        correlation_flags = [k for k in a_mu_cont_fit[n_ens_key][t_thr_key][t_cut_strategy][dt_plateau_key].keys() if "corr" in k]
                        for correlation_flag in correlation_flags:
                            print(f"      {correlation_flag}")
                            for fit_type in fit_types:
                                print(f"       Fit type: {fit_type}")
                                for tm_ansatz, OS_ansatz in fit_pairs:
                                    print(f"        {tm_ansatz}, {OS_ansatz}")
                                    err_a_mu_tm = a_mu_tm.error()
                                    err_a_mu_OS = a_mu_OS.error()
                                    DictFitPairs = a_mu_cont_fit[n_ens_key][t_thr_key][t_cut_strategy][dt_plateau_key][correlation_flag][f"{fit_type}_fit"][f"tm:{tm_ansatz}"][f"OS:{OS_ansatz}"]
                                    par_tm = DictFitPairs["par_tm"]
                                    par_OS = DictFitPairs["par_OS"]
                                    a_mu_fit_tm  = DictFitPairs["a_mu"]["tm"]
                                    a_mu_fit_OS  = DictFitPairs["a_mu"]["OS"]
                                    ch2_values  = DictFitPairs["minimization"]["ch2_values"]
                                    n_dof = DictFitPairs["minimization"]["n_dof"]
                                    a_fm_tm_dense = a_mu_cont_fit[n_ens_key]["a_fm_dense"]["tm"]
                                    a_fm_OS_dense = a_mu_cont_fit[n_ens_key]["a_fm_dense"]["OS"]
                                    # plotting
                                    a_mu_tm_dense = DictFitPairs["a_mu_tm_dense"]
                                    a_mu_OS_dense = DictFitPairs["a_mu_OS_dense"]
                                    plt.figure(figsize=(16,9))
                                    plt.plot(a_fm_tm_dense**2, a_mu_tm_dense.unbiased_mean(), label="tm-fit", color="red")
                                    plt.fill_between(
                                        x=a_fm_tm_dense**2, 
                                        y1=a_mu_tm_dense.unbiased_mean()+a_mu_tm_dense.error(), 
                                        y2=a_mu_tm_dense.unbiased_mean()-a_mu_tm_dense.error(), 
                                        color="red",
                                        alpha=0.1
                                        )
                                    # print(a_fm_dense.shape, a_mu_OS_dense.shape, OS_ansatz, par_tm)
                                    plt.plot(a_fm_OS_dense**2, a_mu_OS_dense.unbiased_mean(), label="OS-fit", color="blue")
                                    plt.fill_between(
                                        x=a_fm_OS_dense**2, 
                                        y1=a_mu_OS_dense.unbiased_mean()+a_mu_OS_dense.error(), 
                                        y2=a_mu_OS_dense.unbiased_mean()-a_mu_OS_dense.error(), 
                                        color="blue",
                                        alpha=0.1
                                        )
                                    plt.errorbar(x=(a_fm_tm**2).unbiased_mean(), y=a_mu_tm.unbiased_mean(), xerr=a_fm_tm.error(), yerr=err_a_mu_tm, label="tm", capsize=2)
                                    plt.errorbar(x=(a_fm_OS**2).unbiased_mean(), y=a_mu_OS.unbiased_mean(), xerr=a_fm_OS.error(), yerr=err_a_mu_OS, label="OS", capsize=2)
                                    plt.plot([0.0], a_mu_fit_tm.unbiased_mean(), marker="x", label=f"$a_\\mu^{{tm}} \\times 10^{{10}}={np.round(1e+10*a_mu_fit_tm.unbiased_mean() ,2)}\\pm {np.round(1e+10 * a_mu_fit_tm.error() ,2)}$")
                                    plt.plot([0.0], a_mu_fit_OS.unbiased_mean(), marker="x", label=f"$a_\\mu^{{OS}} \\times 10^{{10}}={np.round(1e+10*a_mu_fit_OS.unbiased_mean() ,2)}\\pm {np.round(1e+10 * a_mu_fit_OS.error() ,2)}$")
                                    plt.xlabel("$(a[\\mathrm{fm}])^2$")
                                    plt.ylabel(f"$a_\\mu^{{\\text{{{window}}}}}$")
                                    plt.legend()
                                    ch2_red_str = f"{np.round(ch2_values.unbiased_mean()/n_dof ,2)}" if n_dof!=0 else "NaN"
                                    plt.title(f"Fit: {fit_type}, tm: {tm_ansatz}, OS: {OS_ansatz}, $\\chi^2_\\mathrm{{d.o.f.}}={ch2_red_str}$")

                                    plt.tight_layout()

                                    plots_fld=f"./{plots_dir}/{fpi_suffix}/continuum_limit/fits/{correlation_flag}/{window_key}/{n_ens_key}/{t_thr_key}-{t_cut_strategy}-{dt_plateau_key}/"
                                    os.makedirs(plots_fld, exist_ok=True)
                                    pdf_file = f"{plots_fld}/{fit_type}_tm-{tm_ansatz}_OS-{OS_ansatz}.svg"
                                    # print(pdf_file)
                                    plt.savefig(pdf_file)
                                    # plt.show()
                                    plt.close()
#------------------------------


