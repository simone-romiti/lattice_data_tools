print("""
      -------------------------------------
      Plots for the autocorrelation of V(t)
      -------------------------------------
      """
      )

import math
import pandas as pd
import os
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt

from lattice_data_tools.dictionaries import NestedDict
from lattice_data_tools import uwerr
from lattice_data_tools.bootstrap import BootstrapSamples, auto_binning, binning, correlated_confs_to_bts, uncorrelated_confs_to_bts
from lattice_data_tools.io import with_dill, with_yaml

ens_info = with_yaml.load("ensembles.yaml")
aux_dir = ens_info["data"]["auxiliary"]
plots_dir = ens_info["data"]["plots"]

ens_list = ens_info["data"]["ens_list"]
N_bts = ens_info["N_bts"]
RNG_seed=ens_info["RNG_seed"]
np.random.seed(RNG_seed)


f_pi_list = ens_info["data"]["f_pi_MeV"]
for f_pi in f_pi_list:
    print(f"f_pi: {f_pi}")
    VKVK_bts = NestedDict()
    fpi_suffix = f"fpi_{f_pi}_MeV"
    fpi_dir = f"{aux_dir}/{fpi_suffix}/"
    VKVK_confs = with_dill.load(f"{fpi_dir}/VKVK_confs.pkl")
    for ens_name in ens_list:
        print(" Ensemble: ", ens_name)
        T = ens_info[ens_name]["T"]
        T_ext = T // 2 +1
        times = np.arange(0, T_ext)
        for fermion_type in ["OS", "tm"]:
            print("  Fermions: ", fermion_type)
            corr_confs = np.average(VKVK_confs[ens_name][fermion_type], axis=0) # average over the stochastic sources
            Ng = corr_confs.shape[0]
            # 1st plot: autocorrelation as a function of time
            tau_int, dtau_int = [], []
            for t in times:
                uwerr_res = uwerr.uwerr_primary(corr_confs[:,t])
                tau_int.append(uwerr_res["tauint"])
                dtau_int.append(uwerr_res["dtauint"])
            #---
            pdf_fld = f"./{plots_dir}/{fpi_suffix}/VKVK/tau_int/"
            os.makedirs(pdf_fld, exist_ok=True)
            plt.errorbar(x=times[1:], y=tau_int[1:], yerr=dtau_int[1:], marker="^", capsize=1, linestyle="None")
            plt.yscale("log")
            plt.title("tau_int over time")
            plt.xlabel("$t/a$")
            plt.ylabel("$\\tau_\\mathrm{int}$")
            plt.tight_layout()
            plt.savefig(f"{pdf_fld}/{ens_name}-{fermion_type}-t_VS_tauint.svg")
            plt.close()
            # 2nd plot: for each time, tau_int as a function of bin size
            for t in times:
                tau0 = math.ceil(uwerr.uwerr_primary(corr_confs[:,t])["tauint"])
                Cg_bts = uncorrelated_confs_to_bts(x=corr_confs[:,t], N_bts=N_bts, seed=RNG_seed) # computing bootstraps over uncorrelated configurations
                bin_arr, err_corr = [1], [Cg_bts.error()]
                bin_optimal = 1
                i = 1
                tauint_i, dtauint_i = 1.0, 0.0
                n_extra, n_extra_flag = 4, False # extra point after optimal bin size
                while n_extra >= 0:
                    bin_size = math.ceil(i * tau0) # updating bin size
                    Cg_binned = binning(Cg=corr_confs[:,t], bin_size=bin_size) # binning the correlator
                    uwerr_res = uwerr.uwerr_primary(Cg_binned)
                    tauint_i = uwerr_res["tauint"]
                    dtauint_i = uwerr_res["dtauint"]
                    if tauint_i-dtauint_i <= 0.5 and n_extra_flag==False:
                        bin_optimal = bin_size
                        n_extra_flag = True #  start decreasing n_extra 
                    #---
                    Cg_bts = uncorrelated_confs_to_bts(x=Cg_binned, N_bts=N_bts, seed=RNG_seed) # computing bootstraps
                    bin_arr.append(bin_size)
                    err_corr.append(Cg_bts.error())
                    i += 1
                    if n_extra_flag:
                        n_extra -= 1
                    #---
                #---
                plt.scatter(x=bin_arr, y=err_corr, marker="o", label="$\\sigma$", color="red")
                plt.vlines(x=(bin_optimal), ymin=0.0, ymax=max(err_corr), label=f"optimal bin size = {bin_optimal}", color="green") 
                plt.title(f"$t={t}$ : correlator error as a function of bin size")
                plt.xlabel("bin size")
                plt.ylabel("$\\Delta C(t)$")
                plt.legend()
                plt.tight_layout()
                pdf_fld_t = f"./{plots_dir}/{fpi_suffix}/VKVK/binning/{ens_name}/{fermion_type}/"
                os.makedirs(pdf_fld_t, exist_ok=True)
                plt.savefig(f"{pdf_fld_t}/t{t}.svg")
                plt.close()
#-------




