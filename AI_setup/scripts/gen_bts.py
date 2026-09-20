print("""
      -------------------------------------
      Bootstrap samples generation for V(t)
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
RNG_seed = ens_info["RNG_seed"]
np.random.seed(RNG_seed)

print("# Generating the bootstraps of V(t)")

# t_dict = {"cB.72.64": 25 , "cB.72.96": 25, "cC.06.80": 30, "cC.06.112": 30, "cD.54.96": 35, "cE.44.112": 40}

f_pi_list = ens_info["data"]["f_pi_MeV"]
for f_pi in f_pi_list:
    print(f"f_pi: {f_pi}")
    fpi_suffix = f"fpi_{f_pi}_MeV"
    VKVK_confs = with_dill.load(f"{aux_dir}/{fpi_suffix}/VKVK_confs.pkl")
    VKVK_bts = NestedDict()
    opt_binsize_dict = dict({"ensemble": [], "fermions": [], "Ng": [], "bin_size": [], "N_bins": []})
    for ens_name in ens_list:
        print("Ensemble: ", ens_name)
        T = ens_info[ens_name]["T"]
        T_ext = T // 2 + 1
        times = np.arange(0, T_ext)
        for fermion_type in ["OS", "tm"]:
            print("  Fermions: ", fermion_type)
            corr_confs = np.average(VKVK_confs[ens_name][fermion_type], axis=0) # average over the stochastic sources
            Ng = corr_confs.shape[0]
            tau_int = []
            pdf_fld = f"./{plots_dir}/{fpi_suffix}/VKVK/MC_history/{ens_name}/{fermion_type}/"
            os.makedirs(pdf_fld, exist_ok=True)
            #
            # TEST: assuming uncorrelated confs
            # VKVK_uncorr = BootstrapSamples([uncorrelated_confs_to_bts(corr_confs[:,t], N_bts=N_bts, seed=RNG_seed) for t in times]).transpose()
            #
            # OUTDATED: automatic procedure to find the optimal bin size
            # # NOTE: t=0 is not used in the analysis, there is no need to use its value to determine the optimal bin size
            # for t in times[1:]:
            #     # tau_int.append(uwerr.uwerr_primary(corr_confs[:,t])["tauint"])
            #     tau_int.append(uwerr.uwerr_primary(corr_confs[:,t], output_file=f"{pdf_fld}/MC_history-t{t}.svg")["tauint"])
            # #---
            # t_taumax = 1+np.argmax(tau_int) # time where tau_int is maximum
            # N_bins_optimal = auto_binning(corr_confs[:,t_taumax]).shape[0]
            # optimal_bin_size = Ng // N_bins_optimal
            # VKVK_corr = BootstrapSamples([correlated_confs_to_bts(Cg=corr_confs[:,t], N_bts=N_bts, seed=RNG_seed, bin_size=optimal_bin_size) for t in times]).transpose()
            # opt_binsize_dict["t_taumax"].append(t_taumax)
            # opt_binsize_dict["optimal_binsize"].append(N_bins_optimal)
            # opt_binsize_dict["N_bins"].append(N_bins_optimal)
            #
            bin_size = ens_info[ens_name]["bin_size"][fermion_type]
            N_bins = Ng//bin_size + (1 - int(Ng%bin_size == 0))
            VKVK_corr = BootstrapSamples([correlated_confs_to_bts(Cg=corr_confs[:,t], N_bts=N_bts, seed=RNG_seed, bin_size=bin_size) for t in times]).transpose()
            VKVK_bts[ens_name][fermion_type] = VKVK_corr
            #
            opt_binsize_dict["ensemble"].append(ens_name)
            opt_binsize_dict["fermions"].append(fermion_type)
            opt_binsize_dict["Ng"].append(Ng)
            opt_binsize_dict["bin_size"].append(bin_size)
            opt_binsize_dict["N_bins"].append(N_bins)
            pdf_fld = f"./{plots_dir}/{fpi_suffix}/VKVK/"
            os.makedirs(pdf_fld, exist_ok=True)
            pd.DataFrame(opt_binsize_dict).to_csv(f"./{plots_dir}/{fpi_suffix}/VKVK/binning.csv")

            with_dill.dump(VKVK_bts, f"{aux_dir}/{fpi_suffix}/VKVK_bts.pkl")
#-------




