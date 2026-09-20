print("""
      ---------------------------------
      Plotting: a_\\mu boundings curves
      ---------------------------------
      """
      )

import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
import os

from lattice_data_tools.io import with_dill, with_yaml
from lattice_data_tools.bootstrap import BootstrapSamples

ens_info = with_yaml.load("ensembles.yaml")
aux_dir = ens_info["data"]["auxiliary"]
plots_dir = ens_info["data"]["plots"]

ens_names = ens_info["data"]["ens_list"]
N_bts = ens_info["N_bts"]

windows = ens_info["windows"]
fermion_types = ens_info["fermion_types"]

boundings_names = ["ZeroTail", "2pions", "MV_tail"]

f_pi_list = ens_info["data"]["f_pi_MeV"]
for f_pi in f_pi_list:
    print(f"f_pi:{f_pi}")
    fpi_suffix = f"fpi_{f_pi}_MeV"
    fpi_dir = f"{aux_dir}/{fpi_suffix}/"

    a_fm_dict = with_dill.load(f'{fpi_dir}/a_fm.pkl') # lattice spacing bootstraps

    a_mu_bounding_bts_dict = with_dill.load(f'{fpi_dir}/a_mu_bounding.pkl') # a_mu for each bounding method and window
    boundings_criterion = with_dill.load(f"{fpi_dir}/boundings_criterion.pkl")

    a_mu_eff_dict = {b: with_dill.load(f'{fpi_dir}/a_mu-{b}.pkl') for b in boundings_names}

    for ens_name in ens_names:
        print("Ensemble:", ens_name)
        a_fm_mean = a_fm_dict[ens_name].mean()
        for fermion_type in ["tm", "OS"]:
            print(" Fermions:", fermion_type)

            T = ens_info[ens_name]["T"]
            T_half = int(T / 2)
            L = ens_info[ens_name]["L"]
            ti = np.arange(0, T_half + 1)

            for corr in ["pp", "sim"]:
                if corr == "sim" and ens_name[0:2] != "cB":
                    continue
                #---
                print(f"  Correction: {corr}")
                for window in windows:
                    print("    Window:", window)
                    t_thr_keys = list(a_mu_eff_dict["MV_tail"][ens_name][fermion_type][corr].keys())
                    for t_thr_key in t_thr_keys:
                        print(f"   {t_thr_key}")
                        t_cut_strategies = list(boundings_criterion[ens_name][fermion_type][corr][f"{window}_window"][t_thr_key].keys())
                        for t_cut_strategy in t_cut_strategies:
                            print(f"     {t_cut_strategy}")
                            dt_plateau_strategies = list(boundings_criterion[ens_name][fermion_type][corr][f"{window}_window"][t_thr_key][t_cut_strategy].keys())
                            for dt_plateau_key in dt_plateau_strategies:
                                print(f"     {dt_plateau_key}")
                                fig, ax = plt.subplots(figsize=(16,9))
                                if window in ["full", "LD"]:
                                    t_cut = boundings_criterion[ens_name][fermion_type][corr][f"{window}_window"][t_thr_key][t_cut_strategy][dt_plateau_key]["t_cut_fit"]
                                    t_end = boundings_criterion[ens_name][fermion_type][corr][f"{window}_window"][t_thr_key][t_cut_strategy][dt_plateau_key]["t_end_fit"]
                                    dt_L = int(0.5/a_fm_mean) # empirical value
                                    dt_R = int(0.3/a_fm_mean) # empirical value
                                    tmin = t_cut - dt_L
                                    tmax = t_end + dt_R
                                else:
                                    t_cut = T_half
                                    t_end = T_half+1 # empirical value (better visualization)
                                    tmin = t_end-52 # empirical value (better visualization)
                                    tmax = t_end 

                                # Top: a_mu bounding methods (effective curves of a_\mu)
                                for bounding_name in boundings_names:
                                    if bounding_name == "MV_tail":
                                        curve = a_mu_eff_dict[bounding_name][ens_name][fermion_type][corr][t_thr_key][f"{window}_window"]
                                    else:
                                        curve = a_mu_eff_dict[bounding_name][ens_name][fermion_type][corr][f"{window}_window"]
                                    #---
                                    ax.errorbar(
                                        x=ti[tmin:], y=BootstrapSamples(curve[:,tmin:]).unbiased_mean(), 
                                        yerr=BootstrapSamples(curve[:,tmin:]).error(),
                                        fmt='o', capsize=5, linestyle='--', label=bounding_name
                                        )
                                #---
                                # Plot a_mu_fit in the specified range
                                a_mu_fit = a_mu_bounding_bts_dict[ens_name][fermion_type][corr][f"{window}_window"][t_thr_key][t_cut_strategy][dt_plateau_key]["a_mu_fit"]
                                ax.errorbar(
                                    x=ti[t_cut:t_end],
                                    y=np.full_like(ti[t_cut:t_end], a_mu_fit.unbiased_mean(), dtype=np.float64),
                                    yerr=np.full_like(ti[t_cut:t_end], a_mu_fit.error(), dtype=np.float64),
                                    color='red',
                                    linestyle='-',
                                    linewidth=2,
                                    capsize=5,
                                    label=f'$a_\\mu \\times 10^{{10}}={np.round(1e+10*a_mu_fit.unbiased_mean(),2)} \\pm {np.round(1e+10*a_mu_fit.error(),2)}$'
                                )

                                fm_to_lat = lambda t_fm: (t_fm/a_fm_mean)
                                lat_to_fm = lambda t_lat: (t_lat*a_fm_mean)
                                secax = ax.secondary_xaxis('top', functions=(lat_to_fm, fm_to_lat))
                                secax.set_xlabel('$t$ [fm]', fontsize=15)

                                # Adjust tick locations for readability
                                secax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))  # up to ~6 ticks to avoid overlap
                                secax.tick_params(axis='x', labelrotation=0)                
                                ax.set_xlabel('$t/a$', fontsize=15)

                                ax.tick_params(direction='in')
                                ax.set_ylabel('$a_\\mu(\\ell)$', fontsize=15)
                                ax.set_title(f'{ens_name}-{fermion_type}-{window}_window')
                                ax.grid(True)
                                ax.legend()

                                plt.tight_layout()

                                plots_fld=f"./{plots_dir}/{fpi_suffix}/boundings/fit/{window}_window/{ens_name}_{fermion_type}_{corr}/"
                                os.makedirs(plots_fld, exist_ok=True)

                                plt.savefig(f"{plots_fld}/{t_thr_key}-{t_cut_strategy}-{dt_plateau_key}.svg")
                                #plt.show()
                                plt.close()

