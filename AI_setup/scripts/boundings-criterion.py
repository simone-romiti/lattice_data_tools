print("""
      ----------------------------------------
      Bounding method: find time region to fit 
      ----------------------------------------
      """
      )

import numpy as np
import matplotlib.pyplot as plt
import os

from lattice_data_tools.bootstrap import BootstrapSamples
from lattice_data_tools.dictionaries import NestedDict
from lattice_data_tools.io import with_dill, with_yaml

# Loading input files

ens_info = with_yaml.load("ensembles.yaml")

ens_names = ens_info["data"]["ens_list"]
aux_dir = ens_info["data"]["auxiliary"]
plots_dir = ens_info["data"]["plots"]
N_bts = ens_info["N_bts"]

windows = ens_info["windows"]
fermion_types = ens_info["fermion_types"]

N_int = ens_info["N_int_QED_kernel"]



# number of sigmas of difference for the matching point of the correlated difference and error on the mean
t_cut_strategy_dict =  {k: v for k, v in ens_info["bounding"]["tcut_strategy"].items()} # {"aggressive_t_cut": 0.0, "moderate_t_cut": 1.0, "conservative_t_cut": 2.0}

t_cut_fm_list = {fermion_type: {window: {t_cut_strategy: [] for t_cut_strategy in t_cut_strategy_dict.keys()} for window in windows} for fermion_type in fermion_types}
N_t_cut = 0

dt_fm_plateau = ens_info["bounding"]["dt_fm"]


f_pi_list = ens_info["data"]["f_pi_MeV"]
for f_pi in f_pi_list:
    print(f"f_pi:{f_pi}")
    fpi_suffix = f"fpi_{f_pi}_MeV"
    fpi_dir = f"{aux_dir}/{fpi_suffix}/"

    a_fm_dict = with_dill.load(f'{fpi_dir}/a_fm.pkl') # lattice spacing bootstraps
    a_mu_eff_dict = {b: with_dill.load(f'{fpi_dir}/a_mu-{b}.pkl') for b in ["ZeroTail", "2pions", "MV_tail"]}

    plots_fld=f"./{plots_dir}/{fpi_suffix}/boundings/criterion/"
    os.makedirs(plots_fld, exist_ok=True)

    boundings_criterion = NestedDict()
    # boundings_fit = ens_info["bounding"]["methods_fit"]
    for ens_name in ens_names:
        print(" Ensemble:", ens_name)
        a_fm = a_fm_dict[ens_name]
        a_fm_mean = a_fm.mean()
        T = ens_info[ens_name]["T"]
        T_half = int(T/2)
        L = ens_info[ens_name]["L"] 
        ti = np.arange(0, T_half+1)
        dt_fm = ens_info["bounding"]["dt_fm"]
        for fermion_type in fermion_types:
            print(" Fermions:", fermion_type)

            print("# Loading the correlator, and including charge factors")
            for corr in ["sim", "pp"]:
                if corr == "sim" and ens_name[0:2] != "cB":
                    continue
                #---
                print(f"  Correction: {corr}")
                for window in windows:
                    print("   Window:", window)
                    b0, b1 = "2pions", "MV_tail" # boundings_fit
                    # a_mu_ZeroTail = a_mu_ZeroTail_dict[ens_name][fermion_type][corr][f"{window}_window"]["ZeroTail"]     
                    a_mu_upper = a_mu_eff_dict[b0][ens_name][fermion_type][corr][f"{window}_window"]
                    t_thr_keys = list(a_mu_eff_dict[b1][ens_name][fermion_type][corr].keys())
                    for t_thr_key in t_thr_keys:
                        print(f"    {t_thr_key}")
                        a_mu_lower = a_mu_eff_dict[b1][ens_name][fermion_type][corr][t_thr_key][f"{window}_window"]
                        a_mu_between = (a_mu_lower+a_mu_upper)/2.0
                        # finding the t_start of the fit to a constant. NOTE: it must be upper-lower bound, NOT viceversa
                        delta_a_mu = BootstrapSamples((a_mu_upper - a_mu_lower)) 
                        assert( not np.isnan(delta_a_mu).any()) # Check for NaNs in delta_a_mu
                        
                        # avoiding considering t=0 --> we need to add 1 to the resulting index
                        t_cut_dict = NestedDict()
                        for t_cut_strategy in t_cut_strategy_dict.keys():
                            print(f"     {t_cut_strategy}")
                            n_sigma = t_cut_strategy_dict[t_cut_strategy]
                            t_cut = np.where((delta_a_mu.mean()+n_sigma*delta_a_mu.error())[1:] < a_mu_between.error()[1:])[0][0] + 1
                        
                            t_cut_fm_list[fermion_type][window][t_cut_strategy].append(t_cut*a_fm_mean)
                            N_t_cut += 1
                            
                            t_end_plot = t_cut + int(0.8/a_fm_mean) # empirical value for good plot visualization
                            dt_L = int(0.5/a_fm_mean) # empirical value
                            dt_R = int(0.3/a_fm_mean) # empirical value
                        
                            fig, ax = plt.subplots()
                            ax.errorbar(
                                x=ti[(t_cut-dt_L):(t_end_plot+dt_R)], y=a_mu_between.error()[(t_cut-dt_L):(t_end_plot+dt_R)],
                                label="Error on avg.", marker="o", markersize=3, mfc='w', linestyle="None")
                            ax.errorbar(
                                x=ti[(t_cut-dt_L):(t_end_plot+dt_R)], y=delta_a_mu.mean()[(t_cut-dt_L):(t_end_plot+dt_R)], yerr=delta_a_mu.error()[(t_cut-dt_L):(t_end_plot+dt_R)],
                                label="Correlated difference", capsize=2, marker="o", markersize=3, mfc='w', linestyle="None")
                            ax.vlines(
                                x=[t_cut], ymin=np.min(delta_a_mu[:,(t_cut-dt_L):(t_end_plot+dt_R)]), ymax=np.max(delta_a_mu[:,(t_cut-dt_L):(t_end_plot+dt_R)]),
                                color="red", label="$t_\\mathrm{start}$")
                            ax.hlines(y=0.0, xmin=t_cut-dt_L, xmax=t_end_plot+dt_R, color="pink")
                            ax.set_title(f"Boundings: {b0}, {b1}")

                            secax = ax.secondary_xaxis('top', functions=(lambda t_lat: (t_lat*a_fm_mean), lambda t_fm: (t_fm/a_fm_mean)))
                            secax.set_xlabel('$t$ [fm]', fontsize=15)

                            # Adjust tick locations for readability
                            secax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))  # up to ~6 ticks to avoid overlap
                            secax.tick_params(axis='x', labelrotation=0)                
                            ax.set_xlabel('$t/a$', fontsize=15)

                            ax.set_ylabel("$\\Delta a_\\mu(t)$")
                            ax.set_xlabel("$t/a$")
                            ax.legend()
                            plt.tight_layout()
                            plt.savefig(f"{plots_fld}/{ens_name}_{fermion_type}-{corr}-{window}-{t_thr_key}-{t_cut_strategy}-{b0}_{b1}.svg")
                            # plt.show()
                            plt.close()
                            for dt_fm in dt_fm_plateau:
                                t_end = t_cut + int(dt_fm/a_fm_mean) # last point of the plateau
                                dt_fm_key = f"dt={dt_fm}[fm]"
                                print(f"      {dt_fm_key}")
                                boundings_criterion[ens_name][fermion_type][corr][f"{window}_window"][t_thr_key][t_cut_strategy][dt_fm_key]["t_cut_fit"] = t_cut
                                boundings_criterion[ens_name][fermion_type][corr][f"{window}_window"][t_thr_key][t_cut_strategy][dt_fm_key]["t_end_fit"] = t_end
    #---------------------------

    with_dill.dump(boundings_criterion, f'{fpi_dir}/boundings_criterion.pkl') # Save to pickle

    print("# Plotting the ranges of t_cut")

    colors = ["blue", "red", "green", "brown", "cyan", "orange", "purple", "pink", "olive", "gray"]
    linestyles = 2*['-', '--', '-.', ':']

    for t_cut_strategy in t_cut_strategy_dict.keys():
        fig, ax = plt.subplots(2,1)
        for fermion_type in fermion_types:
            ax_i = ax[0] if fermion_type=="tm" else ax[1]
            i_g = 0
            for window in windows:
                ti = t_cut_fm_list[fermion_type][window][t_cut_strategy]
                t_min, t_max = min(ti), max(ti)
                ax_i.axvspan(t_min, t_max, color=colors[i_g], alpha=0.15)
                ax_i.axvline(t_min, color=colors[i_g], linewidth=2, linestyle=linestyles[i_g], label=f"{window} window, $\\bar{{t}}_\\text{{cut}}={np.round(np.average(ti),2)}$ fm")
                ax_i.axvline(t_max, color=colors[i_g], linewidth=2, linestyle=linestyles[i_g])
                ax_i.set_yticks([])
                i_g += 1
            #---
            ax_i.set_title(f"{fermion_type} fermions")
            ax_i.legend()
            ax_i.set_xlabel("$t$ [fm]")
        #-------

        fig.suptitle("Range of $t_\\text{cut}$: start of the fit for the bounding method")
        plt.tight_layout()
        # plt.show()
        os.makedirs(f"./{plots_dir}/{fpi_suffix}/boundings/t_cut/", exist_ok=True)
        plt.savefig(f"./{plots_dir}/{fpi_suffix}/boundings/t_cut/{t_cut_strategy}-distribution.svg")
        plt.close()
    #---
#---
