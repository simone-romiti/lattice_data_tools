print("""
      -------------------------------------------------
      B ensembles interpolation to the reference volume
      -------------------------------------------------
      """
      )



import numpy as np
import matplotlib.pyplot as plt
import os

from lattice_data_tools.bootstrap import BootstrapSamples
from lattice_data_tools.io import with_dill, with_yaml
import lattice_data_tools.constants as constants
from lattice_data_tools.dictionaries import NestedDict

ens_info = with_yaml.load("ensembles.yaml")
aux_dir = ens_info["data"]["auxiliary"]
plots_dir = ens_info["data"]["plots"]

ens_names = ens_info["data"]["ens_list"]
N_bts = ens_info["N_bts"]

Mpi_iso_MeV = constants.masses_MeV["pi"]["isoQCD"]["Edinburgh"]

L_ref_fm = ens_info["L_ref_fm"]
MpiL_ref = constants.MeV_to_fm_inv(Mpi_iso_MeV)*L_ref_fm

B64_name = "cB.72.64"
B96_name = "cB.72.96"

def fL(L, ML):
    return L*np.exp(-ML)
#---

f_pi_list = ens_info["data"]["f_pi_MeV"]
for f_pi in f_pi_list:
    print(f"f_pi:{f_pi}")
    fpi_suffix = f"fpi_{f_pi}_MeV"
    fpi_dir = f"{aux_dir}/{fpi_suffix}/"

    a_fm_dict = with_dill.load(f'{fpi_dir}/a_fm.pkl') # lattice spacing bootstraps
    L_B64 = ens_info[B64_name]["L"]*a_fm_dict[B64_name].mean()
    L_B96 = ens_info[B96_name]["L"]*a_fm_dict[B96_name].mean()

    MpiL_B64 = Mpi = constants.MeV_to_fm_inv(Mpi_iso_MeV)*L_B64
    MpiL_B96 = Mpi = constants.MeV_to_fm_inv(Mpi_iso_MeV)*L_B96

    a_mu_bounding_bts_dict = with_dill.load(f'{fpi_dir}/a_mu_bounding.pkl') # a_mu for each bounding method and window

    e_short_list = []
    a_fm_Lref = NestedDict()
    a_mu_Lref_dict = NestedDict()
    for ens_name in [e for e in ens_names if e not in [B64_name, B96_name]]:
        print("Ensemble:", ens_name)
        e_short = ens_name.split(".")[0][-1]
        e_short_list.append(e_short)
        a_fm_Lref[e_short] = a_fm_dict[ens_name]
        for fermion_type in ["tm", "OS"]:
            print(" Fermions:", fermion_type)
            t_thr_keys = list(a_mu_bounding_bts_dict[ens_name][fermion_type]["pp"].keys())
            windows_keys = list(a_mu_bounding_bts_dict[B96_name][fermion_type]["pp"].keys())
            for window_key in windows_keys:
                print("   Window_key:", window_key)
                t_thr_keys = list(a_mu_bounding_bts_dict[B96_name][fermion_type]["pp"][window_key].keys())
                for t_thr_key in t_thr_keys:
                    print(f"  {t_thr_key}")
                    t_cut_strategies = list(a_mu_bounding_bts_dict[B96_name][fermion_type]["pp"][window_key][t_thr_key].keys())
                    for t_cut_strategy in t_cut_strategies:
                        dt_plateau_strategies = list(a_mu_bounding_bts_dict[B96_name][fermion_type]["pp"][window_key][t_thr_key][t_cut_strategy].keys())
                        for dt_plateau_key in dt_plateau_strategies:
                            print(f"     {dt_plateau_key}")
                            a_mu_Lref_dict[e_short][fermion_type][window_key][t_thr_key][t_cut_strategy][dt_plateau_key] = a_mu_bounding_bts_dict[ens_name][fermion_type]["pp"][window_key][t_thr_key][t_cut_strategy][dt_plateau_key]["a_mu_fit"]
    #-----------------------

    e_short_list.append("B")
    a_mu_Lref_dict["ens_list"] = sorted(e_short_list)
    a_fm_Lref["B"] = a_fm_dict["cB.72.64"] # it's the same for cB.72.96


    print("=== interpolating ===")



    ens_ref_keys = "C" # ensemble used as a reference for the dictionary keys (also "D" and "E") are valid options
    for fermion_type in ["tm", "OS"]:
        print(" Fermions:", fermion_type)
        window_keys_keys = list(a_mu_Lref_dict[ens_ref_keys][fermion_type].keys())
        for window_key in window_keys_keys:
            print("   Window_key:", window_key)
            plots_fld=f"./{plots_dir}/{fpi_suffix}/volume_interpolation-B_ensembles/{window_key}/{fermion_type}/"
            os.makedirs(plots_fld, exist_ok=True)
            t_thr_keys = list(a_mu_Lref_dict[ens_ref_keys][fermion_type][window_key].keys())
            for t_thr_key in t_thr_keys:
                print(f"  {t_thr_key}")
                t_cut_strategies = list(a_mu_Lref_dict[ens_ref_keys][fermion_type][window_key][t_thr_key].keys())
                for t_cut_strategy in t_cut_strategies:
                    print(f"   {t_cut_strategy}") 
                    dt_plateau_strategies = list(a_mu_Lref_dict[ens_ref_keys][fermion_type][window_key][t_thr_key][t_cut_strategy].keys())
                    for dt_plateau_key in dt_plateau_strategies:
                        print(f"     {dt_plateau_key}")
                        # applying the mistuning correction to B96
                        a_mu_B96_sim = a_mu_bounding_bts_dict[B96_name][fermion_type]["sim"][window_key][t_thr_key][t_cut_strategy][dt_plateau_key]["a_mu_fit"]
                        a_mu_B64_sim = a_mu_bounding_bts_dict[B64_name][fermion_type]["sim"][window_key][t_thr_key][t_cut_strategy][dt_plateau_key]["a_mu_fit"]
                        a_mu_B64_pp = a_mu_bounding_bts_dict[B64_name][fermion_type]["pp"][window_key][t_thr_key][t_cut_strategy][dt_plateau_key]["a_mu_fit"]
                        mistuning_a_mu_B96 = a_mu_B64_pp - a_mu_B64_sim # we impose it to be the same as B64 (approximation, we don't have the mistuning data for B96)
                        a_mu_B96_pp = a_mu_B96_sim + mistuning_a_mu_B96

                        # determining A, where we assume a_mu(L) = a_mu(infinity) + A*L*exp(-Mpi*L) 
                        A = (a_mu_B96_pp - a_mu_B64_pp)/(fL(L_B96, MpiL_B96) - fL(L_B64, MpiL_B64))  # Slope of the linear interpolation
                        a_mu_Lref_dict["B_interpolation"][fermion_type][window_key][t_thr_key][t_cut_strategy][dt_plateau_key]["A_coeff"] = A

                        # finding a_\mu at the reference volume    
                        a_mu_B_Lref = a_mu_B64_pp + A*(fL(L_ref_fm, MpiL_ref) - fL(L_B64, MpiL_B64)) # Interpolated value at the reference volume
                        a_mu_Lref_dict["B"][fermion_type][window_key][t_thr_key][t_cut_strategy][dt_plateau_key] = a_mu_B_Lref

                        # print(0.9*(a_mu_B_Lref - a_mu_B64_pp).mean(), 0.9*(a_mu_B_Lref - a_mu_B64_pp).error())

                        # Plotting B64, B96, interpolation, and Lref point
                        L_values = np.linspace(L_B64, L_B96, 100)
                        MpiL_values = constants.MeV_to_fm_inv(Mpi_iso_MeV) * L_values
                        a_mu_curve = BootstrapSamples(a_mu_B64_pp[:,np.newaxis] + A[:,np.newaxis] * (fL(L_values, MpiL_values) - fL(L_B64, MpiL_B64))[np.newaxis,:])

                        plt.figure(figsize=(7, 5))
                        plt.fill_between(x=L_values, y1=a_mu_curve.unbiased_mean() + a_mu_curve.error(), y2=a_mu_curve.unbiased_mean() - a_mu_curve.error(), label="Interpolation", color="green", alpha=0.3)
                        plt.errorbar(
                            x=[L_B64, L_B96], y=[a_mu_B64_pp.unbiased_mean(), a_mu_B96_pp.unbiased_mean()], yerr=[a_mu_B64_pp.error(), a_mu_B96_pp.error()], 
                            capsize=2, marker="o", markersize=3, mfc='w', linestyle="None", color="red", label="B64 & B96")
                        plt.errorbar(
                            x=[L_ref_fm], y=[a_mu_B_Lref.unbiased_mean()], yerr=[a_mu_B_Lref.error()], 
                            capsize=2, marker="o", markersize=3, mfc='w', linestyle="None", color="green", label="Interpolated at Lref", zorder=5)
                        plt.xlabel("L [fm]")
                        plt.ylabel("$a_\\mu$")
                        plt.title(f"Volume interpolation for {fermion_type}, {window_key} window (just mean values)")
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(f"{plots_fld}/{fermion_type}-{t_thr_key}-{t_cut_strategy}-{dt_plateau_key}.svg")
                        # plt.show()
                        plt.close()
    #---------------



    with_dill.dump(a_fm_Lref, f'{fpi_dir}/a_fm_Lref.pkl')
    with_dill.dump(a_mu_Lref_dict, f'{fpi_dir}/a_mu_Lref.pkl')
