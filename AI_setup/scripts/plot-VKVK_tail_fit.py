print("""
      ----------------------
      Plotting the V(t) tail
      ----------------------
      """
      )

import numpy as np
import matplotlib.pyplot as plt
import os

from lattice_data_tools.io import with_dill, with_yaml
# from lattice_data_tools.bootstrap import BootstrapSamples

ens_info = with_yaml.load("ensembles.yaml")
aux_dir = ens_info["data"]["auxiliary"]
plots_dir = ens_info["data"]["plots"]

ens_names = ens_info["data"]["ens_list"]
N_bts = ens_info["N_bts"]

windows = ens_info["windows"]
fermion_types = ens_info["fermion_types"]


f_pi_list = ens_info["data"]["f_pi_MeV"]
for f_pi in f_pi_list:
    print(f"f_pi:{f_pi}")
    fpi_suffix = f"fpi_{f_pi}_MeV"
    fpi_dir = f"{aux_dir}/{fpi_suffix}/"
    os.makedirs(fpi_dir, exist_ok=True)
    #
    a_fm_bts_dict = with_dill.load(f"{fpi_dir}/a_fm.pkl")
    VKVK_eff_curves = with_dill.load(f'{fpi_dir}/VKVK_eff_curves.pkl') # effective mass of the Vector meson
    VKVK_tail_model_avg = with_dill.load(f'{fpi_dir}/VKVK_tail_model_avg.pkl')

    plots_fld=f"./{plots_dir}/{fpi_suffix}/VKVK/effective/"
    os.makedirs(plots_fld, exist_ok=True)

    for ens_name in ens_names:
        print("Ensemble:", ens_name)
        a_fm_mean = a_fm_bts_dict[ens_name].mean()
        T = ens_info[ens_name]["T"]
        T_half = int(T / 2)
        L = ens_info[ens_name]["L"]
        ti = np.arange(0, T_half + 1)
        for fermion_type in ["tm", "OS"]:
            print(" Fermions:", fermion_type)

            for corr in ["sim", "pp"]:
                if corr == "sim" and ens_name[0:2] != "cB":
                    continue
                #---
                # Compute mean and bootstrap error over the samples
                res_dict = VKVK_eff_curves[ens_name][fermion_type][corr]
                tmin_fm = ens_info["MV"][fermion_type]["t_min_fm"]
                tmax_fm = ens_info["MV"][fermion_type]["t_max_fm"]
                tmin = int(tmin_fm / a_fm_mean)
                tmax = int(tmax_fm / a_fm_mean)
                tL = tmin-4 # left bound for plot
                tR = tmax+3 # right bound for plot
                dt_plateau = int(ens_info["MV"][fermion_type]["dt_plateau_fm"] / a_fm_mean)
                Nd = 100
                t_fit=np.linspace(tmin, tmax, Nd)
                for X in ["MV"]: # ["MV", "A"]:
                    # print(f"X = {X}")
                    X_TeX = "M_V" if X=="MV" else X
                    X_dict = res_dict[f"{X}"]
                    fig, ax = plt.subplots()
                    X_eff = X_dict["eff"]  # Shape: (N_bts, T_ext)
                    X_eff_mean = X_eff.unbiased_mean()
                    X_eff_err = X_eff.error()
                    ax.errorbar(x=ti[tL:tR], y=X_eff_mean[tL:tR], yerr=X_eff_err[tL:tR], capsize=2, marker="o", markersize=3, mfc='w', linestyle="None", label="Eff. mass")
                    for fit_type in ["correlated_fit", "uncorrelated_fit"]:
                        X_AIC = VKVK_tail_model_avg[ens_name][fermion_type][corr][f"{X}"][fit_type]

                        X_fit_mean = X_AIC.unbiased_mean()
                        X_fit_err = X_AIC.error()
                        X_fit_down = np.array([X_fit_mean-X_fit_err for t in t_fit])
                        X_fit_up = np.array([X_fit_mean+X_fit_err for t in t_fit])
                        
                        ax.fill_between(
                            x=t_fit, y1=X_fit_down, y2=X_fit_up, 
                            alpha=0.3, label=f"{fit_type}: ${X_TeX}={X_fit_mean:.4e} \\pm {X_fit_err:.4e}$"
                            )
                    #---
                    fm_to_lat = lambda t_fm: (t_fm/a_fm_mean)
                    lat_to_fm = lambda t_lat: (t_lat*a_fm_mean)
                    secax = ax.secondary_xaxis('top', functions=(lat_to_fm, fm_to_lat))
                    secax.set_xlabel('$t$ [fm]', fontsize=15)

                    # Adjust tick locations for readability
                    secax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))  # up to ~6 ticks to avoid overlap
                    secax.tick_params(axis='x', labelrotation=0)                
                    ax.set_xlabel('$t/a$', fontsize=15)
                    ax.set_ylabel(f'${X_TeX}^\\mathrm{{eff}}(t)$', fontsize=15)
                    ax.set_title(f'{ens_name}-{fermion_type}')
                    ax.grid(True)
                    ax.legend()

                    plt.tight_layout()
                    pdf_file = f"{plots_fld}/{X}_eff-{ens_name}_{fermion_type}_{corr}.svg"
                    # print(pdf_file)
                    plt.savefig(pdf_file)
                    # plt.show()
                    plt.close()

