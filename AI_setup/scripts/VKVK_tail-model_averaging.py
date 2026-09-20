print("""
      -------------------------------
      V(t) tail fits: model averaging
      -------------------------------
      """
      )


""" Effective mass for the Vector correlator and its fit """

import numpy as np
import matplotlib.pyplot as plt
import os


from lattice_data_tools.bootstrap import BootstrapSamples, parametric_gaussian_bts
from lattice_data_tools.dictionaries import NestedDict
from lattice_data_tools.io import with_dill, with_yaml
import lattice_data_tools.model_averaging.IC as IC
from lattice_data_tools.plotting.with_matplotlib.IC import FromBootstraps as IC_from_bts

# Loading input files

ens_info = with_yaml.load("ensembles.yaml")

ens_names = ens_info["data"]["ens_list"]
aux_dir = ens_info["data"]["auxiliary"]
plots_dir = ens_info["data"]["plots"]

N_bts = ens_info["N_bts"]
RNG_seed = ens_info["RNG_seed"] 

windows = ens_info["windows"]
fermion_types = ens_info["fermion_types"]



f_pi_list = ens_info["data"]["f_pi_MeV"]
for f_pi in f_pi_list:
    print(f"f_pi:{f_pi}")
    fpi_suffix = f"fpi_{f_pi}_MeV"
    fpi_dir = f"{aux_dir}/{fpi_suffix}/"
    os.makedirs(fpi_dir, exist_ok=True)
    #
    plots_fld=f"./{plots_dir}/{fpi_suffix}/VKVK/AIC_on_MV/"
    os.makedirs(plots_fld, exist_ok=True)

    VKVK_fit_dict = with_dill.load(f'{fpi_dir}/VKVK_fit_tail.pkl')
    VKVK_tail_model_avg = NestedDict()
    for ens_name in ens_names:
        print("Ensemble:", ens_name)
        for fermion_type in fermion_types:
            print(" Fermions:", fermion_type)
            for corr in ["sim", "pp"]:
                if corr == "sim" and ens_name[0:2] != "cB":
                    continue
                #---
                print(f"  Correction: {corr}")
                plateaus = VKVK_fit_dict[ens_name][fermion_type][corr]["plateaus"]
                n_plateaus = len(plateaus)
                n_pts = np.array([VKVK_fit_dict[ens_name][fermion_type][corr][i_plat]["n_pts"] for i_plat in range(n_plateaus)])
                for X in ["MV"]: # ["MV", "A"]:
                    print(f"    X : {X}")
                    for fit_type in ["correlated_fit", "uncorrelated_fit"]:
                        print(f"     Fit type: {fit_type}")
                        X_dict = VKVK_fit_dict[ens_name][fermion_type][corr]
                        X_fit = BootstrapSamples([X_dict[i_plat][f"{X}"][fit_type]["fit"] for i_plat in range(n_plateaus)]).T
                        X_ch2 = BootstrapSamples([X_dict[i_plat][f"{X}"][fit_type]["ch2_fit"].unbiased_mean() for i_plat in range(n_plateaus)]).T
                        n_par = np.full(shape=(n_plateaus), fill_value=1) # fit to a constant
                        w = IC.get_weights(ch2=X_ch2, n_par=n_par, n_data=n_pts, IC="AIC")
                        res_AIC = IC.get_P_from_bootstraps(y = X_fit, w=w, lam=1.0)
                        y = res_AIC["y"]
                        P = res_AIC["P"]
                        # plotting the AIC
                        fig, ax = IC_from_bts.plot_cdf(y=y, P=P)
                        plt.tight_layout()
                        fig.savefig(f"{plots_fld}/{X}_AIC-{ens_name}-{fermion_type}-{fit_type}.svg")
                        plt.close()
                        Q = IC.with_CDF.get_quantiles(y=y, P=P)
                        y16 = Q["16%"]
                        y50 = Q["50%"]
                        y84 = Q["84%"]
                        X_mean = y50
                        X_err_up = y84 - y50
                        X_err_down = y50 - y16
                        X_err = (y84 - y16)/2
                        X_bts_model_avg = parametric_gaussian_bts(mean=X_mean, error=X_err, N_bts=N_bts, seed=RNG_seed)
                        VKVK_tail_model_avg[ens_name][fermion_type][corr][f"{X}"][fit_type] = BootstrapSamples(X_bts_model_avg)
    #---------------

    with_dill.dump(VKVK_tail_model_avg, f'{fpi_dir}/VKVK_tail_model_avg.pkl')
#---

