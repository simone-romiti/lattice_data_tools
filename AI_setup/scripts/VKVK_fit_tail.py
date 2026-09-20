print("""
      ----------------
      Fitting the V(t)
      ----------------
      """
      )
""" Effective mass for the Vector correlator and its fit """

import numpy as np


from lattice_data_tools.bootstrap import BootstrapSamples
from lattice_data_tools.dictionaries import NestedDict
from lattice_data_tools.io import with_dill, with_yaml
from lattice_data_tools.fit.xyey import fit_xyey
from lattice_data_tools import statistics_tools

# Loading input files

ens_info = with_yaml.load("ensembles.yaml")

ens_names = ens_info["data"]["ens_list"]
aux_dir = ens_info["data"]["auxiliary"]
N_bts = ens_info["N_bts"]

windows = ens_info["windows"]
fermion_types = ens_info["fermion_types"]

f_pi_list = ens_info["data"]["f_pi_MeV"]
for f_pi in f_pi_list:
    print(f"f_pi:{f_pi}")
    fpi_suffix = f"fpi_{f_pi}_MeV"
    fpi_dir = f"{aux_dir}/{fpi_suffix}/"

    VKVK_eff_curves = with_dill.load(f'{fpi_dir}/VKVK_eff_curves.pkl')
    a_fm_dict = with_dill.load(f'{fpi_dir}/a_fm.pkl') # lattice spacing bootstraps

    VKVK_fit_tail = NestedDict()
    for ens_name in ens_names:
        print("Ensemble:", ens_name)
        for fermion_type in fermion_types:
            print(" Fermions:", fermion_type)
            a_fm = a_fm_dict[ens_name]
            T = ens_info[ens_name]["T"]
            T_half = int(T/2)
            L = ens_info[ens_name]["L"] 
            ti = np.arange(0, T_half+1)
            for corr in ["sim", "pp"]:
                if corr == "sim" and ens_name[0:2] != "cB":
                    continue
                #---
                print(f"  Correction: {corr}")
                MV_eff = VKVK_eff_curves[ens_name][fermion_type][corr]["MV"]["eff"]
                # A_eff  = VKVK_eff_curves[ens_name][fermion_type][corr]["A"]["eff"]
                t_min = int(ens_info["MV"][fermion_type]["t_min_fm"] / a_fm.mean())
                t_max = int(ens_info["MV"][fermion_type]["t_max_fm"] / a_fm.mean())
                dt_plateau = int(ens_info["MV"][fermion_type]["dt_plateau_fm"] / a_fm.mean())
                t_shift_plat = ens_info["MV"][fermion_type]["t_shift_lat"]
                plateaus = [
                    (t1, t2)
                    for t1 in range(t_min, t_max, t_shift_plat)
                    for t2 in range(t1 + dt_plateau, t_max, t_shift_plat)
                ]
                plateaus = [p for p in plateaus if len(p)>0] # only non-empty cases
                assert(len(plateaus) > 0)
                VKVK_fit_tail[ens_name][fermion_type][corr]["plateaus"] = plateaus
                print(f"# Loop over {len(plateaus)} plateau combinations")
                for i_plat, plateau in enumerate(plateaus):
                    t1, t2 = plateau
                    # print(i_plat, t1, t2)
                    VKVK_fit_tail[ens_name][fermion_type][corr][i_plat]["t1"] = t1
                    VKVK_fit_tail[ens_name][fermion_type][corr][i_plat]["t2"] = t2
                    VKVK_fit_tail[ens_name][fermion_type][corr][i_plat]["n_pts"] = (t2-t1)+1
                    for fit_type in ["correlated_fit", "uncorrelated_fit"]:
                        MV_Cov_inv = None
                        # AV_Cov_inv = None
                        if fit_type=="correlated_fit":
                            MV_Cov = statistics_tools.rooting(BootstrapSamples(MV_eff[:,t1:(t2+1)]).covariance_matrix())
                            MV_Cov_inv = np.linalg.inv(MV_Cov)
                            # A_Cov = BootstrapSamples(A_eff[:,t1:(t2+1)]).covariance_matrix()
                            # A_Cov_inv = np.linalg.inv(A_Cov)
                        #---
                        def fit_function(i, X_eff, X_Cov_inv):
                            res = fit_xyey(
                                ansatz=lambda x, p: p[0],
                                x=ti[t1:(t2+1)],
                                y=X_eff[i, t1:(t2+1)],
                                ey=X_eff.error()[t1:(t2+1)],
                                guess=[np.mean(X_eff.mean()[t1:(t2+1)])],
                                method="Nelder-Mead",
                                Cov_y_inv=MV_Cov_inv
                            )
                            return res
                        #---
                        MV_mini = BootstrapSamples.bts_list_from_lambda(N_bts=N_bts, fun=lambda i: fit_function(i, MV_eff, MV_Cov_inv), parallel=False)
                        MV_fit = BootstrapSamples.from_lambda(N_bts=N_bts, fun=lambda i: MV_mini[i]["par"][0], parallel=False)
                        VKVK_fit_tail[ens_name][fermion_type][corr][i_plat]["MV"][fit_type]["fit"] = MV_fit
                        VKVK_fit_tail[ens_name][fermion_type][corr][i_plat]["MV"][fit_type]["ch2_fit"] = BootstrapSamples.from_lambda(N_bts=N_bts, fun=lambda i: MV_mini[i]["ch2"], parallel=False)
                        VKVK_fit_tail[ens_name][fermion_type][corr][i_plat]["MV"][fit_type]["N_dof"] = MV_mini[0]["N_dof"]
                        VKVK_fit_tail[ens_name][fermion_type][corr][i_plat]["MV"][fit_type]["ch2_dof"] = BootstrapSamples.from_lambda(N_bts=N_bts, fun=lambda i: MV_mini[i]["ch2_dof"], parallel=False)
                        # print(f"[{t1}, {t2}] : ch2/dof, {fit_type}:", VKVK_fit_tail[ens_name][fermion_type][corr][i_plat]["MV"][fit_type]["ch2_dof"].unbiased_mean(),  VKVK_fit_tail[ens_name][fermion_type][corr][i_plat]["MV"][fit_type]["N_dof"], " | ", MV_fit.mean(), MV_fit.error())
                        # A_mini = BootstrapSamples.bts_list_from_lambda(N_bts=N_bts, fun=lambda i: fit_function(i, A_eff, A_Cov_inv))
                        # A_fit = BootstrapSamples.from_lambda(N_bts=N_bts, fun=lambda i: A_mini[i]["par"][0])
                        # VKVK_fit_tail[ens_name][fermion_type][corr][i_plat]["A"][fit_type]["fit"] = A_fit
                        # VKVK_fit_tail[ens_name][fermion_type][corr][i_plat]["A"][fit_type]["ch2_fit"] = BootstrapSamples.from_lambda(N_bts=N_bts, fun=lambda i: A_mini[i]["ch2"])
                        # VKVK_fit_tail[ens_name][fermion_type][corr][i_plat]["A"][fit_type]["ch2_dof"] = BootstrapSamples.from_lambda(N_bts=N_bts, fun=lambda i: A_mini[i]["ch2_dof"])
    #-------------------

    # Save to pickle
    with_dill.dump(VKVK_fit_tail, f'{fpi_dir}/VKVK_fit_tail.pkl')
#---

