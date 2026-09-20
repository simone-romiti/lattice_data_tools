print("""
      ---------------------------------------------
      Applying UV corrections (perturbation theory)
      ---------------------------------------------
      """
      )


import numpy as np
import pandas as pd

from lattice_data_tools.io import with_dill, with_yaml
from lattice_data_tools.dictionaries import NestedDict
# from lattice_data_tools.sampling import parametric_gaussian_bts

ens_info = with_yaml.load("ensembles.yaml")

raw_data_dir = ens_info["data"]["main_dir"]
aux_dir = ens_info["data"]["auxiliary"]
fermion_types = ens_info["fermion_types"]
windows = ens_info["windows"]

f_pi_list = ens_info["data"]["f_pi_MeV"]
for f_pi in f_pi_list:
    print(f"f_pi:{f_pi}")
    fpi_suffix = f"fpi_{f_pi}_MeV"
    fpi_dir = f"{aux_dir}/{fpi_suffix}/"


    a_mu_Lref = with_dill.load(f'{fpi_dir}/a_mu_Lref.pkl')

    e_short = a_mu_Lref["ens_list"]
    df_UV = pd.read_csv(f"{raw_data_dir}/UV_corrections.dat", sep=r"\s+", comment='#')

    a_mu_UV_corrected = NestedDict()
    a_mu_UV_corrected["ens_list"] = e_short
    for e in e_short:
        print(f"Ensemble: {e}")
        df_e = df_UV.loc[df_UV['Ensemble'] == e] # filter the row we are interested in
        for fermion_type in fermion_types:
            print(f" Fermion type: {fermion_type}")
            delta_UV = 1e-10 * df_e[fermion_type].to_numpy()[0]
            windows_keys = list(a_mu_Lref[e][fermion_type].keys())
            for window_key in windows_keys:
                print("   Window:", window_key)
                t_thr_keys = list(a_mu_Lref[e][fermion_type][window_key].keys())
                for t_thr_key in t_thr_keys:
                    print(f"  {t_thr_key}")
                    t_cut_strategies = list(a_mu_Lref[e][fermion_type][window_key][t_thr_key].keys())
                    for t_cut_strategy in t_cut_strategies:
                        print(f"   {t_cut_strategy}")
                        dt_plateau_strategies = list(a_mu_Lref[e][fermion_type][window_key][t_thr_key][t_cut_strategy].keys())
                        for dt_plateau_key in dt_plateau_strategies:
                            print(f"     {dt_plateau_key}")
                            a_mu_val = a_mu_Lref[e][fermion_type][window_key][t_thr_key][t_cut_strategy][dt_plateau_key]
                            """ 
                            NOTE:
                            The UV contribution is a SD (Short Distance) effect. 
                            We add it only there (and to the full) such that at every step we have consistently that 
                            the sum of the 3 windows gives the full contribution.
                            """
                            a_mu_UV_corrected[e][fermion_type][window_key][t_thr_key][t_cut_strategy][dt_plateau_key] = a_mu_val + delta_UV if window_key in ["full", "SD"] else a_mu_val
    #---------------


    with_dill.dump(a_mu_UV_corrected, f"{fpi_dir}/a_mu-UV_corrected.pkl")
#---


