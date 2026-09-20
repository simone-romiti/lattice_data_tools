print("""
      ----------------------------------------------------
      Applying residual mistunings (m_cr and sea m_s, m_c)
      ----------------------------------------------------
      """
      )

import pandas as pd
import numpy as np

from lattice_data_tools.io import with_dill, with_yaml
from lattice_data_tools.dictionaries import NestedDict
from lattice_data_tools.bootstrap import parametric_gaussian_bts

ens_info = with_yaml.load("ensembles.yaml")

raw_data_dir = ens_info["data"]["main_dir"]
aux_dir = ens_info["data"]["auxiliary"]
N_bts = ens_info["N_bts"]
RNG_seed=ens_info["RNG_seed"]
np.random.seed(RNG_seed)

fermion_types = ens_info["fermion_types"]
windows = ens_info["windows"]

f_pi_list = ens_info["data"]["f_pi_MeV"]
for f_pi in f_pi_list:
    print(f"f_pi:{f_pi}")
    fpi_suffix = f"fpi_{f_pi}_MeV"
    fpi_dir = f"{aux_dir}/{fpi_suffix}/"

    a_mu_UV_corrected = with_dill.load(f'{fpi_dir}/a_mu-UV_corrected.pkl')
    e_short = a_mu_UV_corrected["ens_list"]
    print(e_short)

    a_mu_pp_UVcorr = NestedDict()
    a_mu_pp_UVcorr["ens_list"] = e_short
    for i_e, e in enumerate(e_short):
        print(f"Ensemble: {e}")
        for fermion_type in fermion_types:
            print(f" Fermions: {fermion_type}")
            a_fm_dict = with_dill.load(f'{fpi_dir}/a_fm.pkl') # lattice spacing bootstraps
            print(with_dill.load(f'{fpi_dir}/dVkVk_{fermion_type}_mus_sea.pkl').keys())
            da_mu_strange = with_dill.load(f'{fpi_dir}/dVkVk_{fermion_type}_mus_sea.pkl')[e]
            da_mu_charm = with_dill.load(f'{fpi_dir}/dVkVk_{fermion_type}_muc_sea.pkl')[e]

            da_mu_m0 = with_dill.load(f'{fpi_dir}/dVkVk_{fermion_type}_m0.pkl')[e]

            delta_a_mu_residual = da_mu_strange + da_mu_charm + da_mu_m0
            windows = list(a_mu_UV_corrected[e][fermion_type].keys())
            for window in windows:
                print(f"  {window}")
                t_thr_keys = list(a_mu_UV_corrected[e][fermion_type][windows[0]].keys()) # I take the t_thr keys from the 1st window, they are the same for all
                for t_thr_key in t_thr_keys:
                    print(f"  {t_thr_key}")
                    t_cut_strategies = list(a_mu_UV_corrected[e][fermion_type][window][t_thr_key].keys())
                    for t_cut_strategy in t_cut_strategies:
                        print(f"   {t_cut_strategy}")
                        dt_plateau_strategies = list(a_mu_UV_corrected[e][fermion_type][window][t_thr_key][t_cut_strategy].keys())
                        for dt_plateau_key in dt_plateau_strategies:
                            print(f"     {dt_plateau_key}")
                            # the correction is supposed to be applied to the full window only
                            # the other windows stay the same
                            a_mu_pp_UVcorr[e][fermion_type][window][t_thr_key][t_cut_strategy][dt_plateau_key] = a_mu_UV_corrected[e][fermion_type][window][t_thr_key][t_cut_strategy][dt_plateau_key]
                            if window == "full_window":
                                print(f"# applying residual mistunings to the: {window}")
                                a_mu_pp_UVcorr[e][fermion_type][window][t_thr_key][t_cut_strategy][dt_plateau_key] += delta_a_mu_residual
    #-------------------

    with_dill.dump(a_mu_pp_UVcorr, f"{fpi_dir}/a_mu-pp_UV_corrected.pkl")
#---
