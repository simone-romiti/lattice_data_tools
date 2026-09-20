print("""
      -----------------------------------------
      Printing mean and error of V(t) at each t
      -----------------------------------------
      """
      )


import os
import pandas as pd

from lattice_data_tools.dictionaries import NestedDict
from lattice_data_tools.io import with_dill, with_yaml

ens_info = with_yaml.load("ensembles.yaml")
aux_dir = ens_info["data"]["auxiliary"]
plots_dir = ens_info["data"]["plots"]

ens_list = ens_info["data"]["ens_list"]

f_pi_list = ens_info["data"]["f_pi_MeV"]
for f_pi in f_pi_list:
    print(f"f_pi: {f_pi}")
    fpi_suffix = f"fpi_{f_pi}_MeV"
    plots_fld=f"./{plots_dir}/{fpi_suffix}/VKVK/correlator/"
    VKVK_sim_bts = with_dill.load(f"{aux_dir}/{fpi_suffix}/VKVK_bts.pkl")
    os.makedirs(plots_fld, exist_ok=True)
    for ens_name in ens_list:
        print(" Ensemble: ", ens_name)
        for fermion_type in ["OS", "tm"]:
            print("  Fermions: ", fermion_type)
            VKVK_sim = VKVK_sim_bts[ens_name][fermion_type] # correlator at simulation point
            res_dict = NestedDict()
            VKVK_mean = VKVK_sim.unbiased_mean()
            VKVK_error = VKVK_sim.error()
            res_dict["V(t)"] = VKVK_mean
            res_dict["dV(t)"] = VKVK_error
            pd.DataFrame(res_dict).to_csv(f"{plots_fld}/{ens_name}-{fermion_type}-VKVK_bare.csv")
#-----------


