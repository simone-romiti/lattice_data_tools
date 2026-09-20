print("""
      ----------------------------------------------------
      Blinding: comparing the results against other groups
      ----------------------------------------------------
      """
      )


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lattice_data_tools.io import with_yaml

ens_info = with_yaml.load("ensembles.yaml")

ens_names = ens_info["data"]["ens_list"]
fermion_types = ens_info["fermion_types"]
plots_dir = ens_info["data"]["plots"]

dVKVK_other_groups = pd.read_csv(f"raw_data/compare_with_other_groups/dC_tstar.csv", sep=r"\s+", comment="#")


output_fld = f"./{plots_dir}/blinding/"
os.makedirs(output_fld, exist_ok=True)

F_VKVK = open(f"./{plots_dir}/blinding/dVKVK.csv", "w")
F_VKVK.write("ensemble fermion Bern ToV Cyprus\n")
for ens_name in ens_names:
    if ens_name == "cB.72.96":
        continue
    #---
    ens_name_splitted = ens_name.split(".")
    e_short = "".join([ens_name_splitted[i] for i in [0,2]]).split("c")[1]
    for fermion_type in ["tm", "OS"]:
        df_og = dVKVK_other_groups.loc[dVKVK_other_groups['ensemble'] == e_short]
        df_og = df_og.loc[df_og['fermion'] == fermion_type]
        t_star = df_og["t/a"].to_numpy()[0]
        dV_ToV = df_og["ToV"].to_numpy()[0]
        dV_Cyprus = df_og["Cyprus"].to_numpy()[0]
        VKVK_Bern = pd.read_csv(f"./{plots_dir}/VKVK/correlator/{ens_name}-{fermion_type}-VKVK_bare.csv")
        factor = (9/10)*(1/9 + 4/9) # 0.5 = (9/10)*(q_u**2 + q_d**2)
        dV_Bern = factor*VKVK_Bern["dV(t)"][t_star]
        F_VKVK.write(f"{ens_name} {fermion_type} {dV_Bern} {dV_ToV} {dV_Cyprus}\n")




da_mu_other_groups = pd.read_csv(f"raw_data/compare_with_other_groups/da_mu.csv", sep=r"\s+", comment="#")

F_a_mu = open(f"./{plots_dir}/blinding/da_mu.csv", "w")
F_a_mu.write("ensemble fermion Bern ToV Cyprus\n")

i_ens = 0
for ens_name in ens_names:
    if ens_name == "cB.72.96":
        continue
    #---
    ens_name_splitted = ens_name.split(".")
    e_short = "".join([ens_name_splitted[i] for i in [0,2]]).split("c")[1]
    for fermion_type in ["tm", "OS"]:
        df_og = da_mu_other_groups.loc[da_mu_other_groups['ensemble'] == e_short]
        df_og = df_og.loc[df_og['fermion'] == fermion_type]
        da_mu_ToV    = df_og["ToV"].to_numpy()[0]
        da_mu_Cyprus = df_og["Cyprus"].to_numpy()[0]
        df_Bern = pd.read_csv(f"./{plots_dir}/continuum_limit/points/t_thr_fm=1.8-aggressive_t_cut-dt=0.25[fm].csv")
        factor = (9/10) # 0.5 = (9/10)*(q_u**2 + q_d**2)
        da_mu_Bern = factor*df_Bern[f"da_mu-{fermion_type}"].to_numpy()[i_ens]
        F_a_mu.write(f"{ens_name} {fermion_type} {da_mu_Bern:.04e} {da_mu_ToV} {da_mu_Cyprus}\n")
    #---
    i_ens += 1
#---

