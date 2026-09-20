print("""
      -------------------------
      Effective curves for V(t)
      -------------------------
      """
      )

import numpy as np


from lattice_data_tools.bootstrap import BootstrapSamples
from lattice_data_tools.effective_curves import get_m_eff, get_A_eff
import lattice_data_tools.constants as constants
from lattice_data_tools.dictionaries import NestedDict
from lattice_data_tools.io import with_dill, with_yaml
from lattice_data_tools.fit.xyey import fit_xyey

# Loading input files

ens_info = with_yaml.load("ensembles.yaml")

ens_names = ens_info["data"]["ens_list"]
aux_dir = ens_info["data"]["auxiliary"]
N_bts = ens_info["N_bts"]

windows = ens_info["windows"]
fermion_types = ens_info["fermion_types"]


Q_light = constants.q_u**2 + constants.q_d**2 ## charge factor for the VKVK correlator


f_pi_list = ens_info["data"]["f_pi_MeV"]
for f_pi in f_pi_list:
    print(f"f_pi:{f_pi}")
    fpi_suffix = f"fpi_{f_pi}_MeV"
    fpi_dir = f"{aux_dir}/{fpi_suffix}/"
    VKVK_dict = with_dill.load(f'{fpi_dir}/VKVK_pp_tuned.pkl')
    VKVK_eff_curves_dict = NestedDict()
    for ens_name in ens_names:
        print("Ensemble:", ens_name)
        for fermion_type in fermion_types:
            print(" Fermions:", fermion_type)
            T = ens_info[ens_name]["T"]
            T_half = int(T/2)
            L = ens_info[ens_name]["L"] 
            ti = np.arange(0, T_half+1)
            for corr in ["sim", "pp"]:
                if corr == "sim" and ens_name[0:2] != "cB":
                    continue
                #---
                print(f"  Correction: {corr}")
    
                VkVk_charge = Q_light * VKVK_dict[ens_name][fermion_type][corr] #["VKVK"]
                MV_eff = BootstrapSamples.from_lambda(N_bts=N_bts, fun=lambda i: get_m_eff(C=VkVk_charge[i,:], strategy="cosh", T=T, avoid_instability=True))
                VKVK_eff_curves_dict[ens_name][fermion_type][corr]["MV"]["eff"] = MV_eff
                # A_eff = BootstrapSamples.from_lambda(N_bts=N_bts, fun=lambda i: get_A_eff(C=VkVk_charge[i,:], m_eff=MV_eff[i,:], strategy="cosh", T=T))
                # VKVK_eff_curves_dict[ens_name][fermion_type][corr]["A"]["eff"] = A_eff
    #-----------
    # Save to pickle
    with_dill.dump(VKVK_eff_curves_dict, f'{fpi_dir}/VKVK_eff_curves.pkl')
#---

