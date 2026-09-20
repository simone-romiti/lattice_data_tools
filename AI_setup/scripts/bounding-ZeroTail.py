print("""
      ---------------------------
      Bounding methods: Zero Tail
      ---------------------------
      """
      )

import numpy as np

from lattice_data_tools.bootstrap import BootstrapSamples
from lattice_data_tools.gm2.HVP.bounding_methods import ZeroTail
import lattice_data_tools.constants as constants
from lattice_data_tools.dictionaries import NestedDict
from lattice_data_tools.io import with_dill, with_yaml

# Loading input files

ens_info = with_yaml.load("ensembles.yaml")

ens_names = ens_info["data"]["ens_list"]
aux_dir = ens_info["data"]["auxiliary"]
N_bts = ens_info["N_bts"]

windows = ens_info["windows"]
fermion_types = ens_info["fermion_types"]

N_int = ens_info["N_int_QED_kernel"]

Q_light = constants.q_u**2 + constants.q_d**2 ## charge factor for the VKVK correlator
Z_ren_mapping = {"tm": "ZA", "OS": "ZV"}

f_pi_list = ens_info["data"]["f_pi_MeV"]
for f_pi in f_pi_list:
    print(f"f_pi:{f_pi}")
    fpi_suffix = f"fpi_{f_pi}_MeV"
    fpi_dir = f"{aux_dir}/{fpi_suffix}/"

    K_dict = with_dill.load(f'{fpi_dir}/precomputed_kernel-N_int{N_int}.pkl') # precomputed kernel for each ensemble
    VKVK_dict = with_dill.load(f'{fpi_dir}/VKVK_pp_tuned.pkl')
    a_fm_dict = with_dill.load(f'{fpi_dir}/a_fm.pkl') # lattice spacing bootstraps

    a_mu_ZeroTail = NestedDict()
    for ens_name in ens_names:
        print("Ensemble:", ens_name)
        for fermion_type in fermion_types:
            print(" Fermions:", fermion_type)
            Z_ren = ens_info[ens_name][Z_ren_mapping[fermion_type]]        
            a_fm = a_fm_dict[ens_name]
            a_MeV_inv = constants.fm_to_MeV_inv(a_fm)
            Mpi = constants.masses_MeV["pi"]["isoQCD"]["Edinburgh"]
            aMpi_phys = a_MeV_inv*Mpi
            T = ens_info[ens_name]["T"]
            T_half = int(T/2)
            L = ens_info[ens_name]["L"] 
            ti = np.arange(0, T_half+1)
            
            # Pre-loaded kernel per ensemble
            K = K_dict[ens_name]   
            
            print("# Loading the correlator, and including charge factors")
            for corr in ["sim", "pp"]:
                if corr == "sim" and ens_name[0:2] != "cB":
                    continue
                #---
                print(f"  Correction: {corr}")
                VkVk_charge = Q_light * VKVK_dict[ens_name][fermion_type][corr] #["VKVK"]
                for window in windows:
                    print("   Window:", window)
                    lambda_bts = lambda i: [ZeroTail(window=window, a_fm=a_fm[i], V=VkVk_charge[i,:], K=K[i,:], Z_ren=Z_ren, t0=t0, strategy="trapezoidal") for t0 in ti]
                    a_mu_eff = BootstrapSamples.from_lambda(N_bts=N_bts, fun=lambda_bts, parallel=True)
                    a_mu_ZeroTail[ens_name][fermion_type][corr][f"{window}_window"] = a_mu_eff
    #---------------
    with_dill.dump(a_mu_ZeroTail, f'{fpi_dir}/a_mu-ZeroTail.pkl') # Save to pickle
#---

