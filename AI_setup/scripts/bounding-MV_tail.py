print("""
      -------------------------
      Bounding method: M_V tail 
      -------------------------
      """
      )

import time
import numpy as np


from lattice_data_tools.bootstrap import BootstrapSamples
from lattice_data_tools.gm2.HVP.bounding_methods import M_eff_t0_Tail
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

t_thr_fm_values = ens_info["MV"]["t_thr_fm"]

f_pi_list = ens_info["data"]["f_pi_MeV"]
for f_pi in f_pi_list:
    print(f"f_pi:{f_pi}")
    fpi_suffix = f"fpi_{f_pi}_MeV"
    fpi_dir = f"{aux_dir}/{fpi_suffix}/"

    K_dict = with_dill.load(f'{fpi_dir}/precomputed_kernel-N_int{N_int}.pkl') # precomputed kernel for each ensemble
    VKVK_dict = with_dill.load(f'{fpi_dir}/VKVK_pp_tuned.pkl')
    a_fm_dict = with_dill.load(f'{fpi_dir}/a_fm.pkl') # lattice spacing bootstraps
    VKVK_eff_curves = with_dill.load(f'{fpi_dir}/VKVK_eff_curves.pkl')
    VKVK_tail_model_avg = with_dill.load(f'{fpi_dir}/VKVK_tail_model_avg.pkl')
    
    a_mu_MV_tail = NestedDict()
    
    for ens_name in ens_names:
        print(" Ensemble:", ens_name)
        a_fm = a_fm_dict[ens_name]
        T = ens_info[ens_name]["T"]
        T_half = int(T/2)
        L = ens_info[ens_name]["L"] 
        ti = np.arange(0, T_half+1)

        # Precomputed kernel for the ensemble
        K = K_dict[ens_name]   
        for fermion_type in fermion_types:
            print(" Fermions:", fermion_type)
            Z_ren = ens_info[ens_name][Z_ren_mapping[fermion_type]]        
            
            print("# Loading the correlator, and including charge factors")
            for corr in ["sim", "pp"]:
                if corr == "sim" and ens_name[0:2] != "cB":
                    continue
                #---
                print(f"  Correction: {corr}")
                VkVk_charge = Q_light * VKVK_dict[ens_name][fermion_type][corr] #["VKVK"]
                # MV_fit = VKVK_tail_model_avg[ens_name][fermion_type][corr]["MV"]["correlated_fit"]
                # MV_arr = BootstrapSamples.from_lambda(N_bts=N_bts, fun=lambda i: [MV_fit[i] for t in range(VkVk_charge.shape[1])])
                MV_eff = VKVK_eff_curves[ens_name][fermion_type][corr]["MV"]["eff"]
                for t_thr_fm in t_thr_fm_values:
                    print("   t_thr [fm] =", t_thr_fm)
                    t_thr = int(t_thr_fm/a_fm.mean()) # in lattice units
                    MV_arr = BootstrapSamples.from_lambda(N_bts=N_bts, fun=lambda i: [(MV_eff[i,t] if t<t_thr else MV_eff[i,t_thr]) for t in range(VkVk_charge.shape[1])])
                    for window in windows:
                        print("    Window:", window)
                        lambda_bts = lambda i: [M_eff_t0_Tail(M_eff=MV_arr[i,:], window=window, a_fm=a_fm[i], V=VkVk_charge[i,:], K=K[i,:], Z_ren=Z_ren, t0=t0, strategy="trapezoidal") for t0 in ti]
                        a_mu_M_eff_t0_Tail = BootstrapSamples.from_lambda(N_bts=N_bts, fun=lambda_bts, parallel=True)
                        a_mu_MV_tail[ens_name][fermion_type][corr][f"t_thr_fm={t_thr_fm}"][f"{window}_window"] = a_mu_M_eff_t0_Tail
    #---------------
    # Save to pickle
    with_dill.dump(a_mu_MV_tail, f'{fpi_dir}/a_mu-MV_tail.pkl')
#---

