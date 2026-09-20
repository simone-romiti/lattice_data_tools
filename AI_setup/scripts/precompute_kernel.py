print("""
      ------------------------------------------
      Precomputing: QED kernel for each ensemble
      ------------------------------------------
      """
      )
# precomputing the QED Kernel for each ensemble

import os
import numpy as np

# from lattice_data_tools.bootstrap import BootstrapSamples
from lattice_data_tools.gm2.HVP import kernel
import lattice_data_tools.constants as constants
# from lattice_data_tools.bootstrap import parametric_gaussian_bts
from lattice_data_tools.io import with_dill, with_yaml
# from lattice_data_tools.constants import alpha_EM

m_mu_MeV = constants.masses_MeV["mu"]

ens_info = with_yaml.load("ensembles.yaml")
ens_names = ens_info["data"]["ens_list"]
aux_dir = ens_info["data"]["auxiliary"]

N_bts = ens_info["N_bts"]
RNG_seed=ens_info["RNG_seed"]
np.random.seed(RNG_seed)

print("# Precomputing the QED Kernel")

N_int = ens_info["N_int_QED_kernel"]

f_pi_list = ens_info["data"]["f_pi_MeV"]
for f_pi in f_pi_list:
    print(f"f_pi:{f_pi}")
    fpi_dir = f"{aux_dir}/fpi_{f_pi}_MeV/"
    os.makedirs(fpi_dir, exist_ok=True)
    #
    a_fm_bts_dict = with_dill.load(f"{fpi_dir}/a_fm.pkl")
    K_dict = {}
    for ens_name in ens_names:
        print(f" Ensemble: {ens_name}")
        #
        a_fm = a_fm_bts_dict[ens_name]
        a_MeV_inv = constants.fm_to_MeV_inv(a_fm)
        am_mu = a_MeV_inv*m_mu_MeV # muon mass
        T = ens_info[ens_name]["T"]
        T_half = int(T/2)
        T_ext = T_half+1
        ti = np.arange(1, T_ext)
        am_mu_avg = am_mu.mean()
        am_mu_err = am_mu.error()
        K_mean = np.array([0.0]+[kernel.K(mt=am_mu_avg*t, N=N_int) for t in range(1, T_half+1)])
        dK_da_mean = np.array([0.0]+[kernel.dK_da(a=a_MeV_inv.unbiased_mean(), m=m_mu_MeV, mt=am_mu_avg*t, N=N_int) for t in range(1, T_half+1)])
        da_MeV_inv_bts = a_MeV_inv - a_MeV_inv.unbiased_mean()
        K_err = (dK_da_mean[np.newaxis,:]) * (da_MeV_inv_bts[:,np.newaxis])
        K_bts = K_mean[np.newaxis,:] + K_err
        K_dict[ens_name] = np.expand_dims(K_mean, axis=0) + K_err
    #---
    print(f"Saving the output in {fpi_dir}")
    with_dill.dump(K_dict, f'{fpi_dir}/precomputed_kernel-N_int{N_int}.pkl')
#---


