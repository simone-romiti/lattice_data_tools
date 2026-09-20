print("""
      ----------------------------------------------------
      Correlated bootstrap samples for a[fm] and mistunings
      ----------------------------------------------------
      """
      )

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from lattice_data_tools.dictionaries import NestedDict
from lattice_data_tools.bootstrap import ParametricBootstraps, BootstrapSamples
from lattice_data_tools.io import with_dill, with_yaml
from lattice_data_tools.statistics_tools import covariance_to_correlation

ens_info = with_yaml.load("ensembles.yaml")
mistunings_dir = f'{ens_info["data"]["mistuning_correlation"]}/'
aux_dir = ens_info["data"]["auxiliary"]

ens_names = ens_info["data"]["ens_list"]
ens_short = ["".join((e.split(".")[0])[1:] + e.split(".")[2]) for e in ens_names]

N_bts = ens_info["N_bts"]
RNG_seed=ens_info["RNG_seed"]
np.random.seed(RNG_seed)

f_pi_list = ens_info["data"]["f_pi_MeV"]
for f_pi in f_pi_list:
    print(f"f_pi:{f_pi}")
    f_pi_dir = f"{mistunings_dir}/fpi_{f_pi}_MeV/"
    outdir = f"{aux_dir}/fpi_{f_pi}_MeV/"
    os.makedirs(outdir, exist_ok=True)
    a_fm_dict = NestedDict()
    dmul_dict = NestedDict()
    da_mu_dict = NestedDict()
    for ie, e in enumerate(ens_short):
        print(f" Ensemble:{e}")
        ens_name = ens_names[ie]
        e_actual = e if e!="B96" else "B64"
        txt_path = f"{f_pi_dir}/{e_actual}_lattice_spacing_and_HVP_mistuning_corrections.txt"
        f = open(txt_path, "r")
        pairs = [l.split(":") for l in f.readlines() if l[0] != "#"]
        dict_pairs = {pair[0]: pair[1] for pair in pairs}
        a_fm_mean = dict_pairs["a [fm]                                         "]
        a_fm_err  = dict_pairs["error on a [fm]                                "]
        dmu_l_mean = dict_pairs["a * delta mu_l                                 "]
        dmu_l_err = dict_pairs["error on a * delta mu_l                        "]
        dVkVk_tm_mus_sea_mean = (1e-10) * float(dict_pairs["HVP tm light-connected*10^10 sea-ms shift      "])
        dVkVk_tm_mus_sea_err  = (1e-10) * float(dict_pairs["HVP tm light-connected*10^10 sea-ms shift error"])
        dVkVk_OS_mus_sea_mean = (1e-10) * float(dict_pairs["HVP OS light-connected*10^10 sea-ms shift      "])
        dVkVk_OS_mus_sea_err  = (1e-10) * float(dict_pairs["HVP OS light-connected*10^10 sea-ms shift error"])
        dVkVk_tm_muc_sea_mean = (1e-10) * float(dict_pairs["HVP tm light-connected*10^10 sea-mc shift      "])
        dVkVk_tm_muc_sea_err  = (1e-10) * float(dict_pairs["HVP tm light-connected*10^10 sea-mc shift error"])
        dVkVk_OS_muc_sea_mean = (1e-10) * float(dict_pairs["HVP OS light-connected*10^10 sea-mc shift      "])
        dVkVk_OS_muc_sea_err  = (1e-10) * float(dict_pairs["HVP OS light-connected*10^10 sea-mc shift error"])
        dVkVk_tm_m0_sea_mean  = (1e-10) * float(dict_pairs["HVP tm light-connected*10^10     m0 shift      "])
        dVkVk_tm_m0_sea_err   = (1e-10) * float(dict_pairs["HVP tm light-connected*10^10     m0 shift error"])
        dVkVk_OS_m0_sea_mean  = (1e-10) * float(dict_pairs["HVP OS light-connected*10^10     m0 shift      "])
        dVkVk_OS_m0_sea_err   = (1e-10) * float(dict_pairs["HVP OS light-connected*10^10     m0 shift error"])
        means = np.array([float(s) for s in [a_fm_mean, dmu_l_mean, dVkVk_tm_mus_sea_mean, dVkVk_OS_mus_sea_mean, dVkVk_tm_muc_sea_mean, dVkVk_OS_muc_sea_mean]])
        errors = np.array([float(s) for s in [a_fm_err, dmu_l_err, dVkVk_tm_mus_sea_err, dVkVk_OS_mus_sea_err, dVkVk_tm_muc_sea_err, dVkVk_OS_muc_sea_err]])
        Cov = np.empty(shape=(6,6)) # covariance matrix
        for i in range(6):
            Cov[i,:] = np.array([float(s) for s in dict_pairs[f"C_{i}j "].strip().split(" ")])
        #---
        # building the correlated bootstrap samples
        rho  = covariance_to_correlation(C=Cov)
        bts_matrix = ParametricBootstraps.correlated_from_rho(x_mean=means, x_error=errors, rho=rho, N_bts=N_bts, seed=RNG_seed, method="Cholesky", decimals=15)
        rho_estimated = bts_matrix.correlation_matrix()
        # print(rho )
        # print(rho_estimated)
        a_fm_dict[ens_name] = BootstrapSamples(bts_matrix[:,0])
        dmul_dict[ens_name] = BootstrapSamples(bts_matrix[:,1])

        # using only "B","C","D" because they will be used after the volume interpolation
        da_mu_dict["dVkVk_tm_mus_sea"][e[0:1]] =  BootstrapSamples(bts_matrix[:,2])
        da_mu_dict["dVkVk_OS_mus_sea"][e[0:1]] =  BootstrapSamples(bts_matrix[:,3])
        da_mu_dict["dVkVk_tm_muc_sea"][e[0:1]] =  BootstrapSamples(bts_matrix[:,4])
        da_mu_dict["dVkVk_OS_muc_sea"][e[0:1]] =  BootstrapSamples(bts_matrix[:,5])

        # m0 correction is considered uncorrelated from the rest
        # ACHTUNG: typo in Lorenzo's file: the shift is sea+valence, even though it says m0_sea throughout
        da_mu_dict["dVkVk_tm_m0"][e[0:1]] =  ParametricBootstraps.Gaussian(mean=dVkVk_tm_m0_sea_mean, error=dVkVk_tm_m0_sea_err, N_bts=N_bts, seed=RNG_seed)
        da_mu_dict["dVkVk_OS_m0"][e[0:1]] =  ParametricBootstraps.Gaussian(mean=dVkVk_OS_m0_sea_mean, error=dVkVk_OS_m0_sea_err, N_bts=N_bts, seed=RNG_seed)

    #---
    print(f"Saving data in {outdir}")
    with_dill.dump({e: a_fm_dict[e] for e in ens_names}, f"{outdir}/a_fm.pkl")
    with_dill.dump({e: dmul_dict[e] for e in ens_names}, f"{outdir}/dmu_l.pkl")
    for k in da_mu_dict.keys():
        with_dill.dump(da_mu_dict[k], f"{outdir}/{k}.pkl")
#---
 

