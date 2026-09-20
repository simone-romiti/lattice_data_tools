print("""
      -------------------------
      EXPERIMENTAL
      Bounding method: GS model
      -------------------------
      """
      )

import numpy as np
import matplotlib.pyplot as plt

from lattice_data_tools.bootstrap import BootstrapSamples
import lattice_data_tools.symmetries as symmetries
from lattice_data_tools.gm2.HVP.bounding_methods import ZeroTail
import lattice_data_tools.constants as constants
from lattice_data_tools.dictionaries import NestedDict
from lattice_data_tools.io import with_dill, with_yaml
from lattice_data_tools.fit.xyey import fit_xyey
from lattice_data_tools.gm2.HVP.PP_finite_volume import get_V_PP_GSmodel


# Loading input files
ens_info = with_yaml.load("ensembles.yaml")

ens_names = ens_info["data"]["ens_list"]
aux_dir = ens_info["data"]["auxiliary"]
N_bts = ens_info["N_bts"]

windows = ens_info["windows"]
fermion_types = ens_info["fermion_types"]

N_int = ens_info["N_int_QED_kernel"]

MV_phys_MeV = 775
g_VPP_phys = 5.95
Gamma_V_phys_MeV = 147.8
Z_00_dict = with_dill.load(f'{aux_dir}/precomputed_Z00.pkl') # precomputed Z_00 function
Z_00_obj = Z_00_dict["Z_00"]
N_lev = Z_00_dict["N_lev"]

K_dict = with_dill.load(f'{aux_dir}/precomputed_kernel-N_int{N_int}.pkl') # precomputed kernel for each ensemble
VKVK_dict = with_dill.load(f'{aux_dir}/VKVK_pp_tuned.pkl')
a_fm_dict = with_dill.load(f'{aux_dir}/a_fm_bts.pkl') # lattice spacing bootstraps

VKVK_eff = with_dill.load(f'{aux_dir}/VKVK_eff_curves.pkl') # effective mass of the vector meson

Q_light = constants.q_u**2 + constants.q_d**2 ## charge factor for the VKVK correlator
a_mu_GS_model = NestedDict()
Z_ren_mapping = {"tm": "ZA", "OS": "ZV"}
for ens_name in ens_names:
    print("Ensemble:", ens_name)
    for fermion_type in fermion_types:
        print(" Fermions:", fermion_type)
        Z_ren = ens_info[ens_name][Z_ren_mapping[fermion_type]]        
        a_fm = a_fm_dict[ens_name].mean()
        a_MeV_inv = constants.fm_to_MeV_inv(a_fm)
        Mpi_MeV = constants.masses_MeV["pi"]["isoQCD"]["Edinburgh"]
        aMpi_phys = a_MeV_inv*Mpi_MeV
        aMV_phys = a_MeV_inv*MV_phys_MeV 
        aMV_data = VKVK_eff[ens_name][fermion_type]["MV"]["fit"].mean()
        aMV = aMV_data # aMV_phys # + aMV_data) / 2
        Gamma_V = a_MeV_inv*Gamma_V_phys_MeV
        g_VPP = g_VPP_phys # np.sqrt((48*(aMV**2)*Gamma_V)/((aMV**2 - 4*(aMpi_phys**2))**(3/2)))
        print(g_VPP)
        T = ens_info[ens_name]["T"]
        T_half = int(T/2)
        L = ens_info[ens_name]["L"] 
        ti = np.arange(0, T_half+1)
        
        # Pre-loaded kernel per ensemble
        K = K_dict[ens_name]   
        
        VkVk_GS = get_V_PP_GSmodel(
            times=ti, MP=aMpi_phys, MV=aMV, g_VPP=g_VPP, L=L, N_lev=N_lev, Z_00_obj=Z_00_obj, eps_roots=1e-3, eps_der=1e-10)["V_PP"]
        # VkVK_GS_bkw = get_V_PP_GSmodel(
        #     times=T-ti, MP=aMpi_phys, MV=aMV_phys, g_VPP=g_VPP, L=L, N_lev=N_lev, Z_00_obj=Z_00_obj, eps_roots=1e-3, eps_der=1e-10)["V_PP"]
        print("# Loading the correlator, and including charge factors")
        for corr in ["sim", "pp"]:
            if corr == "sim" and ens_name[0:2] != "cB":
                continue
            #---
            print(f"  Correction: {corr}")
            VkVk_charge = (9/10) * (Z_ren**2) * Q_light * VKVK_dict[ens_name][fermion_type][corr] #["VKVK"]
            if corr == "pp":
                x=ti
                y_bts=VkVk_charge #-VkVk_GS
                y=y_bts.mean()
                ey=y_bts.error()
                plt.errorbar(x=ti, y=y, yerr=ey, label="VKVK data")
                plt.plot(ti, VkVk_GS, label="VKVK GS model")
                plt.yscale("log")
                plt.legend()
                plt.show()
            # for window in windows:
            #     print("   Window:", window)
            #     a_mu_eff = BootstrapSamples.from_lambda(N_bts=N_bts, fun=lambda i: [ZeroTail(window=window, a_fm=a_fm[i], V=VkVk_charge[i,:], K=K[i,:], Z_ren=Z_ren, t0=t0, strategy="trapezoidal") for t0 in ti])
            #     a_mu_GS_model[ens_name][fermion_type][corr][f"{window}_window"]["ZeroTail"] = a_mu_eff
#---------------

with_dill.dump(a_mu_GS_model, f'{aux_dir}/a_mu-ZeroTail.pkl') # Save to pickle


