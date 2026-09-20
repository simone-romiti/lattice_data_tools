print("""
      ----------------------------------------
      Mistunings: apply valence+sea correction
      ----------------------------------------
      """)
 
 
import os
 
import numpy as np
import matplotlib.pyplot as plt
 
from lattice_data_tools.dictionaries import NestedDict
from lattice_data_tools.io import with_dill, with_yaml
 
 
ens_info = with_yaml.load("ensembles.yaml")
ens_list = ens_info["data"]["ens_list"]
aux_dir = ens_info["data"]["auxiliary"]
plots_dir = ens_info["data"]["plots"]
 
 
print("# Applying mistunings corrections")
 
f_pi_list = ens_info["data"]["f_pi_MeV"]
for f_pi in f_pi_list:
    print(f"f_pi:{f_pi}")
    fpi_suffix = f"fpi_{f_pi}_MeV"
    fpi_dir = f"{aux_dir}/{fpi_suffix}/"
    os.makedirs(fpi_dir, exist_ok=True)
    
    VKVK_sim_bts  = with_dill.load(f"{fpi_dir}/VKVK_bts.pkl")
    a_fm_bts_dict = with_dill.load(f"{fpi_dir}/a_fm.pkl")
 
    delta_VKVK_ml_bts = with_dill.load(f"{fpi_dir}/dVKVK_ml_valence_plus_sea_bts.pkl") # valence + sea
 
    VKVK_pp_dict = NestedDict() # VKVK at the physical point
    
    plots_fld=f"./{plots_dir}/{fpi_suffix}/VKVK/correlator/"
    os.makedirs(plots_fld, exist_ok=True)
 
    for ens_name in ens_list:
        print("Ensemble: ", ens_name)
        a_fm_mean = a_fm_bts_dict[ens_name].mean()
        T = ens_info[ens_name]["T"]
        T_ext = T//2 + 1
        for fermion_type in ["OS", "tm"]:
            print("  Fermions: ", fermion_type)
            VKVK_sim = VKVK_sim_bts[ens_name][fermion_type] # correlator at simulation point
            VKVK_pp_dict[ens_name][fermion_type]["sim"] = VKVK_sim
            
            # Handle broadcasting if correction has fewer bootstrap samples than simulation
            correction = delta_VKVK_ml_bts[ens_name][fermion_type]
            if correction.shape[0] != VKVK_sim.shape[0]:
                # If correction is a mean/baseline (e.g. shape (1, T) or (2, T) where we take mean),
                # broadcast it to match the bootstrap axis of VKVK_sim.
                if correction.shape[0] == 1:
                    corr_val = correction[0]
                else:
                    # use .to_numpy() to avoid BootstrapSamples.mean() restrictions
                    corr_val = np.ndarray.mean(correction[1:].view(np.ndarray), axis=0)
                
                # Expand corr_val to match (N_bts, T_ext)
                corr_broadcasted = np.tile(corr_val, (VKVK_sim.shape[0], 1))
                VKVK_pp = VKVK_sim + corr_broadcasted
            else:
                VKVK_pp = VKVK_sim + correction
                
            VKVK_pp_dict[ens_name][fermion_type]["pp"] = VKVK_pp
    
            # Compute average and error along the bootstrap axis (axis=0)
            VKVK_pp_avg = VKVK_pp.unbiased_mean()
            VKVK_pp_err = VKVK_pp.error()
    
            VKVK_sim_avg = VKVK_sim.unbiased_mean()
            VKVK_sim_err = VKVK_sim.error()
    
            # Plot the curve of the averages along the 1st axis for both VKVK_pp and VKVK_sim
    
            plt.figure()
            times = a_fm_mean*np.arange(VKVK_pp_avg.shape[0]-1)
            plt.errorbar(times, VKVK_pp_avg[1:], yerr=VKVK_pp_err[1:], fmt='o-', label='VKVK_pp')
            plt.errorbar(times, VKVK_sim_avg[1:], yerr=VKVK_sim_err[1:], fmt='s-', label='VKVK_sim')
            plt.xlabel('$t$ [fm]')
            plt.ylabel('$\\langle{V_k(t) V_k(0)}\\rangle$')
            plt.yscale("log")
            plt.title(f'{ens_name} {fermion_type}: VKVK at physical point and simulation point')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{plots_fld}/{ens_name}-{fermion_type}-sim_VS_pp.svg")
            # plt.show()
            plt.close()
    #-------
    # Save VKVK at the physical point (pp) to a .pkl file using dill
    os.makedirs(f"{fpi_dir}", exist_ok=True)
    with_dill.dump(VKVK_pp_dict, f"{fpi_dir}/VKVK_pp_tuned.pkl")
#---
