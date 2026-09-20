print("""
      ---------------------------------------------
      Precomputing: Luescher'z zeta function Z_{00} 
      ---------------------------------------------
      """
      )

import dill
import numpy as np
import yaml

from lattice_data_tools.gm2.HVP.Z_function import Z_00_Calculator

N_gauss = 100  # number of Gauss-Legendre points
Lambda = 1.0
Lambda_Z3 = 5 # cutoff for |n| in Z_00

N_lev = 15
q2_max = (N_lev)**2 # after this value, \phi(q) is not correctly wrapped around the half-circle
Z_00_obj = Z_00_Calculator(Lambda_Z3=Lambda_Z3, Lambda=Lambda, N_gauss=N_gauss, q2_max=q2_max)

Z_00_info = {"N_lev": N_lev, "Z_00": Z_00_obj}

print("Saving Z_00 to a .pkl file")
with open('./auxiliary_data/precomputed_Z00.pkl', 'wb') as file:
    dill.dump(Z_00_info, file)
####
