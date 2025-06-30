""" 
This test script shows how to use the library to find the Vector-Vector correlator (HVP) 
in terms of 2-pion states in a finite volume, in the I=1, J=1 channel. 
The phase shift needed to solve the Luescher's quantization condition is provided by the Gounaris-Sakurai model.

Reference: https://arxiv.org/pdf/1808.00887
"""

import numpy as np
import matplotlib.pyplot as plt    

import sys
sys.path.append('../../')

from lattice_data_tools.gm2.HVP.Z_function import Z_00_Calculator
from lattice_data_tools.gm2.HVP.Gounaris_Sakurai_model import GS_model
from lattice_data_tools.gm2.HVP.PP_finite_volume import PP_model, get_V_PP_GSmodel

N_gauss = 100  # number of Gauss-Legendre points
Lambda = 1.0
Lambda_Z3 = 5 # cutoff for |n| in Z_00


N_lev = 5
q2_max = (N_lev)**2 # after this value, \phi(q) is not correctly wrapped around the half-circle
Z_00_obj = Z_00_Calculator(Lambda_Z3=Lambda_Z3, Lambda=Lambda, N_gauss=N_gauss, q2_max=q2_max)

#
MP_MeV = 320 # pion mass
MV_MeV = 2.77*MP_MeV # rho mass
g_VPP =  5.22 # g_{\rho\pi\pi}
hbarc_MeV_fm = 197.3269631
a_fm = 0.089
a_MeV_inv = a_fm / hbarc_MeV_fm
Nx = 24
L_MeV_inv = a_fm * Nx
aMP = a_MeV_inv*MP_MeV
aMV = a_MeV_inv*MV_MeV
print("aMP:", aMP)
print("aMV:", aMV)

# -------------------
# The following lines reproduce Fig. 5 of https://arxiv.org/pdf/1808.00887
#--------------------

times = np.arange(3, 22+1)
VV_info = get_V_PP_GSmodel(
    times=times, 
    MP=aMP, MV=aMV, g_VPP=g_VPP, 
    L=Nx, N_lev=N_lev, 
    Z_00_obj=Z_00_obj, 
    eps_roots=1e-3, eps_der=1e-10
    )
VV = VV_info["V_PP"]
omega_n = VV_info["omega_n"]
plt.plot(times, VV, label="V(t)")
plt.xlim([2, 22])
plt.ylim([1e-7, 0.0015])
plt.yscale("log")
plt.tick_params(axis='both', which='both', direction='in', right=True, top=True, labelsize=14, length=6, width=1.5)

plt.show()
