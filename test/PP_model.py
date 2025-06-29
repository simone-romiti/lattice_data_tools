

import numpy as np
import matplotlib.pyplot as plt    

import sys
sys.path.append('../../')

from lattice_data_tools.gm2.HVP.Z_function import Z_00_Calculator
from lattice_data_tools.gm2.HVP.Gounaris_Sakurai_model import GS_model
from lattice_data_tools.gm2.HVP.PP_finite_volume import PP_model, get_V_PP_GSmodel

N_gauss = 100  # number of Gauss-Legendre points
Lambda = 1.0
Lambda_Z3 = 10 # cutoff for |n| in Z_00

Z_00_obj = Z_00_Calculator(Lambda_Z3=Lambda_Z3, Lambda=Lambda, N_gauss=N_gauss, q2_max=25.0)

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

# k_vals = np.linspace(0.01, 1.5, 500)
# q_vals = k_vals*Nx/(2.0*np.pi)
# GS_obj = GS_model(MP=aMP, MV=aMV, g_VPP=g_VPP)
# delta_11 = np.array([GS_obj.delta_11(k) for k in k_vals])
# phi_vals = np.array([Z_00_obj.phi(q) for q in q_vals])

# plt.figure(figsize=(8, 6))
# plt.plot(k_vals, delta_11 + phi_vals, label=r'$\delta_{11}(k) + \phi(q)$')
# N_lev = 5
# for n in range(N_lev + 1):
#     plt.axhline(n * np.pi, color='gray', linestyle='--', linewidth=0.8, label=r'$n\pi$' if n == 0 else None)
# plt.xlabel(r'$k$')
# plt.ylabel(r'$\delta_{11}(k) + \phi(q)$')
# plt.title(r'$\delta_{11}(k) + \phi(q)$ and $n\pi$ lines')
# plt.legend()
# plt.tight_layout()
# plt.show()


# -------------------
# The following lines reproduce Fig. 5 of https://arxiv.org/pdf/1808.00887
#--------------------

times = np.arange(3, 22+1)
VV_info = get_V_PP_GSmodel(
    times=times, 
    MP=aMP, MV=aMV, g_VPP=g_VPP, 
    L=Nx, N_lev=5, 
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
