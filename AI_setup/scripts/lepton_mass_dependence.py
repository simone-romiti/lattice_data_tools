

import os
import dill
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from lattice_data_tools.dictionaries import NestedDict
import lattice_data_tools.constants as constants
from lattice_data_tools.gm2.HVP.kernel import K as QED_kernel # QED kernel
from lattice_data_tools.gm2.HVP.amu import get_amu_precomp_K as get_amu # value of the anomaly

M_pi_MeV = constants.masses_MeV["pi"]["isoQCD"]["Edinburgh"]
alpha_em = constants.alpha_EM

with open("auxiliary_data/VKVK_pp_tuned.pkl", "rb") as f:
    VKVK_pp_dict = dill.load(f) # VKVK at the physical point
#---

with open("ensembles.yaml", 'r') as stream:
    ens_info = yaml.safe_load(stream)
#---

ens_list = ens_info["data"]["ens_list"] # list of ensembles
times_lattice = NestedDict()
for ens_name in ens_list:
    print("Ensemble: ", ens_name)
    T = ens_info[ens_name]["T"]
    T_ext = T // 2 + 1 
    times_lattice[ens_name] = np.arange(T_ext)

print("---")
print("Precomputing the QED kernel values")
print("---")
N_integ = int(1e+5)
eps_omega = 1e-12
n_omega = 50
omega_min = -1+eps_omega
omega_max = 1-eps_omega
omega_vals = np.linspace(omega_min, omega_max, n_omega)



import mpmath as mp
def A_sing(omega, L12, R11, prec=100):
    """
    Compute A^sing(ω) = -3 * L_{1,2} * Li5(-ω) - 2 * R_{1,1} * ln(1 - ω).

    Args:
        omega: real or complex number ω (|ω| < 1 ideally).
        L12: constant L_{1,2}.
        R11: constant R_{1,1}.
        prec: (optional) precision in bits or decimal places.

    Returns:
        mpmath.mpc or mp.mpf: value of A^sing(ω).
    """
    mp.mp.dps = prec  # set decimal precision

    # compute Li₅(−ω)
    # li5 = mp.polylog(5, -omega)
    one = L12 * (1/8)*(1 + omega)**4 * np.log(1+omega)

    # compute ln(1 - ω)
    # ln_term = mp.log(1 - omega)
    two = -2 * R11 * np.log(1 - omega)

    # return (alpha_em/np.pi) * (-3 * L12 * li5 - 2 * R11 * ln_term)
    # return (alpha_em/np.pi) * (-3 * L12 * li5)
    return (alpha_em/np.pi) * (one + two)
#---

# mp.mp.dps = 50  # set decimal precision
# plt.plot(omega_vals,  [-3*mp.polylog(5, -omega) for omega in omega_vals])
# plt.plot(omega_vals,  [(1/8)*(1 + omega)**4 * np.log(1+omega) for omega in omega_vals])
# plt.show()
# quit()

A_sing_vals = np.array([A_sing(omega=omega, L12=2.86e-3, R11=5/6, prec=50) for omega in omega_vals])

z_vals = ((1 + omega_vals)/(1 - omega_vals))**2
K_values = NestedDict()
for ens_name in ens_list:
    print("Ensemble: ", ens_name)
    a_fm = ens_info[ens_name]["a_fm"]
    a_MeV_inv = constants.fm_to_MeV_inv(a_fm)
    aM_pi = a_MeV_inv*M_pi_MeV # pion mass
    am_mu_vals = z_vals*aM_pi
    t_vals = times_lattice[ens_name]
    # QED kernel diverges at 0
    K_values[ens_name] = np.array([[QED_kernel(am_mu*t, N_integ) for t in t_vals[1:]] for am_mu in am_mu_vals])
#---

# for ens_name in ens_list:
#     t_vals = times_lattice[ens_name]
#     for i_omega, omega in enumerate(omega_vals):
#         plt.plot(t_vals, K_values[ens_name][i_omega], label=f"$\\omega={omega}")
#     plt.legend()
#     plt.show()
#     plt.close()


print("---")
print("Computing a_\\mu")
print("---")
Z_ren_mapping = {"tm": "ZA", "OS": "ZV"}

colors = {ens_name: cm.rainbow(i*100) for i, ens_name in enumerate(ens_list)}

N_bts = ens_info["N_bts"]
a_mu_val = NestedDict()
for ens_name in ens_list:
    print("Ensemble: ", ens_name)
    T = ens_info[ens_name]["T"]
    T_ext = T // 2 + 1 
    t_vals = np.arange(T_ext)
    a_fm = ens_info[ens_name]["a_fm"]
    m_mu_phys_MeV = constants.masses_MeV["mu"]
    a_MeV_inv = constants.fm_to_MeV_inv(a_fm)
    am_mu = a_MeV_inv*m_mu_phys_MeV # physical muon mass
    for fermion_type in ["OS", "tm"]:
        print("  Fermions: ", fermion_type)
        Q_sqr = 4/9 + 1/9 # charge factors for the light quarks contribution
        VKVK = Q_sqr * VKVK_pp_dict[ens_name][fermion_type]
        Z_ren = ens_info[ens_name][Z_ren_mapping[fermion_type]]
        a_mu_val[ens_name][fermion_type] = np.zeros(shape=(N_bts, n_omega))
        for i_omega, omega in enumerate(omega_vals):
            for i_bts in range(N_bts):
                a_mu_val[ens_name][fermion_type][i_bts][i_omega] = get_amu(ti=t_vals[1:], Vi=VKVK[i_bts,1:], K=K_values[ens_name][i_omega], Z_ren=Z_ren, strategy="trapezoidal")
            #---
        #---
        a_mu_avg = np.average(a_mu_val[ens_name][fermion_type], axis=0)
        a_mu_err = np.std(a_mu_val[ens_name][fermion_type], axis=0, ddof=1)
        y = (1-omega_vals)*a_mu_avg  # / (alpha_em/np.pi)
        ey = (1-omega_vals)*a_mu_err # / (alpha_em/np.pi)
        marker = "o" if fermion_type=="tm" else "^"
        plt.errorbar(x=omega_vals, y=y, yerr=ey, label=f"Ens: {ens_name}, fermions: {fermion_type}", marker=marker, markersize=0.5)
    #---
    plt.vlines([am_mu], ymin=0.0, ymax=np.max(y), label=f"$am_\\mu^\\mathrm{{phys}}$ for {ens_name}", linestyles=["--"], linewidths=[0.75], color=[colors[ens_name]])
#---

# plt.plot(omega_vals, (1 - omega_vals) * A_sing_vals, label="Phenomenological model")


plt.legend()
plt.xlabel("$\\omega$")
plt.ylabel("$a_\\mu$")
plt.title("c.f. Fig. 5 of https://arxiv.org/pdf/2311.11597")
plt.savefig("./a_mu-mass_dependence.svg")
# plt.show()
plt.close()

os.makedirs("./auxiliary_data", exist_ok=True)
with open("auxiliary_data/a_mu_omega.pkl", "wb") as f:
    dill.dump(a_mu_val, f)
