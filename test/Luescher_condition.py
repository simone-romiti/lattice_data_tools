"""
Luescher quantization condition for PP states in a finite volume.
I use the Gounaris-Sakurai model to describe the phase shift
$$ \\delta_11(k) + \\phi(q=kL/(2\\pi)) = n \\pi $$
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
from lattice_data_tools.gm2.HVP.Z_function import Z_00_Calculator
from lattice_data_tools.gm2.HVP.Gounaris_Sakurai_model import GS_model

# --- shared physics constants and objects ---
hbarc_MeV_fm = 197.3269631
a_fm = 0.089
Nx = 24
MP_MeV = 320
MV_MeV = 2.77 * MP_MeV
g_VPP = 5.22

a_MeV_inv = a_fm / hbarc_MeV_fm
aMP = a_MeV_inv * MP_MeV
aMV = a_MeV_inv * MV_MeV

N_gauss = 100
Lambda = 1.0
Lambda_Z3 = 5
N_lev = 10
q2_max = N_lev**2
k_max = 1.0 

Z_00_obj = Z_00_Calculator(Lambda_Z3=Lambda_Z3, Lambda=Lambda, N_gauss=N_gauss, q2_max=q2_max)
GS_obj = GS_model(MP=aMP, MV=aMV, g_VPP=g_VPP)


# --- tests ---

def test_aMP_aMV_in_lattice_units():
    """Lattice masses should be small numbers (<<1 in lattice units)."""
    assert 0.0 < aMP < 0.5, f"aMP={aMP} unexpectedly large"
    assert 0.0 < aMV < 1.0, f"aMV={aMV} unexpectedly large"


def test_rho_heavier_than_pion():
    assert aMV > aMP


def test_delta_11_increases_near_resonance():
    """Phase shift should rise through pi/2 near the rho mass."""
    k_res = np.sqrt((aMV / 2)**2 - aMP**2)
    k_below = k_res * 0.8
    k_above = k_res * 1.2
    assert GS_obj.delta_11(k_below) < np.pi / 2 < GS_obj.delta_11(k_above)


def test_delta_11_range():
    """Phase shift should stay within [0, pi] for physical k values."""
    k_vals = np.linspace(0.01, 1.5, 200)
    deltas = np.array([GS_obj.delta_11(k) for k in k_vals])
    assert np.all(deltas >= 0), "Phase shift went negative"
    assert np.all(deltas <= np.pi + 1e-10), "Phase shift exceeded pi"


def test_phi_increases_between_poles():
    """phi(q) should increase monotonically in each interval between poles.
    Poles occur at q = sqrt(n^2) for integer lattice vectors n, i.e. q ~ 1, sqrt(2), sqrt(3)...
    We test on a safe window well away from any pole.
    """
    # q in (0.1, 0.9) is safely below the first pole at q=1
    q_vals = np.linspace(0.1, 0.9, 50)
    phi_vals = np.array([Z_00_obj.phi(q) for q in q_vals])
    assert np.all(np.diff(phi_vals) > 0), "phi(q) not increasing in (0.1, 0.9)"


# def test_phi_has_poles_at_shells():
#     """phi(q) should diverge near shell crossings (where q^2 = integer).
#     We test that |phi| is large close to the first shell at q=1,
#     and that a sign change occurs somewhere in (0.1, 3.0) indicating a pole.
#     """
#     # phi should be large (diverging) very close to q=1
#     phi_near_pole = abs(Z_00_obj.phi(1.001))
#     assert phi_near_pole > 1.0, \
#         f"phi does not appear to diverge near q=1, got |phi|={phi_near_pole}"

#     # There should be at least one sign change in (0.1, 3.0)
#     q_vals = np.linspace(0.1, 3.0, 500)
#     phi_vals = np.array([Z_00_obj.phi(q) for q in q_vals])
#     sign_changes = np.sum((phi_vals[:-1] * phi_vals[1:]) < 0)
#     assert sign_changes >= 1, "No sign change (pole) found in phi for q in (0.1, 3.0)"
    

def test_quantization_condition_has_expected_levels():
    """delta_11 + phi should cross n*pi at least once."""
    k_vals = np.linspace(0.01, 1.5, 1000)
    q_vals = k_vals * Nx / (2.0 * np.pi)
    lhs = np.array([GS_obj.delta_11(k) for k in k_vals])
    lhs += np.array([Z_00_obj.phi(q) for q in q_vals])
    crossings = 0
    for n in range(1, N_lev + 1):
        diff = lhs - n * np.pi
        crossings += np.sum((diff[:-1] * diff[1:]) < 0)
    assert crossings >= 1, "No energy levels found from quantization condition"


if __name__ == "__main__":
    print("aMP:", aMP)
    print("aMV:", aMV)

    k_vals = np.linspace(0.01, k_max, 500)
    q_vals = k_vals * Nx / (2.0 * np.pi)

    delta_11 = np.array([GS_obj.delta_11(k) for k in k_vals])
    phi_vals  = np.array([Z_00_obj.phi(q) for q in q_vals])

    plt.figure(figsize=(8, 6))
    plt.plot(k_vals, delta_11 + phi_vals, label=r'$\delta_{11}(k) + \phi(q)$')
    for n in range(N_lev + 1):
        plt.axhline(n * np.pi, color='gray', linestyle='--', linewidth=0.8,
                    label=r'$n\pi$' if n == 0 else None)
    plt.xlabel(r'$k$')
    plt.ylabel(r'$\delta_{11}(k) + \phi(q)$')
    plt.title(r'$\delta_{11}(k) + \phi(q)$ and $n\pi$ lines')
    plt.legend()
    plt.tight_layout()
    plt.show()
