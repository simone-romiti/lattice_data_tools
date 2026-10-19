"""
lattice_data_tools.pinn — Physics-Informed Neural Network (PINN) toolkit for
SU(N) pure-gauge lattice Hamiltonians.

This subpackage codifies the conventions, architecture, loss, sampling,
convergence gates, and verification procedures used for the adiabatic
evolution of SU(N) PINN eigenstates from $\beta = 0$ onwards, following
the SOP in the `pinn-sun` opencode skill (paper arXiv:2510.26904).

Key conventions (see README.md for the full exposition):
  * Hamiltonian: $H_{\\rm total} = \\beta H = \\beta\\,H_E + \\beta^{2}\\,H_B$
    (the $\\beta g^{2}/2 = N_c$ rescaling avoids the $\\beta \\to 0$ blowup).
  * Wavefunction: a SINGLE REAL output (NOT complex); wrapped as
    `complex128` with zero imaginary part.
  * Activation: TRIGONOMETRIC (`torch.sin`), NOT `tanh`. USE TRIG THROUGHOUT.
  * Envelope: $\\psi = \\alpha(\\beta)\\,\\psi_{\\rm analytic} + \\psi_{\\rm net}$,
    $\\alpha(\\beta) = e^{-0.5\\beta}$.
  * Loss: SE residue on HAAR configs + normalization on LANGEVIN configs
    with importance reweighting. NOT Metropolis. NOT $\\langle H\\rangle - E$.
  * Convergence gate: |E - <H>_Langevin| < 1e-3 (HARD gate, LANGEVIN
    not Haar; Haar <H> is biased and can hide false convergence).
  * Fresh-config verification: MANDATORY at every beta > 0 step
    (relaxed at beta = 0 via --skip-beta0-se).

DO NOT edit `lattice_data_tools/links/suN.py`. If a change is needed to
the core utilities (`get_exp_iA`, etc.), ask the user first.
"""

from .network import RealMLP, RealLCNNMLP, make_real_wavefunction
from .envelope import EnvelopeWavefunction, analytic_envelope_alpha
from .hamiltonian import KogutSusskindHamiltonian, single_link_HB, compute_HB
from .sampling import haar_configs, langevin_configs
from .loss import se_residue_loss, normalization_loss, total_loss
from .convergence import check_plateau, PLATEAU_REACHED, CONTINUE_TRAINING, STUCK, NOT_CONVERGED
from .verification import fresh_config_verification, VERIFIED, OVERFIT
from .plaquette import plaquette_expectation, plaquette_smoothness_check

__all__ = [
    # network
    "RealMLP",
    "RealLCNNMLP",
    "make_real_wavefunction",
    # envelope
    "EnvelopeWavefunction",
    "analytic_envelope_alpha",
    # hamiltonian
    "KogutSusskindHamiltonian",
    "single_link_HB",
    "compute_HB",
    # sampling
    "haar_configs",
    "langevin_configs",
    # loss
    "se_residue_loss",
    "normalization_loss",
    "total_loss",
    # convergence
    "check_plateau",
    "PLATEAU_REACHED",
    "CONTINUE_TRAINING",
    "STUCK",
    "NOT_CONVERGED",
    # verification
    "fresh_config_verification",
    "VERIFIED",
    "OVERFIT",
    # plaquette
    "plaquette_expectation",
    "plaquette_smoothness_check",
]
