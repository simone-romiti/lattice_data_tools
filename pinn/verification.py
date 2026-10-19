import torch
import numpy as np
from .envelope import EnvelopeWavefunction, analytic_envelope_alpha
from .hamiltonian import KogutSusskindHamiltonian, electric_coefficient
from .links.configuration import GaugeConfiguration
import random

def fresh_config_verification(
    model: KogutSusskindHamiltonian,
    E: float,
    Nc: int,
    L_mu: list[int] = [],
    n_fresh: int = 20,
    seed: int = 0,
    elec_coeff: float = 1.0,
    eps_eh_fresh: float = 1e-3,
):
    """
    Sample NEW Langevin configs (different seed) and compute expectation ⟨H_total⟩.
    A global expectation value ⟨H_total⟩ is computed by averaging ⟨ψ|H_total|ψ⟩ over configurations from fresh_configs.
    It returns a dict with verification metrics.
    """
    rng = random.Random(seed)
    configs = langevin_configs("temp_dir", Nc=Nc, L_mu=L_mu, n_fresh=n_fresh, seed=seed)
    # Generate new configs with different seed
    fresh_configs = gauge_conf.sample_step(config=random.choice(configs))
    model.set_beta(beta=0.5)  # For sanity check, set a mid-beta
    psi_batch = [model.sample_step(config=config) for config in fresh_configs]
    H_total_batch = [model.H_psi(psi_val, gauge_conf=config) for psi_val in psi_batch]
    mean_he = np.mean(torch.stack(H_total_batch))
    std_he = np.std(torch.stack(H_total_batch))
    return {
        "verified": mean_he < eps_eh_fresh,
        "HE_fresh": mean_he,
        "|E - HE_fresh|": abs(E - mean_he),
        "std_Hloc": np.std(torch.stack(H_total_batch))
    }

def gate_a_truth_test(
    Nc: int,
    L_mu: list[int] = [],
    eigenstates: list[torch.Tensor] = [],
    elec_coeff: float = 1.0,
):
    """
    Verify analytic eigenstates at beta=0: for each eigenstate, compute H_E·ψ/ψ and check it equals E_0.
    """
    for state_idx, state in enumerate(eigenstates):
        # Theoretical H_E_psi / psi_analytic should equal E(state)[n_state_idx]
        # For each eigenstate at beta=0, compute H_E_psi at low beta
        # Need analytic_envelope_alpha(beta=0)=1.0 so should converge to E(state)
        if beta == 0:
            he_psi_theo = ...  # need implementation to compute H_E_psi at beta≈0
            he_sim = model.H_psi(psi_analytic(eigenstate, beta=0.0), U_conf=U_conf)
            if not torch.allclose(he_sim, E[state_idx], atol=1e-6):
                return {"pass": False, "state_idx": state_idx}
    return {"pass": True, "state_indices": []}