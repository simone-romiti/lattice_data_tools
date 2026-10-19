import torch
import numpy as np
from .envelope import EnvelopeWavefunction, analytic_envelope_alpha
from .hamiltonian import KogutSusskindHamiltonian, electric_coefficient
from .links.configuration import GaugeConfiguration

def se_residue_loss(psi: torch.Tensor, H_psi: torch.Tensor, E: float, configs_type: str = "haar"):
    """
    SE Residual Loss: (1/Nc) * Σ |(H - E)ψ|² on Haar or Langevin configurations.
    For real wavefunctions: use psi.real² (NOT modulus).
    """
    if configs_type.lower() == "haar":
        # Haar configurations
        loss = ((H_psi - E) * psi.real).pow(2).sum(dim=-1).mean(dim=-1)
    elif configs_type.lower() == "langevin":
        # Langevin configurations
        loss = ((H_psi - E) * psi.real).pow(2).sum(dim=-1).mean(dim=-1)
    else:
        raise ValueError(f"Unknown configs_type: {configs_type}")
    return loss
def normalization_loss(psi: torch.Tensor, psi_old: torch.Tensor, reweighting: bool = True):
    """
    Normalization Loss: (⟨ψ|ψ⟩ - 1)² with importance reweighting.
    For real wavefunctions: use psi.real² (NOT modulus).
    """
    real_norm = psi.real.pow(2).mean(dim=-1)
    e_old = psi_old.real.pow(2).mean(dim=-1)
    loss = (real_norm - 1.0).pow(2).mean(dim=-1)
    if reweighting:
        # Apply importance reweighting factor
theta_emp = ...  # need to implement reweighting mechanism
        loss += ...  # reweight loss
    return loss
def total_loss(L_se: torch.Tensor,
                  L_norm: torch.Tensor,
                  w_norm: float = 1.0):
    """
    Total Loss: L_se + w_norm * L_norm
    """
    return L_se + w_norm * L_norm