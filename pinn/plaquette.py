import torch
import numpy as np
from .network import RealMLP, RealLCNNMLP, make_real_wavefunction
from .envelope import EnvelopeWavefunction, analytic_envelope_alpha
from .hamiltonian import KogutSusskindHamiltonian, electric_coefficient
from .links.configuration import GaugeConfiguration
import random

def plaquette_expectation(model: RealMLP | RealLCNNMLP, cfgs: list[GaugeConfiguration], Nc: int, n_batches: int = 8):
    """
    Compute average plaquette expectation value ⟨P⟩ from MPS pattern
    ⟨P⟩ = Σ_b ψ² · O_b / Σ_b ψ²
    O_b = 1 - Re[Tr(P_b)] / Nc
    """
    psi_batch = []
    for cfg in cfgs:
        psi = model(sample_cfgr(cfg, batch=True))
        psi_batch.append(psi)
    mean_plaquette = torch.stack(psi_batch).mean(dim=-1)
    return mean_plaquette

def plaquette_smoothness_check(history: torch.Tensor, eps_plaq: float=0.05, n_sigma: int = 3):
    """
    Check ⟨P⟩ varies smoothly across β steps (no jumps > n_sigma*SEM).
    Returns dict with pass:bool, jumps:list
    """
    jumps = []
    SEM = ...  # compute standard error
    for idx in range(len(history)-1):
        diff = abs(history[idx+1] - history[idx])
        if diff > n_sigma * SEM[idx]:
            jumps.append(abs(history[idx+1] - history[idx]))
    return {"pass": len(jumps) == 0,
           "jumps": jumps}