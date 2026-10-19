"""
Envelope wavefunction: psi(U; beta) = alpha(beta) * psi_analytic(U) + psi_net(U)

The analytic envelope provides a STRONG prior that keeps the optimization
in the correct basin:

  * At beta = 0, alpha = 1 and psi_net -> 0 (trivial MSE loss), so
    psi = psi_analytic exactly. This is the strong-coupling fixed point.
  * At beta > 0, alpha < 1 and psi_net learns the SMALL deviation from
    the analytic eigenstate.
  * The default schedule is alpha(beta) = exp(-0.5 * beta), PROVEN on the
    U(1) single link. An alternative alpha = 1/(1+beta) is also valid.

Do NOT make alpha a learnable parameter — it is a fixed schedule. The
network trains psi_net only.

For the single U(1)/SU(N) link at beta = 0 the analytic eigenstates are
  psi_n = sqrt(2) * Re[Tr(U)^n]   (REAL EVEN, normalized to <psi|psi>=1
                                  on the Haar measure)
which tracks the a_{2n} (even, ce_{2n}) branch throughout the adiabatic
evolution. For 2d_L2+ the LCNN isolates the plaquette trace and the same
envelope applies with the plaquette as the "link".
"""

import math
import typing

import torch

from lattice_data_tools.links.configuration import GaugeConfiguration


def analytic_envelope_alpha(beta: float, c: float = 0.5) -> float:
    """Default envelope schedule alpha(beta) = exp(-c * beta).

    At beta = 0, alpha = 1 (full analytic). At beta = 0.5 (default
    beta_max), alpha ~ 0.78, giving the network ~22% freedom. The
    alternative `alpha = 1/(1+beta)` is also valid; pass `c=0` and use
    that schedule explicitly if desired.
    """
    return math.exp(-c * beta)


class EnvelopeWavefunction(torch.nn.Module):
    """psi(U; beta) = alpha(beta) * psi_analytic(U) + psi_net(U).

    Parameters
    ----------
    psi_analytic : callable
        EXACT beta = 0 eigenstate, psi_analytic(U) -> complex128 tensor.
        Fixed (no gradient through network params) but differentiable w.r.t.
        U through the trace.
    psi_net : torch.nn.Module
        Trainable correction, initially ~ 0. Must output a SINGLE REAL
        number (RealMLP or RealLCNNMLP from this package).
    c : float, default 0.5
        Envelope decay rate. alpha(beta) = exp(-c * beta).

    Attributes
    ----------
    alpha_value : float
        The current envelope coefficient (set via `set_beta`).
    """

    def __init__(
        self,
        psi_analytic: typing.Callable,
        psi_net: torch.nn.Module,
        c: float = 0.5,
    ) -> None:
        super().__init__()
        self.psi_analytic = psi_analytic
        self.psi_net = psi_net
        self.c = c
        self.alpha_value: float = 1.0  # set by set_beta()

    def set_beta(self, beta: float) -> None:
        """Update the envelope coefficient for the current beta step."""
        self.alpha_value = analytic_envelope_alpha(beta, c=self.c)
        self.beta = float(beta)

    def forward(self, U: typing.Union[GaugeConfiguration, torch.Tensor]) -> torch.Tensor:
        """Return psi(U) = alpha * psi_analytic(U) + psi_net(U) (complex128)."""
        psi_a = self.psi_analytic(U)
        psi_n = self.psi_net(U)
        # alpha is a fixed scalar (no grad); broadcast over the batch.
        alpha = torch.as_tensor(
            self.alpha_value, dtype=psi_a.dtype, device=psi_a.device
        )
        return alpha * psi_a + psi_n

    def envelope_weight(self) -> float:
        """Return the current analytic weight alpha (for logging)."""
        return self.alpha_value
