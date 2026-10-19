"""
Kogut-Susskind Hamiltonian for SU(N) pure-gauge lattice PINNs.

CONVENTION (paper arXiv:2510.26904; user 2026-09-16):
    H_total = beta * H
            = (beta * g^2 / 2) * sum_{x,mu,a} L_a^2
              + beta^2 * sum_{square} (1 - Re[Tr(P_square)])
            = N_c * H_E_raw  +  beta^2 * H_B

Key property: the eigenfunctions of `beta * H` are IDENTICAL to the
paper's eigenfunctions of `H`; only the eigenvalues are rescaled by
`beta`. So the network architecture, sampling, and loss are unchanged;
only the energy values differ by a factor of `beta`.

  * H_E (electric): coefficient is the FIXED constant N_c
    (since beta * g^2 / 2 = N_c), NOT the running g^2/2.
    Use `La_squared_per_link` (links/lie_derivatives.py) for the
    2nd-order gauge-link derivative. Forbidden: 1st-order La/Ra
    approximations of the electric term.
    Code convention: `La_squared_per_link` returns n^2/2 for U(1). To
    get N_c * n^2 use `ELECTRIC_COEFF = 2 * Nc`.
  * H_B (magnetic, Wilson): coefficient is the CURRENT beta^2
    (NOT beta). For the single U(1) plaquette, H_B = 1 - Re[Tr(U)] =
    1 - cos(theta) != 0 (the single link IS the plaquette, paper sec 4.1).
    Use `single_link_HB(U)` (this module), NOT `compute_HB(cfg)` which
    returns 0 for 1D lattices (a latent bug that hides beta-dependent
    physics).
  * The SE residue MUST use the FULL Hamiltonian
    `H_psi = N_c * H_E_psi + beta^2 * H_B * psi` with the CURRENT beta,
    even where H_B happens to vanish (multi-link). A placeholder such as
    `Hpsi = H_E_psi + 0.0*beta` is a latent bug for the next phase.

DO NOT edit `lattice_data_tools/links/suN.py`. If a change is needed to
`get_exp_iA` or other core utilities, ask the user first.
"""

import typing

import torch

from lattice_data_tools.links.configuration import GaugeConfiguration

# Coefficient for the electric term in the beta-rescaled Hamiltonian:
#   H_E = (beta * g^2 / 2) * sum L_a^2 = N_c * sum L_a^2
# `La_squared_per_link` returns n^2/2 for U(1) (generator Tr(t_0^2)=1/2),
# so to get N_c * n^2 we multiply by ELECTRIC_COEFF = 2 * Nc.
def electric_coefficient(Nc: int) -> float:
    """Return the FIXED electric coefficient 2 * Nc for the beta-rescaled H."""
    return 2.0 * float(Nc)


def single_link_HB(U: torch.Tensor) -> torch.Tensor:
    """Magnetic term for the SINGLE plaquette (= the link itself).

    For the single U(1)/SU(N) plaquette, H_B = 1 - Re[Tr(U)]
    (paper sec 4.1). This is NONZERO even though a 1D lattice has no
    plaquettes in the Wilson-loop sense — the single link IS the
    plaquette. `compute_HB(cfg)` returns 0 for 1D lattices and is a
    latent bug; use THIS function for the single link.
    """
    # Re[Tr(U)]; U has shape (..., Nc, Nc).
    re_tr = suN_ReTr(U)
    return 1.0 - re_tr


def compute_HB(U: GaugeConfiguration) -> torch.Tensor:
    """Magnetic term summed over all plaquettes of a multi-link lattice.

    Returns the SUM of (1 - Re[Tr(P_square)]) over all plaquettes.
    For 1D lattices (no plaquettes) this returns 0 — for the single
    link, use `single_link_HB(U)` instead, because the single link IS
    the plaquette (paper sec 4.1).
    """
    from lattice_data_tools.links.loops import WilsonLoopsGenerator

    plaq = WilsonLoopsGenerator.plaquettes(U=U)  # (..., L1..Ld, ndims**2, Nc, Nc)
    re_tr = suN_ReTr(plaq)
    # Sum over plaquette indices. re_tr has shape (..., n_plaquettes).
    return (1.0 - re_tr).sum(dim=-1)


def suN_ReTr(W: torch.Tensor) -> torch.Tensor:
    """Real part of the trace of a batch of color matrices.

    Thin wrapper around `lattice_data_tools.links.suN.get_ReTr` so the
    public API of this module does not require callers to know about suN.
    """
    import lattice_data_tools.links.suN as suN

    return suN.get_ReTr(W=W)


class KogutSusskindHamiltonian:
    """beta-rescaled Kogut-Susskind Hamiltonian.

    H_total = N_c * H_E_raw + beta^2 * H_B

    where:
      * H_E_raw = sum_{x,mu,a} L_a^2  (computed via La_squared_per_link)
      * H_B     = sum_{square} (1 - Re[Tr(P_square)])  (single_link_HB
        for the single link, compute_HB for multi-link)

    Parameters
    ----------
    Nc : int
        Number of colors (N_c). Determines the FIXED electric
        coefficient via `electric_coefficient(Nc) = 2 * Nc`.
    geometry : str
        "single_link" or any multi-link geometry (e.g. "2d_L2"). For
        "single_link" the magnetic term uses `single_link_HB`; for
        multi-link it uses `compute_HB`.
    La_squared_per_link : callable, optional
        The canonical 2nd-order Lie derivative routine. If None, it is
        built lazily from the gauge configuration on first use (requires
        `from lattice_data_tools.links.lie_derivatives import ...`).
    """

    def __init__(
        self,
        Nc: int,
        geometry: str = "single_link",
        La_squared_per_link: typing.Optional[typing.Callable] = None,
    ) -> None:
        self.Nc = int(Nc)
        self.geometry = geometry
        self.electric_coeff = electric_coefficient(self.Nc)
        self._La_squared_per_link = La_squared_per_link
        self.beta = 1.0  # set by set_beta()

    def set_beta(self, beta: float) -> None:
        """Update the current beta (used for the beta^2 H_B coefficient)."""
        self.beta = float(beta)

    def H_E_psi(self, psi: typing.Callable, U) -> torch.Tensor:
        """Electric term applied to psi: N_c * sum_{x,mu,a} L_a^2 psi.

        `psi` is a callable taking U and returning a complex128 tensor.
        `La_squared_per_link` is the canonical 2nd-order Lie derivative
        (links/lie_derivatives.py); 1st-order approximations are
        forbidden.
        """
        if self._La_squared_per_link is None:
            # Lazy construction from the canonical momenta machinery.
            from lattice_data_tools.links.lie_derivatives import (
                WithAutodifferentiation,
            )

            cm = WithAutodifferentiation(U=U)
            self._La_squared_per_link = cm.get_La2_per_link

        # La_squared_per_link returns (batchsize, Ng) of d^2 psi / d omega_a^2.
        La2 = self._La_squared_per_link(psi, U)
        # Sum over the generator index a, then apply the FIXED electric
        # coefficient 2 * Nc (so H_E = N_c * sum_a L_a^2 for U(1) where
        # La^2 returns n^2/2).
        return self.electric_coeff * La2.sum(dim=-1)

    def H_B_value(self, U) -> torch.Tensor:
        """Magnetic term (a scalar observable per config), NOT applied to psi."""
        if self.geometry == "single_link":
            return single_link_HB(U)
        return compute_HB(U)

    def H_psi(self, psi: typing.Callable, U) -> torch.Tensor:
        """FULL Hamiltonian applied to psi, with the CURRENT beta.

        H_psi = N_c * H_E_psi(psi, U) + beta^2 * H_B_value(U) * psi(U)

        Even where H_B happens to vanish (multi-link 1D, which is
        DISCARDED anyway), the beta^2 * H_B * psi term MUST be present
        with the real `H_B_value(U)` and the current beta. A placeholder
        such as `H_psi = H_E_psi + 0.0 * beta` is a latent bug for the
        next phase.
        """
        H_E = self.H_E_psi(psi, U)
        H_B = self.H_B_value(U)
        psi_val = psi(U)
        return H_E + (self.beta ** 2) * H_B * psi_val
