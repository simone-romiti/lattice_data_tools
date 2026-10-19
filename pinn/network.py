"""
Real-output wavefunction networks for SU(N) PINNs.

CONVENTIONS (do not regress):
  * SINGLE REAL output: the network returns a real number, wrapped as
    `complex128` with zero imaginary part: `psi = out_real + 0j`.
    The legacy complex (Re, Im) pair trained an unconstrained imaginary
    part via |psi|^2 = Re^2 + Im^2 and |(H-E)psi|^2 = Re_res^2 + Im_res^2,
    causing instabilities. With a single real output, both losses train
    ONLY the physical real eigenstate.
  * TRIGONOMETRIC activation (`torch.sin`), NOT `tanh`. The Mathieu
    eigenstates at beta=0 are cos(n*theta), sin(n*theta) — literally
    trigonometric — so a sin activation is a natural basis and PREVENTS
    variational collapse (tanh collapsed at beta=0.45 for n=1).
    Improvement factors (U(1) single-link, beta=0->0.5, real+envelope):
      n=0: 6.9x, n=1: 32.3x, n=2: 1.7x.
    USE TRIG THROUGHOUT. Do not regress to tanh.

The single-link case uses a plain `RealMLP` on the trace features
Re[Tr(U^n)]. The multi-link case (2d_L2, 2d_L3, ...) uses `RealLCNNMLP`,
which wraps the existing `lattice_data_tools.machine_learning.lcnn_mlp.LCNN_MLP`
with `act_fun_MLP = torch.sin`. The LCNN feature extractor learns to
ISOLATE the plaquette trace Re[Tr(P)]; once isolated, the problem reduces
to the single-link case where the "link" is the plaquette.
"""

import typing

import torch

from lattice_data_tools.links.configuration import GaugeConfiguration

# Default activation. Do NOT change to torch.tanh — see module docstring.
TRIG_ACTIVATION: typing.Callable = torch.sin


class RealMLP(torch.nn.Module):
    """Plain MLP with trigonometric (sin) activation and a SINGLE REAL output.

    Intended for the single-link case, where the input is a feature vector
    of real traces (e.g. [Re Tr(U), Re Tr(U^2), ...] for a chosen set of
    powers). The output is a real scalar wrapped as complex128 with zero
    imaginary part, so it composes with the Hamiltonian (which expects a
    complex wavefunction).
    """

    def __init__(
        self,
        n_in: int,
        n_hidden: int = 32,
        n_layers: int = 2,
        activation: typing.Callable = TRIG_ACTIVATION,
        dtype: torch.dtype = torch.complex128,
    ) -> None:
        super().__init__()
        self.activation = activation
        self.dtype = dtype

        layers = [torch.nn.Linear(n_in, n_hidden), self.activation]
        for _ in range(n_layers - 1):
            layers += [torch.nn.Linear(n_hidden, n_hidden), self.activation]
        # Final layer: SINGLE real output (NOT a complex Re,Im pair).
        layers += [torch.nn.Linear(n_hidden, 1)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return psi(x) as a complex128 tensor with zero imaginary part.

        The network itself outputs a real number; we wrap it so the
        Hamiltonian operator (which is defined on complex wavefunctions)
        receives a well-typed input.
        """
        out_real = self.net(x).squeeze(-1)
        return out_real.to(torch.complex128)


class RealLCNNMLP(torch.nn.Module):
    """LCNN + MLP wavefunction with trig activation and a SINGLE REAL output.

    Wraps `lattice_data_tools.machine_learning.lcnn_mlp.LCNN_MLP` with
    `act_fun_MLP = torch.sin` (the default here; the upstream default of
    `torch.nn.Tanh()` is a known regression — see module docstring).

    The LCNN (arXiv:2012.12901) produces gauge-invariant features by
    taking traces of parallel-transported Wilson loops. On 2D L>=2 lattices
    the LCNN feature extractor learns to ISOLATE the plaquette trace
    Re[Tr(P)]; once isolated, the multi-link problem reduces to the
    single-link case where the "link" is the plaquette.
    """

    def __init__(
        self,
        U: GaugeConfiguration,
        LCNN_layer,
        LCNN_N_in: int,
        LCNN_N_out: int,
        N_hidden: int = 2,
        N_neurons: typing.List[int] = (32, 32),
        seed: int = 0,
        activation: typing.Callable = TRIG_ACTIVATION,
    ) -> None:
        super().__init__()
        # Local import to avoid a hard dependency at module import time
        # (LCNN_MLP pulls in the full lattice_data_tools ML stack).
        from lattice_data_tools.machine_learning.lcnn_mlp import LCNN_MLP

        # Use trig (sin) as the MLP activation, NOT the upstream default tanh.
        self._base = LCNN_MLP(
            U=U,
            LCNN_layer=LCNN_layer,
            LCNN_N_in=LCNN_N_in,
            LCNN_N_out=LCNN_N_out,
            N_hidden=N_hidden,
            N_neurons=list(N_neurons),
            seed=seed,
            act_fun_MLP=activation,
        )

    def forward(self, U: GaugeConfiguration) -> torch.Tensor:
        # The wrapped LCNN_MLP already outputs a single real number (its
        # MLP ends with Linear(N_neurons[-1], 1)); we just ensure the
        # complex128 dtype with zero imaginary part for downstream use.
        out_real = self._base(U=U).squeeze(-1)
        return out_real.to(torch.complex128)


def make_real_wavefunction(
    geometry: str,
    U: typing.Optional[GaugeConfiguration] = None,
    n_in: int = 8,
    n_hidden: int = 32,
    n_layers: int = 2,
    activation: typing.Callable = TRIG_ACTIVATION,
    **lcnn_kwargs,
) -> torch.nn.Module:
    """Factory selecting the right real wavefunction network.

    Parameters
    ----------
    geometry : str
        One of "single_link", "2d_L2", "2d_L3", "3d_L2", ... For
        "single_link" a plain RealMLP is returned; for any multi-link
        geometry a RealLCNNMLP is returned (requires `U` and `LCNN_layer`).
    U : GaugeConfiguration, optional
        Required for multi-link geometries (to build the LCNN). Ignored
        for single_link.
    n_in : int
        Number of real-trace input features for the single-link MLP.
    """
    if geometry == "single_link":
        return RealMLP(
            n_in=n_in, n_hidden=n_hidden, n_layers=n_layers,
            activation=activation,
        )
    if U is None:
        raise ValueError(
            f"make_real_wavefunction(geometry={geometry!r}) requires a "
            f"GaugeConfiguration `U` to build the LCNN feature extractor."
        )
    return RealLCNNMLP(U=U, activation=activation, **lcnn_kwargs)
