## Fixed SU(3) Casimir Check — PyTorch version
import torch
import torch.func as TF   # vmap, jvp

# ── dtype ─────────────────────────────────────────────────────────────────────
DTYPE  = torch.complex128
FDTYPE = torch.float64

# ── Gell-Mann matrices ────────────────────────────────────────────────────────
def get_gell_mann() -> torch.Tensor:
    j = 1j
    lams = [
        [[0,  1,  0 ], [ 1,  0,  0], [0,  0,  0]],
        [[0, -j,  0 ], [ j,  0,  0], [0,  0,  0]],
        [[1,  0,  0 ], [ 0, -1,  0], [0,  0,  0]],
        [[0,  0,  1 ], [ 0,  0,  0], [1,  0,  0]],
        [[0,  0, -j ], [ 0,  0,  0], [j,  0,  0]],
        [[0,  0,  0 ], [ 0,  0,  1], [0,  1,  0]],
        [[0,  0,  0 ], [ 0,  0, -j], [0,  j,  0]],
    ]
    l8 = (1 / 3**0.5) * torch.tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=DTYPE
    )
    return torch.stack([torch.tensor(m, dtype=DTYPE) for m in lams] + [l8])

LAMBDAS  = get_gell_mann()   # (8, 3, 3)
C2_EXACT = 4 / 3.0

# ── Left rotation ─────────────────────────────────────────────────────────────
def left_rotation(t: torch.Tensor, gen: torch.Tensor) -> torch.Tensor:
    """exp(i * t/2 * gen).  t is a real scalar, gen is (3,3) complex."""
    return torch.linalg.matrix_exp(1j * (t / 2.0) * gen)

# ── Per-generator metrics (JVP-based) ────────────────────────────────────────
def _metrics_one_gen(t0: torch.Tensor, U: torch.Tensor,
                     gen: torch.Tensor, psi_fn) -> tuple:
    """First and second Lie derivative of psi along one generator."""
    one = torch.ones((), dtype=FDTYPE)

    def f(t):
        return psi_fn(left_rotation(t, gen) @ U)

    # L_a psi
    _, g = TF.jvp(f, (t0,), (one,))

    # L_a^2 psi  (mirrors the JAX double-jvp pattern)
    def df_dt(t):
        _, gt = TF.jvp(f, (t,), (one,))
        return 1j * gt

    _, jl2 = TF.jvp(df_dt, (t0,), (one,))
    return g, 1j * jl2

# ── Per-sample metrics: vmap over all 8 generators ───────────────────────────
def get_physics_metrics(U: torch.Tensor, psi_fn) -> tuple:
    t0 = torch.zeros((), dtype=FDTYPE)

    def per_gen(gen):
        return _metrics_one_gen(t0, U, gen, psi_fn)

    # vmap over (8, 3, 3) → grads/laps each shape (8,)
    grads, laps = TF.vmap(per_gen)(LAMBDAS)

    grad_sq   = torch.sum(torch.abs(grads) ** 2)
    laplacian = torch.sum(laps)
    return grad_sq, laplacian

# ── Haar-random SU(3) samples ─────────────────────────────────────────────────
def sample_su3(N: int, seed: int = 42) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    z = (torch.randn(N, 3, 3, dtype=FDTYPE, generator=rng)
       + 1j * torch.randn(N, 3, 3, dtype=FDTYPE, generator=rng)).to(DTYPE) / 2**0.5
    Q, R = torch.linalg.qr(z)
    phases = torch.diagonal(R, dim1=-2, dim2=-1)
    Q = Q * (phases / torch.abs(phases))[:, None, :]          # phase-fix
    return Q / torch.linalg.det(Q)[:, None, None] ** (1 / 3)  # → SU(3)

# ── Main check ────────────────────────────────────────────────────────────────
def run_check(name: str, psi_fn, N: int = 50_000):
    samples = sample_su3(N)                                    # (N, 3, 3)

    # vmap psi over all N samples
    psi_vals = TF.vmap(psi_fn)(samples)                        # (N,)

    # vmap [grad_sq, laplacian] over all N samples
    # (each call already vmaps internally over the 8 generators)
    def per_sample(U):
        return get_physics_metrics(U, psi_fn)

    grad_sq_vals, lap_vals = TF.vmap(per_sample)(samples)      # (N,), (N,)

    # 1. Pointwise Check: (L^2 psi) / psi ≈ -C2
    mask         = torch.abs(psi_vals) > 1e-2
    pointwise_c2 = torch.mean((lap_vals[mask] / psi_vals[mask]).real)

    # 2. Integrated Check: <|grad|^2> / <|psi|^2> ≈ C2
    mean_psi_sq  = torch.mean(torch.abs(psi_vals) ** 2)
    mc_energy    = torch.mean(grad_sq_vals) / mean_psi_sq

    print(f"--- {name} ---")
    print(f"Pointwise C2: {pointwise_c2.item():10.6f}")
    print(f"MC Energy:    {mc_energy.item():10.6f}")
    print(f"Exact C2:     {C2_EXACT:10.6f}")
    print(f" norm         {mean_psi_sq.item()}")
    print("-" * 20)

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_check("Complex Trace: Tr(U)",    lambda u: torch.trace(u))
    run_check("Real Trace: Re[Tr(U)]",   lambda u: torch.trace(u).real)
