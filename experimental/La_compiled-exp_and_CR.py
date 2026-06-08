"""
The compilation with torch.compile requires that I use all real numbers to be effective.
This file contains a minimal implementation of the canonical momenta,
for a function f(U) that is non-trivial in gauge configuration of links.
The latter is casted into real by extending the last dimension to contain real and imaginary part
"""
import sys
sys.path.append("../../")
import time
import torch
import typing
import warnings
#warnings.filterwarnings("always")
from lattice_data_tools.links.configuration import GaugeConfiguration
import lattice_data_tools.links.suN as suN
from lattice_data_tools.autodifferentiation.with_torch_func_grad import get_compiled_function
from lattice_data_tools.links.canonical_momenta import CanonicalMomenta


def complex2ri(x):
    return torch.stack([x.real, x.imag], dim=-1)


def adjoint(M):
    M_T = torch.transpose(M, -3, -2)
    M_H = torch.stack([M_T[..., 0], -M_T[..., 1]], dim=-1)
    return M_H


def complex_matmul(A, B):
    ReA, ImA = A[..., 0], A[..., 1]
    ReB, ImB = B[..., 0], B[..., 1]
    Re_AB = ReA @ ReB - ImA @ ImB
    Im_AB = ReA @ ImB + ImA @ ReB
    AB = torch.stack([Re_AB, Im_AB], dim=-1)
    return AB


class La_Generator_exp:
    """
    Object generating the momenta L_a, for each a and each link in the configuration, and each configuration
    The output is a lambda function (potentially compiled with `torch.compile`), which takes U in the usual shape (it is flattened internally)
    """
    def __init__(self, f: typing.Callable, U: GaugeConfiguration, do_compile: bool):
        """
        Initialize the lambda function for the La,
        using the function f,
        an example gauge configuration (for the number of links). U.shape==(batchsize, n_links)
        and when `do_compile==True` it is compiled
        """
        batchsize = U.shape[0]
        n_links = U.n_links
        Nc = U.Nc
        Ng = U.Ng
        device = U.device
        dtype = U.dtype
        # diagonalization of the tau_a
        tau = suN.get_generators(Nc=Nc, device=device, dtype=dtype)
        tau_eigh_complex = torch.linalg.eigh(tau)
        # keep tau_eigh as a tuple of real tensors -- passed explicitly, not captured
        tau_eigh = (tau_eigh_complex[0], complex2ri(tau_eigh_complex[1]))
        Id = torch.stack(
            [
                torch.eye(Nc, device=U.device, dtype=U.real.dtype),
                torch.zeros(Nc, Nc, device=U.device, dtype=U.real.dtype)
            ], dim=-1)
        Id_arr = Id.expand(n_links, Nc, Nc, 2)
        single_conf_shape = (1,) + U.shape[1:] + (2,)
        def f_shift(tau_a_eigh, Ub, eps, i, Id_arr):
            d, M = tau_a_eigh
            exp_iD = torch.stack(
                [
                    torch.diag_embed(torch.cos(eps * d)),
                    -torch.diag_embed(torch.sin(eps * d))
                ],
                dim=-1)
            Va = complex_matmul(complex_matmul(M, exp_iD), adjoint(M))
            ei = (torch.arange(n_links, device=Ub.device) == i).to(Ub.dtype)
            Va_arr = Id_arr + torch.einsum("abC,i->iabC", Va - Id_arr[0], ei)
            VaU_i = complex_matmul(Va_arr, Ub).reshape(*single_conf_shape)
            res = f(VaU_i)  # shape==(1,1,...,2)
            return res[0, :]
        def Re_f_shift(tau_a_eigh, Ub, eps, i, Id_arr):
            return f_shift(tau_a_eigh, Ub, eps, i, Id_arr)[0]
        def Im_f_shift(tau_a_eigh, Ub, eps, i, Id_arr):
            return f_shift(tau_a_eigh, Ub, eps, i, Id_arr)[1]
        # grad wrt eps (argnums=2), Id_arr passed explicitly (argnums=4)
        Re_df = torch.func.grad(Re_f_shift, argnums=2)
        Im_df = torch.func.grad(Im_f_shift, argnums=2)
        eps = torch.tensor(0.0, device=device, dtype=U.real.dtype)
        idx_links = torch.arange(n_links, device=device)
        La_f_shape = (batchsize, Ng, *(U.shape[1:-2]))
        def make_vmapped(df_i):
            return torch.func.vmap(
                torch.func.vmap(
                    torch.func.vmap(
                        df_i,
                        in_dims=(None, None, None, 0, None)  # over link index
                    ),
                    in_dims=(0, None, None, None, None)       # over generators
                ),
                in_dims=(None, 0, None, None, None)           # over batch
            )
        Re_df_vmapped = make_vmapped(Re_df)
        Im_df_vmapped = make_vmapped(Im_df)
        # Single function combining Re and Im -- this is what we compile as one graph
        def uncompiled_df(U_arr, tau_eigh, eps, idx_links, Id_arr):
            U_flat = U_arr.view(batchsize, -1, Nc, Nc, 2)
            Re = Re_df_vmapped(tau_eigh, U_flat, eps, idx_links, Id_arr)
            Im = Im_df_vmapped(tau_eigh, U_flat, eps, idx_links, Id_arr)
            # factor of -i: -i*(Re + i*Im) = Im - i*Re, kept as real tensor (Re, Im parts)
            result_re = Im.reshape(La_f_shape)
            result_im = -Re.reshape(La_f_shape)
            return result_re, result_im
        U_tens = complex2ri(U.as_subclass(torch.Tensor))
        if do_compile:
            compiled_df = get_compiled_function(
                uncompiled_df, U_tens, tau_eigh, eps, idx_links, Id_arr
            )
        else:
            compiled_df = uncompiled_df
        # Store everything needed at call time
        self._compiled_df = compiled_df
        self._tau_eigh = tau_eigh
        self._eps = eps
        self._idx_links = idx_links
        self._Id_arr = Id_arr

    @property
    def df_function(self):
        """
        Returns a callable that takes U_ri (real tensor) and returns
        the canonical momenta as a complex tensor of shape (batchsize, Ng, *lattice_shape).
        """
        def _call(U):
            re, im = self._compiled_df(
                U, self._tau_eigh, self._eps, self._idx_links, self._Id_arr
            )
            return re + 1j * im
        return _call


class La_Generator_CR:
    """
    Left canonical momenta L_a via the chain rule, using only real tensors
    and ``torch.func`` (no autograd), compatible with ``torch.compile``.

    The chain-rule identity is:

        L_a f(U) = -i * d/deps  f(U + eps * A_L_a(U)) |_{eps=0}

    where  A_L_a(U) = (-i * tau_a) @ U  is the first-order (linear) tangent
    vector of the left-shift.  This is the chain-rule linearisation of the
    exponential shift used in ``La_Generator_exp``; at eps=0 the two are
    identical, so the derivatives agree exactly.

    Structure mirrors ``La_Generator_exp`` closely:

    * ``f_shift(tau_a_ri, Ub, eps, i)`` builds  U_shifted  for a single
      generator ``a`` and a single link ``i``, then calls ``f``.
    * ``Re_f_shift`` / ``Im_f_shift`` return the Re/Im output channels.
    * ``torch.func.grad`` differentiates w.r.t. ``eps`` (argnums=2).
    * Three nested ``torch.func.vmap`` calls vectorise over links, generators
      and batch, exactly as in ``La_Generator_exp``.
    * The factor ``-i`` is applied at the end:
          -i * (Re + i*Im) = Im - i*Re
    """

    def __init__(self, f: typing.Callable, U: GaugeConfiguration, do_compile: bool):
        batchsize = U.shape[0]
        n_links   = U.n_links
        Nc        = U.Nc
        Ng        = U.Ng
        device    = U.device

        # ── generators in real notation, passed explicitly (not captured) ──
        tau_cplx = suN.get_generators(Nc=Nc, device=device, dtype=U.dtype)  # (Ng, Nc, Nc) complex
        tau_ri   = complex2ri(tau_cplx)  # (Ng, Nc, Nc, 2)

        # (-i * tau_a) in Re/Im split:  -i*(Re+iIm) = Im - i*Re
        neg_i_tau_ri = torch.stack(
            [tau_ri[..., 1], -tau_ri[..., 0]], dim=-1
        )  # (Ng, Nc, Nc, 2)

        single_conf_shape = (1,) + U.shape[1:] + (2,)
        La_f_shape = (batchsize, Ng, *(U.shape[1:-2]))

        # ── f_shift: single generator a, single link i ────────────────────
        # neg_i_tau_a : (-i*tau_a) for one generator, shape (Nc, Nc, 2)
        # Ub          : one configuration, shape (n_links, Nc, Nc, 2)
        # eps         : scalar
        # i           : link index (scalar int tensor)
        def f_shift(neg_i_tau_a, Ub, eps, i):
            # ei selects link i without data-dependent indexing (vmap-safe)
            ei    = (torch.arange(n_links, device=Ub.device) == i).to(Ub.dtype)  # (n_links,)
            # U_i = sum_j ei[j] * Ub[j]  →  shape (Nc, Nc, 2)
            U_i   = torch.einsum("i,iabC->abC", ei, Ub)
            # A_L_a for link i:  (-i*tau_a) @ U_i
            A_L_i = complex_matmul(neg_i_tau_a, U_i)           # (Nc, Nc, 2)
            # Scatter A_L_i back to all links, zero everywhere except link i
            A_arr  = torch.einsum("abC,i->iabC", A_L_i, ei)    # (n_links, Nc, Nc, 2)
            shifted = Ub + eps * A_arr                           # (n_links, Nc, Nc, 2)
            res = f(shifted.reshape(*single_conf_shape))         # (1, 2) or (1, 1)
            return res[0, :]                                      # (2,) or (1,)

        def Re_f_shift(neg_i_tau_a, Ub, eps, i):
            return f_shift(neg_i_tau_a, Ub, eps, i)[0]

        def Im_f_shift(neg_i_tau_a, Ub, eps, i):
            return f_shift(neg_i_tau_a, Ub, eps, i)[1]

        # ── torch.func.grad over eps (argnums=2) ──────────────────────────
        Re_df = torch.func.grad(Re_f_shift, argnums=2)
        Im_df = torch.func.grad(Im_f_shift, argnums=2)

        eps      = torch.tensor(0.0, device=device, dtype=U.real.dtype)
        idx_links = torch.arange(n_links, device=device)

        # ── vmap over links, generators, batch  (same nesting as La_Generator_exp) ──
        def make_vmapped(df_i):
            return torch.func.vmap(
                torch.func.vmap(
                    torch.func.vmap(
                        df_i,
                        in_dims=(None, None, None, 0)   # over link index
                    ),
                    in_dims=(0, None, None, None)        # over generators
                ),
                in_dims=(None, 0, None, None)            # over batch
            )

        Re_df_vmapped = make_vmapped(Re_df)
        Im_df_vmapped = make_vmapped(Im_df)

        # ── function to (optionally) compile ──────────────────────────────
        def uncompiled_df(U_arr, neg_i_tau_ri, eps, idx_links):
            U_flat = U_arr.view(batchsize, -1, Nc, Nc, 2)
            Re = Re_df_vmapped(neg_i_tau_ri, U_flat, eps, idx_links)
            Im = Im_df_vmapped(neg_i_tau_ri, U_flat, eps, idx_links)
            # factor -i:  -i*(Re + i*Im) = Im - i*Re
            result_re = Im.reshape(La_f_shape)
            result_im = -Re.reshape(La_f_shape)
            return result_re, result_im

        U_tens = complex2ri(U.as_subclass(torch.Tensor))
        if do_compile:
            compiled_df = get_compiled_function(
                uncompiled_df, U_tens, neg_i_tau_ri, eps, idx_links
            )
        else:
            compiled_df = uncompiled_df

        self._compiled_df    = compiled_df
        self._neg_i_tau_ri   = neg_i_tau_ri
        self._eps            = eps
        self._idx_links      = idx_links

    @property
    def df_function(self):
        """
        Returns a callable

            L_a(U_ri) → complex tensor of shape (batchsize, Ng, *lattice_shape)

        where ``U_ri`` is a real tensor of shape
        ``(batchsize, *lattice_shape, Nc, Nc, 2)``.
        """
        def _call(U_ri):
            re, im = self._compiled_df(
                U_ri, self._neg_i_tau_ri, self._eps, self._idx_links
            )
            return re + 1j * im
        return _call



class La_Generator_CR_Delta:
    """
    Left canonical momenta using the exact delta trick from with_chain_rule.
    Only the df/d(delta) part is compiled.
    Works with U that does not require grad.
    """
    def __init__(self, f: typing.Callable, U: GaugeConfiguration, do_compile: bool):
        self.batchsize = U.shape[0]
        self.n_links = U.n_links
        self.Nc = U.Nc
        self.Ng = U.Ng
        self.device = U.device
        self.dtype = U.real.dtype

        # Base configuration in real form (captured)
        self.U_ri_base = complex2ri(U.as_subclass(torch.Tensor)).detach()

        # (-i * tau_a) in real form
        tau_cplx = suN.get_generators(Nc=self.Nc, device=self.device, dtype=U.dtype)
        tau_ri = complex2ri(tau_cplx)
        self.neg_i_tau_ri = torch.stack([tau_ri[..., 1], -tau_ri[..., 0]], dim=-1)  # (Ng, Nc, Nc, 2)

        self.La_shape = (self.batchsize, self.Ng, *U.shape[1:-2])
        self.single_shape = (1,) + U.shape[1:] + (2,)

        # ── Delta trick: df/d(delta)  (this is the compiled core) ───────
        def f_shift_delta(delta_ri):
            """delta_ri shape: (1, n_links, Nc, Nc, 2)"""
            shifted = self.U_ri_base + delta_ri
            out = f(shifted.reshape(self.single_shape))
            # Return real and imag parts separately for torch.func.grad
            if out.ndim == 1 and out.shape[0] == 2:   # already (re, im)
                return out[0], out[1]
            else:                                     # real scalar
                return out[0, 0], torch.zeros_like(out[0, 0])

        # Separate gradients for Re and Im parts
        grad_re = torch.func.grad(lambda d: f_shift_delta(d)[0], argnums=0)
        grad_im = torch.func.grad(lambda d: f_shift_delta(d)[1], argnums=0)

        def df_ddelta_core(dummy_U_ri):
            """Compute df/d(delta) for all batch elements"""
            delta0 = torch.zeros((self.batchsize, self.n_links, self.Nc, self.Nc, 2),
                                 device=self.device, dtype=self.dtype)

            # vmap over batch
            re_part = torch.func.vmap(grad_re)(delta0)
            im_part = torch.func.vmap(grad_im)(delta0)

            # Stack into (B, n_links, Nc, Nc, 2)
            return torch.stack([re_part, im_part], dim=-1)

        # Compile only this derivative
        dummy = complex2ri(U.as_subclass(torch.Tensor))
        if do_compile:
            self._df_dU = get_compiled_function(df_ddelta_core, dummy)
        else:
            self._df_dU = df_ddelta_core

    @torch.no_grad()
    def df_function(self, U_ri: torch.Tensor):
        """
        U_ri: real tensor of shape (B, ..., Nc, Nc, 2)
        Returns L_a as complex tensor (B, Ng, *lattice)
        """
        # 1. df/dU via delta trick (the only compiled part)
        df_dU = self._df_dU(U_ri)                          # (B, n_links, Nc, Nc, 2)

        # 2. A_L = (-i τ_a) @ U   (cheap, outside compiled region)
        U_flat = U_ri.view(self.batchsize, self.n_links, self.Nc, self.Nc, 2)
        A_L = complex_matmul_ri(
            self.neg_i_tau_ri.unsqueeze(0).unsqueeze(1),   # (1, 1, Ng, Nc, Nc, 2)
            U_flat.unsqueeze(2)                            # (B, n_links, 1, Nc, Nc, 2)
        )  # (B, n_links, Ng, Nc, Nc, 2)

        # 3. Wirtinger-style contraction: Re[ conj(df_dU) * A_L ]
        re_df = df_dU[..., 0]
        im_df = df_dU[..., 1]
        re_A = A_L[..., 0]
        im_A = A_L[..., 1]

        inner = re_df.unsqueeze(2) * re_A + im_df.unsqueeze(2) * im_A
        df_domega = inner.sum(dim=(-2, -1))                # (B, n_links, Ng)

        # Apply -i factor: -i * df_domega = Im - i*Re
        momenta_re = df_domega[..., 1]
        momenta_im = -df_domega[..., 0]

        return momenta_re.reshape(self.La_shape) + 1j * momenta_im.reshape(self.La_shape)
    
# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def perf(fun, info: str):
    torch.cuda.synchronize()
    t1 = time.time()
    res = fun()
    torch.cuda.synchronize()
    t2 = time.time()
    print(f"dt ({info}): {t2-t1} sec.")
    return res


def f(U_ri):
    U_xp1 = torch.roll(U_ri, shifts=(1, 3), dims=(0, 2))
    n = len(U_ri.shape)
    res = (U_xp1 * U_ri).sum(dim=tuple(torch.arange(1, n - 1)))
    return res


def f_from_conf(U):
    ri_res = f(complex2ri(U.as_subclass(torch.Tensor)))
    cmplx_res = (ri_res[:, 0] + 1j * ri_res[:, 1]).unsqueeze(-1)
    return cmplx_res


# ══════════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════════

f_is_real = False
B     = 1
L_mu  = [4,4,4]
Nc    = 3
device = torch.device("cpu")

U = GaugeConfiguration.from_hotstart(
    batchsize=B, L_mu=L_mu, Nc=Nc,
    seed=20260502, dtype=torch.complex128, device=device,
    requires_grad=False
)
U_tens = U.as_subclass(torch.Tensor)
U_ri   = complex2ri(U_tens)

# ── CanonicalMomenta reference (complex tensors, chain rule) ──────────────
CM         = CanonicalMomenta(U=U)
momenta_cr = perf(lambda: CM.with_chain_rule(f=f_from_conf, U=U, f_is_real=f_is_real),
                  "L_a & R_a from chain rule (reference, complex)")
momenta_exp = perf(lambda: CM.with_exponential(f=f_from_conf, U=U, f_is_real=f_is_real),
                  "L_a & R_a from exp (reference, complex)")


# ── La_Generator_exp (vmap + exponential, reference) ─────────────────────
LaG          = La_Generator_exp(f=f, U=U, do_compile=False)
La_arr_vmap  = perf(lambda: LaG.df_function(U=U_ri), "La_exp not compiled")

LaG_compiled     = La_Generator_exp(f=f, U=U, do_compile=True)
La_arr_vmap_comp = perf(lambda: LaG_compiled.df_function(U=U_ri), "La_exp compiled")

# ── La_Generator_CR (new: vmap + chain rule, no autograd) ────────────────
LaCR         = La_Generator_CR(f=f, U=U, do_compile=False)
La_arr_CR    = perf(lambda: LaCR.df_function(U_ri), "La_CR not compiled")

LaCR_compiled    = La_Generator_CR(f=f, U=U, do_compile=True)
La_arr_CR_comp   = perf(lambda: LaCR_compiled.df_function(U_ri), "La_CR compiled")


# ── Shapes ────────────────────────────────────────────────────────────────
print("\n--- shapes ---")
print("momenta_cr (ref)  :", momenta_cr.shape)    # (B, 2, Ng, *lattice)
print("La_arr_vmap       :", La_arr_vmap.shape)    # (B, Ng, *lattice)
print("La_arr_CR         :", La_arr_CR.shape)      # (B, Ng, *lattice)

# ── Correctness ───────────────────────────────────────────────────────────
print("\n--- correctness ---")
print("La_exp    VS reference CR :",
      torch.allclose(momenta_cr[:, 0, ...], La_arr_vmap,    atol=1e-10))
print("La_CR     VS reference CR :",
      torch.allclose(momenta_cr[:, 0, ...], La_arr_CR,      atol=1e-10))
print("La_CR     VS La_exp       :",
      torch.allclose(La_arr_vmap,           La_arr_CR,      atol=1e-10))
print("La_CR compiled VS uncompiled:",
      torch.allclose(La_arr_CR,             La_arr_CR_comp, atol=1e-10))

