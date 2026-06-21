"""
Implementation of the squared Lie derivatives
\\sum_a L_a^2 = \\sum_a R_a^2
on functions of the gauge links,
using autodifferentiation:

$${ L^2_a f(U) = -i\\frac{d^2}{d \\omega^2} f( e^{-i \\omega \\tau_a } U ) |_{\\omega = 0} }$$
$${ R^2_a f(U) = +i\\frac{d^2}{d \\omega^2} f( U e^{-i \\omega \\tau_a } ) |_{\\omega = 0} }$$

The $L_a$ and $R_a$ are the canonical momenta associated to the link $U$,
and satisfy:

$${ [L_a , U] = - \\tau_a U }$$
$${ [R_a , U] = + U \\tau_a }$$

"""

import typing
import torch

from lattice_data_tools.autodifferentiation.with_torch_autograd_grad import my_autograd
import lattice_data_tools.links.suN as suN
from lattice_data_tools.links.canonical_momenta import CanonicalMomenta
from lattice_data_tools.links.configuration import GaugeConfiguration
from lattice_data_tools.links.canonical_momenta import CanonicalMomenta


import torch
import typing

def chain_rule(A, dRef, dImf):
    df_re = (dRef.conj() * A).real # d Re(f)/domega
    df_im = (dImf.conj() * A).real # d Im(f)/domega
    return df_re + 1j * df_im # d( Re(f) + i*Im(f) )/domega

            
class WithAutodifferentiation(CanonicalMomenta):
    def __init__(self, U: GaugeConfiguration):
        super().__init__(U=U)
    #---
    def get_La2_per_link(self, f: typing.Callable, U: GaugeConfiguration):
        """
        Returns, for each configuration and each `a`,
        ${ \\sum_{x,\\mu} L_a^2(x,\\mu) f(U) }$
        NOTE: the index `a` is NOT summed over.
        
        `f(U)` should return a tensor of shape (batchsize,1)

        The return shape is (batchsize, Ng).

        The derivative is a Laplacian with respect to `omega` in:
        $$
          L^2_a(x,\\mu) f(...,U(x,\\mu),...)
          = -i\\frac{d^2}{d \\omega^2}
            f(..., e^{-i \\omega \\tau_a } U(x,\\mu), ...)|_{\\omega = 0}
        $$
        """
        Nc = self.Nc # number of colors
        Ng = self.Ng # number of generators in the Lie algebra
        n_links = self.n_links # number of links
        batchsize = self.batchsize # number of configurations
        Id = torch.eye(Nc).to(device=U.device)
        Id_arr = Id.expand(n_links, Nc, Nc)
        omega = torch.tensor(0.0, requires_grad=True, dtype=U.real.dtype, device=U.device)
        V = self.get_V(omega=omega, tau=self.tau)
        La2_per_link = []
        for a in range(Ng):
            Va = V[a,:,:]
            sum_La_squared = []
            for b in range(batchsize):
                conf_shape = (1,)+ U[b,...].shape
                U_b = U[b,...].reshape(n_links, Nc, Nc)
                def f_i(i):
                    e_i = torch.nn.functional.one_hot(i, n_links).to(dtype=U.dtype, device=U.device)
                    Va_arr = Id_arr + torch.einsum("ab,i->iab", Va-Id, e_i)
                    VaU_i = (Va_arr @ U_b).reshape(*conf_shape)
                    f_VaU = f(GaugeConfiguration(VaU_i)) / n_links
                    return f_VaU
                #---
                # vectorize over all indices 0..N-1
                sum_f = torch.vmap(f_i)( torch.arange(n_links) ).sum()
                # \\sum_i \\partial_{x_i} f : directional derivative along (1,...,1)


                # dir_der = my_autograd(sum_f, omega, create_graph=True, retain_graph=True)
                # laplacian_b = my_autograd(dir_der, omega, create_graph=True, retain_graph=True)

                f_is_real = not torch.is_complex(sum_f)
                if f_is_real:
                    dir_der = torch.autograd.grad(sum_f, omega, create_graph=True)[0]
                    laplacian_b = torch.autograd.grad(dir_der, omega, create_graph=True)[0]
                else:
                    Re_dir_der = torch.autograd.grad(sum_f.real, omega, create_graph=True)[0]
                    Im_dir_der = torch.autograd.grad(sum_f.imag, omega, create_graph=True)[0]
                    Re_laplacian_b = torch.autograd.grad(Re_dir_der, omega, create_graph=True)[0]
                    Im_laplacian_b = torch.autograd.grad(Im_dir_der, omega, create_graph=True)[0]
                    laplacian_b = Re_laplacian_b + 1j*Im_laplacian_b
                #---
                sum_La_squared.append(laplacian_b)
            #---
            La2_per_link.append(torch.stack(sum_La_squared, dim=0))
        #---
        return -torch.stack(La2_per_link, dim=1)

    def with_La_twice(self, f: typing.Callable, U: GaugeConfiguration) -> torch.Tensor:
        """
        La^2 applying La twice and using the chain rule

        The Wirtinger derivatives df/dU are obtained as df/deps|_eps=0, where f=f(U + eps).

        1. I construct the sum s=f(U1+eps1,U2,...)+f(U1,U2+eps2)+...
        2. I derive with respect to eps_i and multiply componentwise with (V1,V2,...), where V_i = -i*tau_a @ (U_i+eps_i). I get a scalar "r" (which I multiply by "-i").
        3. I derive "r" with respect to eps_i, and multiply c-wise with the same vector of V_i above. I multiply the final result by "-i".
        
        """
        CM = CanonicalMomenta(U=U)
        Nc = U.Nc
        Ng = U.Ng
        batchsize = U.batch_size
        n_links = U.n_links

        # use this to have the result per link
        eps = torch.zeros(
            size=U.shape, dtype=U.dtype, device=U.device,
            requires_grad=True
        )
        # use this to have the sum over links
        # eps = torch.tensor(
        #     0.0+0.0*1j, dtype=U.dtype, device=U.device,
        #     requires_grad=True
        # )

        eps_flat = eps.view(-1)
        # omega = torch.zeros(
        #     size=U.shape, dtype=U.dtype, device=U.device,
        #     requires_grad=True
        # )
        #omega_flat = omega.view(-1)
        N = U.numel()
        U_flat = U.view(-1).as_subclass(torch.Tensor)
        def fi(i):
            e_i = (torch.arange(N, device=U.device) == i).to(U.dtype)
            U_pert_flat = U_flat + eps_flat * e_i  # scalar * one-hot
            U_pert = U_pert_flat.reshape(batchsize, *U.shape[1:])
            return f(GaugeConfiguration(U_pert))
        #---
        s = torch.vmap(fi, in_dims=0)(torch.arange(N, device=U.device)).sum() + 0.0*1j
        #assert s.shape == (U.batch_size, 1)
        dRes_deps = my_autograd(y=s.real, x=eps_flat, create_graph=True,retain_graph=True)
        dIms_deps = my_autograd(y=s.imag, x=eps_flat, create_graph=True,retain_graph=True) 
        Up = (U+eps).as_subclass(torch.Tensor)
        #Up_flat = Up.view(-1) #U_flat+eps_flat
        tau_Up = torch.einsum("aij,...jk->a...ik", self.tau, Up).reshape(Ng, -1)
        A = -1j*tau_Up # for all a=1,...,Ng
        def get_La2f(a: int):
            A_a = A[a,:]
            r = -1j*chain_rule(A, dRes_deps, dIms_deps).sum() #  (A_a.conj()*ds_deps).real.sum()
            dRer_domega = my_autograd(y=r.real, x=eps_flat, create_graph=True,retain_graph=True)
            dImr_domega = my_autograd(y=r.imag, x=eps_flat, create_graph=True,retain_graph=True)
            La2_f = -1j*chain_rule(A_a, dRer_domega, dImr_domega).reshape(batchsize,n_links, Nc,Nc).sum(dim=(-2,-1))
            return La2_f
        #---
        # La2_f_arr = torch.vmap(get_La2f, in_dims=0)(A)
        La2_f_arr = torch.stack([get_La2f(a) for a in range(Ng)], dim=1)
        return La2_f_arr
        
        
class WithFiniteDifferences(CanonicalMomenta):
    def __init__(self, U: GaugeConfiguration):
        super().__init__(U=U)
    #---
    def get_La2_per_link(self, f: typing.Callable, U: GaugeConfiguration, eps: float = 1e-8):
        """
        Returns ${ \\sum_{x,\\mu} L_a^2(x,\\mu) f(U) }$ via Finite Differences.
        NOTE: the index `a` is NOT summed over.

        Uses the second-order central difference approximation:
            L_a^2 f(U) ≈ [f(e^{+i eps tau_a} U) - 2f(U) + f(e^{-i eps tau_a} U)] / eps^2

        `f(U)` should return a tensor of shape (batchsize,1)
        """
        Nc = U.Nc # number of colors
        Ng = U.Ng # number of generators
        n_links = U.n_links # number of links
        batchsize = U.batch_size
        Id = torch.eye(Nc, dtype=U.dtype, device=U.device)
        Id_arr = Id.expand(n_links, Nc, Nc)
        omega = torch.tensor(0.0, requires_grad=True, dtype=U.real.dtype, device=U.device)
        Va_plus  = self.get_V(+eps+omega, tau=self.tau) # e^{+i eps tau_a}
        Va_minus = self.get_V(-eps+omega, tau=self.tau) # e^{-i eps tau_a}
        f_unpert = f(U) 
        La2_per_link = []
        for a in range(Ng):
            sum_La_squared = []
            for b in range(batchsize):
                U_b = U[b, ...].reshape(n_links, Nc, Nc)
                shape_b = U[b, ...].shape
                def make_perturbed_U(Va):
                    """Apply Va to link i, leaving all other links unchanged."""
                    def f_i(i):
                        e_i = torch.nn.functional.one_hot(i, n_links).to(dtype=U.dtype, device=U.device)
                        Va_arr = Id_arr + torch.einsum("ab,i->iab", Va - Id, e_i)
                        VaU_i = (Va_arr @ U_b).reshape(*shape_b).unsqueeze(0)
                        return f(GaugeConfiguration(VaU_i))/n_links
                    #---
                    indices = torch.arange(n_links)
                    return torch.vmap(f_i)(indices).sum()  # sum over all links
                #---
                f_plus  = make_perturbed_U(Va_plus[a,:,:]) # sum_i f(Va(+eps) . U_i)
                f_minus = make_perturbed_U(Va_minus[a,:,:]) # sum_i f(Va(-eps) . U_i)

                dir_der = (f_plus - f_minus)/(2.0*eps) # 1st derivative found with finite difference
                laplacian_b = my_autograd(dir_der, omega, grad_outputs=torch.ones_like(dir_der), create_graph=False, retain_graph=True) # 2nd derivative through autodifferentiation
                sum_La_squared.append(laplacian_b)
            #---
            La2_per_link.append(torch.stack(sum_La_squared, dim=0))
        #---
        return -torch.stack(La2_per_link, dim=1) # `-1` factor coming from the product of the 2 `-i` in front of the L_a

    
    def TODO_get_sum_La_squared_per_link_fast(self, a: int, f: typing.Callable, U: GaugeConfiguration, f_is_real: bool, eps: float = 1e-8):
        """ Faster implementation of La_squared_per_link_FD() but that uses more memory """
        Nc = U.Nc
        n_links = U.n_links
        batchsize = U.batch_size
        tau_a = self.tau[a, :, :]
        Id = torch.eye(Nc, dtype=U.dtype, device=U.device)

        # --- Precompute Va(±eps) once ---
        d, M = torch.linalg.eigh(tau_a)
        def make_Va(omega):
            exp_iD = torch.diag_embed(torch.exp(-1j * omega * d).to(dtype=U.dtype))
            return M.to(dtype=U.dtype) @ exp_iD @ M.adjoint().to(dtype=U.dtype)

        omega = torch.tensor(0.0, requires_grad=True, dtype=U.real.dtype, device=U.device)
        Va_plus  = make_Va(+eps+omega)
        Va_minus = make_Va(-eps+omega)

        # --- Build all perturbed link arrays at once ---
        # For each link i, apply Va to link i only.
        # Va_arr shape: (n_links, n_links, Nc, Nc)
        # Va_arr[i, j] = Va if i==j else Id
        Id_arr = Id.expand(n_links, Nc, Nc)  # (n_links, Nc, Nc)

        def make_Va_arr(Va):
            base = Id.expand(n_links, n_links, Nc, Nc).clone()  # (n_links, n_links, Nc, Nc)
            idx = torch.arange(n_links)
            base[idx, idx] = Va.expand(n_links, Nc, Nc)
            return base

        Va_arr_plus  = make_Va_arr(Va_plus)   # (n_links, n_links, Nc, Nc)
        Va_arr_minus = make_Va_arr(Va_minus)

        # --- Process all batch elements at once ---
        # U_all: (batchsize, n_links, Nc, Nc)
        U_all = U.data.reshape(batchsize, n_links, Nc, Nc)

        # Apply: (n_links, n_links, Nc, Nc) @ (batchsize, n_links, Nc, Nc)
        # → (batchsize, n_links, n_links, Nc, Nc)  [each config, each perturbed link]
        U_exp = U_all.unsqueeze(1)                          # (batchsize, 1, n_links, Nc, Nc)
        VaU_plus  = (Va_arr_plus  @ U_exp).reshape(batchsize * n_links, *U.shape[1:])
        VaU_minus = (Va_arr_minus @ U_exp).reshape(batchsize * n_links, *U.shape[1:])
        U_unpert  = U_all.unsqueeze(1).expand(-1, n_links, -1, -1, -1).reshape(batchsize * n_links, *U.shape[1:])

        # --- Single f() call for all configs ---
        # all_configs = torch.cat([VaU_plus, VaU_minus, U_unpert], dim=0)
        # f_plus_all, f_minus_all, f_0_all = all_results.chunk(3, dim=0)

        all_configs = torch.cat([VaU_plus, VaU_minus], dim=0)
        all_results = f(GaugeConfiguration(all_configs))/n_links   # (3 * batchsize * n_links, 1)
        f_plus_all, f_minus_all = all_results.chunk(2, dim=0)

        # Sum over links, then central difference
        f_plus_sum  = f_plus_all.reshape(batchsize, n_links).sum(dim=1)
        f_minus_sum = f_minus_all.reshape(batchsize, n_links).sum(dim=1)
        # f_0_sum     = f_0_all.reshape(batchsize, n_links).sum(dim=1)

        dir_der = (f_plus_sum - f_minus_sum)/(2.0*eps)
        if f_is_real:
            laplacian = torch.autograd.grad(dir_der.real, omega)[0]
        else:
            Re_laplacian = torch.autograd.grad(dir_der.real, omega, create_graph=True)[0]
            Im_laplacian = torch.autograd.grad(dir_der.imag, omega)[0]
            laplacian = Re_laplacian + 1j*Im_laplacian
           
        return laplacian.unsqueeze(-1)  # (batchsize, 1)


