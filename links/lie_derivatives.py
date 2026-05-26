"""
Implementation of Lie derivatives on functions of the gauge links,
using autodifferentiation:

$${ L_a f(U) = -i\\frac{d}{d \\omega} f( e^{-i \\omega \\tau_a } U ) |_{\\omega = 0} }$$
$${ R_a f(U) = +i\\frac{d}{d \\omega} f( U e^{-i \\omega \\tau_a } ) |_{\\omega = 0} }$$

The $L_a$ and $R_a$ are the canonical momenta associated to the link $U$,
and satisfy:

$${ [L_a , U] = - \\tau_a U }$$
$${ [R_a , U] = + U \\tau_a }$$

"""

import typing

import torch

import lattice_data_tools.links.suN as suN
from lattice_data_tools.links.configuration import GaugeConfiguration, ColorMatrix


class LieDerivatives:
    """
    Class for the calculation of the Lie derivatives (canonical momenta of gauge links)
    """
    def __init__(self, U: GaugeConfiguration):
        """
        Initialization of tensors needed
        
        Main idea: this object is initialized before the training,
        and allows to compute efficiently the derivatives with respect to $\\omega_a(x,\\mu)$ in $0$.
        For instance:

        $${ L_a(x,\\mu) f(..., U(x,\\mu), ...) = \\frac{d}{d\\omega} f(..., V_a U(x,\\mu), ...)|_{\\omega=0} }$$

        where ${ V_a = exp(-i \\omega_a(x,\\mu) \\tau_a) }$

        Input:
            U: GaugeConfiguration object, used only to infer the shape of `omega`, `dtype` and `device`.

        where `Ng` is the number of generators of `SU(Nc)`.
        """
        self.Nc = U.Nc # number of colors of the group
        self.Ng = U.Ng # number of generators in the algebra
        self.omega_shape = (*U.shape[1:-2],) # (L1,...,Ld, d): one omega per link, no batch, no color
        self.omega = torch.zeros(size=self.omega_shape, dtype=U.real.dtype, device=U.device, requires_grad=True)
        self.tau = suN.get_generators(Nc=self.Nc, device=U.device, dtype=U.dtype)
        # list of exponentials exp(-i*omega*tau_a)
        self.V = suN.get_exp_i_omega_tau_a(
            omega = - self.omega,
            tau_a =   self.tau
        ) # exp(-i*omega*tau_a)
    #---

    def L_a_chain_rule(self, a: int, f: typing.Callable, U: GaugeConfiguration, f_is_real: bool):
        """
        Left canonical momenta using the chain rule:

        $$
        -i (d/d omega) f(e^{-i* omega *tau_a} @ U) |_{omega=0} =
        -i * (-i*\\tau_a U)_{ab} (\\partial f / \\partial U_{ab})
        $$

         Notes:
         1. `df/dU=df(U+\\delta)/d\\delta |_{\\delta=0}`
         2. If `w=w1+i*w2`, `z=z1+iz2` --> `w1*z1+w2+z2=Re(w*conj(z))`

         3. Here we compute -i d/domega with the chain rule over the real and imaginary parts of the color indices of the arguments

         For `f=f_1+i*f_2`, `i=1,2`, this gives:

        $$
           (df_i/d_\\omega)|_{\\omega=0} =
           dRe(-1j*\\tau_a*U)_{ab} Re( df_i/dRe(U_{ab})) + dIm(-1j*\\tau_a*U_{ab}) Im( df_i/dIm(U_{ab})) =
           Re[ (-1j*\\tau_a*U) * conj(df_i/dU) ]
        $$
        """
        tau_a = self.tau[a]  # (Nc, Nc)
        delta = torch.zeros(size=U.shape[1:], dtype=U.dtype, device=U.device, requires_grad=True) # Note: delta should be complex
        f_U = f(GaugeConfiguration(U + delta.unsqueeze(0))) # f(U+delta)
        f_U_flat = f_U.view(-1) # flattened view
        batchsize = f_U_flat.numel() # f(U) is a scalar --> one element for each batch

        A = -1j * torch.einsum("ij,...jk->...ik", tau_a, U.as_subclass(torch.Tensor)) # d(e^{-i*omega*tau_a})/domega at omega==0

        df_domega_list = []
        for i in range(batchsize):
            if f_is_real:
                df_dU_i = torch.autograd.grad(
                    outputs=f_U_flat[i].real,
                    inputs=delta,
                    create_graph=False, # graph not needed later
                )[0] # real part of df(U)/dU
                df = (df_dU_i.conj() * A[i,:]).sum(dim=(-2, -1)).real # df/domega
            else:
                Re_df_dU_i = torch.autograd.grad(
                    outputs=f_U_flat[i].real,
                    inputs=delta,
                    create_graph=True,
                    retain_graph=True,
                )[0] # real part of df(U)/dU
                Im_df_dU_i = torch.autograd.grad(
                    outputs=f_U_flat[i].imag,
                    inputs=delta,
                    create_graph=True,
                    retain_graph=True,
                )[0] # imaginary part of df(U)/dU

                df_re = (Re_df_dU_i.conj() * A[i,:]).sum(dim=(-2, -1)).real # d Re(f)/domega
                df_im = (Im_df_dU_i.conj() * A[i,:]).sum(dim=(-2, -1)).real # d Im(f)/domega
                df = df_re + 1j * df_im
            #---
            df_domega_list.append(df)
        #---
        df_domega = torch.stack(df_domega_list, dim=0)
        return -1j * df_domega
    
    def L_a(self, a: int, f: typing.Callable, U: GaugeConfiguration, f_is_real: bool):
        """
        Left canonical momenta from the definition of the Lie derivative
        """
        f_U = f(GaugeConfiguration(self.V[...,a,:,:] @ U))
        f_U_flat = f_U.view(-1)
        batch_size = f_U_flat.numel() # f(U) is a scalar --> one component for each batch
        df_domega_list = []
        for i in range(batch_size):
            if f_is_real:
                df_domega_i = torch.autograd.grad(
                    outputs=f_U_flat[i],
                    inputs=self.omega,
                    create_graph=False, # no need to create graph
                    grad_outputs = torch.ones_like(f_U_flat[i]),
                )[0]
            else:
                Re_df_domega_i = torch.autograd.grad(
                    outputs=f_U_flat[i].real,
                    inputs=self.omega,
                    create_graph=True, # graph needed for imaginary part
                    grad_outputs = torch.ones_like(f_U_flat[i].real),
                )[0]
                Im_df_domega_i = torch.autograd.grad(
                    outputs=f_U_flat[i].imag,
                    inputs=self.omega,
                    create_graph=True,
                    grad_outputs = torch.ones_like(f_U_flat[i].imag),
                )[0]
                df_domega_i = Re_df_domega_i + 1j*Im_df_domega_i
            #---
            df_domega_list.append(df_domega_i)
        #---
        df_domega = torch.stack(df_domega_list, dim=0)
        return -1j*df_domega

    
    def La_squared_per_link(self, a: int, f: typing.Callable, U: GaugeConfiguration, f_is_real: bool):
        """
        Returns ${ \\sum_{x,\\mu} L_a^2(x,\\mu) f(U) }$
        NOTE: the index `a` is NOT summed over.
        
        `f(U)` should return a tensor of shape (batchsize,1)
        """
        Nc = U.Nc # number of colors
        n_links = U.n_links # number of links
        tau_a = self.tau[a,:,:]
        d, M = torch.linalg.eigh(tau_a)  # diagonalization of the generator tau_a
        omega = torch.tensor(0.0, requires_grad=True, dtype=U.real.dtype, device=U.device)
        phase = omega * d
        exp_iphase = torch.exp(-1j * phase)
        exp_iD = torch.diag_embed(exp_iphase)
        Va = (M @ exp_iD @ M.adjoint()) #.expand(n_links, Nc, Nc)
        batchsize = U.batch_size # number of configurations
        Id = torch.eye(Nc).to(device=U.device)
        Id_arr = Id.expand(n_links, Nc, Nc)
        sum_La_squared = []
        for b in range(batchsize):
            U_b = U[b,...].reshape(n_links, Nc, Nc)
            def f_i(i):
                e_i = torch.nn.functional.one_hot(i, n_links).to(dtype=U.dtype, device=U.device)
                Va_arr = Id_arr + torch.einsum("ab,i->iab", Va-Id, e_i)
                VaU_i = (Va_arr @ U_b).reshape(*(U[b,...].shape)).unsqueeze(0)
                f_VaU = f(GaugeConfiguration(VaU_i))/n_links
                return f_VaU
            #---
            # vectorize over all indices 0..N-1
            indices = torch.arange(n_links)
            sum_f = torch.vmap(f_i)(indices).sum()
    
            # \\sum_i \\partial_{x_i} f : directional derivative along (1,...,1)
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
        return torch.stack(sum_La_squared, dim=0)


    def La_squared_per_link_FD(self, a: int, f: typing.Callable, U: GaugeConfiguration, f_is_real: bool, eps: float = 1e-8):
        """
        Returns ${ \\sum_{x,\\mu} L_a^2(x,\\mu) f(U) }$ via Finite Differences.
        NOTE: the index `a` is NOT summed over.

        Uses the second-order central difference approximation:
            L_a^2 f(U) ≈ [f(e^{+i eps tau_a} U) - 2f(U) + f(e^{-i eps tau_a} U)] / eps^2

        `f(U)` should return a tensor of shape (batchsize,1)
        """
        Nc = U.Nc
        n_links = U.n_links
        tau_a = self.tau[a, :, :]
        batchsize = U.batch_size
        Id = torch.eye(Nc, dtype=U.dtype, device=U.device)
        Id_arr = Id.expand(n_links, Nc, Nc)

        # Compute Va(+eps) and Va(-eps): the group elements e^{±i eps tau_a}
        # Use matrix exponential via diagonalization: tau_a = M D M†, e^{i w tau_a} = M e^{i w D} M†
        d, M = torch.linalg.eigh(tau_a)

        def make_Va(omega: float):
            phase = omega * d
            exp_iD = torch.diag_embed(torch.exp(-1j * phase).to(dtype=U.dtype))
            return M @ exp_iD @ M.adjoint()

        omega = torch.tensor(0.0, requires_grad=True, dtype=U.real.dtype, device=U.device)
        Va_plus  = make_Va(+eps+omega)   # e^{+i eps tau_a}
        Va_minus = make_Va(-eps+omega)   # e^{-i eps tau_a}

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

            f_plus  = make_perturbed_U(Va_plus)   # sum_i f(Va(+eps) . U_i)
            f_minus = make_perturbed_U(Va_minus)  # sum_i f(Va(-eps) . U_i)
            # f_0     = f(GaugeConfiguration(U[b, ...].unsqueeze(0))) * n_links  # n_links * f(U)

            # Central difference: [f(+eps) - 2f(U) + f(-eps)] / eps^2
            # laplacian_b = (f_plus - 2 * f_0 + f_minus) / (eps ** 2)
            dir_der = (f_plus - f_minus)/(2.0*eps)

            if f_is_real:
                laplacian_b = torch.autograd.grad(dir_der, omega)[0]
            else:
                Re_laplacian_b  = torch.autograd.grad(dir_der.real, omega, create_graph=True)[0]
                Im_laplacian_b  = torch.autograd.grad(dir_der.imag, omega)[0]
                laplacian_b  = Re_laplacian_b + 1j*Im_laplacian_b

            sum_La_squared.append(laplacian_b)

        return torch.stack(sum_La_squared, dim=0)

    
    def La_squared_per_link_FD_fast(self, a: int, f: typing.Callable, U: GaugeConfiguration, f_is_real: bool, eps: float = 1e-8):
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

    # def La_squared_per_link_funcgrad(
    #     self,
    #     a: int,
    #     f: typing.Callable,
    #     U: GaugeConfiguration,
    #     f_is_real: bool,
    # ):
    #     """
    #     Computes

    #         sum_i L_a(i)^2 f(U)

    #     using torch.func.grad + jvp.

    #     Requires f(U) -> (batchsize,1)
    #     and currently supports real-valued outputs.
    #     """

    #     if not f_is_real:
    #         raise NotImplementedError(
    #             "Complex-valued output not implemented yet."
    #         )

    #     Nc = U.Nc
    #     n_links = U.n_links
    #     batchsize = U.batch_size

    #     tau_a = self.tau[a]

    #     d, M = torch.linalg.eigh(tau_a)

    #     U_links = U.as_subclass(torch.Tensor).reshape(
    #         batchsize,
    #         n_links,
    #         Nc,
    #         Nc,
    #     )

    #     omega0 = torch.zeros(
    #         n_links,
    #         dtype=U.real.dtype,
    #         device=U.device,
    #     )

    #     @torch.compile
    #     def laplacian_single(Ub):

    #         def scalar_function(omega):

    #             phase = omega[:, None] * d[None, :]

    #             exp_iD = torch.diag_embed(
    #                 torch.exp(-1j * phase)
    #             )

    #             Va = (
    #                 M[None]
    #                 @ exp_iD
    #                 @ M.adjoint()[None]
    #             )

    #             VaU = Va @ Ub

    #             cfg = GaugeConfiguration(
    #                 VaU.reshape((1,) + U.shape[1:])
    #             )

    #             #
    #             # grad() requires scalar output
    #             #
    #             return f(cfg).real.squeeze()

    #         grad_f = torch.func.grad(scalar_function)

    #         basis = torch.eye(
    #             n_links,
    #             dtype=omega0.dtype,
    #             device=omega0.device,
    #         )

    #         def diag_entry(v):

    #             _, hv = torch.func.jvp(
    #                 grad_f,
    #                 (omega0,),
    #                 (v,),
    #             )

    #             return torch.dot(hv, v)

    #         diag = torch.vmap(diag_entry)(basis)

    #         return diag.sum()

    #     return torch.vmap(laplacian_single)(U_links).unsqueeze(-1)

    def R_a(self, f: typing.Callable[[GaugeConfiguration], ColorMatrix], U: GaugeConfiguration):
        pass
        # f_prime_values = lambda V:  f(U @ V)
        # df_domega = componentwise_autodiff(y=f_prime_values, x=self.omega)
        # return +1j*df_domega



        
