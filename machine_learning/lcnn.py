"""
Implementation of the L-CNN as described in:
https://arxiv.org/abs/2012.12901

"""

import torch
from torch.nn import parallel

def get_plaquettes(U: torch.Tensor):
    """
    Returns the array of plaquettes, assuming periodic boundary conditions on a L^d lattice.
    
    U: batch of gauge configurations U^{(b)}(x, mu) as Nc \\times Nc matrices, with x=(x_1,...x_d).
       It is passed as a multi-dimensional array U[b, L1,L2,...,Ld, d, Nc, Nc],
       where N_c is the number of colors.
    """
    assert(U.is_complex()) # check that the gauge configuration is complex-valued
    d = U.shape[-3] # number of dimensions of the lattice
    plaqs = []
    # loop over the positive-oriented plaquettes
    for mu in range(d):
        for nu in range(mu + 1, d):
            U_mu = U[..., mu, :, :] # U_\\mu(\cdot) : all links along the direction \\mu
            U_nu = U[..., nu, :, :] # U_\\nu(\cdot) : all links along the direction \\nu
            # Spatial dim \\mu is dim \\mu+1 in the full tensor (batch at 0)
            U_mu_fwd = torch.roll(U_mu, -1, dims=1+nu) # U_\\mu(x + \\nu): shifting **backwards**
            U_nu_fwd = torch.roll(U_nu, -1, dims=1+mu) # U_\\nu(x + \\mu): shifting **backwards**
            P_munu = U_mu @ U_nu_fwd @ U_mu_fwd.adjoint() @ U_nu.adjoint() # U_\\mu(x)*U_\\nu(x+\mu)*U_\\mu(x+\\nu)^\dagger*U_\\mu(x)^\dagger
            plaqs.append(P_munu) # appending the plaquette to the list
    #-------
    plaqs = torch.stack(plaqs, dim=-3)
    return plaqs
#---


def get_Polyakov_loops(U: torch.Tensor):
    """
    Computes the Polyakov loops for each direction mu.
    
    U: [batch, L1, L2, ..., Ld, d, Nc, Nc]
    Returns: A tensor of shape [batch, L1, ..., Ld, d, Nc, Nc] 
             where the L_mu dimension is reduced or contains the loop.
             Commonly, it returns [batch, (transverse_dims), d, Nc, Nc].
    """
    assert(U.is_complex())
    d = U.shape[-3] # number of dimensions
    lattice_shape = U.shape[1:-3] # shape of the lattice points grid
    Poly = U.clone() # copy of the links. In the loop, each is extended to its Polyakov loop
    for mu in range(d):
        L_mu = lattice_shape[mu] # extension of the lattice over the \\mu-th direction
        for k in range(1, L_mu):
            Poly[..., mu, :, :] @= torch.roll(U, -k, dims=1+mu)[..., mu, :,:] # P_\\mu(x) --> P_\\mu(x)*U_\\mu(x+mu)
    #-------
    return Poly
#---

def get_ParallelTransporters(k: int):
    """
    Returns the array of the parallel transporters \\prod_{i=0}^{j} U_\mu(x+i*\\mu) , j=0,...,k

    Namely, the ones corresponding to a parallel transport along an axis, for up to "k" steps in the lattice spacing.
    """
    assert(U.is_complex())
    d = U.shape[-3] # number of dimensions
    lattice_shape = U.shape[1:-3] # shape of the lattice points grid
    new_shape = tuple(U.shape[0:-2]) + (k,) + tuple(U.shape[-2:]) # shape: (batch_size,L1,...,Ld,d,k,3,3)
    ParallelTransporters = U.unsqueeze(-2).expand(*new_shape).clone() # array of parallel transporters V_{\mu,i}(x)
    for mu in range(d):
        L_mu = lattice_shape[mu] # extension of the lattice over the \\mu-th direction
        assert(k<=L_mu)
        for i in range(1, k):
            ParallelTransporters_fwd = torch.roll(ParallelTransporters, -i, dims=1+mu) # V_{\\mu,i}(x+i*\\mu)
            ParallelTransporters[..., mu, i, :, :] @= ParallelTransporters_fwd[..., mu, i, :,:] # iterative contruction
    #-------
    return ParallelTransporters
#---    

class LCNN(torch.nn):
    def __init__(self) -> None:
        super().__init__()
    #---

    def get_W(U: torch.tensor):
        """ List of locally transforming variables, as a tensor of shape (batch_size, n_variables, 3,3) """
        Plaq = torch.flatten(get_plaquettes(U), start_dim=1, end_dim=-3) # all plaquettes. shape: (batch_size, n_plaq, 3,3)
        Poly = torch.flatten(get_Polyakov_loops(U), start_dim=1, end_dim=-3) # all Polyakov loops. shape: (batch_size, n_poly, 3,3)
        W = torch.stack((Plaq, Poly), dim=1)
        return W
    #---
    
    def LGE(W: torch.tensor, omega: torch.tensor, K: int):
        """ Eq. 5 of https://arxiv.org/pdf/2012.12901"""
        d = W.shape[-3] # number of dimensions
        lattice_shape = W.shape[1:-3] # shape of the lattice points grid
        W_conv = torch.zeros_like(W)
        for k in range(1, K + 1):
            ParallelTransporters = get_ParallelTransporters(k=k)
            for mu in range(d):
                W_shifted = torch.roll(W, shifts=-k, dims=1+mu)[...,mu,:,:] # W_\\mu(x+k*\\mu)
                U_parall = ParallelTransporters[...,mu,k,:,:] # parallel transporters
                W_transported = U_parall @ W_shifted @ (U_parall.adjoint()) #
                W_out += torch.einsum('ijmk,bs...in->bs...on', omega, transported)
                


if __name__ == "__main__":
    print("===========================")
    print("L-CNN implementation script")
    print("===========================")
    B = 1
    L1, L2, L3 = 16, 16, 16
    d = 3
    Nc = 3
    U = torch.randn(B, L1, L2, L3, d, Nc, Nc).to(torch.complex64)
    Plaq = get_plaquettes(U)
    Poly = get_Polyakov_loops(U)
    print(U.shape, Plaq.shape, Poly.shape)
