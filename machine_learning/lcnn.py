"""
Implementation of the L-CNN as described in:
https://arxiv.org/abs/2012.12901

"""

import torch

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
