
import torch
from lattice_data_tools.links.configuration import GaugeConfiguration, LocallyGaugeCovariant

class WilsonLoopsGenerator:
    def __init__(self, U: GaugeConfiguration):
        """
          U: batch of gauge configurations U^{(b)}(x, mu) as Nc \\times Nc matrices, with x=(x_1,...x_d).
          It is passed as a multi-dimensional array U[b, L1,L2,...,Ld, d, Nc, Nc],
          where N_c is the number of colors.
        """
        self.U = U
    #---
    def plaquettes(self):
        """
        Plaquettes (as loops, no trace or real part), assuming periodic boundary conditions on a L^d lattice.

        Returns: tensor of shape (batch_size, L1,...,Ld, N_plaq, Nc, Nc)
        """
        assert(self.U.is_complex()) # check that the gauge configuration is complex-valued
        d = self.U.shape[-3] # number of dimensions of the lattice
        plaqs = []
        # loop over the positive-oriented plaquettes
        for mu in range(d):
            for nu in range(mu + 1, d):
                U_mu = self.U[..., mu, :, :] # U_\\mu(\cdot) : all links along the direction \\mu
                U_nu = self.U[..., nu, :, :] # U_\\nu(\cdot) : all links along the direction \\nu
                # Spatial dim \\mu is dim \\mu+1 in the full tensor (batch at 0)
                U_mu_fwd = torch.roll(U_mu, -1, dims=1+nu) # U_\\mu(x + \\nu): shifting **backwards**
                U_nu_fwd = torch.roll(U_nu, -1, dims=1+mu) # U_\\nu(x + \\mu): shifting **backwards**
                P_munu = U_mu @ U_nu_fwd @ U_mu_fwd.adjoint() @ U_nu.adjoint() # U_\\mu(x)*U_\\nu(x+\mu)*U_\\mu(x+\\nu)^\dagger*U_\\mu(x)^\dagger
                plaqs.append(P_munu) # appending the plaquette to the list
        #-------
        plaqs = LocallyGaugeCovariant(torch.stack(plaqs, dim=-3))
        return plaqs
    #---
    def Polyakov_loops(self):
        """
        Polyakov loops for each direction mu.

        Returns: A tensor of shape (batch, L1, ..., Ld, d, Nc, Nc)
        """
        assert(self.U.is_complex())
        d = self.U.shape[-3] # number of dimensions
        lattice_shape = self.U.shape[1:-3] # shape of the lattice points grid
        Poly = self.U.clone() # copy of the links. In the loop, each is extended to its Polyakov loop
        for mu in range(d):
            L_mu = lattice_shape[mu] # extension of the lattice over the \\mu-th direction
            for k in range(1, L_mu):
                Poly[..., mu, :, :] @= torch.roll(self.U, -k, dims=1+mu)[..., mu, :,:] # P_\\mu(x) --> P_\\mu(x)*U_\\mu(x+mu)
        #-------
        return LocallyGaugeCovariant(Poly)
    #---


    
