
import torch
from lattice_data_tools.links.configuration import GaugeConfiguration, LocallyGaugeCovariant

class WilsonLoopsGenerator:
    """
    Generation of Wilson loops from the gauge configuration.

      U: batch of gauge configurations U^{(b)}(x, mu) as Nc \\times Nc matrices, with x=(x_1,...x_d).
      It is passed as a multi-dimensional array U[b, L1,L2,...,Ld, d, Nc, Nc],
      where N_c is the number of colors.
    """
    @staticmethod
    def plaquettes(U: GaugeConfiguration):
        """
        Plaquettes (as loops, no trace or real part), assuming periodic boundary conditions on a L^d lattice.

        Returns: tensor of shape (batch_size, L1,...,Ld, N_plaq, Nc, Nc)
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
        plaqs = LocallyGaugeCovariant(torch.stack(plaqs, dim=-3))
        return plaqs
    #---
    @staticmethod
    def Polyakov_loops(U: GaugeConfiguration):
        """
        Polyakov loops for each direction mu.

        Returns: A tensor of shape (batch, L1, ..., Ld, d, Nc, Nc)
        """
        assert(U.is_complex())
        d = U.shape[-3] # number of dimensions
        lattice_shape = U.shape[1:-3] # shape of the lattice points grid
        poly_list = []
        for mu in range(d):
            L_mu = lattice_shape[mu] # extension of the lattice over the \\mu-th direction
            P = U[..., mu, :, :]
            for k in range(1, L_mu):
                P @= torch.roll(U, -k, dims=1+mu)[..., mu, :,:] # P_\\mu(x) --> P_\\mu(x)*U_\\mu(x+mu)
            #---
            poly_list.append(P)
        #---
        Poly = torch.stack(poly_list, dim=-3)
        return LocallyGaugeCovariant(Poly)



    
