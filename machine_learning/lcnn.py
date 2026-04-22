"""
Implementation of the L-CNN as described in:
https://arxiv.org/abs/2012.12901

"""

from itertools import chain
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
#---

def get_ParallelTransporters(U: torch.tensor, K: int):
    """
    Returns the array of the parallel transporters \\prod_{i=0}^{j} U_\\mu(x+i*\\mu) , j=0,...,k

    Namely, the ones corresponding to a parallel transport along an axis, for up to "k" steps in the lattice spacing.
    """
    assert(U.is_complex())
    lattice_shape = U.shape[1:-3] # shape of the lattice points grid
    assert all(K <= Lmu for Lmu in lattice_shape) # the parallel transporters for |k|>L_\mu would be a redundancy
    d = U.shape[-3] # number of dimensions
    # array of parallel transporters V_{\mu,i}(x)
    # i=-K,...,K --> 2*K+1 components
    new_shape = tuple(U.shape[0:-2]) + (2*K+1,) + tuple(U.shape[-2:]) # shape: (batch_size,L1,...,Ld,d,2*K+1,Nc,Nc)
    ParallelTransporters = U.unsqueeze(-3).expand(*new_shape).clone()
    for mu in range(d):
        # loop over k=-K,...,-1,+1,...,K.
        # NOTE: k=K corresponds to the link U_\mu(x): a parallel transport over 1 lattice site
        for k in chain(range(1, K+1), range(-1, -K-1, -1)):
            U_parall = torch.roll(ParallelTransporters, -k, dims=1+mu) # \prod_{i=0}^{k} U_{\\mu}(x+k*\\mu)
            i_prev = k if (k>0) else K+k
            i_next = K+k
            PT_k = ParallelTransporters[..., mu, K+k, :, :]
            PT_k = PT_k @ U_parall[..., mu, K+k-1, :,:] # iterative contruction
    #-------
    return ParallelTransporters
#---    


def get_W_shifted(U: torch.tensor, U_PT: torch.tensor, W: torch.tensor):
    """
    Returns W_\\mu(x+k*\\mu), as a tensor of shape  (batch_size,L1,...,Ld,d,2*K+1,Nc,Nc)

    U: gauge configuration. shape:  (batch_size,L1,...,Ld,d,Nc,Nc)
    U_PT: parallel transporters. shape:  (batch_size,L1,...,Ld,d,2*K+1,Nc,Nc)
    W: locally gauge-transforming variables. shape  (batch_size,L1,...,Ld, N_var, Nc,Nc)
    
    """
    d = U.shape[-3] # number of dimensions
    nK = U_PT.shape[-3] # makimum length of paraller transport along an axis
    K = (nK-1)//2 # nK = 2*K+1
    new_shape = tuple(W.shape[0:-2]) + (d,nK,) + tuple(W.shape[-2:]) # shape: (batch_size,L1,...,Ld,N_var,d,2*K+1,Nc,Nc) 
    # W_shifted = W.unsqueeze(-3).unsqueeze(-3).expand(*new_shape) # W_\\mu(x+k*\\mu)
    W_shifted = torch.empty(new_shape, dtype=W.dtype, device=W.device)
    for k in range(-K, K+1):
        i_k = k+K
        for mu in range(d):
            W_shifted[...,mu,i_k,:,:] = torch.roll(W, shifts=-k, dims=1+mu) # W_\\mu(x+k*\\mu)
    #-------
    return W_shifted
#---



class LCNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    #---

    def get_W(self, U: torch.tensor):
        """ List of locally transforming variables, as a tensor of shape (batch_size, n_variables, Nc,Nc) """
        Plaq = get_plaquettes(U) # torch.flatten(get_plaquettes(U), start_dim=1, end_dim=-3) # all plaquettes. shape: (batch_size, n_plaq, Nc,Nc)
        Poly = get_Polyakov_loops(U) # torch.flatten(get_Polyakov_loops(U), start_dim=1, end_dim=-3) # all Polyakov loops. shape: (batch_size, n_poly, Nc,Nc)
        W = torch.cat((Plaq, Poly), dim=-3)
        return W
    #---
    
    def L_conv(self, U: torch.tensor, W: torch.tensor, omega: torch.tensor, K: int, N_out: int):
        """
        Eq. 5 of https://arxiv.org/pdf/2012.12901

        U: gauge configuration. shape: (batchsize, L1,...Ld, d, Nc, Nc)
        
        Def. N_in: number of input channels (inferred from W shape)
        W: array of W objects. shape (batchsize, L1,...Ld, N_in)

        K: size of the convolution kernel
        N_out: output channels
        omega: convolution coefficients. shape (N_out,N_in,d,K)

        """
        d = U.shape[-3] # number of dimensions
        # batch_size = W.shape[0]
        # N_in = W.shape[-3] # number of W objects
        # lattice_shape = U.shape[1:-3] # shape of the lattice points grid
        W_conv = torch.zeros(*(U.shape[0:-3] +  (N_out,) + U.shape[-2:])).type(U.type())
        ParallelTransporters = get_ParallelTransporters(U=U, K=K)
        for k in range(-K, K+1):
            i_k = k+K
            for mu in range(d):
                W_shifted = torch.roll(W, shifts=-k, dims=1+mu) # W_\\mu(x+k*\\mu)
                U_parall = ParallelTransporters[...,mu,i_k,:,:] # parallel transporters
                W_conv += torch.einsum('ij,...ac,...jcd,...db->...iab', omega[...,mu,i_k], U_parall, W_shifted, U_parall.adjoint())
        #-------
        return W_conv

    def L_conv_new(self, U: torch.tensor, W: torch.tensor, omega: torch.tensor, K: int, N_out: int):
        """
        Eq. 5 of https://arxiv.org/pdf/2012.12901

        U: gauge configuration. shape: (batchsize, L1,...Ld, d, Nc, Nc)
        
        Def. N_in: number of input channels (inferred from W shape)
        W: array of W objects. shape (batchsize, L1,...Ld, N_in)

        K: size of the convolution kernel
        N_out: output channels
        omega: convolution coefficients. shape (N_out,N_in,d,K)

        """
        U_PT = get_ParallelTransporters(U=U, K=K)
        W_shifted = get_W_shifted(U=U, U_PT=U_PT, W=W) # W_\\mu(x+k*\\mu)
        W_conv = torch.einsum('ijmk,...mkac,...jmkcd,...mkdb->...iab', omega, U_PT, W_shifted, U_PT.adjoint())
        return W_conv
 #-------
                


if __name__ == "__main__":
    import time
    print("===========================")
    print("L-CNN implementation script")
    print("===========================")
    B = 1
    Lmu = [12, 12, 12]
    d = 3
    Nc = 3
    t1 = time.time()
    U = torch.randn(B, *Lmu[0:d], d, Nc, Nc).to(torch.complex64)
    t2 = time.time()
    print(f"{t2-t1} sec.")
    Plaq = get_plaquettes(U)
    t3 = time.time()
    print(f"{t3-t2} sec.")
    Poly = get_Polyakov_loops(U)
    t4 = time.time()
    print(f"{t4-t3} sec.")
    lcnn1 = LCNN()
    t5 = time.time()
    print(f"{t5-t4} sec.")
    W = lcnn1.get_W(U=U)
    t6 = time.time()
    print(f"{t6-t5} sec.")
    N_in = W.shape[-3]
    N_out = 100
    K = 5
    # omega = torch.rand(N_out, N_in, d, 2*K+1) # convolution coefficients
    omega = torch.rand(N_out, N_in, d, 2*K+1, dtype=U.dtype, device=U.device)
    #omega = omega.type(U.type())
    t7 = time.time()
    print(f"{t7-t6} sec.")
    PT = get_ParallelTransporters(U=U, K=K)
    t8 = time.time()
    print(f"{t8-t7} sec.")
    W_conv = lcnn1.L_conv(U=U, W=W, omega=omega, K=K, N_out=N_out)
    t9 = time.time()
    print(f"{t9-t8} sec.")
    W_conv_new = lcnn1.L_conv_new(U=U, W=W, omega=omega, K=K, N_out=N_out)
    t10 = time.time()
    print(f"{t10-t9} sec.")
    print(f"N_in={N_in}, N_out={N_out}, K={K}")
    print(U.shape)
    print(Plaq.shape)
    print(Poly.shape)
    print(PT.shape)
    print(W.shape)
    print(W_conv.shape)
    print(torch.max(torch.abs(W_conv - W_conv_new)))
