
import torch
from lattice_data_tools.links.configuration import GaugeConfiguration

def get_ParallelTransporters(U: GaugeConfiguration, K: int):
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
    # I extend the original configuration and iteratively build the parallel transporters for each \\mu and "k"
    new_shape = tuple(U.shape[0:-2]) + (2*K+1,) + tuple(U.shape[-2:]) # shape: (batch_size,L1,...,Ld,d,2*K+1,Nc,Nc)
    ParallelTransporters = U.unsqueeze(-3).expand(*new_shape).clone()
    for mu in range(d):
        # loop over k=-K,...,-1,+1,...,K.
        # NOTE: k=K corresponds to the link U_\mu(x): a parallel transport over 1 lattice site
        for k in range(1, K+1):
            i_bkw = K-k # index of the backward parallel transporter
            i_fwd = K+k # index of the forward parallel transporter
            # U_shift = torch.roll(ParallelTransporters, -k, dims=1+mu) # U_{\\mu}(x+k*\\mu)
            PT_k_bkw = ParallelTransporters[..., mu, i_bkw, :, :]
            PT_k_bkw = ParallelTransporters[..., mu, i_bkw+1, :, :] @ PT_k_bkw # iterative contruction
            PT_k_fwd = ParallelTransporters[..., mu, i_fwd, :, :]
            PT_k_fwd = ParallelTransporters[..., mu, i_fwd-1, :, :] @ PT_k_fwd # iterative contruction
    #-------
    return ParallelTransporters
#---    


def get_W_shifted(U: GaugeConfiguration, U_PT: torch.tensor, W: torch.tensor):
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
    W_shifted = torch.empty(new_shape, dtype=W.dtype, device=W.device)
    for k in range(-K, K+1):
        i_k = k+K
        for mu in range(d):
            W_shifted[...,mu,i_k,:,:] = torch.roll(W, shifts=-k, dims=1+mu) # W_\\mu(x+k*\\mu)
    #-------
    return W_shifted
#---
