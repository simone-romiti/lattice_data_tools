
import torch
from lattice_data_tools.links.configuration import GaugeConfiguration, LocallyGaugeCovariant
import lattice_data_tools.links.suN as suN

def get_ParallelTransporters(U: GaugeConfiguration, K: int):
    """
    Array of the parallel transporters \\prod_{i=0}^{j} U_\\mu(x+i*\\mu) , j=0,...,k

    Namely, the ones corresponding to a parallel transport along an axis, for up to "k" steps in the lattice spacing.

    Returns:
      Array of shape (batch_size,L1,...,Ld,d,2K+1,Nc,Nc)    
    """
    assert(U.is_complex())
    lattice_shape = U.lattice_shape # shape of the lattice points grid
    assert all(K <= Lmu for Lmu in lattice_shape) # the parallel transporters for |k|>L_\mu would be a redundancy
    d = U.n_dims # number of dimensions
    # array of parallel transporters V_{\mu,i}(x)
    # i=-K,...,K --> 2*K+1 components
    # I extend the original configuration and iteratively build the parallel transporters for each \\mu and "k"
    new_shape = tuple(U.shape[0:-2]) + (2*K+1,) + tuple(U.shape[-2:]) # shape: (batch_size,L1,...,Ld,d,2*K+1,Nc,Nc)
    ParallelTransporters = U.unsqueeze(-3).expand(*new_shape).clone()
    # setting identity element for k=0 --> no transport
    U0 = U.clone()
    suN.apply_coldstart(U=U0)
    ParallelTransporters[...,K,:,:] = U0
    for mu in range(d):
        # loop over k=-K,...,-1,+1,...,K.
        # NOTE: k=K corresponds to the link U_\mu(x): a parallel transport over 1 lattice site
        Lmu = U.shape[1 + mu] # extent of the lattice along the direction \\mu
        for k in range(1, K+1):
            # backward transporters
            i_bkw = K-k # index of the backward parallel transporter
            idx_bkw = (torch.arange(Lmu, device=U.device) - k + Lmu) % Lmu
            U_bkw = U.index_select(1 + mu, idx_bkw) # U_\\mu(x - \\mu)
            ParallelTransporters[..., mu, i_bkw, :, :] = ParallelTransporters[..., mu, i_bkw+1, :, :] @ U_bkw[...,mu,:,:].adjoint() # iterative construction
            # forward transporters
            i_fwd = K+k # index of the forward parallel transporter
            idx_fwd = (torch.arange(Lmu, device=U.device) + k-1) % Lmu
            U_fwd = U.index_select(1 + mu, idx_fwd) # U_\\mu(x+\\mu)
            ParallelTransporters[..., mu, i_fwd, :, :] = ParallelTransporters[..., mu, i_fwd-1, :, :] @  U_fwd[...,mu,:,:] # iterative construction
    #-------
    U_PT = ParallelTransporters.as_subclass(torch.Tensor) # downgrading to not confuse with a gauge configuration
    return U_PT
#---    



def get_W_shifted(U: GaugeConfiguration, W: LocallyGaugeCovariant, K: int):
    """
    W_\\mu(x+k*\\mu)

    Input:
      U: gauge configuration. shape:  (batch_size,L1,...,Ld,d,Nc,Nc)
      U_PT: parallel transporters. shape:  (batch_size,L1,...,Ld,d,2*K+1,Nc,Nc)
      W: locally gauge-transforming variables. shape  (batch_size,L1,...,Ld, N_var, Nc,Nc)

    Returns:
      W_shifted: shifted W. shape: (batch_size, L1,...,Ld, N_var, d, nK, Nc, Nc)
    
    """
    d = U.shape[-3] # number of dimensions
    nK = 2*K+1
    new_shape = tuple(W.shape[0:-2]) + (d,nK,) + tuple(W.shape[-2:]) # shape: (batch_size,L1,...,Ld,N_var,d,2*K+1,Nc,Nc) 
    W_shifted = torch.empty(new_shape, dtype=W.dtype, device=W.device)
    for k in range(-K, K+1):
        i_k = k+K
        for mu in range(d):
            W_shifted[...,mu,i_k,:,:] = torch.roll(W, shifts=-k, dims=1+mu) # W_\\mu(x+k*\\mu)
    #-------
    return W_shifted
#---
