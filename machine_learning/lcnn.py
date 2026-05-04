"""
Implementation of the L-CNN as described in:
https://arxiv.org/abs/2012.12901

TODO:
- check gauge covariance (random gauge transformation)
- implement L-CNN in one go
- compare with line 198 of https://gitlab.com/openpixi/lge-cnn/-/blob/master/lge_cnn/nn/layers.py?ref_type=heads

"""

import typing
from itertools import chain
import torch

import sys
sys.path.append("../../")

from lattice_data_tools.links.suN import get_Tr, get_ReTr, get_U_from_theta
from lattice_data_tools.links.configuration import GaugeConfiguration
from lattice_data_tools.links.loops import WilsonLoopsGenerator
from lattice_data_tools.links.parallel_transport import get_ParallelTransporters, get_W_shifted


class LCNN(torch.nn.Module):
    def __init__(self, U: GaugeConfiguration) -> None:
        super().__init__()
        self.U = U
        self.loops_generator = WilsonLoopsGenerator(U)
    #---

    def get_W(self, U: torch.tensor):
        """ List of locally transforming variables, as a tensor of shape (batch_size, n_variables, Nc,Nc) """
        Plaq = self.loops_generator.plaquettes()
        Poly = self.loops_generator.Polyakov_loops() 
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
        W_conv = torch.zeros(*(U.shape[0:-3] +  (N_out,) + U.shape[-2:])).type(U.type())
        ParallelTransporters = get_ParallelTransporters(U=U, K=K)
        for k in range(-K, K+1):
            i_k = k+K
            for mu in range(d):
                W_shifted = torch.roll(W, shifts=-k, dims=1+mu) # W_\\mu(x+k*\\mu)
                U_parall = ParallelTransporters[...,mu,i_k,:,:] # parallel transporters
                W_conv += torch.einsum("ij,...ac,...jcd,...db->...iab", omega[...,mu,i_k], U_parall, W_shifted, U_parall.adjoint())
        #-------
        return W_conv
    #---

    def L_conv_einsum(self, U: torch.tensor, W: torch.tensor, omega: torch.tensor, K: int):
        """
        Same as L_conv(), but using Einstein summation.
        It is slower that L_conv(), because it needs to compute W_shifted, which have to be allocated
        """
        U_PT = get_ParallelTransporters(U=U, K=K)
        W_shifted = get_W_shifted(U=U, U_PT=U_PT, W=W) # W_\\mu(x+k*\\mu)
        W_conv = torch.einsum("ijmk,...mkac,...jmkcd,...mkdb->...iab", omega, U_PT, W_shifted, U_PT.adjoint())
        return W_conv
    #---

    def L_Bilin(W: torch.tensor, Wprime: torch.tensor, alpha: torch.tensor):
        """
        Eq. 6 of https://arxiv.org/pdf/2012.12901
        
        alpha: parameters. shape: (N_out,N_in1, N_in2)
        """
        return torch.einsum("ijk,...jac,...kcb->...iab", alpha, W, Wprime)
    #---

    def L_act(self, U: torch.tensor, W: torch.tensor, act_func: typing.Callable = lambda U, W: torch.tanh(get_ReTr(W=W))):
        """
        Eq. 7 of https://arxiv.org/pdf/2012.12901

        NOTE: act_func() should be scalar-valued --> we use componentwise multiplication
        """
        return act_func(U,W) * W
    #---

    def L_exp(self, U: torch.tensor, W: torch.tensor, beta: torch.tensor):
        """
        Eqs. 8 and 9 of https://arxiv.org/pdf/2012.12901
        beta: parameters. shape: (d, N_ch)

        U: gauge configuration: (batch, L1, ..., Ld, d, Nc, Nc)
        W: N_ch locally transforming variables (obtained after L_conv()). shape: (batch, L1, ..., Ld, N_ch, Nc, Nc)
        """
        # building the anti-hermitian part of W --> i*W_ah lies in the algebra su(N)
        W_ah = W - W.adjoint() # taking the anti-hemitian part
        W_ah -= get_Tr(W_ah) # subtracting the trace
        arg_exp = torch.einsum("mi,...iab->...mab", beta, W_ah)
        E = exponentiate_suN(1j*arg_exp) # eq. 9 of https://arxiv.org/pdf/2012.12901
        EU = E @ U # eq. 8 of https://arxiv.org/pdf/2012.12901
        return EU
    #---
 #---
                


if __name__ == "__main__":
    import time
    print("===========================")
    print("L-CNN implementation script")
    print("===========================")
    device = torch.device("cpu")
    B = 1
    d = 3
    Lmu = d*[8]
    Nc = 3
    t1 = time.time()
    Ng = Nc**2 - 1
    theta = -torch.pi + (2*torch.pi)*torch.rand(B, *Lmu[0:d], d, Ng).to(device).type(torch.float64) # random angles in [-\\pi,\\pi]
    U = GaugeConfiguration.from_theta(theta)
    loops_generator = WilsonLoopsGenerator(U=U)
    # U = get_U_from_theta(theta=theta, N=Nc)
    # U = torch.randn(B, *Lmu[0:d], d, Nc, Nc).to(device).type(torch.complex64)
    print(U.shape)
    t2 = time.time()
    print(f"t2-t1: {t2-t1} sec.")
    Plaq = loops_generator.plaquettes()
    t3 = time.time()
    print(f"t3-t2: {t3-t2} sec.")
    Poly = loops_generator.Polyakov_loops()
    t4 = time.time()
    print(f"t4-t3: {t4-t3} sec.")
    lcnn1 = LCNN(U=U)
    t5 = time.time()
    print(f"t5-t4: {t5-t4} sec.")
    W = lcnn1.get_W(U=U)
    t6 = time.time()
    print(f"t6-t5: {t6-t5} sec.")
    N_in = W.shape[-3]
    N_out = 100
    K = 5
    # omega = torch.rand(N_out, N_in, d, 2*K+1) # convolution coefficients
    omega = torch.rand(N_out, N_in, d, 2*K+1, dtype=U.dtype, device=U.device)
    #omega = omega.type(U.type())
    t7 = time.time()
    print(f"t7-t6: {t7-t6} sec.")
    U_PT = get_ParallelTransporters(U=U, K=K)
    t8 = time.time()
    print(f"t8-t7: {t8-t7} sec.")
    W_shifted = get_W_shifted(U=U, U_PT=U_PT, W=W) # W_\\mu(x+k*\\mu)
    t9 = time.time()
    print(f"t9-t8: {t9-t8} sec.")
    W_conv = lcnn1.L_conv(U=U, W=W, omega=omega, K=K, N_out=N_out)
    t10 = time.time()
    print(f"t10-t9: {t10-t9} sec.")
    W_conv_einsum = lcnn1.L_conv_einsum(U=U, W=W, omega=omega, K=K)
    t11 = time.time()
    print(f"t11-t10: {t11-t10} sec.")
    print(f"N_in={N_in}, N_out={N_out}, K={K}")
    print(U.shape)
    print(Plaq.shape)
    print(Poly.shape)
    print(U_PT.shape)
    print(W.shape)
    print(W_conv.shape)
    print("Checking the einsum implementation of the convolution")
    dW_conv = W_conv - W_conv_einsum
    print("Absolute values")
    dW_conv_abs = torch.abs(dW_conv)
    print("Hello", type(dW_conv_abs))
    print(torch.max(torch.abs(W_conv)))
    print(torch.max(dW_conv_abs))
