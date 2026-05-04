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

from lattice_data_tools.links.suN import get_Tr, get_ReTr, get_U_from_theta
from lattice_data_tools.links.configuration import GaugeConfiguration
from lattice_data_tools.links.loops import WilsonLoopsGenerator
from lattice_data_tools.links.parallel_transport import get_ParallelTransporters, get_W_shifted


class LCNN(torch.nn.Module):
    def __init__(self, U: GaugeConfiguration, K: int) -> None:
        """
        U: tensor representing a gauge configuration. shape: (batchsize, L1,...Ld, d, Nc, Nc)
        K: cf. eq. 5 of https://arxiv.org/pdf/2012.12901
        N_in, N_out: cf. eq. 6 of https://arxiv.org/pdf/2012.12901

        """
        super().__init__()
        self.U = U
        self.WLG = WilsonLoopsGenerator(U)
        self.K = K # K <= k <= K in the parallel transporters of length "k"
    #---
    def get_W(self):
        """
        List of locally transforming variables,
        as a tensor of shape (batch_size, n_variables, Nc,Nc)
        """
        Plaq = self.WLG.plaquettes()
        Poly = self.WLG.Polyakov_loops() 
        return torch.cat((Plaq, Poly), dim=-3)
    #---
    def get_Wprime(self, U: torch.Tensor, U_PT: torch.Tensor,  W: torch.Tensor):
        """ W' as in eq. 11 of https://arxiv.org/pdf/2012.12901 """
        Wprime = get_W_shifted(U=U, U_PT=U_PT, W=W) # W_\\mu(x+k*\\mu)
        return Wprime        
    #---
    def apply_L_conv(self, U: torch.Tensor, W: torch.Tensor, omega: torch.Tensor, K: int, N_out: int):
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
        ParallelTransporters = get_ParallelTransporters(U=U, K=self.K)
        for k in range(-K, K+1):
            i_k = k+K
            for mu in range(d):
                W_shifted = torch.roll(W, shifts=-k, dims=1+mu) # W_\\mu(x+k*\\mu)
                U_parall = ParallelTransporters[...,mu,i_k,:,:] # parallel transporters
                W_conv += torch.einsum("ij,...ac,...jcd,...db->...iab", omega[...,mu,i_k], U_parall, W_shifted, U_parall.adjoint())
        #-------
        return W_conv
    #---
    def L_conv(self, U: torch.Tensor, U_PT: torch.Tensor, Wprime: torch.Tensor, omega: torch.Tensor, K: int):
        """
        Eq. 5 of https://arxiv.org/pdf/2012.12901, using Wprime as in eq. 11.
        """
        W_conv = torch.einsum("ijmk,...mkac,...jmkcd,...mkdb->...iab", omega, U_PT, Wprime, U_PT.adjoint())
        return W_conv
    #---
    def L_Bilin(W: torch.Tensor, Wprime: torch.Tensor, alpha: torch.Tensor):
        """
        Eq. 6 of https://arxiv.org/pdf/2012.12901
        
        alpha: parameters. shape: (N_out,N_in1, N_in2)
        """
        return torch.einsum("ijk,...jac,...kcb->...iab", alpha, W, Wprime)
    #---
    def L_act(self, U: torch.Tensor, W: torch.Tensor, act_func: typing.Callable = lambda U, W: torch.tanh(get_ReTr(W=W))):
        """
        Eq. 7 of https://arxiv.org/pdf/2012.12901

        NOTE: act_func() should be scalar-valued --> we use componentwise multiplication
        """
        return act_func(U,W) * W
    #---
    def L_exp(self, U: torch.Tensor, W: torch.Tensor, beta: torch.Tensor):
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
    def all_layers(self, alpha: torch.Tensor, beta: torch.Tensor, act_fun: typing.Callable):
        U_PT = get_ParallelTransporters(U=U, K=self.K)
        W = self.get_W() # set of locally transforming variables
        Wprime = self.get_Wprime(U=self.U, W=W) # W' as in eq. 11 of https://arxiv.org/pdf/2012.12901
        W_conv = self.L_conv(U=self.U, U_PT=U_PT, Wprime=Wprime, omega=omega, K=self.K) # W after eq. 5 of https://arxiv.org/pdf/2012.12901
        W_bilin = self.L_Bilin(Wprime=Wprime, alpha=alpha) #  W after eq. 6 of https://arxiv.org/pdf/2012.12901
        W_act = self.L_act(U=U, W=W_bilin, act_fun=act_fun) # W after eq. 7 of https://arxiv.org/pdf/2012.12901
        EU = self.L_exp(U=self.U, W=W_act, beta=beta)
        return EU
 #---
                


    
    
