"""
Implementation of the L-CNN as described in:
https://arxiv.org/abs/2012.12901

TODO:
- check gauge covariance on all layers (random gauge transformation)
- implement Bilinear+Convolution single layer
- add unity and daggers to the set of W
- compare with line 198 of https://gitlab.com/openpixi/lge-cnn/-/blob/master/lge_cnn/nn/layers.py?ref_type=heads

"""

import typing
from itertools import chain
import torch

import lattice_data_tools.links.suN as suN
from lattice_data_tools.links.configuration import ColorMatrix, GaugeConfiguration, LocallyGaugeCovariant
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
        List of locally transforming variables.

        Returns: tensor of shape (batch_size, L1,...,Ld, N_var, Nc,Nc),
          where N_var is the total number of variables considered
        
        """
        Plaq = self.WLG.plaquettes()
        Poly = self.WLG.Polyakov_loops() 
        return torch.cat((Plaq, Poly), dim=-3)
    #---
    def get_Wprime(self, U_PT: torch.Tensor, W: torch.Tensor):
        """
        W' as in eq. III.11 of https://arxiv.org/pdf/2012.12901

        Input:
          U: gauge configuration. shape=(batch_size,L1,...,Ld,d,Nc,Nc)
          U_PT: parallel transporters.  shape=(batch_size,L1,...,Ld,d,2*K+1,Nc,Nc)
          W: parallel transporters.  shape=(batch_size, L1,...,Ld, N_var, Nc,Nc)

        Return:
          W_prime: parallel-transported W. shape=(batch_size, L1,...,Ld, d, 2*K+1, N_var, Nc,Nc)

        """
        W_shifted = get_W_shifted(U=self.U, W=W, K=self.K) # W_\\mu(x+k*\\mu)
        W_prime = torch.einsum("... m k a b, ... i m k b c, ... m k c d -> ... m k i a d", U_PT, W_shifted, U_PT.adjoint())
        return W_prime        
    #---
    def apply_L_conv(self, U: GaugeConfiguration, W: torch.Tensor, omega: torch.Tensor, K: int, N_out: int):
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
    def gen_random_omega(self, Nch_out: int, Nch_in: int, seed: int):
        """
        Initialize the \\omega for Eq. 5 of https://arxiv.org/pdf/2012.12901,
        such that the order of magnitude does not spoil gauge-covariance because of numerical rounding.
        This is achieved by setting them all to be O(1) per component
        """
        d = self.U.n_dims
        K = self.K
        torch.manual_seed(seed=seed)
        den = Nch_in * d * (2*K+1) 
        omega = (torch.rand(*(Nch_out,Nch_in,d,2*K+1)) / den).type(self.U.type())
        return omega
    #---
    def L_conv(self, U: GaugeConfiguration, U_PT: torch.Tensor, Wprime: torch.Tensor, omega: torch.Tensor, K: int):
        """
        Eq. 5 of https://arxiv.org/pdf/2012.12901, using Wprime as in eq. 11.

          omega: convolution parameters. shape=(N_out,N_in,d,2K+1)
          Wprime: parallel transported W. shape=(batch_size, L1,...,Ld, N_var, d, 2*K+1, Nc,Nc)

        Returns:
          W_conv: shape=(batchsize,L1,...,Ld,N_out,Nc,Nc)
        """
        # W_conv = torch.einsum("ijmk,...mkac,...jmkcd,...mkdb->...iab", omega, U_PT, Wprime, U_PT.adjoint())
        W_conv = LocallyGaugeCovariant(torch.einsum("ijmk,...mkjab->...iab", omega, Wprime))
        return W_conv
    #---
    def gen_random_alpha(self, N_out: int, N_in1: int, N_in2: int, seed: int):
        """
        Initialize the \\alpha for Eq. 6 of https://arxiv.org/pdf/2012.12901,
        such that the order of magnitude does not spoil gauge-covariance because of numerical rounding.
        This is achieved by setting them all to be O(1) per component
        """
        torch.manual_seed(seed=seed)
        den = N_in1 * N_in2 
        alpha = (torch.randn(N_out, N_in1, N_in2) / den).type(self.U.type())
        return alpha
    #---
    def L_Bilin(self, W: torch.Tensor, Wprime: torch.Tensor, alpha: torch.Tensor):
        """
        Eq. 6 of https://arxiv.org/pdf/2012.12901
        NOTE: In this layer, the W' are evaluated for no shift --> any \\mu and  shift index "K", corresponding to k=0 steps

        Input:
          alpha: parameters of the layer. shape: (N_out,N_in1, N_in2)
          W: locally transforming objects. shape=(batchsize, L1,...,Ld, N_in1, Nc, Nc)
          Wprime_x: locally transforming objects. shape=(batchsize, L1,...,Ld, N_in2, Nc, Nc)

        Returns:
          W_bilin: locally tranforming object. shape=(batchsize, L1,...,Ld, N_out, Nc, Nc)
        
        """
        W_bilin = torch.einsum("ijk,...jac,...kcb->...iab", alpha, W, Wprime)
        return W_bilin
    #---
    def L_act(self, W: torch.Tensor, act_fun: typing.Callable):
        """
        Eq. 7 of https://arxiv.org/pdf/2012.12901

        NOTE: act_fun() should be scalar-valued --> we use componentwise multiplication
        """
        f_U = act_fun(self.U,W).unsqueeze(dim=-1).unsqueeze(dim=-1).to_tensor()
        res = LocallyGaugeCovariant(f_U * W.to_tensor()) # componentwise operation, not matrix multiplication 
        return res
    #---
    def gen_random_beta(self, N_out: int, seed: int):
        """
        Initialize the \\beta for Eq. 8,9 of https://arxiv.org/pdf/2012.12901.
        The \\betas are multiplied componentwise, so we can initialize all of them to be O(1)
        to not spoil gauge-covariance because of numerical rounding.
        """
        d = self.U.n_dims
        torch.manual_seed(seed=seed)
        beta =  torch.rand(*(d, N_out)).type(self.U.type())
        return beta
    #---
    def L_exp(self, W: torch.Tensor, beta: torch.Tensor):
        """
        Eqs. 8 and 9 of https://arxiv.org/pdf/2012.12901
        beta: parameters. shape: (d, N_ch)

        U: gauge configuration: (batch, L1, ..., Ld, d, Nc, Nc)
        W: N_ch locally transforming variables (obtained after L_conv()). shape: (batch, L1, ..., Ld, N_ch, Nc, Nc)
        beta: shape: (d,N_ch)
        """
        # building the anti-hermitian part of W --> i*W_ah lies in the algebra su(N)
        W_ah_traceless = ColorMatrix(W).get_ah_traceless()
        arg_exp = torch.einsum("mi,...iab->...mab", 1j*beta, W_ah_traceless) # NOTE: arg_exp has to be hermitean --> we include the ""1j"
        E = suN.get_exp_iA(arg_exp.to_tensor()) # eq. 9 of https://arxiv.org/pdf/2012.12901
        EU = E @ self.U # eq. 8 of https://arxiv.org/pdf/2012.12901
        return EU
    #---
    def all_layers(self, omega: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, act_fun: typing.Callable = lambda U, W: suN.get_ReTr(W=W)):
        """
          omega: shape=(N_out,N_in,d,2K+1)
          alpha: shape=(N_out,N_in1,N_in2)
          beta: shape=(d,N_ch)
        """
        U_PT = get_ParallelTransporters(U=self.U, K=self.K)
        W = self.get_W() # set of locally transforming variables 
        Wprime = self.get_Wprime(W=W, U_PT=U_PT) # W' as in eq. III.11 of https://arxiv.org/pdf/2012.12901
        W_conv = self.L_conv(U=self.U, U_PT=U_PT, Wprime=Wprime, omega=omega, K=self.K) # W after eq. 5 of https://arxiv.org/pdf/2012.12901
        # Wprime_conv = self.get_Wprime(W=W_conv, U_PT=U_PT) # updated W' after L-Conv
        ## in Eq. 6, W' is just W (k=0 in eq. III.11)
        W_bilin = self.L_Bilin(W=W_conv, Wprime=W_conv, alpha=alpha) #  W after eq. 6 of https://arxiv.org/pdf/2012.12901
        W_act = self.L_act(W=W_bilin, act_fun=act_fun) # W after eq. 7 of https://arxiv.org/pdf/2012.12901
        EU = self.L_exp(W=W_act, beta=beta)
        W_res = LCNN(U=EU, K=self.K).get_W() 
        return W_res
 #---
                


    
    
