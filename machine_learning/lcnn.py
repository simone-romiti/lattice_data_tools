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
#from itertools import chain
import torch

import lattice_data_tools.links.suN as suN
from lattice_data_tools.links.configuration import ColorMatrix, GaugeConfiguration, LocallyGaugeCovariant
from lattice_data_tools.links.loops import WilsonLoopsGenerator
from lattice_data_tools.links.parallel_transport import get_ParallelTransporters, get_W_shifted

def default_activation_function(U: GaugeConfiguration, W: LocallyGaugeCovariant):
    return suN.get_ReTr(W=W)
#---    

class LCNN(torch.nn.Module):
    def __init__(
            self,
            U: GaugeConfiguration,
            K: int,
            act_fun: typing.Callable = default_activation_function
    ) -> None:
        """
        U: tensor representing a gauge configuration. shape: (batchsize, L1,...Ld, d, Nc, Nc). Used only to infer general properties.
        
        K: cf. eq. 5 of https://arxiv.org/pdf/2012.12901
        N_in, N_out: cf. eq. 6 of https://arxiv.org/pdf/2012.12901

        """
        super().__init__()
        self.Nc = U.Nc
        self.n_dims = U.n_dims
        self.dtype_U = U.dtype
        self.K = K # K <= k <= K in the parallel transporters of length "k"
        self.act_fun = act_fun # activation function
    #---
    def nK(self):
        return 2*self.K+1
    #---
    def get_W(self, U: GaugeConfiguration):
        """
        List of locally transforming variables.

        Returns: tensor of shape (batch_size, L1,...,Ld, N_var, Nc,Nc),
          where N_var is the total number of variables considered
        
        """
        Plaq = WilsonLoopsGenerator.plaquettes(U=U)
        Poly = WilsonLoopsGenerator.Polyakov_loops(U=U)
        Nc = self.Nc
        Unity = torch.eye(Nc).expand(*Plaq.shape[0:-3], 1, Nc, Nc)
        res = torch.cat((Plaq, Poly, Unity, Plaq.adjoint(), Poly.adjoint()), dim=-3)
        res_LGC = LocallyGaugeCovariant(res)
        return res_LGC
    #---
    def get_Wprime(self, U: GaugeConfiguration, U_PT: torch.Tensor, W: LocallyGaugeCovariant):
        """
        W' as in eq. III.11 of https://arxiv.org/pdf/2012.12901

        Input:
          U: gauge configuration. shape=(batch_size,L1,...,Ld,d,Nc,Nc)
          U_PT: parallel transporters.  shape=(batch_size,L1,...,Ld,d,nK,Nc,Nc)
          W: parallel transporters.  shape=(batch_size, L1,...,Ld, N_var, Nc,Nc)

        Return:
          W_prime: parallel-transported W. shape=(batch_size, L1,...,Ld, d, nK, N_var, Nc,Nc)

        """
        W_shifted = get_W_shifted(U=U, W=W, K=self.K) # W_\\mu(x+k*\\mu)
        Wprime = torch.einsum("... m k a b, ... i m k b c, ... m k c d -> ... m k i a d", U_PT, W_shifted, U_PT.adjoint())
        return LocallyGaugeCovariant(Wprime)        
    #---
    def gen_random_omega(self, Nch_out: int, Nch_in: int, seed: int):
        """
        Initialize the \\omega for Eq. 5 of https://arxiv.org/pdf/2012.12901,
        such that the order of magnitude does not spoil gauge-covariance because of numerical rounding.
        This is achieved by setting them all to be O(1) per component
        """
        d = self.n_dims
        K = self.K
        torch.manual_seed(seed=seed)
        nK = self.nK()
        den = Nch_in * d * nK 
        omega = (torch.rand(*(Nch_out,Nch_in,d,nK)) / den).to(self.dtype_U)
        return omega
    #---
    def L_conv(self, Wprime: torch.Tensor, omega: torch.Tensor):
        """
        Eq. 5 of https://arxiv.org/pdf/2012.12901, using Wprime as in eq. III.11.

          omega: convolution parameters. shape=(N_out,N_in,d,nK)
          Wprime: parallel transported W. shape=(batch_size, L1,...,Ld, d, nK, N_var, Nc,Nc)

        Returns:
          W_conv: shape=(batchsize,L1,...,Ld,N_out,Nc,Nc)
        """
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
        alpha = (torch.randn(N_out, N_in1, N_in2) / den).to(self.dtype_U)
        return alpha
    #---
    def L_Bilin(self, W: LocallyGaugeCovariant, Wprime: LocallyGaugeCovariant, alpha: torch.Tensor):
        """
        Eq. 6 of https://arxiv.org/pdf/2012.12901
        NOTE: In this layer, the W' are evaluated for no shift --> any \\mu and  shift index "K", corresponding to k=0 steps

        Input:
          alpha: parameters of the layer. shape: (N_out,N_in1, N_in2)
          W: locally transforming objects. shape=(batchsize, L1,...,Ld, N_in1, Nc, Nc)
          Wprime: locally transforming objects. shape=(batchsize, L1,...,Ld, N_in2, Nc, Nc)

        Returns:
          W_bilin: locally tranforming object. shape=(batchsize, L1,...,Ld, N_out, Nc, Nc)
        
        """
        W_bilin = torch.einsum("ijk,...jac,...kcb->...iab", alpha, W, Wprime)
        return W_bilin
    #---
    def gen_random_omega_CB(self, N_out: int, N_in: int, seed: int):
        """
        Initialize \\omega_{CB} for eq. 18 of https://arxiv.org/pdf/2401.06481
        shape: (N_out, N_in1, N_in2, d, nK) [NOTE: \\mu before k]
        """
        torch.manual_seed(seed=seed)
        nK = self.nK()
        d = self.n_dims
        den = (N_in**2) * d * nK
        omega_CB = (torch.randn(N_out, N_in, N_in, d, nK) / den).to(self.dtype_U)
        return omega_CB
    #---
    def L_CB(self, W: LocallyGaugeCovariant, Wprime: LocallyGaugeCovariant, omega_CB: torch.Tensor):
        """
        Combination of L-Conv and L-Bilin into L-CB
        as in eq. 18 https://arxiv.org/pdf/2401.06481

        Input:
          omega_CB: parameters of the layer. shape: (N_out, N_in1, N_in2, d, nK) [NOTE: \\mu before k]
          W: locally transforming objects. shape=(batchsize, L1,...,Ld, N_in1, Nc, Nc)
          Wprime: parallel transported W. shape=(batch_size, L1,...,Ld, d, nK, N_in2, Nc,Nc)

        Returns:
          W_CB: locally tranforming object. shape=(batchsize, L1,...,Ld, N_out, Nc, Nc)

        """
        W_CB = torch.einsum("abc mk,...b pq,...mk c qr->...a pr", omega_CB, W, Wprime)
        return LocallyGaugeCovariant(W_CB)
    #---
    def L_act(self, U: GaugeConfiguration, W: LocallyGaugeCovariant, act_fun: typing.Callable = default_activation_function):
        """
        Eq. 7 of https://arxiv.org/pdf/2012.12901

        NOTE: act_fun() should be scalar-valued --> we use componentwise multiplication
        """
        f_U = act_fun(U, W).unsqueeze(dim=-1).unsqueeze(dim=-1).to_tensor()
        res = LocallyGaugeCovariant(f_U * W.to_tensor()) # componentwise operation, not matrix multiplication 
        return res
    #---
    def gen_random_beta(self, N_out: int, seed: int):
        """
        Initialize the \\beta for Eq. 8,9 of https://arxiv.org/pdf/2012.12901.
        The \\betas are multiplied componentwise, so we can initialize all of them to be O(1)
        to not spoil gauge-covariance because of numerical rounding.
        """
        d = self.n_dims
        torch.manual_seed(seed=seed)
        beta =  torch.rand(*(d, N_out)).to(self.dtype_U)
        return beta
    #---
    def exp_ibetaWah(self, W: LocallyGaugeCovariant, beta: torch.Tensor):
        """
        Eq. 9 of https://arxiv.org/pdf/2012.12901
        beta: parameters. shape: (d, N_ch)

        U: gauge configuration: (batch, L1, ..., Ld, d, Nc, Nc)
        W: N_ch locally transforming variables (obtained after L_conv()). shape: (batch, L1, ..., Ld, N_ch, Nc, Nc)
        beta: shape: (d,N_ch)
        """
        # building the anti-hermitian part of W --> i*W_ah lies in the algebra su(N)
        W_ah_traceless = ColorMatrix(W).get_ah_traceless()
        arg_exp = torch.einsum("mi,...iab->...mab", 1j*beta, W_ah_traceless) # NOTE: arg_exp has to be hermitean --> we include the ""1j"
        E = suN.get_exp_iA(arg_exp.to_tensor()) # eq. 9 of https://arxiv.org/pdf/2012.12901
        return LocallyGaugeCovariant(E)
    #---
    def L_exp(self, U: GaugeConfiguration, W: LocallyGaugeCovariant, beta: torch.Tensor):
        """ Eq. 8 of https://arxiv.org/pdf/2012.12901 """
        E = self.exp_ibetaWah(W=W, beta=beta) # eq. 9 of https://arxiv.org/pdf/2012.12901
        EU = E @ U # eq. 8 of https://arxiv.org/pdf/2012.12901
        return GaugeConfiguration(EU)
    #---
    def all_layers(self, U: GaugeConfiguration, omega: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor):
        """
          omega: shape=(N_out,N_in,d,nK)
          alpha: shape=(N_out,N_in1,N_in2)
          beta: shape=(d,N_ch)
        """
        U_PT = get_ParallelTransporters(U=U, K=self.K)
        W = self.get_W(U=U) # set of locally transforming variables 
        Wprime = self.get_Wprime(U=U, U_PT=U_PT, W=W) # W' as in eq. III.11 of https://arxiv.org/pdf/2012.12901
        W_conv = self.L_conv(Wprime=Wprime, omega=omega) # W after eq. 5 of https://arxiv.org/pdf/2012.12901
        # Wprime_conv = self.get_Wprime(W=W_conv, U_PT=U_PT) # updated W' after L-Conv
        ## in Eq. 6, W' is just W (k=0 in eq. III.11)
        W_bilin = self.L_Bilin(W=W_conv, Wprime=W_conv, alpha=alpha) #  W after eq. 6 of https://arxiv.org/pdf/2012.12901
        W_act = self.L_act(U=U, W=W_bilin, act_fun=self.act_fun) # W after eq. 7 of https://arxiv.org/pdf/2012.12901
        EU = self.L_exp(U=U, W=W_act, beta=beta)
        W_res = self.get_W(U=EU) 
        return W_res
    #---
    def all_layers_AND_Tr(self, U: GaugeConfiguration, omega: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor):
        return suN.get_Tr(self.all_layers(U=U, omega=omega,alpha=alpha,beta=beta))
    #---
    def all_layers_with_CB(self, U: GaugeConfiguration, omega_CB: torch.Tensor, beta: torch.Tensor):
        """
          omega: shape=(N_out,N_in,d,nK)
          omega_CB: shape: (N_out, N_in1, N_in2, d, nK) [NOTE: \\mu before k]
          beta: shape=(d,N_ch)
        """
        U_PT = get_ParallelTransporters(U=U, K=self.K)
        W = self.get_W(U=U) # set of locally transforming variables
        Wprime = self.get_Wprime(U=U, W=W, U_PT=U_PT) # W' as in eq. III.11 of https://arxiv.org/pdf/2012.12901
        W_CB = self.L_CB(W=W, Wprime=Wprime, omega_CB=omega_CB) # W after eq. 18 of https://arxiv.org/pdf/2401.06481
        W_act = self.L_act(U=U, W=W_CB, act_fun=self.act_fun) # W after eq. 7 of https://arxiv.org/pdf/2012.12901
        EU = self.L_exp(U=U, W=W_act, beta=beta)
        W_res = self.get_W(U=EU) 
        return W_res
    #---
    def all_layers_with_CB_AND_Tr(self, U: GaugeConfiguration, omega_CB: torch.Tensor, beta: torch.Tensor):
        trace = suN.get_Tr(self.all_layers_with_CB(U=U, omega_CB=omega_CB,beta=beta))
        return trace
    #---
#---
                


    
    
