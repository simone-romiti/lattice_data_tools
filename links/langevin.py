""" Langevin dynamics for a configuration of gauge links """


import numpy as np
import torch
import typing
from lattice_data_tools.links.configuration import GaugeConfiguration
from lattice_data_tools.links.canonical_momenta import CanonicalMomenta


class LangevinDynamics:
    """ Class defining a Langevin evolution of gauge configurations of links U_\\mu(x)"""
    def __init__(self, U: GaugeConfiguration, log_p: typing.Callable) -> None:
        self.Ng = U.Ng # number of Lie algebra generators
        self.log_p = log_p # log of probability density function p(U)
        self.CM = CanonicalMomenta(U=U) # generator of canonical momenta 
        
    def La(self, f: typing.Callable, U: GaugeConfiguration):
        return self.CM.La_chain_rule(f=f, U=U)
        
    def evolve(self, U: GaugeConfiguration, eps: float, N: int, seed: int, omeas: typing.Callable = lambda U_i : None):
        """
        Evolve the gauge configurations for `N` time steps of size `eps`, 
        as in eq. 11 of https://arxiv.org/pdf/2601.19552
        with (beta/beta_0)*s_theta^a = L_a log(p(U))

        At each step `i`, I make online measurements according to the lambda function `omeas`
        """
        Oi = [] # observables
        Ui = U.clone().detach()
        n_idx = len(U.shape)
        for i in range(N):
            Oi_value = omeas(Ui.detach())
            print(f"i={i}, O={Oi_value}")
            Oi.append(Oi_value)
            # evolution step
            Ui.requires_grad_(True)
            Da = 1j * self.La(f=self.log_p, U=Ui).detach()
            perm = (0,*[i for i in range(2, n_idx-1)], 1)
            Da_perm = torch.permute(input=Da, dims=perm)
            drift = - eps * Da_perm
            torch.manual_seed(seed+i) #
            eta_a = torch.randn(*U.shape[0:-2], self.Ng)
            noise = +np.sqrt(2.0*eps)*eta_a
            theta = drift + noise
            exp_iW = GaugeConfiguration.from_theta(theta=theta)
            Ui = GaugeConfiguration(exp_iW @ Ui.detach())
        #---
        return Oi
        
