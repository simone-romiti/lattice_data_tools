""" Langevin dynamics for a configuration of gauge links """


import numpy as np
import torch
import typing
from lattice_data_tools.links.configuration import GaugeConfiguration
from lattice_data_tools.links.canonical_momenta import CanonicalMomenta


class LagevinDynamics:
    """ Class defining a Langevin evolution of gauge configurations of links U_\\mu(x)"""
    def __init__(self, U: GaugeConfiguration, copy: bool, log_p: typing.Callable) -> None:
        if copy:
            self.U = torch.clone(U) # copy of gauge configurations
        else:
            self.U = U # inplace evolution, U is overwritten at each step
        #---
        self.Ng = self.U.Ng # number of Lie algebra generators
        self.log_p = log_p # log of probability density function p(U)
        self.CM = CanonicalMomenta(U=U) # TODO define
    def evolve(self, eps: float, N: int, seed: int):
        """
        Evolve the gauge configurations for `N` time steps of size `eps`, 
        as in eq. 11 of https://arxiv.org/pdf/2601.19552
        with (beta/beta_0)*s_theta^a = L_a log(p(U))
        """
        for i in range(N):
            U_i = self.U # just a view of the configuration
            drift = - self.MomentaGenerator.La(f=self.log_p, U=U_i)
            torch.manual_seed(seed+i) #
            eta_a = torch.randn(*self.U.shape[0:-2], self.Ng)
            noise = + torch.sqrt(2*eps)*eta_a
            theta = drift + noise
            exp_iW = GaugeConfiguration.from_theta(theta=theta)
            self.U = exp_iW @ U_i
        #---
