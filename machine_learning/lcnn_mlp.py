"""
This file contains the implementation of a prototype of a gauge-invariant wavefunction \\psi(U) (pure-gauge SU(N) theory),
as a combination of a L-CNN and a MLP networks
"""


import typing
import torch

import lattice_data_tools.links.suN as suN
from lattice_data_tools.machine_learning.lcnn import LCNN
from lattice_data_tools.links.configuration import GaugeConfiguration


class LCNN_MLP(torch.nn.Module):
    """
    Network combining L-CNN (https://arxiv.org/pdf/2012.12901)
    and a Multi-Layer Perceptron (MLP).

    The output is automatically gauge-invariant because of the L-CNN
    """
    def __init__(
            self,
            U: GaugeConfiguration,
            LCNN_layer: LCNN,
            LCNN_N_in: int, LCNN_N_out: int,
            N_hidden: int, N_neurons: typing.List[int],
            seed: int,
            act_fun_MLP: typing.Callable = torch.nn.Tanh()
    ):
        super(LCNN_MLP, self).__init__()

        self.LCNN_layer = LCNN_layer
        self.beta  = LCNN_layer.gen_random_beta(N_out=LCNN_N_out, seed=seed)
        self.omega_CB = LCNN_layer.gen_random_omega_CB(N_out=LCNN_N_out, N_in=LCNN_N_in, seed=seed)
        dtype = self.omega_CB.dtype
        
        example_output = LCNN_layer.all_layers_with_CB_AND_Tr(U=U, omega_CB=self.omega_CB, beta=self.beta).flatten(start_dim=1)
        N_input_MLP = example_output.shape[1]

        self.MLP_layer = torch.nn.Sequential(
            torch.nn.Linear(N_input_MLP, N_neurons[0], dtype=dtype),
            act_fun_MLP,
            *[torch.nn.Sequential(torch.nn.Linear(N_neurons[i], N_neurons[i+1], dtype=dtype), act_fun_MLP) for i in range(N_hidden - 1)],
            torch.nn.Linear(N_neurons[-1], 1, dtype=dtype)
        )

    def forward(self, U: GaugeConfiguration):
        after_LCNN = self.LCNN_layer.all_layers_with_CB_AND_Tr(U=U, omega_CB=self.omega_CB, beta=self.beta).flatten(start_dim=1)
        after_MLP = self.MLP_layer(after_LCNN.to_tensor())
        return after_MLP
#-------
    
