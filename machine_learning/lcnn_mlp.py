"""
This file contains the implementation of a prototype of a gauge-invariant wavefunction \\psi(U) (pure-gauge SU(N) theory)
"""


import torch
import lattice_data_tools.links.suN as suN
from lattice_data_tools.machine_learning import LCNN, default_activation_function


class LCNN_MLP(torch.nn.Module):
    """
    Network combining L-CNN (https://arxiv.org/pdf/2012.12901)
    and a Multi-Layer Perceptron (MLP).

    The output is automatically gauge-invariant because of the L-CNN
    """
    def __init__(
            self,
            LCNN_layer: LCNN,
            N_hidden: int, N_neurons: int,
            seed: int,
            act_fun_MLP: typing.Callable = torch.nn.Tanh
    ):
        super().__init__()

        self.LCNN_layer = LCNN_layer
        self.beta  = LCNN_layer.gen_random_beta(N_out=N_out, seed=seed)
        self.omega_CB = LCNN_layer.gen_random_omega_CB(N_out=N_out, N_in=N_in, seed=seed)
        example_output = LCNN_layer.all_layers_with_CB_AND_Tr(omega_CB=self.omega_CB, beta=self.beta).flatten(start_dim=1)
        N_input_MLP = example_output.shape[1]

        self.MLP_layer = nn.Sequential(
            nn.Linear(N_input_MLP, N_neurons[0]),
            act_fun_MLP,
            *[nn.Sequential(nn.Linear(N_neurons[i], N_neurons[i]), act_fun_MLP) for _ in range(n_hidden - 2)],
            nn.Linear(N_neurons[-1], 2)
        )

    def forward(self, U):
        after_LCNN = self.LCNN_layer.all_layers_with_CB_AND_Tr(omega_CB=self.omega_CB, beta=self.beta).flatten(start_dim=1)
        after_MLP = self.MLP_layer(after_LCNN)
        return after_MLP
#-------
    
