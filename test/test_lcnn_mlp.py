"""
Testing the implementation of the Lattice Convolutional Neural Network (L-CNN)

https://arxiv.org/pdf/2012.12901

"""

import numpy as np
import torch
import time
import sys
sys.path.append("../../")

# import lattice_data_tools.links.suN as suN
from lattice_data_tools.links.configuration import GaugeConfiguration
# from lattice_data_tools.links.loops import WilsonLoopsGenerator
# from lattice_data_tools.links.parallel_transport import get_ParallelTransporters, get_W_shifted
from lattice_data_tools.machine_learning.lcnn import LCNN
from lattice_data_tools.machine_learning.lcnn_mlp import LCNN_MLP

print("===============================")
print("L-CNN + MLP implementation test")
print("===============================")

device = torch.device("cpu")
B = 1
d = 2
L =10
Lmu = d*[L]
K = 6
Nc = 3
t1 = time.time()
Ng = Nc**2 - 1
seed = 20260511

torch.manual_seed(seed=seed)
theta = -torch.pi + (2*torch.pi)*torch.rand(B, *Lmu[0:d], d, Ng).to(device).type(torch.float64) # random angles in [-\\pi,\\pi]
U = GaugeConfiguration.from_theta(theta)
LCNN_layer = LCNN(U=U, K=K)

W = LCNN_layer.get_W()

N_in = W.shape[-3]
N_out = 25

N_hidden = 3
N_neurons = [20,40,20]

LCNN_MLP_obj = LCNN_MLP(
    LCNN_layer=LCNN_layer,
    LCNN_N_in=N_in, LCNN_N_out=N_out,
    N_hidden=N_hidden, N_neurons=N_neurons,
    seed=seed)
