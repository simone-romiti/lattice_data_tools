import numpy as np
import torch
import time

import sys
sys.path.append("../../")

import lattice_data_tools.links.suN as suN
from lattice_data_tools.links.configuration import GaugeConfiguration, ColorMatrix, LocallyGaugeCovariant
from lattice_data_tools.io import with_dill
from lattice_data_tools.links.loops import WilsonLoopsGenerator as WLG
# from lattice_data_tools.links.parallel_transport import get_ParallelTransporters, get_W_shifted
# from lattice_data_tools.machine_learning.lcnn import LCNN

print("=====================")
print("L-CNN comparison test")
print("=====================")

device = torch.device("cpu")
B = 2
d = 4
L = 8
Lmu = d*[L]
# K = 6
Nc = 3
Ng = Nc**2 - 1
seed = 20260504

torch.manual_seed(seed=seed)
torch.set_num_threads(1)

U = GaugeConfiguration.from_hotstart(batchsize=B, L_mu=Lmu, Nc=Nc, seed=seed,  dtype=torch.complex128, device=device, requires_grad=False)

# with_dill.dump(torch.Tensor(U), "./U_hotstart.pkl")

plaquettes= torch.Tensor(WLG.plaquettes(U=U))
avg_plaquette = torch.mean(plaquettes)

polyakov_loops = WLG.Polyakov_loops(U=U)


print(suN.get_Tr(polyakov_loops).shape)
avg_Polyakov_loops = torch.mean(torch.Tensor(suN.get_Tr(polyakov_loops)))

print("avg_plaquette:", avg_plaquette)
print("avg_Polyakov loop:", avg_Polyakov_loops)
