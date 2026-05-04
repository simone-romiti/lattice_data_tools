"""
Testing the implementation of the Lattice Convolutional Neural Network (L-CNN)

https://arxiv.org/pdf/2012.12901

"""

import torch
import time
import sys
sys.path.append("../../")

import lattice_data_tools.links.suN as suN
from lattice_data_tools.links.configuration import GaugeConfiguration, ColorMatrix
from lattice_data_tools.links.loops import WilsonLoopsGenerator
from lattice_data_tools.links.parallel_transport import get_ParallelTransporters, get_W_shifted
from lattice_data_tools.machine_learning.lcnn import LCNN

print("=========================")
print("L-CNN implementation test")
print("=========================")

device = torch.device("cpu")
B = 1
d = 3
Lmu = d*[8]
Nc = 3
t1 = time.time()
Ng = Nc**2 - 1
seed = 20260504

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
K = 5
lcnn1 = LCNN(U=U, K=K)
t5 = time.time()
print(f"t5-t4: {t5-t4} sec.")
W = ColorMatrix(lcnn1.get_W())
t6 = time.time()
print(f"t6-t5: {t6-t5} sec.")
N_in = W.shape[-3]
N_out = 100
# omega = torch.rand(N_out, N_in, d, 2*K+1) # convolution coefficients
omega = torch.rand(N_out, N_in, d, 2*K+1, dtype=U.dtype, device=U.device)
#omega = omega.type(U.type())
t7 = time.time()
print(f"t7-t6: {t7-t6} sec.")
U_PT = get_ParallelTransporters(U=U, K=K)
t8 = time.time()
print(f"t8-t7: {t8-t7} sec.")
Wprime = lcnn1.get_Wprime(U=U, U_PT=U_PT, W=W) # W_\\mu(x+k*\\mu)
t9 = time.time()
print(f"t9-t8: {t9-t8} sec.")
W_conv = lcnn1.L_conv(U=U, U_PT=U_PT, Wprime=Wprime, omega=omega, K=K)
t10 = time.time()

# print(f"t10-t9: {t10-t9} sec.")
# W_conv_einsum = lcnn1.L_conv_einsum(U=U, W=W, omega=omega, K=K)
# t11 = time.time()
# print(f"t11-t10: {t11-t10} sec.")
print(f"N_in={N_in}, N_out={N_out}, K={K}")
print(U.shape)
print(Plaq.shape)
print(Poly.shape)
print(U_PT.shape)
print(W.shape)
print(W_conv.shape)
#print("Checking the einsum implementation of the convolution")
# dW_conv = W_conv - W_conv_einsum
# print("Absolute values")
# dW_conv_abs = torch.abs(dW_conv)
# print("Hello", type(dW_conv_abs))
# print(torch.max(torch.abs(W_conv)))
# print(torch.max(dW_conv_abs))

# checking the gauge covariance
V = suN.gen_random_gauge_transformation(shape=W.shape, dtype=W.dtype, seed=seed)
W.gauge_transformation(V=V) # gauge-transformed W
U.gauge_transformation(V=V)
W_gt = ColorMatrix(LCNN(U=U, K=K).get_W()) # W computed on the gauge-transformed links
print(W - W_gt)
