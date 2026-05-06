"""
Testing the implementation of the Lattice Convolutional Neural Network (L-CNN)

https://arxiv.org/pdf/2012.12901

"""

import numpy as np
import torch
import time
import sys
sys.path.append("../../")

import lattice_data_tools.links.suN as suN
from lattice_data_tools.links.configuration import GaugeConfiguration, ColorMatrix, LocallyGaugeCovariant
from lattice_data_tools.links.loops import WilsonLoopsGenerator
from lattice_data_tools.links.parallel_transport import get_ParallelTransporters, get_W_shifted
from lattice_data_tools.machine_learning.lcnn import LCNN

print("=========================")
print("L-CNN implementation test")
print("=========================")

device = torch.device("cpu")
B = 1
d = 2
L =10
Lmu = d*[L]
K = 6
Nc = 3
t1 = time.time()
Ng = Nc**2 - 1
seed = 20260504

torch.manual_seed(seed=seed)
theta = -torch.pi + (2*torch.pi)*torch.rand(B, *Lmu[0:d], d, Ng).to(device).type(torch.float64) # random angles in [-\\pi,\\pi]
U = GaugeConfiguration.from_theta(theta)
WLG = WilsonLoopsGenerator(U=U)
lcnn1 = LCNN(U=U, K=K)

# W = lcnn1.get_W() # locally tranforming objects
V = U.gen_random_gauge_transformation(seed=seed)
# LocallyGaugeCovariant.check_gauge_covariance(U=U, V=V, fun=lambda U_conf: LCNN(U=U_conf,K=K).get_W())

# print("Testing parallel transporters")
# U_PT1 = get_ParallelTransporters(U=U, K=K)
# print("Backwards: ",torch.allclose(U_PT1[0, 0,0, 0, K-3, :,:], U[0, L-1,0, 0, :,:].adjoint() @ U[0, L-2,0, 0, :,:].adjoint() @ U[0, L-3,0, 0, :,:].adjoint(), atol=1e-15)) #@U[0, 2,0, 0, :,:])

# # checking that undoing the gauge transformation returns the original parallel transporters
# U.gauge_transformation(V=V) # gauge-transformed U
# U_PT2 = get_ParallelTransporters(U=U, K=K)
# U.gauge_transformation(V=V.adjoint()) # gauge-transformed U
# U_PT3 = get_ParallelTransporters(U=U, K=K)


# print("Gauge covariance of parallel transporters:", torch.allclose(U_PT1, U_PT3, atol=1e-15))



#print(V[0, 6,0, :,:] @ U_PT1[0, 0,0, 0, K-1, :,:] @ V[0, 6-1,0, :,:].adjoint() - U_PT2[0, 0,0, 0, K-1, :,:])



W1 = lcnn1.get_W()
#Wprime1 = lcnn1.get_Wprime(U_PT=U_PT1, W=W1)
Nch_in = W1.shape[-3]
Nch_out = 17

N_out = 23
N_in1 = Nch_out
N_in2 = N_in1
omega = lcnn1.gen_random_omega(Nch_out=Nch_out, Nch_in=Nch_in, seed=seed)
alpha = lcnn1.gen_random_alpha(N_out=N_out, N_in1=N_in1, N_in2=N_in2, seed=seed)
beta  = lcnn1.gen_random_beta(N_out=N_out, seed=seed)

print("Checking that the L-CNN produces a locally gauge-covariant object")
LocallyGaugeCovariant.check_gauge_covariance(
    U=U, V=V,
    fun=lambda U_conf: LCNN(U=U_conf,K=K).all_layers(omega=omega, alpha=alpha, beta=beta)
)
print("Done.")






# # U = get_U_from_theta(theta=theta, N=Nc)
# # U = torch.randn(B, *Lmu[0:d], d, Nc, Nc).to(device).type(torch.complex64)
# print(U.shape)
# t2 = time.time()
# print(f"t2-t1: {t2-t1} sec.")
# Plaq = WLG.plaquettes()
# t3 = time.time()
# print(f"t3-t2: {t3-t2} sec.")
# Poly = WLG.Polyakov_loops()
# t4 = time.time()
# print(f"t4-t3: {t4-t3} sec.")
# t5 = time.time()
# print(f"t5-t4: {t5-t4} sec.")
# W = lcnn1.get_W()
# t6 = time.time()
# print(f"t6-t5: {t6-t5} sec.")
# N_in = W.shape[-3]
# N_out = 100
# # omega = torch.rand(N_out, N_in, d, 2*K+1) # convolution coefficients
# omega = torch.rand(N_out, N_in, d, 2*K+1, dtype=U.dtype, device=U.device)
# #omega = omega.type(U.type())
# t7 = time.time()
# print(f"t7-t6: {t7-t6} sec.")
# U_PT = get_ParallelTransporters(U=U, K=K)
# t8 = time.time()
# print(f"t8-t7: {t8-t7} sec.")
# Wprime = lcnn1.get_Wprime(W=W) # W_\\mu(x+k*\\mu)
# t9 = time.time()
# print(f"t9-t8: {t9-t8} sec.")
# W_conv = lcnn1.L_conv(U=U, U_PT=U_PT, Wprime=Wprime, omega=omega, K=K)
# t10 = time.time()

# # print(f"t10-t9: {t10-t9} sec.")
# # W_conv_einsum = lcnn1.L_conv_einsum(U=U, W=W, omega=omega, K=K)
# # t11 = time.time()
# # print(f"t11-t10: {t11-t10} sec.")
# print(f"N_in={N_in}, N_out={N_out}, K={K}")
# print(U.shape)
# print(Plaq.shape)
# print(Poly.shape)
# print(U_PT.shape)
# print(W.shape)
# print(W_conv.shape)
# #print("Checking the einsum implementation of the convolution")
# # dW_conv = W_conv - W_conv_einsum
# # print("Absolute values")
# # dW_conv_abs = torch.abs(dW_conv)
# # print("Hello", type(dW_conv_abs))
# # print(torch.max(torch.abs(W_conv)))
# # print(torch.max(dW_conv_abs))


# # checking the gauge covariance
# W1 = lcnn1.get_W()

# N_ch = W1.shape[-3]
# N_in1, N_in2, N_out = 5, 7, 11

# omega = torch.rand(*(N_out,N_in,d,2*K+1))
# alpha = torch.rand(*(N_out,N_in1,N_in2))
# beta =  torch.rand(*(d,N_ch))


# W_shifted = get_W_shifted(U=U, W=W, K=1)
# U_PT = get_ParallelTransporters(U=U, K=1)
# W_prime = torch.einsum("... m k a b, ... i m k b c, ... m k c d -> ... i m k a d", U_PT, W_shifted, U_PT.adjoint())
# print(W_shifted.shape, W_prime.shape)
# print("Tr(W1) invariance:", torch.allclose(suN.get_Tr(W_shifted), suN.get_Tr(W_prime), atol=1e-15))
# print(torch.einsum("... m k a b, ... m k b d -> ... m k a d", U_PT, U_PT.adjoint()))



# EU1 = lcnn1.all_layers(omega=omega, alpha=alpha, beta=beta)
# # V_trivial = GaugeConfiguration.from_theta(0.0*theta)[...,0,:,:]

# #print(V_trivial)
# V = suN.gen_random_gauge_transformation(shape=W[...,0,:,:].shape, dtype=W.dtype, seed=seed)
# W1.gauge_transformation(V=V) # gauge-transformed W
# print(EU1.shape, "shape of EU1")
# EU1.gauge_transformation(V=V) # gauge-transformed EU1
# U.gauge_transformation(V=V)

# U_PT = get_ParallelTransporters(U=U, K=1)
# W_shifted = get_W_shifted(U=U, W=W1, K=1)



# W2 = lcnn1.get_W() # W computed on the gauge-transformed links
# EU2 = lcnn1.all_layers(omega=omega, alpha=alpha, beta=beta)
# print("Gauge covariance for W:", torch.allclose(W1, W2, atol=1e-15))
# print("Gauge covariance for all layers:", torch.allclose(EU1, EU2, atol=1e-15))


# # W' = W(U')
# # V(x) W(x) Vdag(x) = W[ V(x) U_\mu(x) Vdag(x+mu) ]
