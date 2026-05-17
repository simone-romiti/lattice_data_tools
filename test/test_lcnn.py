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
from lattice_data_tools.links.loops import WilsonLoopsGenerator as WLG
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
theta = -torch.pi + (2*torch.pi)*torch.rand(B, *Lmu[0:d], d, Ng).to(device).to(torch.float64) # random angles in [-\\pi,\\pi]
U = GaugeConfiguration.from_theta(theta)
lcnn1 = LCNN(U=U, K=K)

V = U.gen_random_gauge_transformation(seed=seed)

W1 = lcnn1.get_W(U=U)
Nch_in = W1.shape[-3]
Nch_out = 17

N_out = 23
N_in1 = Nch_out
N_in2 = N_in1
omega = lcnn1.gen_random_omega(Nch_out=Nch_out, Nch_in=Nch_in, seed=seed)
alpha = lcnn1.gen_random_alpha(N_out=N_out, N_in1=N_in1, N_in2=N_in2, seed=seed)
beta  = lcnn1.gen_random_beta(N_out=N_out, seed=seed)

print("Checking that the layers of L-CNN produce a locally gauge-covariant object")

print("Check W(U)")
LocallyGaugeCovariant.check_gauge_covariance(U=U, V=V, fun=lambda U_conf: LCNN(U=U_conf,K=K).get_W(U=U_conf))

print("Check W_conv(U)")
def get_W_conv_local(U_conf):
    lcnn_loc = LCNN(U=U_conf,K=K)
    U_PT = get_ParallelTransporters(U=U_conf, K=K)
    W = lcnn_loc.get_W(U=U_conf) # set of locally transforming variables 
    Wprime = lcnn_loc.get_Wprime(U=U_conf, W=W, U_PT=U_PT) # W' as in eq. III.11 of https://arxiv.org/pdf/2012.12901
    W_conv = lcnn_loc.L_conv(Wprime=Wprime, omega=omega) 
    return W_conv

LocallyGaugeCovariant.check_gauge_covariance(U=U, V=V, fun=get_W_conv_local)

print("Check W_bilin(U)")
def get_W_bilin_local(U_conf):
    lcnn_loc = LCNN(U=U_conf,K=K)
    U_PT = get_ParallelTransporters(U=U_conf, K=K)
    W = lcnn_loc.get_W(U=U_conf) # set of locally transforming variables 
    Wprime = lcnn_loc.get_Wprime(U=U_conf, W=W, U_PT=U_PT) # W' as in eq. III.11 of https://arxiv.org/pdf/2012.12901
    W_conv = lcnn_loc.L_conv(Wprime=Wprime, omega=omega) 
    W_bilin = lcnn_loc.L_Bilin(W=W_conv, Wprime=W_conv, alpha=alpha)
    return W_bilin

LocallyGaugeCovariant.check_gauge_covariance(U=U, V=V, fun=get_W_bilin_local)

print("Check W_act")
def get_W_act_local(U_conf):
    lcnn_loc = LCNN(U=U_conf,K=K)
    U_PT = get_ParallelTransporters(U=U_conf, K=K)
    W = lcnn_loc.get_W(U=U_conf) # set of locally transforming variables 
    Wprime = lcnn_loc.get_Wprime(U=U_conf, W=W, U_PT=U_PT) # W' as in eq. III.11 of https://arxiv.org/pdf/2012.12901
    W_conv = lcnn_loc.L_conv(Wprime=Wprime, omega=omega) 
    W_bilin = lcnn_loc.L_Bilin(W=W_conv, Wprime=W_conv, alpha=alpha)
    W_act = lcnn_loc.L_act(U=U_conf, W=W_bilin) # default act_fun
    return W_act

LocallyGaugeCovariant.check_gauge_covariance(U=U, V=V, fun=get_W_act_local)


print("Check E")
def get_E_local(U_conf):
    lcnn_loc = LCNN(U=U_conf,K=K)
    U_PT = get_ParallelTransporters(U=U_conf, K=K)
    W = lcnn_loc.get_W(U=U_conf) # set of locally transforming variables 
    Wprime = lcnn_loc.get_Wprime(U=U_conf, W=W, U_PT=U_PT) # W' as in eq. III.11 of https://arxiv.org/pdf/2012.12901
    W_conv = lcnn_loc.L_conv(Wprime=Wprime, omega=omega) 
    W_bilin = lcnn_loc.L_Bilin(W=W_conv, Wprime=W_conv, alpha=alpha)
    W_act = lcnn_loc.L_act(U=U_conf,W=W_bilin) # default act_fun
    E = lcnn_loc.exp_ibetaWah(W=W_act, beta=beta)
    return E

LocallyGaugeCovariant.check_gauge_covariance(U=U, V=V, fun=get_E_local)


print("All layers")
LocallyGaugeCovariant.check_gauge_covariance(
    U=U, V=V,
    fun=lambda U_conf: LCNN(U=U_conf,K=K).all_layers(U=U_conf, omega=omega, alpha=alpha, beta=beta)
)
print("Done.")

print("All layers with L-CB")

N_in = Nch_in
omega_CB = lcnn1.gen_random_omega_CB(N_out=N_out, N_in=N_in, seed=seed)

LocallyGaugeCovariant.check_gauge_covariance(
    U=U, V=V,
    fun=lambda U_conf: LCNN(U=U_conf,K=K).all_layers_with_CB(U=U_conf, omega_CB=omega_CB, beta=beta)
)
print("Done.")

print(U.shape)
t2 = time.time()
print(f"t2-t1: {t2-t1} sec.")
Plaq = WLG.plaquettes(U=U)
t3 = time.time()
print(f"t3-t2: {t3-t2} sec.")
Poly = WLG.Polyakov_loops(U=U)
t4 = time.time()
print(f"t4-t3: {t4-t3} sec.")
t5 = time.time()
print(f"t5-t4: {t5-t4} sec.")
W = lcnn1.get_W(U=U)
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
W_conv = lcnn1.L_conv(Wprime=Wprime, omega=omega)
t10 = time.time()

