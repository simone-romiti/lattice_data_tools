

import torch
import sys
sys.path.append("../../")
import lattice_data_tools.links.suN as suN
from lattice_data_tools.links.configuration import GaugeConfiguration
from lattice_data_tools.links.loops import WilsonLoopsGenerator


if __name__ == "__main__":
    device = torch.device("cpu")
    B = 1
    d = 2
    Lmu = d * [4]
    Nc = 3
    Ng = suN.get_Ng(Nc=Nc)
    # random angles in [-\pi, \pi]
    theta = -torch.pi + (2 * torch.pi) * torch.rand(B, *Lmu, d, Ng).type(torch.float64)
    U = GaugeConfiguration.from_theta(theta)
    seed = 12345
    U.hotstart(seed=seed)
    Udag = U.adjoint()
    print("Type and shape of the gauge configuration")
    print("U:", type(U), U.shape)
    print("Udag:", type(Udag), U.shape)
    print("Unitarity check:", torch.allclose(U @ Udag, torch.eye(Nc).type(U.type())))
    print("B=", U.batch_size)
    print("Lattice shape: ", U.lattice_shape)
    print("d=", U.n_dims)
    print("n_links=", U.n_links)
    print("Nc=", U.Nc)
    # behaviour checks
    print("\n behaviour checks \n")
    print("U + U type:", type(U+U))
    print("U - U type:", type(U-U))
    print("U @ Udag type:", type(U @ Udag))
    try:
        _ = U * U
    except TypeError as e:
        print(f"U * U correctly raised TypeError: {e}")
    try:
        _ = U.dim()
    except AttributeError as e:
        print(f"U.dim() correctly raised AttributeError: {e}")
    #---
    print("Applying gauge transformations")
    WLG = WilsonLoopsGenerator(U=U)
    U.hotstart(seed=seed+1)
    Plaq1 = WLG.plaquettes()
    Poly1 = WLG.Polyakov_loops()
    U.random_gauge_transformation(seed=seed+2)
    Plaq2 = WLG.plaquettes()
    Poly2 = WLG.Polyakov_loops()
    Tr_Plaq1 = suN.get_Tr(Plaq1)
    Tr_Plaq2 = suN.get_Tr(Plaq2)
    b_Plaq = torch.allclose(Tr_Plaq1, Tr_Plaq2, atol=1e-15)
    print("Invariant trace:", b_Plaq and not (torch.allclose(Plaq1, Plaq2, atol=1e-15)))
    Tr_Poly1 = suN.get_Tr(Poly1)
    Tr_Poly2 = suN.get_Tr(Poly2)
    b_Poly = torch.allclose(Tr_Poly1, Tr_Poly2, atol=1e-15)
    print("Invariant Polyakov loop:", b_Plaq and not (torch.allclose(Poly1, Poly2, atol=1e-15)))
    
    
    
