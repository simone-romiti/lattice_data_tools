"""
Routines for elements of the su(N) Lie algebra using Generalized Gell-Mann matrices.

Commutation relations: [tau_a, tau_b]  = i f_{abc} tau_c
Number of generators of the Lie algebra: N^2-1.
Basis: Generalized Gell-Mann matrices
Normalization: Tr(tau_a tau_b) = 2 \\delta_{ab}

Special cases:
- U(1)  : only one generator: τ = 1/sqrt{2}
- SU(2) : \\tau_a = \\sigma_a/2  (half-Pauli matrices).
- SU(3) : \\tau_a = \\lambda_a/2  (half-Gell-Mann matrices).

Reference for Generalize Gell-Mann matrices: https://arxiv.org/pdf/0806.1174
"""

import torch
import numpy as np
from typing import List

def get_Ng(Nc: int) -> int:
    """
    Returns of su(Nc): Nc^2-1 for Nc >= 2, 1 for U(1).

    Nc: number of colors 
    """
    Ng = 1 if Nc == 1 else (Nc * Nc - 1)
    return Ng
#---

def get_Nc(Ng: int):
    """ Number of colors Nc given the number of generators Ng

    Ng = 1 for U(1)
    Ng = Nc^2 -1 for SU(Nc) (Nc > 1)
    """
    Nc = 1 if Ng==1 else int(np.sqrt(Ng+1))
    return Nc
#---


def get_Tr(W: torch.tensor):
    """
    Eq. 10 of https://arxiv.org/pdf/2012.12901

    Trace of a set of Nc \\times Nc matrices.

    W: tensor of Nc \\times Nc matrices. shape: (..., Nc, Nc)    
    """
    return torch.einsum("...ii", W)
#---

def get_ReTr(W: torch.tensor):
    """
    Real part of the trace of each W object.


    W: tensor of Nc \\times Nc matrices. shape: (..., Nc, Nc)
    """
    ReTr = get_Tr(W=W).real
    return ReTr
#---


def get_generalized_GellMann_matrices_suN(Nc: int, device: torch.device, dtype=torch.complex128) -> torch.Tensor:
    """
    Return the N^2-1 Generalized Gell-Mann matrices as (N,N) complex
    numpy arrays.

    Reference: Eqs. 3,4,5 of https://arxiv.org/pdf/0806.1174
    """
    Ng = get_Ng(Nc=Nc) # number of generators in the algebra
    matrices = []
    # non-diagonal matrices
    for j in range(0, Nc):
        for k in range(j+1, Nc):
            # Symmetric GGM: eq. 3 of https://arxiv.org/pdf/0806.1174
            lam_s = torch.zeros((Nc, Nc), dtype=dtype) 
            lam_s[j, k] = 1.0
            lam_s[k, j] = 1.0
            matrices.append(lam_s)

            # Antisymmetric GGM: eq 4 of https://arxiv.org/pdf/0806.1174
            lam_a = torch.zeros((Nc, Nc), dtype=dtype) 
            lam_a[j, k] = -1j
            lam_a[k, j] = +1j
            matrices.append(lam_a)
        #---
    #---
    for l in range(1, Nc):
        # diagonal matrices: eq. 5 of https://arxiv.org/pdf/0806.1174
        lam_l =  np.sqrt(2.0 / (l * (l + 1))) * torch.diag(torch.tensor(l*[1.0] + [-l] + (Nc-l-1)*[0.0])).type(dtype)
        matrices.append(lam_l)
    #---
    GMM = torch.stack(matrices, dim=0)
    assert(GMM.shape[0] == Ng)
    return GMM
#---

def get_generators(Nc: int, device: torch.device, dtype=torch.complex128):
    """
    Wrapper for the generators of U(1) and SU(Nc) matrices, properly normalized
    The case of U(1) is returned when Nc==1
    """
    if Nc == 1:
        # U(1): single 1×1 generator = 1/sqrt(2)   #documentation:generators_u1
        return torch.tensor([[1.0 / torch.sqrt(2)]], device=device, dtype=dtype)
    else:
        return get_generalized_GellMann_matrices_suN(Nc=Nc, device=device, dtype=dtype) / 2.0
#-------


def get_exp_iA(A: torch.Tensor) -> torch.Tensor:
    """
    Compute exp(A) for a Hermitian matrix A via eigendecomposition.

    1. Diagonalise A = M*D*M^{-1} = M*D*M^{\\dagger}   (A Hermitian --> D real, M unitary via eigh).
    2. Compute exp(i D) element-wise on the real diagonal eigenvalues.
    3. Reconstruct U = M · diag(exp(i D)) · M^{-1}.
    """
    d, M = torch.linalg.eigh(A)        # A = M diag(d) M^\\dagger, d real   #documentation:matrix_exp_diag_algorithm
    exp_iD = torch.diag_embed(torch.exp(1j*d)).type(M.type())  # exp(d_k) for each eigenvalue d_k
    U = M @ exp_iD @ M.adjoint()   # U = M exp(iD) M^\\dagger
    return U
#---


def get_suN_element_from_theta(theta: torch.Tensor, dtype=torch.complex128) -> torch.Tensor:
    """
    Build the su(Nc) algebra element  A = \\theta_a \\tau_a  (implicit sum over a)

    Parameters
    ----------
    theta  : coefficients \\theta_s. shape: (Ng,)
    N      : int. number of colors.
    dtype  : complex dtype for the output matrix.

    Returns
    -------
    A : (Nc,Nc) complex torch.Tensor
    """
    Ng = theta.shape[-1]
    Nc = get_Nc(Ng=Ng)
    tau = get_generators(Nc, device=theta.device, dtype=dtype)
    A = torch.einsum("...a,aij->...ij", theta.type(tau.type()), tau)
    return A
#---

def get_U_from_theta(theta: torch.Tensor, dtype=torch.complex128):
    """ Element of the group SU(N), computed as exp(i*theta_a*tau_a) """
    A = get_suN_element_from_theta(theta=theta,dtype=dtype)
    U = get_exp_iA(A)
    return U
#---

if __name__ == "__main__":
    Nc = 3 # number of colors
    Ng = get_Ng(Nc=Nc) # number of generators
    theta = torch.rand(Ng)
    U = get_U_from_theta(theta=theta)
    Udag = U.adjoint()
    print(torch.allclose(U @ Udag, torch.eye(Nc).type(U.type())))
