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

def get_Ng(N: int) -> int:
    """  Returns of su(N): N^2-1 for N >= 2, 1 for U(1).    """
    Ng = 1 if N == 1 else N * N - 1
    return Ng

def get_generalized_GellMann_matrices_suN(N: int, device: torch.device, dtype=torch.complex128) -> torch.Tensor:
    """
    Return the N^2-1 Generalized Gell-Mann matrices as (N,N) complex
    numpy arrays.

    Reference: Eqs. 3,4,5 of https://arxiv.org/pdf/0806.1174
    """
    Ng = get_Ng(N=N) # number of generators in the algebra
    matrices = []
    # non-diagonal matrices
    for j in range(0, N):
        for k in range(j+1, N):
            # Symmetric GGM: eq. 3 of https://arxiv.org/pdf/0806.1174
            lam_s = torch.zeros((N, N), dtype=dtype) 
            lam_s[j, k] = 1.0
            lam_s[k, j] = 1.0
            matrices.append(lam_s)

            # Antisymmetric GGM: eq 4 of https://arxiv.org/pdf/0806.1174
            lam_a = torch.zeros((N, N), dtype=dtype) 
            lam_a[j, k] = -1j
            lam_a[k, j] = +1j
            matrices.append(lam_a)
        #---
    #---
    for l in range(1, N):
        # diagonal matrices: eq. 5 of https://arxiv.org/pdf/0806.1174
        lam_l =  np.sqrt(2.0 / (l * (l + 1))) * torch.diag(torch.tensor(l*[1.0] + [-l] + (N-l-1)*[0.0])).type(dtype)
        matrices.append(lam_l)
    #---
    GMM = torch.stack(matrices, dim=0)
    assert(GMM.shape[0] == Ng)
    return GMM
#---

def get_generators(N: int, device: torch.device, dtype=torch.complex128) -> List[torch.Tensor]:
    """
    Wrapper for the generators of U(1) and SU(N) matrices, properly normalized
    The case of U(1) is returned when N==1
    """
    if N == 1:
        # U(1): single 1×1 generator = 1/sqrt(2)   #documentation:generators_u1
        return torch.tensor([[1.0 / torch.sqrt(2)]], device=device, dtype=dtype)
    else:
        return get_generalized_GellMann_matrices_suN(N=N, device=device, dtype=dtype) / 2.0
#-------


def get_exp_iA(A: torch.Tensor) -> torch.Tensor:
    """
    Compute exp(A) for a Hermitian matrix A via eigendecomposition.

    1. Diagonalise A = M*D*M^{-1} = M*D*M^{\\dagger}   (A Hermitian --> D real, M unitary via eigh).
    2. Compute exp(i D) element-wise on the real diagonal eigenvalues.
    3. Reconstruct U = M · diag(exp(i D)) · M^{-1}.
    """
    d, M = torch.linalg.eigh(A)        # A = M diag(d) M^\\dagger, d real   #documentation:matrix_exp_diag_algorithm
    exp_iD = torch.diag(torch.exp(1j*d)).type(M.type())  # exp(d_k) for each eigenvalue d_k
    U = M @ exp_iD @ M.adjoint()   # U = M exp(iD) M^\\dagger
    return U
#---


def get_suN_element_from_theta(theta: torch.Tensor, N: int, dtype=torch.complex128) -> torch.Tensor:
    """
    Build the su(N) algebra element  A = \\theta_a \\tau_a  (implicit sum over a)

    Parameters
    ----------
    theta  : coefficients \\theta_s. shape: (Ng,)
    N      : int. number of colors.
    dtype  : complex dtype for the output matrix.

    Returns
    -------
    A : (N,N) complex torch.Tensor
    """
    Ng = theta.shape[0]
    assert(Ng == get_Ng(N=N))
    tau = get_generators(N, device=theta.device, dtype=dtype)
    A = torch.einsum("a,aij->ij", theta.type(tau.type()), tau)
    return A
#---

def get_U_from_theta(theta: torch.Tensor, N: int, dtype=torch.complex128):
    """ Element of the group SU(N), computed as exp(i*theta_a*tau_a) """
    A = get_suN_element_from_theta(theta=theta,N=N,dtype=dtype)
    U = get_exp_iA(A)
    return U
#---

if __name__ == "__main__":
    N = 3 # number of colors
    Ng = get_Ng(N=N) # number of generators
    theta = torch.rand(Ng)
    U = get_U_from_theta(theta=theta, N=N)
    Udag = U.adjoint()
    print(torch.allclose(U @ Udag, torch.eye(N).type(U.type())))
