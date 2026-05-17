"""
Routines for elements of the su(N) Lie algebra using Generalized Gell-Mann matrices: Sec. III.A of https://arxiv.org/pdf/0806.1174 .

Commutation relations: [tau_a, tau_b]  = i f_{abc} tau_c
Number of generators of the Lie algebra: N^2-1.
Basis: Generalized Gell-Mann matrices
Normalization: Tr(tau_a tau_b) = \\delta_{ab} / 2

Special cases:
- U(1)  : only one generator: τ = 1/sqrt{2}
- SU(2) : \\tau_a = \\sigma_a/2  (half-Pauli matrices).
- SU(3) : \\tau_a = \\lambda_a/2  (half-Gell-Mann matrices).

"""

import torch
import numpy as np

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

def subtract_trace(W: torch.tensor):
    Nc = W.shape[-1]
    W_traceless = W - get_Tr(W).unsqueeze(-1) * torch.eye(Nc, dtype=W.dtype, device=W.device)
    return W_traceless


def get_ReTr(W: torch.tensor):
    """
    Real part of the trace of each W object.


    W: tensor of Nc \\times Nc matrices. shape: (..., Nc, Nc)
    """
    ReTr = get_Tr(W=W).real
    return ReTr
#---


def get_generalized_GellMann_matrices_suN(Nc: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
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

def get_generators(Nc: int, device: torch.device, dtype: torch.dtype):
    """
    Wrapper for the generators of U(1) and SU(Nc) matrices, properly normalized
    The case of U(1) is returned when Nc==1

    Note: Eq. A3 of https://arxiv.org/pdf/0806.1174 is normalized such that Tr(\\tau_a \\tau_b)= 2*\\delta_{ab}
          Here we normalize them such that: Tr(\\tau_a \\tau_b)= \\delta_{ab}/2
    """
    if Nc == 1:
        # U(1): single 1×1 generator = 1/sqrt(2)   #documentation:generators_u1
        return torch.tensor([1.0 / np.sqrt(2)], device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
    else:
        return get_generalized_GellMann_matrices_suN(Nc=Nc, device=device, dtype=dtype) / 2.0 # our normalization
#-------


def get_exp_iA(A: torch.Tensor) -> torch.Tensor:
    """
    Compute exp(i*A) for a Hermitian matrix A via eigendecomposition.

    1. Diagonalise A = M*D*M^{-1} = M*D*M^{\\dagger}   (A Hermitian --> D real, M unitary via eigh).
    2. Compute exp(i D) element-wise on the real diagonal eigenvalues.
    3. Reconstruct U = M · diag(exp(i D)) · M^{-1}.
    """
    # A_shape = A.shape
    # A_flat = torch.flatten(A, start_dim=0, end_dim=-3)
    # exp_iA = torch.linalg.matrix_exp(1j*A_flat)
    # return torch.reshape(exp_iA, A_shape)
    d, M = torch.linalg.eigh(A)        # A = M diag(d) M^\\dagger, d real
    exp_iD = torch.diag_embed(torch.exp(1j*d)).type(M.type())  # exp(d_k) for each eigenvalue d_k
    U = M @ exp_iD @ M.adjoint()   # U = M exp(iD) M^\\dagger
    return U
#---


def get_suN_element_from_theta(theta: torch.Tensor) -> torch.Tensor:
    """
    Build the su(Nc) **algebra** element  A = \\theta_a \\tau_a  (implicit sum over a)

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
    theta_complex = theta + 1j*torch.zeros_like(theta) # casting to complex to determine the type of the tau_a and combine them together
    tau = get_generators(Nc, device=theta_complex.device, dtype=theta_complex.dtype).to(theta_complex.device)
    A = torch.einsum("...a,aij->...ij", theta_complex, tau)
    return A
#---

def get_U_from_theta(theta: torch.Tensor):
    """
    Element of the **group** SU(N), computed as U(\\theta_a)=exp(i*theta_a*tau_a)

    Input:
      - theta: array of angles. shape=(...,Ng)

    Returns:
      - `U`: array of links. shape=(...,Nc,Nc)

    !!! NOTES !!!:
      1. The inverse function does not exist, as the map $U(\\theta_a)$ is not injective.
      2. For an arbitrary Nc, the different angles have in general different periodicities, so you can't globally restrict to a fixed interval for all of them
    """
    A = get_suN_element_from_theta(theta=theta) # \\theta_a \\tau_a
    U = get_exp_iA(A) # e^{i*A}
    return U
#---

def get_coldstart(shape: torch.Size, dtype: torch.dtype, device: torch.device, requires_grad: bool) -> torch.Tensor:
    """
    Multi-dimensional array of $\mathrm{SU}(N)$ matrices,
    all equal to $\\mathbb{1}_{N_c \\times N_c}$
    """
    Nc = shape[-1] # number of colors
    U = torch.eye(Nc, dtype=dtype, device=device, requires_grad=requires_grad).expand(*shape)
    return U

def apply_coldstart(U: torch.Tensor) -> None:
    """
    Initialize gauge links $U_\\mu(x)$ to identity matrices of size Nc \\times Nc
    """
    U.copy_(get_coldstart(shape=U.shape, dtype=U.dtype, device=U.device, requires_grad=U.requires_grad))
    return None
#---

def get_hotstart(shape: torch.Size, seed: int, dtype: torch.dtype, device: torch.device, requires_grad: bool) -> torch.Tensor:
    """
    Gauge links initialized to random $SU(N_c)$ elements,
    following the recipe of page 11 of https://arxiv.org/pdf/math-ph/0609050

    Each link matrix is constructed by:
      1. Drawing an Nc×Nc random complex matrix.
      2. Applying row-wise Gram-Schmidt orthonormalization -> U(Nc) matrix.
      3. Dividing the first row by the Nc-th root of det(U) -> SU(Nc) matrix.

    Returns: configuration of links. shape: (..., Nc, Nc).
             The first components of the shape are deduced from the shape of U.
    """
    torch.manual_seed(seed) # seeting the RNG seed for reproducibility
    # generating a random complex matrix
    imag_part = torch.randn(shape, dtype=dtype, device=device, requires_grad=requires_grad) # real part
    real_part = torch.randn(shape, dtype=dtype, device=device, requires_grad=requires_grad) # imaginary part
    Z = (real_part + 1j*imag_part)/np.sqrt(2.0)
    Q, R = torch.linalg.qr(Z) # QR decomposition: https://en.wikipedia.org/wiki/QR_decomposition
    diag = torch.diagonal(R, dim1=-2, dim2=-1) # (..., Nc) --> extracts R_ii
    signs = diag / diag.abs() # R_ii / |R_ii|
    Lam = torch.diag_embed(signs) # shape: (..., Nc, Nc). Eq. 5.12 of https://arxiv.org/pdf/math-ph/0609050
    Qprime = Q @ Lam
    detQprime = torch.linalg.det(Qprime).unsqueeze(-1).unsqueeze(-1) # det(Q). reshaping in order to combine with Q later
    U = (Q / detQprime) # Q is just unitary, we need to impose det(U)==1
    return U
#---

def apply_hotstart(U: torch.Tensor, seed: int):
    """
    Initialize gauge links $U_\\mu(x)$ to random $SU(Nc)$ elements
    """
    U.copy_(get_hotstart(shape=U.shape, seed=seed, dtype=U.dtype, device=U.device, requires_grad=U.requires_grad))
    return None
#---

def gen_random_gauge_transformation(shape: torch.Size, dtype: torch.dtype, seed: int):
    """
    Returns a tensor of links V that generate a gauge transformation,
    using the provided shape.

    U_\\mu(x) --> V(x) U_\\mu(x) V^{\\dagger}(x+\\mu) , \\forall x,\\mu
    """
    V = torch.zeros(size=shape, dtype=dtype)
    apply_hotstart(U=V, seed=seed)
    return V
#---

if __name__ == "__main__":
    Nc = 3 # number of colors
    Ng = get_Ng(Nc=Nc) # number of generators
    theta = -torch.pi + (2 * torch.pi) * torch.rand(Ng).type(torch.float64)
    U = get_U_from_theta(theta=theta)
    Udag = U.adjoint()
    print("Unitarity of U:", torch.allclose(U @ Udag, torch.eye(Nc).type(U.type())))
