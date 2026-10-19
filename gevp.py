## generalized eigenvalue problem

import numpy as np
import scipy
import matplotlib.pyplot as plt

def gevp(C: np.ndarray, t0 = 0):
    """
    Returns eigenvalues and eigenvectors of the GEVP (Generalized EigenValue Problem)

    Input:
      C: np.array of shape (T,N,N)
      t0: index of the time $t_0$
    Output:
      Lam, V:
        Lam = np.array of shape (T,N)
        V = np.array of shape (T,N,N)
    
    """
    assert C.shape[1] == C.shape[2]
    T_ext, N = C.shape[0], C.shape[1]
    Lam = np.array(np.zeros(shape=(T_ext,N))) # \lambda_i(t)=e^{-E_i * t}
    V = np.array(np.zeros(shape=(T_ext,N,N)))
    C0 = np.matrix(C[t0,:,:])
    for t in range(T_ext):
        Ct = np.matrix(C[t,:,:])
        lam, v = scipy.linalg.eigh(a=Ct, b=C0)
        # normalizing the vectors
        v_norm = v/np.sqrt(np.sum(v**2, axis=0))[np.newaxis,:]
        # ordering the eigenvalues (and the eigenvectors accordingly)
        idx_sort = np.argsort(lam)
        Lam[t,:] = lam[idx_sort]
        V[t, :, :] = v_norm[:, idx_sort]
    ####
    return Lam, V
####

