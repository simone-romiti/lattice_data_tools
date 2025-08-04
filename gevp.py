## generalized eigenvalue problem

import numpy as np
import scipy
import matplotlib.pyplot as plt

def gevp(C: np.ndarray, t0 = 0):
    """ Returns eigenvalues and eigenvectors of the GEVP (Generalized EigenValue Problem) """
    assert C.shape[0] == C.shape[1]
    N, T_ext = C.shape[0], C.shape[2]
    Lam = np.array(np.zeros(shape=(N,T_ext))) # \lambda_i(t)=e^{-E_i * t}
    V = np.array(np.zeros(shape=(N,N,T_ext)))
    C0 = np.matrix(C[:,:,t0])
    for t in range(T_ext):
        Ct = np.matrix(C[:,:,t])
        lam, v = scipy.linalg.eigh(a=Ct, b=C0)
        # normalizing the vectors
        v_norm = v/np.sqrt(np.sum(v**2, axis=0))[np.newaxis,:]
        # ordering the eigenvalues (and the eigenvectors accordingly)
        idx_sort = np.argsort(lam)
        Lam[:, t] = lam[idx_sort]
        V[:, :, t] = v_norm[:, idx_sort]
    ####
    return Lam, V
####

