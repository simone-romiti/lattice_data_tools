## generalized eigenvalue problem

import numpy as np
import scipy
import typing
import itertools
# import matplotlib.pyplot as plt



sorting_options = ["lambda", "dot_product", "det", "det_simple"]

def get_sorted_idx(lam, v, v_sort, sorting):
    """
    Sorting the GEVP eigenvalues and eigenvectors using the results at some reference time where the eigenvectors are v_sort
    See Sec. III.A of https://arxiv.org/pdf/2004.10472 and Sec. 3 of https://link.springer.com/article/10.1140/epja/s10050-020-00205-w (same paper. The published version has more details)

    lam: numpy.array of shape (N,). The values are the GEVP eigenvalues lambda_i(t)
    v: numpy.array of shape (N,N). The columns are the eigenvectors v_i(t)
    v_sort: numpy.array of shape (N,N). The columns are the eigenvectors v_i(t0)

    """
    N = lam.shape[0]
    # checking the shapes of the input
    assert(len(lam.shape) == 1)
    assert(v.shape == (N,N) and v_sort.shape == (N,N))
    # finding the orderding based on the criterion
    if sorting == "lambda":
        return np.argsort(lam)  # ordering the eigenvalues (reference-time independent)
    #---
    elif sorting == "dot_product":
        c_kl = np.dot(v_sort.T, v) # matrix of v_k(t0) * v_l(t)
        return np.argmax(np.abs(c_kl), axis=1)
    #---
    elif sorting == "det":
        eps_list = list(itertools.permutations(range(N))) # all permutations of the indices from 0 to (N-1)
        c_eps_list = [] # cf. Eq. 31 of https://link.springer.com/article/10.1140/epja/s10050-020-00205-w
        for eps in eps_list:
            c_eps = 1
            V_eps = v_sort[:,eps]
            for k in range(N):
                V_eps[:,k] = v[:,k]
                c_eps *= np.linalg.det(V_eps)
                V_eps[:,k] = v_sort[:,k]
            #---
            c_eps_list.append(c_eps)
        #---
        return np.argsort(c_eps_list)
    #---
    elif sorting == "det_simple":
        eps_list = list(itertools.permutations(range(N))) # all permutations of the indices from 0 to (N-1)
        c_eps_list = [np.abs(np.diagonal(np.dot(v_sort.T, v[:,eps])).prod(axis=0)) for eps in eps_list] # cf. Eq. 32 of https://link.springer.com/article/10.1140/epja/s10050-020-00205-w
        idx_eps_max = np.argmax(c_eps_list)
        return np.array(eps_list[idx_eps_max]) # returning the permutation that gives the maximal c_\\epsilon
    #---
    else:
        raise ValueError(f"Illegal GEVP sorting method: {sorting}")
    #---
#---

def gevp(C: np.ndarray, t0 = 1, sorting = typing.Literal[*sorting_options], t_sort=2):
    """
    Returns eigenvalues and eigenvectors of the GEVP (Generalized EigenValue Problem): eq. 2.2 of https://arxiv.org/pdf/0902.1265

    Input:
      C: numpy.array of shape (T,N,N)
      t0: index of the time $t_0$
    Output:
      Lam, V:
        Lam = numpy.array of shape (T,N)
        V = numpy.array of shape (T,N,N)
    
    """
    assert C.shape[1] == C.shape[2]
    T_ext, N = C.shape[0], C.shape[1]
    Lam = np.array(np.zeros(shape=(T_ext,N))) # \lambda_i(t)=e^{-E_i * t}
    V = np.array(np.zeros(shape=(T_ext,N,N)))
    C0 = np.matrix(C[t0,:,:])
    # GEVP at the sorting time
    lam_sort, v_sort = scipy.linalg.eigh(a=np.matrix(C[t_sort,:,:]), b=C0)
    # GEVP for all the other times
    for t in range(T_ext):
        Ct = np.matrix(C[t,:,:])
        lam, v = scipy.linalg.eigh(a=Ct, b=C0)
        # normalizing the vectors
        idx_sort = get_sorted_idx(lam=lam, v=v, v_sort=v_sort, sorting=sorting)
        Lam[t,:] = lam[idx_sort]
        v_sorted = v[:, idx_sort]
        v_norm = v_sorted/np.sqrt(np.sum(v_sorted**2, axis=0))[np.newaxis,:]
        V[t, :, :] = v_norm
    ####
    return Lam, V
####

