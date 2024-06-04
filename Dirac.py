
import numpy as np

## Pauli matrices
sigma = {
    1: np.matrix([[0,1], [1,0]], dtype=complex),
    2: np.matrix([[0, -1j], [1j, 0]], dtype=complex),
    3: np.matrix([[1, 0], [0,-1]], dtype=complex)
}

## gamma matrices according to the PLEGMA convention. https://www.jlab.org/sites/default/files/theory/files/JLab_Bacchio.pdf
gamma_PLEGMA = {
    1: -np.matrix(np.kron(sigma[2], sigma[1])),
    2: -np.matrix(np.kron(sigma[1], sigma[1])),
    3: -np.matrix(np.kron(sigma[2], sigma[3])),
    4: +np.matrix(np.kron(sigma[3], np.eye(2))),
    5: -np.matrix(np.kron(sigma[1], np.eye(2)))
}