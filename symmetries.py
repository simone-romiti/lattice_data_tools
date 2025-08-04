

import numpy as np
from typing import Literal


def impose_Thalf_symmety(C: np.ndarray, p: Literal[1, -1]) -> np.ndarray:
    """Impose the symmetry of the correlator at T/2.

    Args:
        C: Input correlator array.
        T: Temporal extent.
        p: Parity, must be 1 or -1.
    """    
    T = C.shape[0]
    assert (T % 2 == 0), "T must be even"
    T_half = T // 2
    C_symm = np.zeros(T_half+1, dtype=C.dtype)
    C_symm[0] = C[0]  # t=0 is always the same
    C_symm[1:(T_half+1)] = (C[1:(T_half+1)] + p*C[::-1][0:T_half])/2
    return C_symm