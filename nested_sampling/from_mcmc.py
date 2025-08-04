""" 
Routines for reading data from the MCMC runs with this library:
https://github.com/HISKP-LQCD/su2

"""

import pandas as pd
import numpy as np


class Plaquette:
    @staticmethod
    def read(path: str, nrows: int = None, skiprows : int = None):
        df = pd.read_csv(path, sep=" ", header=None, nrows=nrows, skiprows=skiprows, dtype=np.float64)
        P = df[1].to_numpy()
        return P
#---

class Polyakov:
    @staticmethod
    def read(path: str, nrows: int = None, skiprows : int = None):
        df = pd.read_csv(path, sep=" ", header=None, nrows=nrows, skiprows=skiprows, dtype=np.float64)
        Re_P = df[0].to_numpy() 
        Im_P = df[1].to_numpy()
        polyakov = Re_P + 1j*Im_P
        return polyakov
    #---
#---
