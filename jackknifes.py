import numpy as np

from lattice_data_tools import uwerr

class JackknifeSamples(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return

    def __array_function__(self, func, types, args, kwargs):
        """
        Block np.mean, np.std, np.average etc.
        The user should call the methods of this class explicitly, to avoid mistakes
        """
        if func in {np.mean, np.std, np.average}:
            raise TypeError(
                f"{func.__name__} is disabled for the {self.__class__.__name__} class . "
                f"Use .mean() or .error() instead."
            )
        return super().__array_function__(func, types, args, kwargs)

    def mean(self):
        """ jackknife mean (unbiased by construction) """
        return np.mean(self.view(np.ndarray), axis=0)

    def error(self):
        N_jkf = self.shape[0]
        return np.sqrt(N_jkf-1)*np.std(self.view(np.ndarray), axis=0, ddof=0)

    @staticmethod
    def from_uncorrelated_confs(x: np.ndarray, N_jkf: int):
        Ng = x.shape[0] ## number of configurations
        b = int(Ng/N_jkf)
        arr = np.array([np.average(np.delete(x, range(i*b, (i+1)*b)), axis=0) for i in range(N_jkf)])
        res = JackknifeSamples(arr)
        return res

    @staticmethod
    def from_correlated_confs(x: np.ndarray, N_jkf: int, output_file=None):
        """
        Jackknife samples from array of correlated configurations
        
        - The configurations are sampled every tau_int, (integrated autocorrelation time)
        - The Jackknife samples are drawn from the uncorrelated configurations
        
        Args:
            x (np.ndarray): 1-dimensional "correlator" (observable computed for each configuration)
            N_jkf (int): Number of Jackknifes

        Returns:
            np.ndarray: Jackknife samples

        """
        Ng = x.shape[0] ## total number of configurations
        tauint = int(uwerr.uwerr_primary(x, output_file=output_file)["tauint"]) ## integrated autocorrelation time
        if tauint == 0:
            tauint = 1 ## uncorrelated data
        #---
        x_uncorr = x[0:Ng:tauint] ## uncorrelated values
        return JackknifeSamples.from_uncorrelated_confs(x=x_uncorr, N_jkf=N_jkf)
#---

