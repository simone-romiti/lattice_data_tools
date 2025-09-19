import numpy as np

from lattice_data_tools import uwerr

class BootstrapSamples(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    @staticmethod
    def from_sample(x: np.ndarray, mean: np.ndarray):
        """ keeping the information about the unbiased mean """
        assert( x.shape[1:] == mean.shape)
        return BootstrapSamples(np.concatenate((np.atleast_1d(mean), x)))
    
    def __array_finalize__(self, obj):
        if obj is None: return

    def __array_function__(self, func, types, args, kwargs):
        """
        Block np.mean, np.std, np.average etc. 
        The user should call the methods of this class explicitly, to avoid mistakes 
        """
        if func in {np.mean, np.std, np.average}:
            raise TypeError(
                f"{func.__name__} is disabled for the BootstrapSamples class . "
                f"Use .mean() or .error() instead."
            )
        return super().__array_function__(func, types, args, kwargs)

    def mean(self, *args, **kwargs):
        """ bootstrap mean (biased) """
        return np.ndarray.mean(self[1:].view(np.ndarray), *args, **kwargs)

    def unbiased_mean(self):
        """ mean of the original random sample """
        return self[0]

    def bias(self):
        return self.unbiased_mean() - self.mean()

    def error(self):
        return np.std(self[1:].view(np.ndarray), axis=0, ddof=1)



def parametric_gaussian_bts(mean: float, std: float, N_bts, seed=12345):
    """Generate paramteric bootstrap samples from a Gaussian distribution
    
    Args:
        mean (float): Mean of the Gaussian distribution
        std (float): Standard deviation of the Gaussian distribution
        N_bts (int): Number of bootstrap samples to generate
        seed (int): Random seed for reproducibility
    
    Returns:
        np.ndarray: Bootstrap samples
    """
    np.random.seed(seed=seed)
    res = BootstrapSamples.from_sample(x=np.random.normal(loc=mean, scale=std, size=N_bts), mean=mean)
    return res
#---

def uncorrelated_confs_to_bts(x, N_bts, seed=12345):
    """Bootstrap samples from array of data
    
    generates Nb bootstrap samples from N configurations,
    It assumes x.shape[0] == N.
    
    Steps:
    - Draw N samples with replacements among the x values
    - Compute the mean. This is a bootstrap sample.
    - Repeat Nb times. The resulting array has standard deviation of the mean (approximately) equal to the one of the original sample.
    
    Args:
        x (np.ndarray): time series. Bootstrapping is done on 1st index
        N_bts (int): Number of bootstraps
        block_size (int): size of the block
    
    Returns:
        np.ndarray: Bootstrap samples
    """
    N = x.shape[0]
    np.random.seed(seed=seed)
    res = BootstrapSamples.from_sample(
        x=np.array([np.average(x[np.random.randint(0,high=N,size=N,dtype=int)]) for i in range(N_bts)]),
        mean = np.mean(x)
    )
    return res
####

def correlated_confs_to_bts(Cg: np.ndarray, N_bts: int, seed=12345, output_file=None) -> np.ndarray:
    """Bootstrap samples from array of correlated configurations

    - The configurations are sampled every tau_int, (integrated autocorrelation time)
    - The bootstrap samples are drawn from the uncorrelated configurations

    Args:
        C (np.ndarray): 1-dimensional "correlator" (observable computed for each configuration)
        N_bts (int): Number of bootstraps

    Returns:
        np.ndarray: Bootstrap samples
    """
    Ng = Cg.shape[0] ## total number of configurations
    tauint = int(uwerr.uwerr_primary(Cg, output_file=output_file)["tauint"]) ## integrated autocorrelation time
    if tauint == 0:
        tauint = 1 ## uncorrelated data
    ####
    Cg_uncorr = Cg[0:Ng:tauint] ## uncorrelated values
    return uncorrelated_confs_to_bts(x=Cg_uncorr, N_bts=N_bts, seed=seed)
####




