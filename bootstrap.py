
import math
import numpy as np
import typing
from joblib import Parallel, delayed


from lattice_data_tools import uwerr

class BootstrapSamples(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __getitem__(self, key):
        out = super().__getitem__(key)

        # Normalize key to a tuple
        if not isinstance(key, tuple):
            key = (key,)

        # If the *first axis* is being indexed in a way that reduces/slices it
        if isinstance(key[0], (int, slice, list, np.ndarray)):
            return np.asarray(out)

        return out
        
    def __array_finalize__(self, obj):
        if obj is None: return

    def N_bts(self):
        return self.shape[0]-1

    def inner_shape(self):
        return self.shape[1:]

    @staticmethod
    def from_sample(x: np.ndarray, mean: np.ndarray):
        """ keeping the information about the unbiased mean

        .. ldt-id:: BOOT-BootstrapSamples-from_sample
        """
        mean_arr = np.atleast_1d(mean)
        if mean_arr.shape == (1,):
            assert( len(x.shape[1:]) == 0)
        else:
            assert( x.shape[1:] == mean.shape)
        #---
        return BootstrapSamples(np.concatenate((mean_arr, x)))

    @staticmethod
    def zeros(N_bts: int, shape: tuple = ()):
        """
        Like numpy.zeros, but with an implicit leading dimension of N_bts+1:
        N_bts samples + the mean on the 0th axis

        .. ldt-id:: BOOT-BootstrapSamples-zeros
        """
        # Ensure shape is a tuple
        if not isinstance(shape, tuple):
            shape = (shape,)
        full_shape = (N_bts + 1,) + shape
        return BootstrapSamples(np.zeros(shape=full_shape))

    @staticmethod
    def bts_list_from_lambda(N_bts: int, fun: typing.Callable[[int], typing.Any], parallel: bool = False):
        """
        loop over the N_bts+1 values: N_bts + mean.
        Advantage: one does not need to manually remember to do a loop over N_bts+1
        This function is needed for those type of objects that are not necessarily numpy arrays, e.g. dict.

        .. ldt-id:: BOOT-BootstrapSamples-bts_list_from_lambda
        """
        if parallel:
            res = Parallel(n_jobs=-1)(delayed(fun)(i) for i in  range(N_bts+1))
        else:
            res = [fun(i) for i in range(N_bts+1)]
        #---
        return res            

    @staticmethod
    def from_lambda(N_bts: int, fun: typing.Callable[[int], typing.Any], parallel: bool = False):
        """ NOTE: only for numeric numpy array objects

        .. ldt-id:: BOOT-BootstrapSamples-from_lambda
        """
        return BootstrapSamples(BootstrapSamples.bts_list_from_lambda(N_bts=N_bts, fun=fun, parallel=parallel))

    @staticmethod
    def run_lambda(N_bts: int, fun: typing.Callable[[int], None]):
        # Note 1: the loop is over the N_bts+1 values: N_bts + mean 
        # Note 2: 
        #   this function does not offer parallelization as for the method `from_lambda()` because 
        #   the user may accidentally modify an external variable, but with joblib it does not work
        for i in range(N_bts+1):
            fun(i)
        #---
        return None

    def __array_function__(self, func, types, args, kwargs):
        """
        Block np.mean, np.std, np.average etc.
        The user should call the methods of this class explicitly, to avoid mistakes

        .. ldt-id:: BOOT-BootstrapSamples-__array_function__
        """
        if func in {np.mean, np.std, np.average}:
            raise TypeError(
                f"{func.__name__} is disabled for the BootstrapSamples class . "
                f"Use .mean() or .error() instead."
            )
        return super().__array_function__(func, types, args, kwargs)

    def to_numpy(self):
        return self.view(np.ndarray) # casting back to mother class

    def mean(self):
        """ bootstrap mean (biased)

        .. ldt-id:: BOOT-BootstrapSamples-mean
        """
        return np.ndarray.mean(self[1:].view(np.ndarray), axis=0)

    def unbiased_mean(self):
        """ mean of the original random sample

        .. ldt-id:: BOOT-BootstrapSamples-unbiased_mean
        """
        return self[0]

    def bias(self):
        """
        The bootstrap mean and the true mean differ by a bias.
        This function can be used a posteriori to check if the bias is comparable with the statistical precision coming from the estimate of the error.

        .. ldt-id:: BOOT-BootstrapSamples-bias
        """
        return self.unbiased_mean() - self.mean()

    def error(self):
        """
        Estimator of the standard error on the mean, computed only on the bootstrap samples,
        i.e. for the rows 1,...,N_bts (the 0-th one is the mean over the original dataset)

        .. ldt-id:: BOOT-BootstrapSamples-error
        """
        return np.std(self[1:].view(np.ndarray), axis=0, ddof=1)

    def covariance_matrix(self):
        """ Bootstrap samples of covariance matrix estimate.

        .. ldt-id:: BOOT-BootstrapSamples-covariance_matrix
        """
        assert(len(self.inner_shape()) == 1) # inner shape must be (N_var,)[number of variables]
        mu = self.unbiased_mean()
        dx = BootstrapSamples(self-mu[np.newaxis,:])
        res = np.cov(dx.transpose())
        return res

    def correlation_matrix(self):
        """ Bootstrap samples of correlation matrix estimate.

        .. ldt-id:: BOOT-BootstrapSamples-correlation_matrix
        """
        assert(len(self.inner_shape()) == 1) # inner shape must be (N_var,)[number of variables]
        mu = self.unbiased_mean()
        dx = BootstrapSamples(self-mu[np.newaxis,:])
        res = np.corrcoef(dx.transpose())
        return res
    
    def covariance_matrix_per_bts(self):
        """ Bootstrap samples of correlation matrix estimate. """
        assert(len(self.inner_shape()) == 1) # inner shape must be (N_var,)[number of variables]
        N_var = self.inner_shape()[0]
        mu = self.unbiased_mean()
        dx = BootstrapSamples(self-mu[np.newaxis,:])
        res = self.from_lambda(N_bts=self.N_bts(), fun=lambda i: [[dx[i,k1]*dx[i,k2] for k2 in range(N_var)] for k1 in range(N_var)])
        return res

    def correlation_matrix_per_bts(self):
        """ Bootstrap samples of correlation matrix estimate. """
        Cov = self.covariance_matrix_per_bts()
        N_var = self.inner_shape()[0]
        C = self.from_lambda(N_bts=self.N_bts(), fun=lambda i: [[Cov[i,k1,k2]/np.sqrt(Cov[i,k1,k1]*Cov[i,k2,k2]) for k2 in range(N_var)] for k1 in range(N_var)])
        return C
    

class ParametricBootstraps:
    @staticmethod
    def Gaussian(mean: float, error: float, N_bts: int, seed: int):
        """Generate parametric bootstrap samples from a Gaussian distribution
        
        Args:
        mean (float): Mean of the Gaussian distribution
        std (float): Standard deviation of the Gaussian distribution
        N_bts (int): Number of bootstrap samples to generate
        seed (int): Random seed for reproducibility
        
        Returns:
        np.ndarray: Bootstrap samples
        """
        np.random.seed(seed=seed)
        res = BootstrapSamples.from_sample(
            x=np.random.normal(loc=mean, scale=error, size=N_bts),
            mean=np.array(mean)
        )
        return res
    #---
    @staticmethod
    def from_x_dx_Cov(x: np.ndarray, dx: np.ndarray, Cov: np.ndarray, N_bts: int, seed: int, with_Cholesky=False):
        """
        Bootstrap samples for the variables x_i,
        from their central values and their covariance matrix Cov.
        """
        # checking that the input arrays have the right sizes
        assert(len(x.shape)==1)
        assert(x.shape == dx.shape)
        assert(x.shape[0]==Cov.shape[0])
        assert(Cov.shape[0]==Cov.shape[1])
        N = x.shape[0] # number of variables
        if with_Cholesky:
            """ using Cholesky decomposition """
            L = np.linalg.cholesky(Cov) # L such that Cov = L * L^T
            y_bts = np.array([ParametricBootstraps.Gaussian(mean=0.0, error=1.0, N_bts=N_bts, seed=seed+i) for i in range(N)]).T # parametric bootstraps for the uncorrelated variables --> different RNG seeds
            x_bts = BootstrapSamples(np.array([x + L @ y_bts[i,:]  for i in range(N_bts+1)])) # bootstrap samples for the correlated data
            return x_bts
        else:
            # x = M y, where "y" are the uncorrelated variables (diagonal correlation matrix)
            ey2, M = np.linalg.eigh(Cov) # var(Y_i) and M_ij such that Cov=M*D*(M^T), with D=diag(var(Y_i))
            assert(np.all(ey2 >= 0)) #  check if Cov_ij is semi-positive definite
            ey = np.sqrt(ey2) # errors as square roots of variances
            y_mean = (M.T) @ x # Cov=Cov^T --> M^{-1}=M^{T}
            y_bts = np.array([ParametricBootstraps.Gaussian(mean=y_mean[i], error=ey[i], N_bts=N_bts, seed=seed+i) for i in range(N)]).T # parametric bootstraps for the uncorrelated variables --> different RNG seeds
            x_bts = BootstrapSamples(np.array([M @ y_bts[i,:]  for i in range(N_bts+1)])) # bootstrap samples for the correlated data
            return x_bts
    #-------
    @staticmethod
    def from_x_dx_rho(x: np.ndarray, dx: np.ndarray, rho: np.ndarray, N_bts: int, seed: int):
        """ correlated bootstrap using means, errors and correlation matrix \\rho """
        N = x.shape[0] # number of variables
        # using Cholesky decomposition
        rhorhoT = (rho @ (rho.T)).round(decimals=15) # positive definite by construction, numerically safe
        ey4, M = np.linalg.eigh(rhorhoT)
        D = np.diag(np.sqrt(ey4))
        rho_rooted = M @ D @ M.T
        L = np.linalg.cholesky(rho_rooted) # L such that Cov = L * L^T
        y_bts = np.array([ParametricBootstraps.Gaussian(mean=0.0, error=1.0, N_bts=N_bts, seed=seed+i) for i in range(N)]).T # parametric bootstraps for the uncorrelated variables --> different RNG seeds
        x_bts = BootstrapSamples(np.array([x + dx * (L @ y_bts[i,:])  for i in range(N_bts+1)])) # bootstrap samples for the correlated data
        return x_bts
#-------        

    
def parametric_gaussian_bts(mean: float, error: float, N_bts, seed=12345):
    """
    Generate parametric bootstrap samples from a Gaussian distribution
    
    Args:
        mean (float): Mean of the Gaussian distribution
        std (float): Standard deviation of the Gaussian distribution
        N_bts (int): Number of bootstrap samples to generate
        seed (int): Random seed for reproducibility

    Returns:
        np.ndarray: Bootstrap samples

    .. ldt-id:: BOOT-parametric_gaussian_bts
    """
    np.random.seed(seed=seed)
    res = BootstrapSamples.from_sample(
        x=np.random.normal(loc=mean, scale=error, size=N_bts),
        mean=np.array(mean)
    )
    return res
#---

def binning(Cg: np.ndarray, bin_size: int):
    Ng = Cg.shape[0]
    if bin_size == 1:
        return np.copy(Cg) # no averaging
    else:
        """
        Applying the binning, i.e. average the values in each bin an returning the result
        
        NOTE: 
        the average of the output is identical to the one of the input 
        only if the bin size is a divisor of Ng
        """
        N_bins = Ng//bin_size # + (1 - int(Ng%bin_size == 0))
        i_next = lambda i: bin_size*(i+1) # min(Ng, bin_size*(i+1))
        return np.array([np.mean(Cg[(bin_size*i):i_next(i)]) for i in range(N_bins)]) # uncorrelated values
#-------

def auto_binning(Cg: np.ndarray, lambda_output_file=lambda i: None):
    """
    Automatic binning of a Monte Carlo (MC) chain according to the autocorrelation time.
    Until it decreases below 0.5 (within its uncertainty), we bin with increasing bin sizes:
    1 (original MC chain), tau, 2*tau, etc.

    The function returns the binned correlator with the optimal bin size as above.

        Cg (np.ndarray): MC chain (e.g. 1-dimensional "correlator" at one spacetime point, computed for each configuration)
        output_files_pattern (str): lambda function that takes 1 argument that is the index of the binning iteration

    .. ldt-id:: BOOT-auto_binning
    """
    i = 0
    bin_size = 1
    tauint_i, dtauint_i = 1, 0
    Cg_binned = np.copy(Cg)
    while tauint_i-dtauint_i > 0.5:
        """ 
        Until \\tau_int is not below 0.5 within the uncertainty, 
        we bin with binsizes multiple of \\tau_0
        """
        Cg_binned = binning(Cg=Cg, bin_size=bin_size)
        res_tauint_i = uwerr.uwerr_primary(Cg_binned, output_file=lambda_output_file(i)) # integrated autocorrelation time
        bin_size = math.ceil((i+1) * tauint_i) # updating bin size
        tauint_i = res_tauint_i["tauint"]
        dtauint_i = res_tauint_i["dtauint"]
        i += 1
    #---
    return Cg_binned
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

    .. ldt-id:: BOOT-uncorrelated_confs_to_bts
    """
    N = x.shape[0]
    np.random.seed(seed=seed)
    res = BootstrapSamples.from_sample(
        x=np.array([np.average(x[np.random.randint(0,high=N,size=N,dtype=int)]) for i in range(N_bts)]),
        mean = np.mean(x)
    )
    return res
#---

def correlated_confs_to_bts(Cg: np.ndarray, N_bts: int, seed=12345, bin_size = None, lambda_output_file=lambda i: None) -> np.ndarray:
    """Bootstrap samples from array of correlated configurations

    - The configurations are binned (block-averaged) every bin_size (determined by tau_int: integrated autocorrelation time)
    - The bootstrap samples are drawn from the uncorrelated configurations

    Args:
        C (np.ndarray): 1-dimensional "correlator" (observable computed for each configuration)
        N_bts (int): Number of bootstraps
        seed (int): seed for RNG (Random Number Generator)

    Returns:
        np.ndarray: Bootstrap samples

    .. ldt-id:: BOOT-correlated_confs_to_bts
    """
    if bin_size is None:
        # Applying optimal binning such that tau_int < 0.5
        Cg_binned = auto_binning(Cg=Cg, lambda_output_file=lambda_output_file)
    else:
        Cg_binned = binning(Cg=Cg, bin_size=bin_size)
    #---
    return uncorrelated_confs_to_bts(x=Cg_binned, N_bts=N_bts, seed=seed)
#---


