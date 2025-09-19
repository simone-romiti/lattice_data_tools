""" PLEASE LOOK AT THE bootstrap.py and jackknife.py """

# ## resampling techniques from gauge configurations

# import numpy as np

# from lattice_data_tools import uwerr



# def parametric_gaussian_bts(mean: float, std: float, N_bts, seed=12345):
#     """Generate paramteric bootstrap samples from a Gaussian distribution
    
#     Args:
#         mean (float): Mean of the Gaussian distribution
#         std (float): Standard deviation of the Gaussian distribution
#         N_bts (int): Number of bootstrap samples to generate
#         seed (int): Random seed for reproducibility
    
#     Returns:
#         np.ndarray: Bootstrap samples
#     """
#     np.random.seed(seed=seed)
#     return np.random.normal(loc=mean, scale=std, size=N_bts)
# #---

# def uncorrelated_confs_to_bts(x, N_bts, seed=12345):
#     """Bootstrap samples from array of data
    
#     generates Nb bootstrap samples from N configurations,
#     It assumes x.shape[0] == N.
    
#     Steps:
#     - Draw N samples with replacements among the x values
#     - Compute the mean. This is a bootstrap sample.
#     - Repeat Nb times. The resulting array has standard deviation of the mean (approximately) equal to the one of the original sample.
    
#     Args:
#         x (np.ndarray): time series. Bootstrapping is done on 1st index
#         N_bts (int): Number of bootstraps
#         block_size (int): size of the block
    
#     Returns:
#         np.ndarray: Bootstrap samples
#     """
#     N = x.shape[0]
#     np.random.seed(seed=seed)
#     return np.array([np.average(x[np.random.randint(0,high=N,size=N,dtype=int)]) for i in range(N_bts)])
# ####

# def correlated_confs_to_bts(Cg: np.ndarray, N_bts: int, seed=12345, output_file=None) -> np.ndarray:
#     """Bootstrap samples from array of correlated configurations

#     - The configurations are sampled every tau_int, (integrated autocorrelation time)
#     - The bootstrap samples are drawn from the uncorrelated configurations

#     Args:
#         C (np.ndarray): 1-dimensional "correlator" (observable computed for each configuration)
#         N_bts (int): Number of bootstraps

#     Returns:
#         np.ndarray: Bootstrap samples
#     """
#     Ng = Cg.shape[0] ## total number of configurations
#     tauint = int(uwerr.uwerr_primary(Cg, output_file=output_file)["tauint"]) ## integrated autocorrelation time
#     if tauint == 0:
#         tauint = 1 ## uncorrelated data
#     ####
#     Cg_uncorr = Cg[0:Ng:tauint] ## uncorrelated values
#     return uncorrelated_confs_to_bts(x=Cg_uncorr, N_bts=N_bts, seed=seed)
# ####

# if __name__ == "__main__":
#     Cg = np.array(100*[list(np.random.normal(0.0, 0.4, 50))]).flatten()
#     seed = 12345
#     N_bts = 1000
#     C_bts = correlated_confs_to_bts(Cg, N_bts=N_bts, seed=seed)

#     print(np.average(Cg))
#     print(np.average(C_bts))

#     np.random.seed(seed=seed)
#     Ng = Cg.shape[0]
#     C_bts = uncorrelated_confs_to_bts(Cg, N_bts=N_bts)
#     print(np.average(Cg))
#     print(np.average(C_bts))
# ####
