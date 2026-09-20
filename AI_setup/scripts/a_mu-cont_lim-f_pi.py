print("""
      ---------------------------------------------------------
      Preparing the dataset of a_\\mu(f_\\pi) for extrapolation
      ---------------------------------------------------------
      """
      )



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Helvetica"
# })

import os

from lattice_data_tools.bootstrap import BootstrapSamples, uncorrelated_confs_to_bts, ParametricBootstraps
from lattice_data_tools.io import with_dill, with_yaml
from lattice_data_tools.dictionaries import NestedDict
import lattice_data_tools.statistics_tools as statistics
from lattice_data_tools.fit.xyey import fit_xyey, polynomial_fit_xyey
import lattice_data_tools.plotting.with_matplotlib.distribution as plot_distribution


ens_info = with_yaml.load("ensembles.yaml")

RNG_seed = ens_info["RNG_seed"]

aux_dir = ens_info["data"]["auxiliary"]
plots_dir = ens_info["data"]["plots"]
N_bts = ens_info["N_bts"]
fermion_types = ens_info["fermion_types"]
windows = ens_info["windows"]

fit_types = ens_info["continuum_limit"]["fit"]["fit_types"]

N_max = 8 # maximum number of points in cont. lim extrapolations

strategies = ["conservative", "preferred", "aggressive", "moderate"]

windows = ens_info["windows"]
n_samples_fpi_extr = ens_info["n_samples_fpi_extr"]

f_pi_list = ens_info["data"]["f_pi_MeV"]
n_fpi = len(f_pi_list)


a_mu_dict = NestedDict()
for f_pi in f_pi_list:
    print(f"f_pi:{f_pi}")
    fpi_suffix = f"fpi_{f_pi}_MeV"
    fpi_dir = f"{aux_dir}/{fpi_suffix}/"

    res_IC_averaging_dict = with_dill.load(f"{fpi_dir}/model_average_continuum_limit.pkl")
    key_combs = res_IC_averaging_dict.get_key_combinations(max_depth=4)
    for key_comb in key_combs:
        print(key_comb)
        window, correlation_flag, strategy, IC_name = key_comb
        EBT = res_IC_averaging_dict[window][correlation_flag][strategy][IC_name]
        y, P = EBT["syst_tot"]["y"], EBT["syst_tot"]["P"]
        a_mu_mean = EBT["syst_tot"]["mean"]
        a_mu_sample = statistics.sample_from_CDF(y=y, P=P, seed=RNG_seed, n_samples=n_samples_fpi_extr)
        a_mu_sample += a_mu_mean - np.mean(a_mu_sample) # removing the bias from the mean
        a_mu_fpi_bts = uncorrelated_confs_to_bts(x=a_mu_sample, N_bts=N_bts, seed=RNG_seed).with_rescaled_error(np.sqrt(n_samples_fpi_extr))
        fix, ax = plt.subplots()
        DP = plot_distribution.DistributionPlotter(fix=fix, ax=ax)
        DP.cdf(a_mu_sample, color="orange", label="Sample from CDF")
        DP.cdf(a_mu_fpi_bts, color="green", label="Bootstraps on the CDF samples")
        ax.axvline(a_mu_mean, color="red", label="Mean: {:.4e}".format(a_mu_mean))
        ax.legend()
        plt.title(f"Distribution of $a_\\mu$ samples for $f_\\pi={f_pi}$ MeV")
        plt.xlabel("$a_\\mu$")
        plt.ylabel("Density")
        plt.tight_layout()
        plt_fld = f"./{plots_dir}/f_pi_dependence/histograms/{window}/{correlation_flag}/{strategy}/{IC_name}/"
        os.makedirs(plt_fld, exist_ok=True)
        outfile = f"{plt_fld}/a_mu_distribution-fpi_{f_pi}_MeV.svg"
        plt.savefig(outfile)
        plt.close()

        a_mu_dict[window][correlation_flag][strategy][IC_name][f_pi] = a_mu_fpi_bts 
#-------

pkl_file = f"{aux_dir}/a_mu-cont_lim-f_pi.pkl"
print("--> Output:", pkl_file)
with_dill.dump(a_mu_dict, pkl_file)

