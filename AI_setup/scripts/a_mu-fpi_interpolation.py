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

from lattice_data_tools.fit.xyey import fit_xyey, polynomial_fit_xyey
import lattice_data_tools.plotting.with_matplotlib.distribution as plot_distribution
import lattice_data_tools.constants as constants

f_pi_FLAG_MeV = constants.Fpi_MeV["isoQCD"]["Edinburgh"]

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

pkl_file = f"{aux_dir}/a_mu-cont_lim-f_pi.pkl"
print("--> Output:", pkl_file)
a_mu_dict = with_dill.load(pkl_file)

print("-------------")
print("interpolation")
print("-------------")

schemes = ["FLAG", "WP25"]

key_combs = a_mu_dict.get_key_combinations(max_depth=2)
for key_comb in key_combs:
    print(key_comb)
    window, correlation_flag = key_comb
    strategies = list(a_mu_dict[key_comb].keys())
    IC_names = list(a_mu_dict[key_comb][strategies[0]].keys())
    for IC_name in IC_names:
        print(f" IC: {IC_name}")
        interp_dict = NestedDict()
        for strategy in strategies:
            print(f"  strategy:{strategy}")
            # FITTING
            a_mu_bts = BootstrapSamples(np.array([a_mu_dict[key_comb][strategy][IC_name][f_pi] for f_pi in f_pi_list]).T)
            f_pi_values = np.array([float(f_pi) for f_pi in f_pi_list])
            N_pts = n_fpi
            a_mu_err = a_mu_bts.error()
            ansatz = lambda x, p: p[0] + p[1] * x
            N_par = 2
            par_bts = BootstrapSamples.zeros(N_bts=N_bts, shape=(N_par)) # parameters of the linear fit
            ch2_bts = BootstrapSamples.zeros(N_bts=N_bts) # parameters of the linear fit
            N_dof = N_pts - N_par
            Cov_inv = None # The points are very correlated --. Cov^{-1} is a numerical artifact
            # if corr_key == "correlated":
            #     Cov = a_mu_bts.covariance_matrix()
            #     Cov_inv = np.linalg.inv(Cov)
            # #---
            for i in range(N_bts+1):
                fit_res = polynomial_fit_xyey(
                    N_deg = 1,
                    x = f_pi_values, y = a_mu_bts[i,:],
                    ey = a_mu_err,
                    Cov_y_inv = Cov_inv
                )
                # guess = fit_res_poly["par"]
                # fit_res = fit_xyey(
                #     ansatz = ansatz, 
                #     x = f_pi_values, y = a_mu_bts[i,:], ey = a_mu_err,
                #     guess = guess, Cov_y_inv = Cov_inv)
                par_bts[i,:] = fit_res['par']
                ch2_bts[i] = fit_res['ch2']
            #---

            # INTERPOLATING
            f_pi_MeV_FLAG_mean = 130.5
            f_pi_MeV_FLAG_err = 0.0
            f_pi_MeV_FLAG_bts = ParametricBootstraps.Gaussian(mean=f_pi_MeV_FLAG_mean, error=f_pi_MeV_FLAG_err, N_bts=N_bts, seed=RNG_seed)

            f_pi_MeV_WP25_mean = 131.30
            f_pi_MeV_WP25_err  = 0.34
            f_pi_MeV_WP25_bts = ParametricBootstraps.Gaussian(mean=f_pi_MeV_WP25_mean, error=f_pi_MeV_WP25_err, N_bts=N_bts, seed=RNG_seed)

            a_mu_WP25_bts = BootstrapSamples.from_lambda(N_bts=N_bts, fun=lambda k: ansatz(f_pi_MeV_WP25_bts[k], par_bts[k,:]))
            a_mu_FLAG_bts = BootstrapSamples.from_lambda(N_bts=N_bts, fun=lambda k: ansatz(f_pi_FLAG_MeV, par_bts[k,:]))
            da_mu_dfpi_bts = BootstrapSamples(par_bts[:,1]) # derivative of a_mu with respect to f_pi
            # Define interpolation points
            interpolation_points = {
                'WP25': {"bts": f_pi_MeV_WP25_bts, 'mean': 131.30, 'err': 0.34},
                'FLAG': {"bts": f_pi_MeV_FLAG_bts, 'mean': f_pi_FLAG_MeV, 'err': 0.0}
            }


            # PLOTTING
            color = "green" # if corr_key == "correlated" else "red"
            plt.errorbar(
                x=f_pi_values, y=a_mu_bts.unbiased_mean(), yerr=a_mu_err, 
                color="black", linestyle="none", marker="o", capsize=3,
                label="data points")
            fpi_dense = np.linspace(np.min(f_pi_values), np.max(f_pi_values), num=250)
            # a_mu_dense = BootstrapSamples.from_lambda(N_bts=N_bts, fun=lambda k: ansatz(fpi_dense,par_bts[k,:]))
            a_mu_dense = BootstrapSamples.from_lambda(N_bts=N_bts, fun=lambda k: ansatz(fpi_dense,par_bts[k,:]))
            a_mu_dense_mean = a_mu_dense.unbiased_mean()
            a_mu_dense_err = a_mu_dense.error()
            #print(a_mu_dense_mean.shape, a_mu_dense_err.shape)
            plt.fill_between(
                x=fpi_dense, y1=a_mu_dense_mean-a_mu_dense_err, y2=a_mu_dense_mean+a_mu_dense_err, 
                alpha=0.5, label=f"fit", color=color)

            a_mu_prediction_bts = BootstrapSamples.from_lambda(N_bts=N_bts, fun=lambda k: ansatz(f_pi_values, par_bts[k,:]))
            plt.plot(f_pi_values, a_mu_prediction_bts.unbiased_mean(), linestyle="--", color=color)
            plt.plot(f_pi_values, a_mu_prediction_bts.unbiased_mean()-a_mu_prediction_bts.error(), linestyle="--", color=color)
            plt.plot(f_pi_values, a_mu_prediction_bts.unbiased_mean()+a_mu_prediction_bts.error(), linestyle="--", color=color)
            #---
            # Plot vertical lines and error bands for interpolation points
            point_colors = {'WP25': 'blue', 'FLAG': 'orange'}
            for point_name, point_params in interpolation_points.items():
                f_pi_mean = point_params['mean']
                f_pi_err  = point_params['err']
                f_pi_bts = ParametricBootstraps.Gaussian(
                    mean = f_pi_mean,
                    error= f_pi_err ,
                    N_bts=N_bts, 
                    seed=RNG_seed
                )
                a_mu_bts_interp = BootstrapSamples.from_lambda(
                    N_bts=N_bts, 
                    fun=lambda k: ansatz(f_pi_bts[k], par_bts[k,:])
                )

                interp_dict[strategy][f"a_mu_{point_name}"]["mean"]         = a_mu_bts_interp.unbiased_mean()
                interp_dict[strategy][f"a_mu_{point_name}"]["error"]        = a_mu_bts_interp.error()
                interp_dict[strategy][f"(9/10)*a_mu_{point_name}"]["mean"]  = (9/10)*interp_dict[strategy][f"a_mu_{point_name}"]["mean"] 
                interp_dict[strategy][f"(9/10)*a_mu_{point_name}"]["error"] = (9/10)*interp_dict[strategy][f"a_mu_{point_name}"]["error"]

                # plotting the interpolations

                a_mu_mean = a_mu_bts_interp.unbiased_mean()
                a_mu_err = a_mu_bts_interp.error()

                color = point_colors[point_name]
                plt.axvline(
                    x=f_pi_mean, 
                    linestyle=":", alpha=0.7, color=color,
                    label=f"{point_name}: $(9/10) \\times a_\\mu=({0.9*a_mu_mean*1e10:.2f}\\pm{0.9*a_mu_err*1e10:.2f})\\times 10^{{-10}}$")

                # Add error band for interpolation points
                if f_pi_err > 0:
                    plt.axvspan(f_pi_mean - f_pi_err, f_pi_mean + f_pi_err, alpha=0.2, color=color)
            #---

            interp_dict[strategy]["a_mu_slope"]["mean"]  = da_mu_dfpi_bts.unbiased_mean()
            interp_dict[strategy]["a_mu_slope"]["error"] = da_mu_dfpi_bts.error()
            interp_dict[strategy]["(9/10)*a_mu_slope"]["mean"] = (9/10)*interp_dict[strategy]["a_mu_slope"]["mean"] 
            interp_dict[strategy]["(9/10)*a_mu_slope"]["error"] = (9/10)*interp_dict[strategy]["a_mu_slope"]["error"]


            plt.ylim(bottom=np.min(a_mu_bts.unbiased_mean())*0.95, top=np.max(a_mu_bts.unbiased_mean())*1.05)

            plt.title("Extrapolation in $f_\\pi$")
            plt.xlabel("$f_\\pi$")
            plt.ylabel("$a_\\mu$")
            plt.legend()
            plt.tight_layout()

            plt_fld = f"./{plots_dir}/f_pi_dependence/interpolation/{window}/{correlation_flag}/plots/"
            os.makedirs(plt_fld, exist_ok=True)
            plt.savefig(f"{plt_fld}/{strategy}-{IC_name}.svg")
            plt.close()
        #---

        # saving sata to a table

        output_data = {
            "strategy": strategies,
            "FLAG_mean": [interp_dict[strategy][f"a_mu_FLAG"]["mean"] for strategy in strategies],
            "FLAG_error": [interp_dict[strategy][f"a_mu_FLAG"]["error"] for strategy in strategies],
            "WP25_mean": [interp_dict[strategy][f"a_mu_WP25"]["mean"] for strategy in strategies],
            "WP25_error": [interp_dict[strategy][f"a_mu_WP25"]["error"] for strategy in strategies],
            "slope_mean": [interp_dict[strategy][f"a_mu_slope"]["mean"] for strategy in strategies],
            "slope_error": [interp_dict[strategy][f"a_mu_slope"]["error"] for strategy in strategies],
            "(9/10)*FLAG_mean": [interp_dict[strategy][f"(9/10)*a_mu_FLAG"]["mean"] for strategy in strategies],
            "(9/10)*FLAG_error": [interp_dict[strategy][f"(9/10)*a_mu_FLAG"]["error"] for strategy in strategies],
            "(9/10)*WP25_mean": [interp_dict[strategy][f"(9/10)*a_mu_WP25"]["mean"] for strategy in strategies],
            "(9/10)*WP25_error": [interp_dict[strategy][f"(9/10)*a_mu_WP25"]["error"] for strategy in strategies],
            "(9/10)*slope_mean": [interp_dict[strategy][f"(9/10)*a_mu_slope"]["mean"] for strategy in strategies],
            "(9/10)*slope_error": [interp_dict[strategy][f"(9/10)*a_mu_slope"]["error"] for strategy in strategies],
        }
        output_df = pd.DataFrame(output_data)
        csv_path = f"./{plots_dir}/f_pi_dependence/interpolation/{window}/{correlation_flag}/tables/{IC_name}.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        output_df.to_csv(csv_path, index=False)

