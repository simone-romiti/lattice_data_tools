
import numpy as np
from typing import Dict, List, Optional

from lattice_data_tools.bootstrap import BootstrapSamples
from lattice_data_tools.dictionaries import NestedDict
from lattice_data_tools.model_averaging.IC import valid_IC, with_CDF, get_weights

def get_unique(L: List) -> List:
    unique = []
    for l in L:
        if l not in unique:
            unique.append(l)
    #-------
    return unique


class ModelAverage:
    @staticmethod
    def get_P(y: BootstrapSamples, w: np.ndarray, lam: np.float64):
        """CDFs from boostrap samples 
        
        We build the CDF numerically, by counting how many occurrencies of y we have before each y_0.
        This is done as follows:
        
        Step 0 (optional): we rescale the variance of the bootstrap samples
        Step 1: We consider all the y values, i.e. from each bootstrap and each each model, all together
        Step 2: We build the CDFs of each model, scaled by its weight (normalized to 1).
                This is done by building, for each model k, a fictitious histogram count (with binning 1). 
                Only for the values of "y" of the k-th model, we set the count to the weight w_k.
        Step 3: The cumulative sums of these fictitious histogram gives the CDF 
                in terms of the whole array of values coming from all the models.

        Arguments:
            y: array of size (N_bts, n_models)
            lam: scaling parameter for the variance --> sqrt(lambda) for the uncertainty
            
        Returns:
            dictionary with the results of this procedure
        """
        N_bts = y.N_bts()
        y_rescaled = BootstrapSamples(np.copy(y))
        # producing bootstraps with same mean but rescaled uncertainty
        if lam != 1.0:
            y_avg = y.unbiased_mean()
            sqrt_lam = np.sqrt(lam)
            y_rescaled = BootstrapSamples.from_lambda(N_bts=N_bts, fun = lambda i : y_avg[:] + sqrt_lam*(y[i,:]-y_avg[:]))
        #---
        y_flat, idx_y = np.unique(y_rescaled.flatten(), return_index=True)
        w_unique = BootstrapSamples.from_lambda(N_bts=N_bts, fun=lambda i: w).flatten()[idx_y]
        P = np.cumsum(w_unique)
        if not (w_unique==0.0).all():
            P /= np.sum(w_unique)
        #---
        return {"y": y_flat, "P": P}
    #---
    @staticmethod
    def error_budget(keys: List[str], y: Dict[str,BootstrapSamples], ch2: Dict[str,np.ndarray], n_par: Dict[str,np.ndarray], n_data: Dict[str,np.ndarray], IC: valid_IC, Nmax: Optional[int] = None):
        """
        Error budget contribution as eq. 16 of https://inspirehep.net/literature/2847988 (law of total variance)
        
        This function splits the contributions to the total error on the variable "y", 
        finding the contribution of a specific one corresponding to the variation of the key in "keys".
        
        This is done by: 
            - Finding the ModelAverage CDF at fixed "key", i.e. finding N_keys CDFs.
            - Model averaging all of them, for 2 values of lambda, in order to isolate the contributions
            - The "statistical" error comes from all the other effects, while the  systematics is the contribution of the variation of "key"
        
        Args:
            keys (List[str]): keys corresponding to the contribution to isolate_
            y (Dict[BootstrapSamples]): bootstrap samples (one for each model), for each model in "keys"
                Example: y = {"model1": [y1_bts, y2_bts, ...], "model2": [y1_bts, y2_bts]}
            w (Dict[np.ndarray]): dictionary of the weights
                Example: w = {"model1": [w1, w2, ...], "model2": [w1, w2, ...]}
        """
        w1 = {k: get_weights(ch2=ch2[k], n_par=n_par[k], n_data=n_data[k], IC=IC, Nmax=Nmax) for k in keys}
        # distance between models not defined --> we use the median as an estimator of the marginal probability density
        w1_keys = np.array([np.sum(w1[k]) for k in keys])
        w1_keys_normalized = w1_keys/np.sum(w1_keys)
        y1_list, P1_list = zip(*[(yP_k["y"], yP_k["P"]) for k in keys for yP_k in [ModelAverage.get_P(y=y[k], w=w1[k], lam=1.0)]])
        y1P1 = with_CDF.get_P(y1_list, w=w1_keys)
        n_models = len(y1_list)
        sigma2_stat = 0.0
        sigma2_syst = 0.0 
        y_avg = np.median(y1P1["y"])
        for i in range(n_models):
            mean_i = np.median(y1_list[i])
            var_i = with_CDF.variance_from_CDF(y=y1_list[i], P=P1_list[i])
            sigma2_stat += (w1_keys_normalized[i] * var_i)
            sigma2_syst += (w1_keys_normalized[i] * (y_avg - mean_i)**2)
        #---
        IPR = np.sum(w1_keys_normalized**4)/(np.sum(w1_keys_normalized**2)**2) # Inverse Participation Ratio
        return {"y": y1P1["y"], "P": y1P1["P"], "sigma2_stat": sigma2_stat, "sigma2_syst": sigma2_syst, "sigma2_tot": sigma2_syst+sigma2_stat, "IPR": IPR}
    #---
    @staticmethod
    def error_budget_table(Y: NestedDict, syst_names: List[str], IC: valid_IC, Nmax: Optional[int] = None):
        """ 
        Computes automatically the error budget contributions for each source of the total error: statistical, total systematic and systematic contributions.
        In practice, this function loops over all systematic effects and computes the marginalized CDFs at fixed value of each systematic effect.
        
        X: nested dictionary such that the innermost level contains the BootstrapSamples for each model, the \\chi^2, the number of points and number of parameters.
            X[model_key_1][model_key_2]...[model_key_n]["y"] = BootstrapSamples
            X[model_key_1][model_key_2]...[model_key_n]["ch2"] = np.ndarray
            X[model_key_1][model_key_2]...[model_key_n]["n_par"] = np.ndarray
            X[model_key_1][model_key_2]...[model_key_n]["n_data"] = np.ndarray
            
        syst_names: list of strings corresponding to the contributions to isolate: This is useful to label the contributions.
        """
        all_key_combs = list(get_unique([kk[:-1] for kk in Y.get_key_combinations()])) # list of all key combinations
        n_comb_tot = len(all_key_combs) # total number of model combinations
        max_depth = max([len(kk) for kk in all_key_combs]) # maximum depth of the dictionary
        subkeys = [[] for _ in range(max_depth)] # subkeys at each depth. empty list, filled below
        for i in range(max_depth):
            """ NOTE: we don't consider the last level, which contains the data (x, ch2, n_par, n_data) """
            for kk in all_key_combs:
                if len(kk) >= i+1:
                    subkeys[i].append(kk[i]) # considering only keys at depth i
            #-------
            subkeys[i] = list(set(subkeys[i])) # unique keys at depth i
        #---
        # Main idea: we loop over all model combinations, and append the data to the lists corresponding to each systematic effect
        y_list, ch2_list, n_par_list, n_pts_list, w_list = [], [], [], [], []
        n_syst = len(syst_names)
        assert(n_syst == max_depth), "Number of systematic names must match the depth of the nested dictionary minus one"
        idx_lists = {syst_names[i]: {k: [] for k in subkeys[i]} for i in range(n_syst)}
        idx_lists["syst_tot"] = {k: [] for k in range(n_comb_tot)}
        for i_c in range(n_comb_tot):
            kk = all_key_combs[i_c]
            Data = Y[kk]
            y_i, ch2_i, n_par_i, n_pts_i = Data["y"], Data["ch2"], Data["n_par"], Data["n_data"]
            w_i = get_weights(ch2=ch2_i, n_par=n_par_i, n_data=n_pts_i, IC=IC, Nmax=Nmax)
            y_list.append(y_i)
            ch2_list.append(ch2_i)
            n_par_list.append(n_par_i)
            n_pts_list.append(n_pts_i)
            w_list.append(np.zeros_like(w_i) if np.isnan(w_i).any() else w_i) # replacing NaNs with zero weight
            idx_lists["syst_tot"][i_c].append(i_c)
            for i in range(n_syst):
                idx_lists[syst_names[i]][kk[i]].append(i_c)
        #-------
        def get_syst_contribution(contribution):
            get_subcase = lambda X, k: np.array([X[i] for i in idx_lists[contribution][k]]).T
            contribution_keys = list(idx_lists[contribution].keys())
            res = ModelAverage.error_budget(
                keys=contribution_keys, 
                y = {k : BootstrapSamples(get_subcase(y_list, k)) for k in contribution_keys}, 
                ch2 = {k : get_subcase(ch2_list, k) for k in contribution_keys},
                n_par = {k : get_subcase(n_par_list, k) for k in contribution_keys},
                n_data = {k : get_subcase(n_pts_list, k) for k in contribution_keys},
                IC = IC,
                Nmax = Nmax
                )
            return res
        #---
        EBT = NestedDict()
        for contribution in syst_names+["syst_tot"]:
            EBT[contribution] = get_syst_contribution(contribution)
        #---
        return EBT
#-------

