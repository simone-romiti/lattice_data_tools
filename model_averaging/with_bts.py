
import numpy as np
from typing import Dict, List

from lattice_data_tools.bootstrap import BootstrapSamples
from lattice_data_tools.model_averaging.AIC import with_CDF, get_weights

class AIC:
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
    def error_budget(keys: List[str], y: Dict[str,BootstrapSamples], ch2: Dict[str,np.ndarray], n_par: Dict[str,np.ndarray], n_data: Dict[str,np.ndarray]):
        """
        Error budget contribution as eq. 1 of https://inspirehep.net/literature/2847988
        
        This function splits the contributions to the total error on the variable "y", 
        finding the contribution of a specific one corresponding to the variation of the key in "keys".
        
        This is done by: 
            - Finding the AIC CDF at fixed "key", i.e. finding N_keys CDFs.
            - Model averaging all of them, for 2 values of lambda, in order to isolate the contributions
            - The "statistical" error comes from all the other effects, while the  systematics is the contribution of the variation of "key"
        
        Args:
            keys (List[str]): keys corresponding to the contribution to isolate_
            y (Dict[BootstrapSamples]): bootstrap samples (one for each model), for each model in "keys"
                Example: y = {"model1": [y1_bts, y2_bts, ...], "model2": [y1_bts, y2_bts]}
            w (Dict[np.ndarray]): dictionary of the weights
                Example: w = {"model1": [w1, w2, ...], "model2": [w1, w2, ...]}
            eps_thr
        """
        w1 = {k: get_weights(ch2=ch2[k], n_par=n_par[k], n_data=n_data[k]) for k in keys}
        w1_keys = np.array([np.sum(w1[k]) for k in keys])
        w1_keys_normalized = w1_keys/np.sum(w1_keys)
        y1_list, P1_list = zip(*[(yP_k["y"], yP_k["P"]) for k in keys for yP_k in [AIC.get_P(y=y[k], w=w1[k], lam=1.0)]])
        y1P1 = with_CDF.get_P(y1_list, w=w1_keys)
        n_models = len(y1_list)
        sigma2_syst = with_CDF.variance_from_CDF(y=y1P1["y"], P=y1P1["P"])
        sigma2_stat = 0.0
        for i in range(n_models):
            var_i = with_CDF.variance_from_CDF(y=y1_list[i], P=P1_list[i])
            sigma2_stat += (w1_keys_normalized[i] * var_i)
        #---
        return {"stat": sigma2_stat, "syst": sigma2_syst, "tot": sigma2_syst+sigma2_stat}
    #---
#---

