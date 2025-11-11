
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
    def error_budget(keys: List[str], y: Dict[str,BootstrapSamples], ch2: Dict[str,np.ndarray], n_par: Dict[str,np.ndarray], n_data: Dict[str,np.ndarray], lam1=1.0, lam2=2.0):
        """
        Error budget contribution as in Section 21 of https://arxiv.org/pdf/2002.12347
        
        This function splits the contributions to the total error on the variable "y", 
        finding the contribution of a specific one corresponding to the variation of the key in "keys".


        NOTE: 
        lam2 is automatically updated in order to return all positive variances.
        In this way both the statistical and systematic variances are positive. 
        Since we don't know a priori the separation, the value of lam2 is found automatically.

        
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
        w1 = {k: np.ones_like(ch2[k]) if lam1==0.0 else  get_weights(ch2=ch2[k]/lam1, n_par=n_par[k], n_data=n_data[k]) for k in keys}
        w2 = {k: get_weights(ch2=ch2[k]/lam2, n_par=n_par[k], n_data=n_data[k]) for k in keys}
        w1_keys = np.array([np.sum(w1[k]) for k in keys])
        w2_keys = np.array([np.sum(w2[k]) for k in keys])
        y1_list, P1_list = zip(*[(with_CDF.get_rescaled_y(yP_k["y"], yP_k["P"], lam=lam1), yP_k["P"]) for k in keys for yP_k in [AIC.get_P(y=y[k], w=w1[k], lam=1.0)]])
        y2_list, P2_list = zip(*[(with_CDF.get_rescaled_y(yP_k["y"], yP_k["P"], lam=lam2), yP_k["P"]) for k in keys for yP_k in [AIC.get_P(y=y[k], w=w2[k], lam=1.0)]])
        y1P1 = with_CDF.get_P(y1_list, w=w1_keys)
        y1, P1 = y1P1["y"], y1P1["P"]
        y2P2 = with_CDF.get_P(y2_list, w=w2_keys)
        y2, P2 = y2P2["y"], y2P2["P"]
        res = with_CDF.get_contributions(y1=y1, y2=y2, P1=P1, P2=P2, lam1=lam1, lam2=lam2)
        sigma2_stat = res["stat"] # this estimate is positive
        if res["syst"] < 0:
            lam1_new, lam2_new = 0.0, 1.0
            res = AIC.error_budget(keys=keys, y=y, ch2=ch2, n_par=n_par, n_data=n_data, lam1=lam1_new, lam2=lam2_new)
            # The new values of "lam1,lam2" might return a negative "stat" variance.
            # We computed it already, so we use it explicitly here.
            res["stat"] = sigma2_stat 
            return res
        else:
            return res
    #---
#---

