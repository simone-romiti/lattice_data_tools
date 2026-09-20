print("""
      ---------------------------------------------
      Bootstrap samples generation: lattice spacing
      ---------------------------------------------
      """
      )

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from lattice_data_tools.dictionaries import NestedDict
from lattice_data_tools.bootstrap import parametric_gaussian_bts
from lattice_data_tools.io import with_dill, with_yaml

ens_info = with_yaml.load("ensembles.yaml")
isoQCD_point = pd.read_csv("isoQCD_point.csv", sep=r"\s+", comment='#')
aux_dir = ens_info["data"]["auxiliary"]

ens_list = ens_info["data"]["ens_list"]
N_bts = ens_info["N_bts"]
RNG_seed=ens_info["RNG_seed"]
np.random.seed(RNG_seed)

print("# Generating the bootstraps of a[fm]")

a_fm_bts_dict = NestedDict()
for i_ens, ens_name in enumerate(ens_list):
    print("Ensemble: ", ens_name)
    en = (ens_name.split(".")[0]+ens_name.split(".")[2])[1:]
    a_fm_mean = isoQCD_point["a[fm]"].to_numpy()[i_ens]
    da_fm = isoQCD_point["err_a[fm]"].to_numpy()[i_ens]
    a_fm_bts_dict[ens_name] = parametric_gaussian_bts(mean=a_fm_mean, error=da_fm, N_bts=N_bts, seed=RNG_seed)
#---

with_dill.dump(a_fm_bts_dict, f"{aux_dir}/a_fm_bts.pkl")

