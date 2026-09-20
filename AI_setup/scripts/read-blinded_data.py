print("""
      ---------------------------
      Reading confs: blinded data
      ---------------------------
      """
      )

import numpy as np
import matplotlib.pyplot as plt
import dill
import yaml
import os
import struct
import sys

from lattice_data_tools.dictionaries import NestedDict
from lattice_data_tools.io import with_dill, with_yaml

from local_libs import read_bin

ens_info = with_yaml.load("ensembles.yaml")
ens_list = ens_info["data"]["ens_list"]
ens_short = ["".join((e.split(".")[0])[1:] + e.split(".")[2]) for e in ens_list]

aux_dir = ens_info["data"]["auxiliary"]

print("# Reading blinded data")

ens_list = ens_info["data"]["ens_list"]
blinded_data_dir = f'{ens_info["data"]["blinded_data"]}'

f_pi_list = ens_info["data"]["f_pi_MeV"]
for f_pi in f_pi_list:
    print(f"f_pi:{f_pi}")
    outdir = f"{aux_dir}/fpi_{f_pi}_MeV/"
    os.makedirs(outdir, exist_ok=True)
    VKVK_dict = NestedDict()
    for ie, e in enumerate(ens_short):
        print(f" Ensemble:{e}")
        ens_name = ens_list[ie]
        for fermion_type in ["OS", "tm"]:
            print("  Fermions: ", fermion_type)
            data = read_bin(f"{blinded_data_dir}/fpi_{f_pi}_MeV/{e}/VkVk_{fermion_type}.bin")
            print(data["confs"].shape)
            VKVK_dict[ens_name][fermion_type] =  data["confs"]
    #-------
    print(f"Saving the output in {outdir}")
    os.makedirs(f"{outdir}", exist_ok=True)
    with_dill.dump(VKVK_dict, f"{outdir}/VKVK_confs.pkl")
#---
