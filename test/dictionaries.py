# testing the nested dictionary routines

import sys
sys.path.append('../../')

from lattice_data_tools.dictionaries import NestedDict, nested_dict

d_old = nested_dict()
d_new = NestedDict()

d_old["a"]["b"] = 1
d_new["ciao"]["h"]["g"] = 30
d_new["hello"]["a"] = "abcdef"

# print(d_old)

key_combs = NestedDict.key_combinations(d_new)
NestedDict.loop_over_key_combinations(key_combs, fun= lambda k1, k2, k3: print(k1, k2, k3))
