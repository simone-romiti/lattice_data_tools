""" Time-modulating functions for the windows contributions to g-2 """


import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../../')
import lattice_data_tools.gm2.HVP.windows as windows


t_fm = np.linspace(0.0, 1.5, 100)
for window in ["SD", "W", "LD"]:
    Theta_window = windows.TimeModulator_dict[window](t_fm=t_fm)
    plt.plot(t_fm, Theta_window, label=window)
#---

plt.legend()
plt.title("Please compare with Fig. 1 of https://arxiv.org/abs/2206.15084")
plt.xlabel("t[fm]")


path = "./gm2_windows-time_modulators.pdf"
plt.savefig(path)
plt.show()