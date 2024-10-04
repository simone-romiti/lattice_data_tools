""" Physical point of quark mass """

import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('../../')
from lattice_data_tools.physical_point.tuning_mf import bts_ml_phys_from_Mpi

ml = np.array([0.01, 0.02])
Mpi = np.array([[0.5, 0.7], [0.53, 0.67]])
dMpi = np.std(Mpi, axis=0)

Mpi_phys = 0.63

linear_ansatz = lambda x, p: p[0] + p[1]*x
guess = np.array([1.0, 1.0])

print(ml, Mpi, dMpi)

ml_phys = bts_ml_phys_from_Mpi(
    ml = ml, Mpi=Mpi,
    Mpi_phys=Mpi_phys,
    ansatz=linear_ansatz, guess=guess)


plt.errorbar(x=ml, y=np.average(Mpi,axis=0), yerr=dMpi)
plt.scatter(np.average(ml_phys, axis=0), Mpi_phys, marker="o", color="red", s=100, label="ml_phys")

plt.xlabel("$m_\ell$")
plt.ylabel("$M_\pi$")

plt.legend()

plt.savefig("./mf_phys.pdf")
plt.show()

