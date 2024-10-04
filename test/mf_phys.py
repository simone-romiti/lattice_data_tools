import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('../../')
from lattice_data_tools.physical_point.tuning_mf import ml_phys_from_Mpi



ml = np.array([0.01, 0.02])
Mpi = np.array([0.5, 0.7])
dMpi = 0.05*Mpi
Mpi_phys = 0.63

linear_ansatz = lambda x, p: p[0] + p[1]*x
guess = np.array([1.0, 1.0])

ml_phys = ml_phys_from_Mpi(
    ml = ml, Mpi=Mpi, dMpi=dMpi,
    Mpi_phys=Mpi_phys,
    ansatz=linear_ansatz, guess=guess)


plt.errorbar(x=ml, y=Mpi, yerr=dMpi)
plt.scatter(ml_phys, Mpi_phys, marker="o", color="red", s=100, label="ml_phys")

plt.xlabel("$m_\ell$")
plt.ylabel("$M_\pi$")

plt.legend()

plt.savefig("./mf_phys.pdf")
# plt.show()

