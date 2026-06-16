"""
Class defining a Nested Sampling run for a lattice gauge theory:

$$Z = \\int dU L[U] = \\int dU e^{-S[U]}$$

"""

import os
import typing
import torch
import numpy as np

from lattice_data_tools.links.configuration import GaugeConfiguration

class NestedSampling_run:
    def __init__(
            self,
            prior: typing.Callable, E: typing.Callable,
            U: GaugeConfiguration, E_values: torch.Tensor,
            n_live: int, seed: int,
            output_dir: str
    ):
        self.prior = prior # normalized prior distribution
        self.E = E # "energy" functional, e.g. E=-log(Likelihood). It is the function defining isocontours
        self.n_live = n_live # number of live points
        self.E_values = E_values # tensor of `n_live` energy values
        self.restart = (self.E_values.numel == 0) # if there are no energies, NS run will restart from scratch

        self.U = U # gauge configuration to be evolved
        self.lattice_shape = U.lattice_shape 
        self.d             = U.d             

        self.seed = seed # seed for random number generation
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        write_mode = "w" if restart else "a"
        self.E_values_file = open(f"{self.output_dir}/E_values.dat", mode=write_mode)
        
    def save_energy(self, E_k: float) -> None:
        self.E_values_file.write(f"{E_k}:e")
        
    def get_constrained_prior(self, E_star: float):
        theta_E = lambda dof: ((self.E(dof) - E_star) > 0)
        pi_E = lambda dof: self.prior(dof)*theta_E(dof)
        return pi_E
    
    def run(self, E_final: float, n_iter: int, save_confs: bool):
        """ Running the NS algorithm for up to `n_iter` steps, stopping if reaching `E_final` """
        if self.restart:
            for k in range(self.n_live):
                self.U.hotstart(seed=self.seed) # n_live points uniformly distributed in the Haar measure
                E_k = torch.Tensor(self.E(self.U)) # measuring the energy
                self.E_values = torch.cat(self.E_values, E_k) # appending the value of the Energy
                self.save_energy(E_K=E_k) # saving the energy measurement
                if save_confs:
                    self.U.save(f"{self.output_dir}/U_{k}.pt")
            #---
        #---
        i_offset = self.n_live # offset of indices of configurations
        i = 0
        E_max = torch.min(self.E_values) # avoiding exiting the loop immediately
        while i < n_iter and E_max < E_final:
            E_min, idx_min = torch.min(self.E_values) # selection of the point with minimum energy
            self.E_values = torch.cat((self.E_values[0:idx_min], self.E_values[(idx_min+1),:]))# removing that point
            
            # selection of one of the remaining points p_1
            idx_new = torch.randint(0, self.n_live-1) # random index of one of the remaining n_live-1 points
            idx_new_conf = idx_new if idx_new<idx_min else 1+idx_new # index of the original configurations
            Ui = GaugeConfiguration.load(f"{self.output_dir}/U_{i}")[idx_new_conf,...]
            
            # drawing a new point according to the constrained prior
            # evolution of p_1 for n_steps, such that acceptance rate is roughly constant
            E_max, idx_max = torch.max(self.E_values)
            i += 1 # moving to the next NS step

        
