"""

Set of routines to read the data produced by `cvc` for the P2gg calculations:

repository: https://github.com/gkanwar/mp-cvc
branch: AlpsFeb2026
commit: ff6e2b9

The functions defined here contain the information on how to process the `.aff` files produced by the `cvc`, accounting for the source positions, average over the momenta on the same orbit, etc.
I have tested them against the preprocessed files obtained by Sebastian Burri.

"""

import os
import numpy as np
import itertools
import typing
import re # regular expressions
import sys
from pathlib import Path

from lattice_data_tools.p2gg.LeviCivita_tensor import get_epsilon

def vector_to_string(v: np.ndarray, v_type: typing.Literal["x_mu","p_i"]):
    if v_type == "x_mu":
        assert( len(v.shape)==1 and v.shape[0]==4 ) # it is a 4-vector
        return f"t{int(v[0]):02d}x{int(v[1]):02d}y{int(v[2]):02d}z{int(v[3]):02d}"
    elif v_type == "p_i":
        return f"px{int(v[0])}py{int(v[1])}pz{int(v[2])}"

def string_to_vector(x: str):
    txyz = np.array([float(x) for x in re.split(r'[a-zA-Z]', x) if x])
    return txyz

class aff_reader:
    """Class for reading from `.aff` files"""
    def __init__(self, path_to_aff: str):
        # 1. Check if 'aff' is already loaded globally
        if "aff" not in sys.path:
            # 2. Use pathlib for safer path manipulation (gets the grandparent directory)
            dir_with_aff = str(Path(path_to_aff).resolve().parents[1])

            if dir_with_aff not in sys.path:
                sys.path.insert(0, dir_with_aff)

        # 3. Only import/bind if it's not already available in this scope
        if "aff" not in sys.modules:
            global aff
            import aff
        else:
            # If already loaded elsewhere, ensure it's accessible globally here
            global aff
            #import sys

            aff = sys.modules["aff"]
        #---
        self.eps_ijk = get_epsilon(3) # \\epsilon_{ijk} (needed later)
        
    def read_connected_3pt(self,
                           aff_file: str,
                           corr_key: str,
                           source: str,
                           q_tot: str,
                           gamma_seq: str,
                           f: str,
                           gamma_i: str, gamma_f: str):
        """
        Reads the array of 3-point functions
        
        Example:
        corr_key: p-lvc-lvc (pseudoscalar meson, product of local-vector currents)
        source: t06x60y08z75
        q_tot: qx00qy00qz00 (total momentum in the TFF. Usually \\vec{0})
        gamma_seq: gseq04 (gamma matrix of the operator at t=t_seq)
        f: fl0 (flavor)
        gamma_f: gf01 (gamma matrix of the bilinear at t=t_final)
        gamma_i: gf02 (gamma matrix of the bilinear at t=t_initial)
        
        """
        path_gamma_seq = f"/{corr_key}/{source}/{q_tot}/{gamma_seq}/"
        # print(path_gamma_seq)
        R = aff.Reader(aff_file) 
        t_seq_list = sorted(R.ls(path_gamma_seq)) # ordered sequential times
        corr = []
        momenta_keys = []
        for t_seq in t_seq_list:
            path_t_seq = f"{path_gamma_seq}/{t_seq}/{f}/{gamma_f}/{gamma_i}/"
            momenta = R.ls(path_t_seq)
            corr_t_seq = []
            momenta_keys = momenta
            for momentum in momenta:
                data = np.array(R.read(f"{path_t_seq}/{momentum}"))
                corr_t_seq.append(data)
                #---
            corr.append(corr_t_seq)
            #---
        corr = np.array(corr)
        return {"correlator": corr, "momenta_keys": momenta_keys, "t_seq": t_seq_list}

    def connected_3pt_to_Atildeij(self, aff_files: list[str], txyz_sources: np.ndarray, corr_key: typing.Literal["p-cvc-cvc", "p-lvc-lvc"]):
        """
        Read the 3pt connected correlation functions to build the $\\tilde{A}_ij$ [with `i,j=1,2,3`] as in eq. 9 of https://arxiv.org/pdf/2308.12458.

        The function returns an array of shape (nf=2, n_sources, n_t_seq, n_momenta, T, 3,3)
        - 
        """
        nf = 2 # number of flavors in the twisted-mass doublet (opposite Wilson parameter), e.g. s_{+}, s_{-}
        q_tot="qx00qy00qz00" # our calculation is done always considering the meson at rest (zero momentum)
        gamma_seq = "gseq04" # in cvc, `04` is the index of `gamma_5`
        n_sources = txyz_sources.shape[0] # number of sources
        assert(len(aff_files) == n_sources) # one aff for each source
        fsij_combinations  = list(
            itertools.product(
                [f for f in range(nf)], # upper or lower element of the flavor doublet
                np.arange(n_sources), # sources for the inversion of the Dirac operator
                [i for i in range(1,4)], # index `i=1,2,3`
                [j for j in range(1,4)]  # index `j=1,2,3`
            )
        )
        Atildeij = [] # list of Atildeij for each flavor, source,
        ij_shape = None # shape at fixed flavor, source and (i,j) combination
        t_seq = None # list of t_seq keys
        momenta_keys = None # list of available momenta (same for all combinations)
        for fsij in fsij_combinations:
            f, s, i,j = fsij # unrolling the combinations
            txyz_source = txyz_sources[s,:] # 4-vector with the coordinates of the source
            aff_file = os.path.abspath(aff_files[s]) # absolute path to the s-th `.aff`
            print(aff_file)
            gamma_i = f"gi0{i}" # key of $\\gamma_i$
            gamma_f = f"gf0{j}" # key of $\\gamma_f$

            data = self.read_connected_3pt(
                aff_file = aff_file,
                corr_key = corr_key, source = vector_to_string(txyz_source, v_type="x_mu"),
                q_tot=q_tot,
                gamma_seq = gamma_seq,
                f = f"fl{f}",
                gamma_i = gamma_i , gamma_f = gamma_f
            )
            corr = data["correlator"]
            Atildeij.append(corr)
            ij_shape = corr.shape
            momenta_keys = data["momenta_keys"]
            t_seq = data["t_seq"]
        #---
        Atildeij = np.array(Atildeij).reshape(nf, n_sources, 3,3, *ij_shape)
        Atildeij = np.moveaxis(Atildeij, [2, 3], [-2, -1]) # (3,3) at the bottom
        res = {
            "correlator": Atildeij,
            "t_seq": t_seq,
            "momenta_keys": momenta_keys
        }
        return res
    
    def Atildeij_to_Btilde(self, Atildeij_dict: dict, L: int, T: int, txyz_sources: np.ndarray, Q_fact: float, corr_key: typing.Literal["p-cvc-cvc", "p-lvc-lvc"]):
        """
        Uses the 3pt function produced on the lattice to generate:

        $$\\tilde{B} = -i m_P \\tilde{A}$$

        where $m_P$ is the mass of the pseudoscalar meson (pion, eta, eta')
        and $\\tilde{A}$ is defined through Eqs. 9 and 25 of https://arxiv.org/pdf/2308.12458.
        In input the user should pass Atilde_ij, obtained with `self.connected_3pt_to_Atildeij()`

        NOTEs:

        - This function generates the estimate at fixed configuration and for each t_sequential
        - m_P, E_P and Z_P are determined through the 2-point function of the meson.
        - The 1st line of Eq. 28 of https://arxiv.org/pdf/2308.12458 provides a better estimator, accounting for a factor due to finite time extent T.
          The factor can be included a posteriori, one one has built $\\tilde{B}$ with this function and determined the meson parameters from the 2-point function.
        - Q_fact: is a factor accounting for the charge factor coming from the electromagnetic currents of the meson (e_u^2 + e_d^2)=5/9 for the light quark and e_s^2=1/9 for the strange

        """
        # --------------------------------------------
        # Construction of the source and orbit average
        # --------------------------------------------
        Atildeij = Atildeij_dict["correlator"]
        # t_seq = Atildeij_dict["t_seq"]
        momenta_keys = Atildeij_dict["momenta_keys"]

        Lo2p = (L/np.pi/2)
        k1 = np.array([string_to_vector(p) for p in momenta_keys]).astype(int) # lattice momenta, in lattice units throughout
        k1_norm_squared = (k1**2).sum(axis=1).astype(int) # $|k_1|^2$
        q1 = k1/Lo2p
        q1_norm_squared = np.linalg.norm(q1, axis=1)**2 # $|q_1|^2$
        r1 = q1 / np.expand_dims(q1_norm_squared, axis=1)

        n_sources = txyz_sources.shape[0]
        xyz_sources = txyz_sources[:,1:4] # only spatial components
        t_sources = np.array(txyz_sources[:,0], dtype=int) # only the times

        Atildeij_time_roll = np.zeros_like(Atildeij)
        t_indices = np.arange(T)
        for i in range(n_sources):
            Atildeij_time_roll[:,i,:,:, t_indices, :] = Atildeij[:,i,:,:, (t_indices + t_sources[i]) % T, :, :]
        # shifted_t_indices = (t_indices[None,:] + t_sources[:, None]) % T
        # # shifted_t_indices shape: (n_sources, T) -> reshape to broadcast along axis 4
        # idx = shifted_t_indices.reshape(1, n_sources, 1, 1, T, 1, 1)

        q1x = np.einsum("qi,xi->qx", q1, xyz_sources)

        # Remark: in eq. 6 of https://arxiv.org/pdf/2308.12458 we sum over $\\vec{x}$.
        # When using a source, the integral has to be manually corrected by the phase induced by the source
        q_phase = np.exp(-1j*q1x) # exp(-i*x*q)
        Atildeij_with_phases = np.einsum("qx,gxSqtij->gxSqtij", q_phase, Atildeij_time_roll)

        if corr_key == "p-cvc-cvc":
            """ If we use electromagnetic currents conserved on the lattice (conserved-vector-current) we get correlators that look like this:
            `Tr(gamma_mu S(x+mu,nu) gamma_nu S(nu,x_seq) gamma_5 S(z, x))`

            where z=(t_seq, \\vec{z}) as in eq. 6 of https://arxiv.org/pdf/2308.12458

            - Since we sum over $\\vec{z}$, but the total momentum of the meson is $\\vec{0}$$, we don't get any phase from the propagator `S(nu, z)`
            - When we sum over $\\vec{x}$, as above, we have to include the phase coming from the shift of the arguments of the 1st propagator S(x+mu,nu)=S(x+mu-nu) [from translational invariance]. Changing the variable in the integral we get a phase:

            $$e^{(i/2) (q_i - q_j)}$$

            !!! STILL UNCLEAR WHY WE HAVE TO DIVIDE BY 2 IN THE PHASE, I WOULD EXPECT exp(i(q_i-q_j))!!!

            """
            phase_qij = np.exp(1j * (q1[:, None, :] - q1[:, :, None])/2.0) # e^{(i/2)*(q_j - q_i)}
            Atildeij_with_phases = np.einsum("qij,gxSqtij->gxSqtij", phase_qij, Atildeij_with_phases)
        #---
        Btilde  = - np.einsum("ijk,qk,gxSqtij->gxSqt", self.eps_ijk, r1, Atildeij_with_phases) # Eq. 3.28 of S. Burri thesis
        Btilde_src_avg  = Btilde.mean(axis=1) # average over the sources
        nf = 2
        Btilde_flav_avg = np.einsum("f,f...->...", np.array([1,-1]), Btilde_src_avg)/nf # flavor average
        # ------------------
        # finding the orbits
        # ------------------
        sort_k1_squared = np.argsort(k1_norm_squared)
        k1_norm_squared_sorted = k1_norm_squared[sort_k1_squared]
        k1_sorted = k1[sort_k1_squared,:] # sorted vectors according to |k1|^2
        k1_squared_unique = np.unique(k1_norm_squared_sorted, axis=0) # sorted values of |k_1|^2, no repetitions
        N_orb = k1_squared_unique.shape[0] # number of |q_1|^2
        k1_orbits = []
        for i in range(N_orb):
            lhs = k1_norm_squared_sorted
            rhs = k1_squared_unique[i]
            k1_orbits.append(k1_sorted[lhs == rhs,:])
        #---
        Btilde_orbits = []
        for k1_orbit in k1_orbits:
            k1_orbit_keys = [vector_to_string(k1_i, v_type="p_i") for k1_i in k1_orbit]
            orbit_idx = [np.where(np.array(momenta_keys, dtype=str) == p)[0][0] for p in k1_orbit_keys]
            Btilde_orbit = Q_fact*Btilde_flav_avg[:,orbit_idx,...].mean(axis=1)
            Btilde_orbits.append(Btilde_orbit)
        #---
        Btilde_orbits = np.array(Btilde_orbits) # (n_orbits, n_seq, T)
        res = {
            "Btilde": Btilde_orbits,
            "k1": k1_orbits,
            "k1_squared": k1_squared_unique
        }
        return res
        


