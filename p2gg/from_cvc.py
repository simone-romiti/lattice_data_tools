"""

Set of routines to read the data produced by `cvc` for the P2gg calculations:

repository: https://github.com/gkanwar/mp-cvc
branch: AlpsFeb2026
commit: ff6e2b9

The functions defined here contain the information on how to process the `.aff` files produced by the `cvc`, accounting for the source positions, average over the momenta on the same orbit, etc.
I have tested them against the preprocessed files obtained by Sebastian Burri.

"""

import sys
import os
import numpy as np
import itertools
import typing
import re # regular expressions

from lattice_data_tools.p2gg.LeviCivita_tensor import get_epsilon

def vector_to_string(v: np.ndarray, v_type: typing.Literal["x_mu","p_i"]):
    if v_type == "x_mu":
        assert( len(v.shape)==1 and v.shape[0]==4 ) # it is a 4-vector
        return f"t{v[0]}v{v[1]}y{v[2]}x{v[3]}"
    elif v_type == "p_i":
        return f"px{v[0]}py{v[1]}pz{v[2]}"

def string_to_vector(x: str):
    txyz = np.array([float(x) for x in re.split(r'[a-zA-Z]', source) if x])
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
            import sys

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

    def connected_3pt_to_Btilde(self, L: int, T: int, aff_files: list[str], txyz_sources: np.ndarray, Q_fact: float, corr_key: typing.Literal["p-cvc-cvc", "p-lvc-lvc"]):
        """
        Uses the 3pt function produced on the lattice to generate:

        $$\\tilde{B} = -i m_P \\tilde{A}$$

        where $m_P$ is the mass of the pseudoscalar meson (pion, eta, eta')
        and $\\tilde{A}$ is defined through Eqs. 9 and 25 of https://arxiv.org/pdf/2308.12458.

        NOTEs:

        - This function generates the estimate at fixed configuration and for each t_sequential
        - m_P, E_P and Z_P are determined through the 2-point function of the meson.
        - The 1st line of Eq. 28 of https://arxiv.org/pdf/2308.12458 provides a better estimator, accounting for a factor due to finite time extent T.
          The factor can be included a posteriori, one one has built $\\tilde{B}$ with this function and determined the meson parameters from the 2-point function.
        - Q_fact: is a factor accounting for the charge factor coming from the electromagnetic currents of the meson (e_u^2 + e_d^2)=5/9 for the light quark and e_s^2=1/9 for the 

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
        C_ij = [] # list of C_ij for each flavor, source,
        ij_shape = None # shape at fixed flavor, source and (i,j) combination
        momenta_keys = None # list of available momenta (same for all combinations)
        for fsij in fsij_combinations:
            f, s, i,j = fsij # unrolling the combinations
            txyz_source = txyz_sources[s,:] # 4-vector with the coordinates of the source
            aff_file = os.path.abspath(aff_files[s]) # absolute path to the s-th `.aff`
            gamma_i = f"gi0{i}" # key of $\\gamma_i$
            gamma_f = f"gf0{j}" # key of $\\gamma_f$
            
            data = self.read_connected_3pt(
                aff_file = aff_file,
                corr_key = corr_key, source = txyz_source,
                q_tot=q_tot,
                gamma_seq = gamma_seq,
                f = f"fl{f}",
                gamma_i = gamma_i , gamma_f = gamma_f
            )
            corr = data["correlator"]
            C_ij.append(corr)
            ij_shape = corr.shape
            momenta_keys = data["momenta_keys"]
            t_seq = data["t_seq"]
        #---
        C_ij = np.array(C_ij).reshape(nf, n_sources, 3,3, *ij_shape)
        C_ij = np.moveaxis(C_ij, [2, 3], [-2, -1]) # (3,3) at the bottom

        # --------------------------------------------
        # Construction of the source and orbit average
        # --------------------------------------------
        n_momenta = len(momenta_keys)

        q1 = (2.0*np.pi/L) * np.array([[float(pi[1:]) for pi in p.split("p")[1:]] for p in momenta_keys]) # lattice momenta, in lattice units throughout
        q1_norm_squared = np.linalg.norm(q1, axis=1)**2 # $|q_1|^2$
        r1 = q1 / np.expand_dims(q1_norm_squared, axis=1)

        n_sources = txyz_sources.shape[0]
        xyz_sources = txyz_sources[:,1:4] # only spatial components
        t_sources = np.array(txyz_sources[:,0], dtype=int) # only the times

        C_ij_time_roll = np.zeros_like(C_ij)
        t_indices = np.arange(T)
        for i in range(n_sources):
            C_ij_time_roll[:,i,:,:, t_indices, :] = C_ij[:,i,:,:, (t_indices + t_sources[i]) % T, :, :]

        # shifted_t_indices = (t_indices[None,:] + t_sources[:, None]) % T

        # # shifted_t_indices shape: (n_sources, T) -> reshape to broadcast along axis 4
        # idx = shifted_t_indices.reshape(1, n_sources, 1, 1, T, 1, 1)

        q1x = np.einsum("qi,xi->qx", q1, xyz_sources)

        # Remark: in eq. 6 of https://arxiv.org/pdf/2308.12458 we sum over $\\vec{x}$.
        # When using a source, the integral has to be manually corrected by the phase induced by the source
        q_phase = np.exp(-1j*q1x) # exp(-i*x*q)
        C_ij_with_phases = np.einsum("qx,gxSqtij->gxSqtij", q_phase, C_ij_time_roll)

        if corr_key == "p-cvc-cvc":
            """ If we use electromagnetic currents conserved on the lattice (conserved-vector-current) we get correlators that look like this:
            `Tr(gamma_mu S(x+mu,nu) gamma_nu S(nu,x_seq) gamma_5 S(z, x))`

            where z=(t_seq, \\vec{z}) as in eq. 6 of https://arxiv.org/pdf/2308.12458

            - Since we sum over $\\vec{z}$, but the total momentum of the meson is $\\vec{0}$$, we don't get any phase from the propagator `S(nu, z)`
            - When we sum over $\\vec{x}$, as above, we have to include the phase coming from the shift of the arguments of the 1st propagator S(x+mu,nu)=S(x+mu-nu) [from translational invariance]. Changing the variable in the integral we get a phase:

            $$e^{(i/2) (q_nu - q_mu)}$$

            !!! STILL UNCLEAR WHY WE HAVE TO DIVIDE BY 2 IN THE PHASE, I WOULD EXPECT exp(i(q_mu-q_nu))!!!

            """
            phase_qij = np.exp(1j * (q1[:, None, :] - q1[:, :, None])/2.0) # e^{(i/2)*(q_j - q_i)}
            C_ij_with_phases = np.einsum("qij,gxSqtij->gxSqtij", phase_qij, C_ij_with_phases)

        Btilde = - np.einsum("ijk,qk,gxSqtij->gxSqt", self.eps_ijk, r1, C_ij_with_phases) # Eq. 3.28 of S. Burri thesis
        Btilde_src_avg = Btilde.mean(axis=1) # average over the sources
        B_tilde_flav_sum = np.einsum("f,f...->...", np.array([1,-1]), Btilde_src_avg)
        # ------------------
        # finding the orbits
        # ------------------
        sort_q1_squared = np.argsort(q1_norm_squared)
        N_orb = np.unique(q1_norm_squared, axis=0).shape[0] # number of |q_1|^2
        q1_sorted = q1[sort_q1_squared,:]
        q1_orbits = np.split(q1_sorted, N_orb)
        Btilde_orbits = []
        for q1_orbit in q1_orbits:
            q1_orbit_keys = [vector_to_string(q1, v_type="p_i") for q1 in q1]
            orbit_idx = [np.where(np.array(momenta_keys, dtype=str) == p)[0][0] for p in q1_orbit_keys]
            # S. Burri wrote that
            # "a factor of 1/2 is included to account for double counting of diagrams due to the use of Osterwalder-Seiler fermions"
            Btilde_orbit = Q_fact*Btilde_f_avg[:,orbit_idx,...].mean(axis=-2)/2
            Btilde_orbits.append(Btilde_orbit)
        #---
        Btilde_orbits = np.array(Btilde_orbits)


