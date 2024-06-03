## routines for Leading Isospin Breaking Effects

import numpy as np

# from lattice_data_tools import effective_curves
from lattice_data_tools import constants

def get_dMpi_eff(C0, C_exch, C_handcuffs=None):
    """ 
    Single Bootstrap sample of the effective curve for the pion mass difference M_{\pi^+} - M_\{\pi^0}
    This is found from eq. 14 of https://journals.aps.org/prd/abstract/10.1103/PhysRevD.106.014502
    """
    e2 = constants.e2
    q_u, q_d = constants.q_u, constants.q_d
    R_exch = C_exch/C0
    R_handcuffs = C_handcuffs/C0
    R_tot = R_exch
    if C_handcuffs == None:
        R_tot -= R_handcuffs
    ####
    R_der = ((e2/2.0)*((q_u - q_d)**2))*np.diff(R_tot)
    return R_der
####

def tuning_dmu_uds_Edinburgh_consensus(a_GeV_inv, dMpi_plus, dMK_plus, dMK_0, dmcr_uds):
    """ 
    Linear system for physical mass tuning with the "Edinburgh consensus".
    This is done for each time --> effective mass curves for the mass counterterms
    dMpi, dMK_plus, dMK_0 are the mass slopes with respect to m_u, m_d, m_s
    dmcr_uds are the critical mass counterterms of the quarks u,d,s
    
    a_GeV_inv = lattice spacing in GeV^{-1}
    
    isoQCD point:
        M_pi^{iso} = 135.0 MeV
        M_K^{iso}  = 494.6 MeV
        
    QCD+QED point: given by the experimental masses
    """
    T_ext = dMpi_plus.shape[0]
    A = np.array([dMpi_plus, dMK_plus,  dMK_0])
    e2 = constants.e2
    dMpi_plus_exp = constants.masses_MeV["pi"]["+"] - constants.isoQCD_point["masses"]["pi"]
    dMK_plus_exp = constants.masses_MeV["K"]["+"] - constants.isoQCD_point["masses"]["K"]
    dMK_0_exp = constants.masses_MeV["K"]["0"] - constants.isoQCD_point["masses"]["K"]
    a_MeV_inv = constants.GeV_inv_to_MeV_inv(a_GeV_inv)
    dmcr_u, dmcr_d, dmcr_s = dmcr_uds
    b1 = a_MeV_inv*dMpi_plus_exp - e2*dMpi_plus - dmcr_u*dMpi_plus - dmcr_d*dMpi_plus
    b2 = a_MeV_inv*dMK_plus_exp  - e2*dMK_plus  - dmcr_u*dMK_plus                     - dmcr_s*dMK_plus
    b3 = a_MeV_inv*dMK_0_exp     - e2*dMK_0                         - dmcr_d*dMK_0    - dmcr_s*dMK_0
    b = b1+b2+b3
    dmu_uds = np.array([np.linalg.solve(A[:,:,t], b[:,t]) for t in range(T_ext)]).transpose()
    return dmu_uds
####






