## physical constants

import numpy as np

## masses

masses_MeV = {
    "e" : 0.5109989461,# electron
    "mu": 105.6583745, # \mu lepton
    "tau": 1776.86, # \tau lepton
    "pi": {
        "+": 139.57061, # \pi^+
        "0": 134.9770   # \pi^0
    },
    "K": {
        "+": 493.664, # K^+
        "0": 497.611  # K^0
    }
}

## units conversion

MeV_to_GeV = lambda x: x/1000.0
GeV_to_MeV = lambda x: 1000.0*x
MeV_inv_to_GeV_inv = lambda x: GeV_to_MeV(x)
GeV_inv_to_MeV_inv = lambda x: MeV_to_GeV(x)

# Conversion from fm to MeV^{-1}
hbarc_fm_MeV = np.float64(197.3269804)

fm_to_MeV_inv = lambda a: a/hbarc_fm_MeV

## electromagnetic fine structure constant
alpha_EM_inv = 137.035999084
alpha_EM = 1.0/alpha_EM_inv
## positron charge: alpha = e^2/(4*\pi)
e2 = 4.0*np.pi*alpha_EM
e = np.sqrt(e2)

## quark charges in units of the positron charge
q_u = + 2.0/3.0 # up
q_d = - 1.0/3.0 # down
q_c = + 2.0/3.0 # charm
q_s = - 1.0/3.0 # strange
q_t = + 2.0/3.0 # top
q_b = - 1.0/3.0 # bottom

