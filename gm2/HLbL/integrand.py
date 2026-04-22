""" Routines involved in the calculation of the HLbL contribution to g-2 in position space """

import numpy as np
from scipy.integrate import simpson
from typing import Literal


def integrated_curve(abs_y_fm: np.ndarray, y3fy: np.ndarray, delta_abs_y: float, integration_scheme: Literal["trapezoidal", "simpson"]):
    """_summary_

    Args:
        abs_y_fm (np.ndarray): |y| in fm
        y3fy (np.ndarray): Integrand for each |y|, already including |y|^3 (NOTE: sometimes it is called simply f(|y|))
        delta_abs_y (np.ndarray): \\Delta |y|
        integration_scheme: numerical integration scheme

    Returns:
        _type_: _description_
    """
    assert (len(abs_y_fm.shape)==1)
    assert (abs_y_fm.shape == y3fy.shape)
    if integration_scheme == "trapezoidal":
        return np.trapezoid(y=y3fy, x=abs_y_fm, dx=delta_abs_y)
    elif integration_scheme == "simpson":
        return simpson(y=y3fy, x=abs_y_fm, dx=delta_abs_y)
#-------

