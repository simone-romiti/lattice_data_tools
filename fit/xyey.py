"""Fit of a 1d function of 1 variable

.. ldt-id:: FIT-xyey-module
"""


import numpy as np
from lattice_data_tools.fit.trajectory import fit_trajectory

def fit_xyey(
    ansatz, 
    x: np.ndarray, y: np.ndarray, ey: np.ndarray, 
    guess: np.ndarray, 
    method = "BFGS",
    Cov_y_inv = None
    ):
    """Fit of y=f(x), there f: \\mathbb{R}^1 \\to \\mathbb{R}^1

    Args:
        ansatz : ansatz taking a float and returning a float
        x (np.ndarray): 1d array of x values
        y (np.ndarray): 1d array of y values
        ey (np.ndarray):1d array of errors on the values
        guess (np.ndarray): 1d array fo guesses for the ansatz_
        maxiter (int, optional): _description_. Defaults to 10000.
        method (str, optional): _description_. Defaults to "BFGS".
        Cov_y_inv: if != None is an estimate of the inverse of the covariance matrix of the y points (e.g. from the bootstrap samples)

    .. ldt-id:: FIT-fit_xyey
    """
    def ansatz_casted(x, p):
        return np.array([ansatz(x[0], p)])
    #---
    xp = np.array([x]).transpose()
    ex = np.zeros(shape=xp.shape)
    yp = np.array([y]).transpose()
    eyp = np.array([ey]).transpose()
    C_inv_full = Cov_y_inv
    if not (Cov_y_inv is None):
        Nx = xp.flatten().shape[0]
        Ny = yp.flatten().shape[0]
        assert(Cov_y_inv.shape == (Ny, Ny))
        C_inv_full = 0.0*np.eye(N=(Nx+Ny)) # ficticious covariance matrix to pass to the general routine
        C_inv_full[Nx:,Nx:] = Cov_y_inv
    #---
    res = fit_trajectory(
        ansatz=ansatz_casted, 
        x=xp, ex=ex, y=yp, ey=eyp, 
        guess=guess, 
        method=method,
        Cov_inv=C_inv_full)
    res["ansatz"] = ansatz
    return res
#---


def polynomial_fit_xyey(
    N_deg: int,
    x: np.ndarray, y: np.ndarray, ey: np.ndarray, 
    Cov_y_inv = None
    ):
    """
    Exact formula for the fit of a polynomial of degree N_deg, with N_deg+1 parameters:

    f(x) = \\sum_{k=0}^{N_deg} \\alpha_k x^k

    The formula can be obtained after some algebra, by minimizing the chi^2 = (y - ansatz(x))^T C^{-1} (y - ansatz(x)).
    A reference is eq. 13 of https://people.duke.edu/~hpgavin/SystemID/CourseNotes/linear-least-squares.pdf

    N_deg: degree of the polynomial (--> number of parameters = N_deg+1)
    x: 1d array of x values
    y: 1d array of y values
    ey: 1d array of errors on the y values
    Cov_y_inv: if != None is an estimate of the inverse of the covariance matrix of the y points

    .. ldt-id:: FIT-polynomial_fit_xyey
    """
    assert(len(x.shape) == 1)
    assert(len(y.shape) == 1)
    assert(len(ey.shape) == 1)
    assert(x.shape == y.shape)
    assert(x.shape == ey.shape)
    N_pts = x.shape[0] # number of points
    N_par = (N_deg+1) # number of parameters of the fit, i.e. the number of coefficients of the polynomial
    #---
    C_inv = np.diag(1.0/ey**2) if Cov_y_inv is None else Cov_y_inv
    # Vandermonde matrix, with increasing powers of x, i.e. B[i,j] = x[i]**j, with j=0,...,N_deg
    B = np.vander(x, N=N_deg+1, increasing=True)
    Left = (B.T) @ C_inv @ B 
    Right = (B.T) @ C_inv @ y
    alpha = np.linalg.inv(Left) @ Right # best fit parameters
    #---
    y_fit = B @ alpha # best fit values of y
    eps = y - y_fit # residuals
    ch2 = eps.T @ C_inv @ eps  # chi^2 of the fit
    n_dof = N_pts - N_par # number of degrees of freedom
    ch2_dof = ch2/n_dof  # chi^2 per degree of freedom
    res = {
        'ansatz': lambda x, p: np.poly1d(p[::-1])(x), # NOTE: np.poly1d takes parameters in descending order of powers
        "N_deg": N_deg,
        'N_par': N_par, 
        'N_pts': N_pts,
        'par': alpha, 
        'ch2': ch2, 
        'N_dof': n_dof, 
        'ch2_dof': ch2_dof}
    return res
