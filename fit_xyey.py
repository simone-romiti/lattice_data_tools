
import numpy as np
import scipy.optimize as opt


#' fit a function f(x) with errors on y (and not on x)
#' N_pts = number of points
#' x = np.array of shape (N_pts)
#' y, dy = list of length N_pts
#' guess = list of N_par guesses (for the ansatz)
#' maxiter = maximum number of iterations for the minimizer
def fit_xyey(ansatz, x: np.ndarray, y: np.ndarray, ey: np.ndarray, guess: np.ndarray, maxiter = 10000, method = "BFGS"):
    assert (x.shape == y.shape and y.shape == ey.shape)
    N_pts = y.shape[0] # number of data points

    N_par = guess.shape[0] # number of parameters of the fit
    N_dof = N_pts - N_par # number of degrees of freedom

    # chi square residual function
    # NOTE: "p" is the array of the external parameters of the function
    def ch2(p):
        res = np.array([(y[i] - ansatz(x[i], p)) / ey[i] for i in range(N_pts)])
        return np.sum(res**2)
    ####

    guess = list(guess)
    mini = opt.minimize(fun = ch2, x0 = guess, method = method)

    ch2_value = ch2(mini.x)

    res = dict({})

    res["ansatz"] = ansatz
    res["N_par"] = N_par
    res["par"] = mini.x
    res["ch2"] = ch2_value
    res["dof"] = N_dof ## degrees of freedom
    res["ch2_dof"] = ch2_value / N_dof

    return(res)
####


