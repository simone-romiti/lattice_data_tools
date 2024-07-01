import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import root, fsolve

def get_m_eff_log(C: np.ndarray) -> np.ndarray:
    """Effective mass curve

    log: log(C[t]/C[t+1])

    Args:
        C (np.ndarray): Correlator C(t)

    Returns:
        np.ndarray: M_eff(t) = log(C(t)/C(t+1))
    """
    T = C.shape[0] ## temporal extent
    return np.array([np.log(C[t]/C[t+1]) if C[t]/C[t+1] > 0 else 0.0 for t in range(T-1)])
####

def get_m_eff_bkw(C: np.ndarray, T: int, p: int):
    """ 
    
    Effective mass including the backward signal 
    see eq. 6.57 of Gattringer & Lang
    
    """
    if T == None:
        raise ValueError("You need to pass T explicitly as an argument!")
    ####
    if p == +1:
        form = lambda x: np.cosh(x)
    elif p == -1:
        form = lambda x: np.sinh(x)
    ####
    T_ext = C.shape[0] ## temporal extent
    T_half = int(T/2)
    t_eff = np.array([t for t in range(T_ext-1)])
    m_eff = np.zeros(shape=(T_ext-1))
    for t in t_eff:
        r = C[t]/C[t+1]
        def func(m_t):
            r_th = form(m_t*(T_half-t))/form(m_t*(T_half-t-1))
            return (r_th - r)
        ####
        if r>=0:
            m_guess = np.log(r)
        else:
            m_guess = 0.0 ## the statistical noise doesn't allow for a better guess
        if t >= T_half:
            m_guess *= -1
        ####
        m_eff[t] = fsolve(func=func, x0=m_guess)[0]
    ####
    return m_eff
####

def get_m_eff(C: np.ndarray, strategy: str, T=None) -> np.ndarray:
    """Effective mass curve from the correlator

    Args:
        C (np.ndarray): correlator C(t)
        strategy (str): computation strategy. Supported: ["log","cosh","sinh"]

    Raises:
        ValueError: if strategy is not in the list of supported types

    Returns:
        np.ndarray: array of effective mass values M_eff(t)
    """
    if strategy == "log":
        return get_m_eff_log(C)
    elif strategy == "cosh":
        return get_m_eff_bkw(C=C, T=T, p=+1)
    elif strategy == "sinh":
        return get_m_eff_bkw(C=C, T=T, p=-1)
    else:
        raise ValueError("Illegal strategy for calculation of effective mass: {strategy}".format(strategy=strategy))
    ####
####

def get_dm_eff_log(C0: np.ndarray, dC: np.ndarray) -> np.ndarray:
    """Effective mass correction assuming C(t) = A*exp(-M*t)

    Args:
        C0 (np.ndarray): Correlator C(t)
        dC (np.ndarray): correction to the correlator C_0(t)

    Returns:
        np.ndarray: dM_eff(t) = - [R(t+1) - R(t)], where R(t) = dC(t)/C0(t)
    """
    # T = C.shape[0] ## temporal extent
    R = dC/C0
    return np.diff(R)
####

def get_dm_eff_bkw(C0: np.ndarray, dC: np.ndarray, M0_eff: np.ndarray, T: int, p: int):
    """ 
    
    Effective mass correction including the backward signal 
    see eq. 11 of https://journals.aps.org/prd/abstract/10.1103/PhysRevD.106.014502
    
    C0 (np.ndarray): Correlator C(t)
    dC (np.ndarray): correction to the correlator C_0(t)
    M0_eff (np.ndarray): 
        effective mass curve for the C_0 correlator.
        NOTE: It is not computed automatically internally because 
              the user may want to fix it for all "t". 
              In this case, pass M0_eff as a constant array
    """
    if T==None:
        raise ValueError("You need to pass T explicitly as an argument!")
    ####
    if p==+1:
        hyp_fun = lambda z: np.tanh(z)
    elif p==-1:
        hyp_fun = lambda z: 1.0/np.tanh(z)
    ####
    F = lambda x, M: x*hyp_fun(M*x) - (x+1)*hyp_fun(M*(x+1))
    T_ext = C0.shape[0] ## temporal extent
    T_half = int(T/2)
    t_eff = np.array([t for t in range(T_ext-1)])
    R = dC/C0
    res = np.diff(R)/F(T_half-t_eff-1, M0_eff)
    return res
####


def get_dm_eff(C0: np.ndarray, dC: np.ndarray, M0_eff: np.ndarray, strategy: str, T=None) -> np.ndarray:
    """Effective curve for a mass correction the correlator

    Args:
        C0 (np.ndarray): correlator C_0(t) in the free theory (e.g. isoQCD)
        dC (np.ndarray): correction to the correlator C_0(t)
        strategy (str): 
            computation strategy. 
            Supported: ["log","tanh","coth"]

    Raises:
        ValueError: if strategy is not in the list of supported types

    Returns:
        np.ndarray: array of effective mass values M_eff(t)
    """
    if strategy == "log":
        return get_dm_eff_log(C0=C0, dC=dC)
    elif strategy == "tanh":
        return get_dm_eff_bkw(C0=C0, dC=dC, M0_eff=M0_eff, T=T, p=+1)
    elif strategy == "coth":
        return get_dm_eff_bkw(C0=C0, dC=dC, M0_eff=M0_eff, T=T, p=-1)
    else:
        raise ValueError("Illegal strategy for calculation of effective mass correction: {strategy}".format(strategy=strategy))
    ####
####


def fit_eff_curve(y_eff: np.ndarray, dy_eff: np.ndarray) -> float:
    """Fit the effective curve (in the plateau) to a constant

    Args:
        y_eff (np.ndarray): array of values for y_eff(t) ONLY in the plateau
        dy_eff (np.ndarray): uncertainty on y_eff(t) in that plateau interval
        
        NOTE: tmin and tmax are not passed because it is assumed assumed to have slices the arrays in the plateau interval

    Returns:
        float: best fit value of y_eff
    """
    T = y_eff.shape[0]
    t = [i for i in range(T)] ## dummy variables, fitting to a constant
    ansatz = lambda x, m0: m0
    par, cov = curve_fit(ansatz, t, y_eff, sigma=dy_eff, p0=np.average(y_eff))
    return par[0]
####


def fit_eff_mass(m_eff: np.ndarray, dm_eff: np.ndarray) -> float:
    """ alias for the fit to a generic effective curve """
    return fit_eff_curve(y_eff=m_eff, dy_eff=dm_eff)
####


## example code
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    T = 24
    t = np.array([i for i in range(T)])
    m = 0.15
    C = np.random.normal(1.0, 0.001, T)*np.exp(-m*t)

    m_eff = get_m_eff(C, strategy="log")[4:16]

    plt.plot(m_eff)
    plt.show()

    print(fit_eff_mass(m_eff=m_eff, dm_eff=0.01*m_eff))
####
