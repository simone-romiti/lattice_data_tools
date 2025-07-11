# Importing the necessary libraries

import numpy as np
from scipy.stats import skewnorm
import matplotlib.pyplot as plt

import sys
sys.path.append('../../')

from lattice_data_tools import uwerr

np.random.seed(123)

N = 100000 # number of points
mu, sigma = 0.0, 3.0


def generate_mc_history(n_steps=10000, dt=0.01, theta_start=1.0, theta_end=0.01):
    """
    Generate a Monte Carlo history with slowly vanishing autocorrelation.
    
    Parameters:
        n_steps (int): Number of time steps.
        dt (float): Time step size.
        theta_start (float): Initial value of the decay rate.
        theta_end (float): Final value of the decay rate.
    
    Returns:
        np.ndarray: The Monte Carlo history.
    """
    history = np.zeros(n_steps)
    noise = np.random.normal(0, 1, n_steps)
    
    # Create a decay schedule for theta
    theta_schedule = np.linspace(theta_start, theta_end, n_steps)

    for t in range(1, n_steps):
        theta_t = theta_schedule[t]
        # OU process with decaying theta
        history[t] = history[t - 1] * (1 - theta_t * dt) + np.sqrt(2 * theta_t * dt) * noise[t]
    
    return history

# Generate and plot
x = generate_mc_history()

MC_history = uwerr.uwerr_primary(x, output_file="./MC_history.pdf")

print("done")