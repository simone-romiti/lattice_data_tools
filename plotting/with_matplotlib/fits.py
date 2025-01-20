import numpy as np
import matplotlib.pyplot as plt

def bts_weighted_fits(x, y_models, weights, n_samples=10, color='blue', marker_size=0.1):
    """Plots of several fit models for a bootstrap analysis

    Args:
        x (np.ndarray): N_pts values of x
        y_models (list): list of numpy arrays of shape (N_bts, N_pts)
        weights (np.ndarray): array fo weights of each model
        n_samples (int): number of bootstraps to sample
        color (str, optional): Color of the traces. Defaults to 'blue'.
    """
    # Normalize weights to range between 0 and 1 for alpha intensities
    weights = np.array(weights)
    weights = weights / weights.max()  # Normalization for alpha scaling
    
    fig, ax = plt.subplots()
    
    # Plot each model curve with alpha determined by its normalized weight
    for y, weight in zip(y_models, weights):
        for i in range(n_samples):
            ax.scatter(x, y[i,:], color=color, alpha=weight, linestyle="None", marker=".", s=marker_size)
    
    return fig, ax
#---

if __name__ == "__main__":
    x = np.linspace(0, 3, 100)
    N_bts = 500
    y_models = [np.array([xi**(shift*(1+np.random.normal(loc=0.0, scale=0.01, size=N_bts))) for xi in x]).transpose() for shift in np.linspace(0.1, 1, 5)]
    weights = [0.2, 0.5, 0.8, 1.0, 0.6]
    fig, ax = bts_weighted_fits(x, y_models, weights, color='blue')
    plt.show()



