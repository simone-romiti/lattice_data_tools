import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def plot_confidence_ellipse(mean, cov_matrix, ax, n_std=1.96, edgecolor='red', **kwargs):
    """
    Plot a confidence ellipse based on the covariance matrix.

    Parameters:
    - mean: The mean or center of the data (2D point, as an array-like or list).
    - cov_matrix: The 2x2 covariance matrix of the data.
    - ax: The matplotlib axis object where the ellipse will be drawn.
    - n_std: The number of standard deviations for the ellipse's radius.
    - edgecolor: The edge color of the ellipse.
    - **kwargs: Additional keyword arguments for `Ellipse`.

    Returns:
    - A matplotlib ellipse patch.
    """
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Compute the angle of rotation of the ellipse in degrees
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

    # The width and height of the ellipse based on the eigenvalues and n_std scaling
    width, height = 2 * n_std * np.sqrt(eigenvalues)

    # Create an ellipse patch
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor=edgecolor, facecolor='none', **kwargs)

    # Add the ellipse to the plot
    ax.add_patch(ellipse)

    return ellipse

# Example usage
mean = [0, 0]  # Center of the ellipse
cov_matrix = np.array([[3, 1], [1, 2]])  # Covariance matrix

fig, ax = plt.subplots(figsize=(6, 6))

# Plot the ellipse
plot_confidence_ellipse(mean, cov_matrix, ax, n_std=2, edgecolor='blue')

# Scatter plot for reference points
points = np.random.multivariate_normal(mean, cov_matrix, 300)
ax.scatter(points[:, 0], points[:, 1], s=5, alpha=0.5)

# Set plot limits and aspect
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_aspect('equal')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Confidence Ellipse with Covariance Matrix")
plt.grid()
plt.show()
