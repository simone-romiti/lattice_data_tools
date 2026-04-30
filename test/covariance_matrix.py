import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


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
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      edgecolor=edgecolor, facecolor='none', **kwargs)
    ax.add_patch(ellipse)
    return ellipse


def test_ellipse_is_added_to_axes():
    mean = [0, 0]
    cov_matrix = np.array([[3, 1], [1, 2]])
    fig, ax = plt.subplots()
    plot_confidence_ellipse(mean, cov_matrix, ax)
    assert len(ax.patches) == 1
    plt.close(fig)


def test_ellipse_center():
    mean = [2, 5]
    cov_matrix = np.array([[1, 0], [0, 1]])
    fig, ax = plt.subplots()
    ellipse = plot_confidence_ellipse(mean, cov_matrix, ax)
    assert list(ellipse.get_center()) == mean
    plt.close(fig)


def test_diagonal_cov_gives_axis_aligned_ellipse():
    """For a diagonal covariance matrix the ellipse axes should align with x/y.
    eigh returns eigenvalues ascending, so for cov=diag(4,1):
    eigenvalues = [1, 4] -> width=2*sqrt(1)=2, height=2*sqrt(4)=4.
    """
    mean = [0, 0]
    cov_matrix = np.array([[4, 0], [0, 1]])
    fig, ax = plt.subplots()
    ellipse = plot_confidence_ellipse(mean, cov_matrix, ax, n_std=1)
    np.testing.assert_allclose(ellipse.get_width(),  2 * np.sqrt(1), rtol=1e-5)
    np.testing.assert_allclose(ellipse.get_height(), 2 * np.sqrt(4), rtol=1e-5)
    plt.close(fig)

def test_n_std_scales_ellipse():
    """Doubling n_std should double width and height."""
    mean = [0, 0]
    cov_matrix = np.array([[3, 1], [1, 2]])
    fig, ax = plt.subplots()
    e1 = plot_confidence_ellipse(mean, cov_matrix, ax, n_std=1)
    e2 = plot_confidence_ellipse(mean, cov_matrix, ax, n_std=2)
    np.testing.assert_allclose(e2.get_width(), 2 * e1.get_width(), rtol=1e-5)
    np.testing.assert_allclose(e2.get_height(), 2 * e1.get_height(), rtol=1e-5)
    plt.close(fig)


def test_edgecolor():
    mean = [0, 0]
    cov_matrix = np.array([[1, 0], [0, 1]])
    fig, ax = plt.subplots()
    ellipse = plot_confidence_ellipse(mean, cov_matrix, ax, edgecolor='blue')
    assert ellipse.get_edgecolor() is not None
    plt.close(fig)


if __name__ == "__main__":
    np.random.seed(42)
    mean = [0, 0]
    cov_matrix = np.array([[3, 1], [1, 2]])

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_confidence_ellipse(mean, cov_matrix, ax, n_std=2, edgecolor='blue')

    points = np.random.multivariate_normal(mean, cov_matrix, 300)
    ax.scatter(points[:, 0], points[:, 1], s=5, alpha=0.5)

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Confidence Ellipse with Covariance Matrix")
    plt.grid()
    plt.show()
