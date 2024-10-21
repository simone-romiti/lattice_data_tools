import numpy as np
import matplotlib.pyplot as plt

def xy(
    x: np.ndarray, y: np.ndarray, 
    title: str,
    marker: str, color: str, 
    xlabel: str, ylabel: str,
    x_error: np.ndarray = None, y_error: np.ndarray = None):
    """
    Plot of x values against y values with optional error bars (using Matplotlib).

    Parameters:
    -----------
    x : np.ndarray
        Array of x values.
    y : np.ndarray
        Array of y values.
    title : str
        Title of the plot.
    marker : str
        Marker style for the scatter plot (e.g., 'o' for circles, 's' for squares).
    color : str
        Color for the markers.
    xlabel : str
        Label for the X-axis.
    ylabel : str
        Label for the Y-axis.
    x_error : np.ndarray, optional
        Array of x-axis error values (default is None).
    y_error : np.ndarray, optional
        Array of y-axis error values (default is None).

    Returns:
    --------
    fig, ax : tuple
        A tuple containing the Matplotlib figure and axes objects.

    Example usage:
    --------------
    # Plot without error bars:
    xy(x, y, 'Plot Title', 'o', 'blue', 'X-Axis', 'Y-Axis')

    # Plot with error bars on x-axis:
    xy(x, y, 'Plot Title', 'o', 'blue', 'X-Axis', 'Y-Axis', x_error=x_errors)

    # Plot with error bars on y-axis:
    xy(x, y, 'Plot Title', 'o', 'blue', 'X-Axis', 'Y-Axis', y_error=y_errors)

    # Plot with error bars on both x and y axes:
    xy(x, y, 'Plot Title', 'o', 'blue', 'X-Axis', 'Y-Axis', x_error=x_errors, y_error=y_errors)
    """
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot with optional error bars
    ax.errorbar(
        x, y, 
        xerr=x_error, yerr=y_error,  # Error bars on x and/or y
        fmt=marker, color=color,  # Marker and color style
        ecolor='gray', elinewidth=1, capsize=2  # Error bar styling
    )

    # Set labels and title
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Return figure and axes
    return fig, ax
#---
