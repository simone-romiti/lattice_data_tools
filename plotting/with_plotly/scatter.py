import numpy as np
import plotly.graph_objs as go

def xy(
    x: np.ndarray, y: np.ndarray, 
    title = "Plot title",
    marker = "x", color = "blue", 
    xlabel = "x", ylabel= "y",
    ex: np.ndarray = None, ey: np.ndarray = None):
    """
    Plot of x values against y values with optional error bars.

    Parameters:
    -----------
    x : np.ndarray
        Array of x values.
    y : np.ndarray
        Array of y values.
    title : str
        Title of the plot.
    marker : str
        Marker style for the scatter plot (e.g., 'circle', 'square').
    color : str
        Color for the markers.
    xlabel : str
        Label for the X-axis.
    ylabel : str
        Label for the Y-axis.
    ex : np.ndarray, optional
        Array of x-axis error values (default is None).
    ey : np.ndarray, optional
        Array of y-axis error values (default is None).

    Returns:
    --------
    fig : plotly.graph_objs.Figure
        A Plotly figure object with the scatter plot and optional error bars.

    Example usage:
    --------------
    # Plot without error bars:
    xy(x, y, 'Plot Title', 'circle', 'blue', 'X-Axis', 'Y-Axis')

    # Plot with error bars on x-axis:
    xy(x, y, 'Plot Title', 'circle', 'blue', 'X-Axis', 'Y-Axis', ex=exs)

    # Plot with error bars on y-axis:
    xy(x, y, 'Plot Title', 'circle', 'blue', 'X-Axis', 'Y-Axis', ey=eys)

    # Plot with error bars on both x and y axes:
    xy(x, y, 'Plot Title', 'circle', 'blue', 'X-Axis', 'Y-Axis', ex=exs, ey=eys)
    """
    # Define scatter plot with error bars if provided
    error_x = dict(type='data', array=ex) if ex is not None else None
    error_y = dict(type='data', array=ey) if ey is not None else None
    
    fig = go.Figure(
        data=go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(symbol=marker, color=color),  # Set color for the markers
            error_x=error_x,
            error_y=error_y
        )
    )
    # Update the layout for labels, title, and axis color
    fig.update_layout(
        title=title,  # Set the title of the plot
        xaxis_title=xlabel,  # X-axis label
        yaxis_title=ylabel,  # Y-axis label
        showlegend=False,    # Hide the legend if not needed
    )
    
    return fig
#---
