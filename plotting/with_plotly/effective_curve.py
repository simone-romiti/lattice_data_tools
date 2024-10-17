
import numpy as np
from typing import List
import plotly.io as pio
import plotly.graph_objects as go

# Golden ratio for figures aspect ratios
phi_golden = (1 + np.sqrt(5)) / 2

def get_figure(
    t: np.ndarray, y: np.ndarray, dy: np.ndarray, 
    title: str, yaxis_title: str, labels: List[str], legend_title: str,
    colors_rgb=List[str],
    tmin = None, tmax=None,
    ymin = None, ymax=None,
    t1_fit = None, t2_fit=None,
    y_fit=None, dy_fit=None,
    width=phi_golden*500, height=500):
    """
    Plot the correlators effective curves y_i(t) as a function of t (in lattice units)
    If  t1_fit, t2_fit, y_fit, dy_fit are passed, the fit band is plotted for each trace.

    """
    if tmin is None:
        tmin = np.min(t)
    if tmax is None:
        tmax = np.max(t)
    if ymin is None:
        ymin = np.min(y)
    if ymax is None:
        ymax = np.max(y)

    fig = go.Figure()
    n_curves = y.shape[0]
    for i in range(n_curves):
        fig.add_trace(go.Scatter(
            x=t,
            y=y[i,:],
            mode='lines+markers',
            name=labels[i],
            error_y=dict(
                type='data',
                array=dy[i,:],
                visible=True
            ),
        ))

        legendgroup = labels[i]
        if not ((t1_fit is None) or (t2_fit is None) or (y_fit is None) or (dy_fit is None)):
            T_ext = t2_fit[i]-t1_fit[i]+1
            # Create trace for the upper boundary
            y_upper = y_fit[i]+dy_fit[i]
            y_lower = y_fit[i]-dy_fit[i]
            fig.add_trace(go.Scatter(
                x=np.arange(t1_fit[i], t2_fit[i]+1),
                y=np.full(shape=(T_ext), fill_value=y_upper),
                mode='lines',
                line=dict(color="rgb({rgb})".format(rgb=colors_rgb[i])),
                showlegend=False,  # Hide legend for upper line
                legendgroup=legendgroup
            ))

            # Create trace for the lower boundary and fill the area to the next y (upper bound)
            fig.add_trace(go.Scatter(
                x=np.arange(t1_fit[i], t2_fit[i]+1),
                y=np.full(shape=(T_ext), fill_value=y_lower),
                mode='lines',
                line=dict(color="rgb({rgb})".format(rgb=colors_rgb[i])),
                fill='tonexty',  # Fill the area between this trace and the previous one
                fillcolor="rgba({rgb}, 0.1)".format(rgb=colors_rgb[i]), 
                opacity=0.1,  # Color for the filled area
                showlegend=True,  # Show legend for the filled band
                legendgroup=legendgroup,
                name=f"Fit of {labels[i]}",

            ))

            # Optionally, plot the average line as a separate trace (optional for clearer visualization)
            fig.add_trace(go.Scatter(
                x=np.arange(t1_fit[i], t2_fit[i]+1),
                y=np.full(shape=(T_ext), fill_value=y_fit),
                mode='lines+markers',
                line=dict(color="rgb({rgb})".format(rgb=colors_rgb[i])),
                showlegend=False,
                legendgroup=legendgroup
            ))
    #-------
    # updating the layout
    fig.update_layout(
        font_family="Ubuntu Mono",
        width=width,
        height=height,
        title=title,
        legend_title=legend_title,
        xaxis={
            'title': '$t/a$', 
            'ticks': 'inside',
            "range": [tmin, tmax],
            "tickformat": ',d'
        },
        yaxis={
            'title': yaxis_title, 
            'tickformat':'.8e', 
            'ticks': 'inside',
            "range": [ymin, ymax]
        },
    )
    return fig
#---


