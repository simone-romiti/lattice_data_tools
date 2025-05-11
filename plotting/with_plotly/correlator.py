
import numpy as np
from typing import List
import plotly.io as pio
import plotly.graph_objects as go

# Golden ratio for figures aspect ratios
phi_golden = (1 + np.sqrt(5)) / 2


def get_figure(
    t: np.ndarray, C: np.ndarray, dC: np.ndarray, 
    title: str, labels: List[str], legend_title:str,
    tmin = None, tmax=None,
    Cmin = None, Cmax=None,
    width=phi_golden*500, height=500,
    log_scale_button=True
    ): 
    """
    Plot the correlators C_i(t) as a function of t (in lattice units)
    

    If `log_scale_button==True` a button is placed in the figure to switch to log scale.
    """
    if tmin is None:
        tmin = np.min(t)
    if tmax is None:
        tmax = np.max(t)
    if Cmin is None:
        Cmin = np.min(C)
    if Cmax is None:
        Cmax = np.max(C)

    fig = go.Figure()

    n_corr = C.shape[0]
    for i in range(n_corr):    
        # Add a trace to the figure
        fig.add_trace(go.Scatter(
            x=t,
            y=C[i,:],
            mode='lines+markers',
            name=labels[i],
            error_y=dict(
                type='data',
                array=dC[i,:],
                visible=True
            )
        ))
    ####
    # Updating the layout
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
        'title': '$C(t)$',
        'tickformat': '.8e',
        'ticks': 'inside',
        "range": [Cmin, Cmax]
    },
    updatemenus=[
        dict(
            x=0.95,
            y=0.95,
            buttons=list([
                dict(label="Linear",
                        method="relayout",
                        args=[{"yaxis.type": "linear"}]),
                dict(label="Log",
                        method="relayout",
                        args=[{"yaxis.type": "log"}])
            ])
        )
    ]
    )
    #
    return fig
#---


