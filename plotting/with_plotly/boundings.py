""" IN PROGRESS

 This script contains routines for fitting curves where the bouding method has been used
"""

import numpy as np
from typing import List
import plotly.io as pio
import plotly.graph_objects as go

# Golden ratio for figures aspect ratios
phi_golden = (1 + np.sqrt(5)) / 2

def get_figure(
    t: np.ndarray, y: np.ndarray, dy: np.ndarray, 
    title: str, yaxis_title: str, labels: List[str], legend_title: str,
    tmin = None, tmax=None,
    ymin = None, ymax=None,
    t1_fit = None, t2_fit=None,
    y_fit=None, dy_fit=None,
    width=phi_golden*500, height=500):
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
            line=dict(color="red"),
        ))

        legendgroup = label[i]
        if not ((t1_fit is None) or (t2_fit is None) or (y_fit is None) or (dy_fit is None)):
            nt_fit = t2_fit-t1_fit+1 ## number of times from the fit
            ti_fit = np.arange(t1_fit, t1_fit + nt_fit)
            
            fig.add_trace(go.Scatter(
                x=ti_fit,
                y=np.full(nt_fit, y_fit[i]),
                mode='lines+markers',
                name=labels[i],
                line=dict(color="gray"),
                legendgroup=f"Fit of {labels[i]}",
                showlegend=True
            ))
            y_upper = y_fit[i]+dy_fit[i]
            y_lower = y_fit[i]-dy_fit[i]
            fig.add_trace(go.Scatter(
                x=ti_fit, y=y_upper,
                fill=None,
                mode='lines',
                name='DummyName',
                showlegend=False,
                legendgroup="Fit",
                line=dict(color="gray"),
            ))

            fig.add_trace(go.Scatter(
                x=ti_fit, y=y_lower,
                fill='tonexty',  # fill area between y2 and y1
                mode='lines',
                name='DummyName',
                line=dict(color="gray"),
                showlegend=False,
                legendgroup="Fit"
            ))


    # Updating the layout
    fig.update_layout(
        width=widht,
        height=height,
        title=title,
        legend_title=legend_title,
        xaxis={
            'title': 't_c', 
            'ticks': 'inside',
            "range": [tmin, tmax],
            "tickformat": ',d'
        },
        yaxis={
            'title': yaxis_title, 
            'tickformat':'.6e', 
            'ticks': 'inside',
            "range": [ymin, ymax]
        }
    )
    return fig
####





 