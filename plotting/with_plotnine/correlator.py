import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_point, geom_line, geom_errorbar, scale_y_continuous, theme, labs, ggtitle, scale_x_continuous, theme_classic
from typing import List

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
    Plot the correlators C_i(t) as a function of t (in lattice units) using plotnine.

    This version uses the `plotnine` package to plot the correlators with error bars.
    """
    
    if tmin is None:
        tmin = np.min(t)
    if tmax is None:
        tmax = np.max(t)
    if Cmin is None:
        Cmin = np.min(C)
    if Cmax is None:
        Cmax = np.max(C)

    # Convert the data to a long-format dataframe for plotnine
    df_list = []
    n_corr = C.shape[0]
    for i in range(n_corr):
        df_temp = pd.DataFrame({
            't': t,
            'C': C[i, :],
            'dC': dC[i, :],
            'label': labels[i]
        })
        df_list.append(df_temp)

    df = pd.concat(df_list)

    # Create the base plot
    p = (
        ggplot(df, aes(x='t', y='C', color='label', group='label'))
        + geom_point()
        + geom_line()
        + geom_errorbar(aes(ymin='C - dC', ymax='C + dC'), width=0.2)
        + theme_classic()
        + scale_x_continuous(limits=(tmin, tmax))
        + scale_y_continuous(limits=(Cmin, Cmax), labels='scientific')
        + ggtitle(title)
        + labs(x="t/a", y="C(t)", color=legend_title)
        + theme(figure_size=(width / 100, height / 100), legend_position='right')
    )

    # Add log scale option if requested
    if log_scale_button:
        p += scale_y_continuous(trans='log10')

    return p
