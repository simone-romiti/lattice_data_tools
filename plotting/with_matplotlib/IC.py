
import matplotlib.pyplot as plt
# import numpy as np

from lattice_data_tools.model_averaging.IC import with_CDF 

class FromBootstraps:
    @staticmethod
    def plot_cdf(y, P, title=None):
        Q = with_CDF.get_quantiles(y=y, P=P)
        y16 = Q["16%"]
        y50 = Q["50%"]
        y84 = Q["84%"]
        fig, ax = plt.subplots()
        ax.plot(y, P, color="black", label="Model average C.D.F")
        ax.axvline(y50, color="purple", linestyle="-", alpha=0.5, label=f"median: $y_M=${y50:.3e}")
        ax.axvline(y16, color="purple", linestyle="--",  alpha=0.25, label=f"16%: $y_L=${y16:.3e}")
        ax.fill_betweenx([0,1], y16, y50, color="pink", alpha=0.5, label=f"$\Delta y_L=${(y50-y16):.3e}")
        ax.axvline(y84, color="purple", linestyle=":",  alpha=0.25, label=f"84%: $y_R=${y84:.3e}")
        ax.fill_betweenx([0,1], y50, y84, color="pink", alpha=0.5, label=f"$\Delta y_R=${(y84-y50):.3e}")
        ax.set_ylabel("P(y)")
        ax.set_xlabel("y")
        title_str = "Cumulative density function from bootstraps" if title is None else title
        ax.set_title(title_str)
        ax.set_ylim(0,1)
        ax.legend(loc="best")
        return fig, ax
