
import matplotlib.pyplot as plt
# import numpy as np

class FromBootstraps:
    @staticmethod
    def plot_cdf(y, P, title=None):
        y16 = y[P <= 0.16][-1]
        y50 = y[P <= 0.50][-1]
        y84 = y[P <= 0.84][-1]
        fig, ax = plt.subplots()
        ax.plot(y, P, color="black", label="C.D.F. from AIC")
        ax.axvline(y50, color="purple", linestyle="-", label=f"median: {y50:.4e}")
        ax.fill_betweenx(P, y16, y84, color="pink", alpha=0.5, label=f"16%: {y16:.4e}, {(y50-y16):.4e} \n84%: {y84:.4e}, {(y84-y50):.4e}")
        ax.set_ylabel("P(y)")
        ax.set_xlabel("y")
        title_str = "Cumulative density function from bootstraps" if title is None else title
        ax.set_title(title_str)
        ax.set_ylim(0,1)
        ax.legend(loc="best")
        return fig, ax
