""" Routnined for plotting probability density functions (PDFs) and cumulative distribution functions (CDFs) of the data. """

import numpy as np
import matplotlib.pyplot as plt
import typing

class DistributionPlotter:
    """Class for plotting probability density functions (PDFs) and cumulative distribution functions (CDFs) of the data."""
    def __init__(self, fix, ax):
        self.fix = fix
        self.ax = ax

    def cdf(self, x: np.ndarray, color='blue', label=None):
        assert x.ndim == 1, "x should be a 1D array of values"
        N = x.shape[0]
        sorted_x = np.sort(x)
        cdf_values = np.arange(1, N+1) / N
        #
        (self.ax).plot(sorted_x, cdf_values, color=color, label=label)
    #---

    def pdf(self, x: np.ndarray, bins=typing.Literal["sqrtN"], color='blue', label=None):
        assert x.ndim == 1, "x should be a 1D array of values"
        N = x.shape[0]
        if bins == "sqrtN":
            bins = int(np.sqrt(N))
        (self.ax).hist(x, bins=bins, density=True, color=color, alpha=0.7, label=label)
    #---
