# Testing the Akaike Information Criterion of a set of fits to determine the systematic ans statistical error
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import sys
sys.path.append('../../')

from lattice_data_tools.plotting.with_matplotlib.fits import bts_weighted_fits
from lattice_data_tools.model_averaging.AIC import get_P_from_bootstraps, get_P

# Seed for reproducibility
np.random.seed(42)

# Define parameters
a_values = np.linspace(0.0001, 0.01, 10)  # Avoid a=1 due to log(1) = 0
true_function = lambda a: a**2 + a**4 + a**2 / np.log(a)
noise_amplitude = 0.1

# Generate data with noise
N_bts = 1000
y_data = [true_function(ai) *(1+ noise_amplitude * np.random.normal(size=N_bts)) for ai in a_values]
y_data = np.array(y_data).transpose()

def f1(a, c1, c2):
    return c1 * a + c2 * a**2 / np.log(a)

def f2(a, c1, c2):
    return c1 * a**2 + c2 * a**4

def f3(a, c1, c2):
    return c1 * a + c2 * a/np.log(a)

# Fit the data to both models
y_fit_1 = []
y_fit_2 = []
y_fit_3 = []

a_dense = np.linspace(np.min(a_values), np.max(a_values), 100)
for i in range(N_bts):
    params_1, _ = curve_fit(f1, a_values, y_data[i])
    params_2, _ = curve_fit(f2, a_values, y_data[i])
    params_3, _ = curve_fit(f3, a_values, y_data[i])

    # Generate fitted data
    y_fit_1.append(f1(a_dense, *params_1))
    y_fit_2.append(f2(a_dense, *params_2))
    y_fit_3.append(f3(a_dense, *params_3))

y_fit_1 = np.array(y_fit_1)
y_fit_2 = np.array(y_fit_2)
y_fit_3 = np.array(y_fit_3)

n_samples = 100

y_models = [y_fit_1, y_fit_2, y_fit_3]
y_extr = np.array([y_model[:,-1] for y_model in y_models]).transpose() ## extrapolation at a=0

weights = np.array([0.1, 0.4, 0.2])
fig1, ax1 = bts_weighted_fits(x=a_dense, y_models=y_models, weights=weights, n_samples=n_samples)

# plt.show()
plt.close()

AIC_pdf = get_P_from_bootstraps(y=y_extr, w=weights, lam=1.0)

# plt.plot(AIC_pdf["y"], AIC_pdf["P"])

y_values = AIC_pdf["y"]
N_pts = y_values.shape[0] 

fig2, ax2 = plt.subplots()
bin_counts, bin_edges, _ = ax2.hist(y_values, bins=int(np.sqrt(N_pts)), facecolor='g')
max_hist_frequency = np.max(bin_counts)/np.sum(bin_counts)

cdf = AIC_pdf["P"]


# fig2, ax2 = plt.subplots()
# y_range = np.arange(-0.0003, 0.0003, 0.000001)
# AIC_pdf = get_P(y=y_range, w=weights, m=np.average(y_extr, axis=0), sigma=np.std(y_extr, axis=0, ddof=1), lam=1.0)
# ax2.plot(y_range[:-1], np.diff(AIC_pdf)/np.diff(y_range))


# plt.show()

plt.close()

import matplotlib.pyplot as plt


phi_golden = (1 + np.sqrt(5))/2

# Create a new figure with two subplots
fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3*phi_golden, 1]})  # Adjust size as needed

# ** Rotated Histogram **
# max_hist_frequency = 0.0
for patch in ax2.patches:  # Iterate through the original histogram's bars
    x = patch.get_x()
    width = patch.get_width()
    height = patch.get_height()
    # print(height, N_pts)
    # max_hist_frequency = max(max_hist_frequency, height/N_pts)
    ax_hist.barh(x + width / 2, height, height=width, color=patch.get_facecolor(), edgecolor=patch.get_edgecolor())  # Rotate as horizontal


# Set the x-axis (vertical in the histogram) limits to match the scatter plot's y-axis
scatter_y_limits = ax1.get_ylim()  # Get the y-axis limits from the scatter plot
ax_hist.set_ylim(scatter_y_limits)  # Apply them as the x-axis limits to the histogram
ax_hist.set_xlim(left=0.0, right=250)

# Adjust labels and title for the rotated histogram
ax_hist.set_title("Rotated Histogram")
ax_hist.set_xlabel(ax2.get_ylabel())  # Original histogram's y-label becomes x-label
ax_hist.set_ylabel(ax2.get_xlabel())  # Original histogram's x-label becomes y-label

# ** Scatter Plot **
# Extract all scatter plot points from ax1
for collection in ax1.collections:
    offsets = collection.get_offsets()  # Get scatter plot data points
    scatter_x = offsets[:, 0]
    scatter_y = offsets[:, 1]
    ax_scatter.scatter(scatter_x, scatter_y, c=collection.get_facecolor(), s=collection.get_sizes()) #, label="Scatter Data")

# Copy labels and title from the original scatter plot
ax_scatter.set_title("Scatter Plot")
ax_scatter.set_xlabel(ax1.get_xlabel())
ax_scatter.set_ylabel(ax1.get_ylabel())
ax_scatter.legend()

# ** Final Layout **
plt.tight_layout()
plt.show()
