# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import gaussian_kde


# # x_list = [
# #     np.array([1,2,3]),
# #     np.array([1, 6, 8])
# # ]

# # ex_list = [
# #     x_list[0]*0.01,
# #     x_list[1]*0.01
# # ]

# # y_list = [
# #     x_list[0]**2,
# #     x_list[1]**2.1
# # ]

# # ey_list = [
# #     y_list[0]*0.01,
# #     y_list[1]*0.01
# # ]


# # fig0, ax0 = plt.subplots()

# # assert(len(x_list) == len(y_list))
# # n_curves = len(x_list)
# # for i in range(n_curves):
# #     x = x_list[i]
# #     ex = ex_list[i]
# #     y = y_list[i]
# #     ey = ey_list[i]
# #     ax0.errorbar(x=x, y=x, xerr=ex, yerr=ey, label=str(i))
# # #---

# # fig, (ax_density, ax_main) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 4]}, figsize=(10, 5))
# # ax_main = ax0

# # # Ensure space between main and density plot
# # fig.subplots_adjust(wspace=0.3)

# # fig.show()

# import matplotlib.pyplot as plt

# # Step 1: Create the original figure and axes with some plot data
# fig1, ax1 = plt.subplots()
# ax1.plot([1, 2, 3], [4, 5, 6], label='Original Plot')
# ax1.set_title("Original Plot")
# ax1.set_xlabel("X-axis")
# ax1.set_ylabel("Y-axis")
# ax1.legend()

# # Step 2: Create a new figure and add the original plot as a subplot
# fig2, (ax2, ax3) = plt.subplots(1,2)

# # Copy data and properties from ax1 to ax2
# for line in ax1.get_lines():
#     ax2.plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), color=line.get_color())

# # Copy axis labels, title, and legend
# ax2.set_title(ax1.get_title())
# ax2.set_xlabel(ax1.get_xlabel())
# ax2.set_ylabel(ax1.get_ylabel())
# ax2.legend()

# # Optionally, close the original figure if it's no longer needed
# plt.close(fig1)

# # Display the new figure with the copied subplot
# plt.show()

# from typing import Literal

# def AIC_density_and_fits(ax_fits, ax_AIC, title: str, side_AIC=Literal["left", "right"]):
#     """Combination of fits and AIC model averaging

#     Args:
#         ax_fits (_type_): _description_
#         ax_AIC (_type_): _description_
#         title (str): _description_
#         side_AIC (_type_, optional): _description_. Defaults to Literal["left", "right"].

#     Returns:
#         _type_: _description_
#     """
#     if side_AIC == "left":
#         fig, (ax_density, ax_main) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 4]}, figsize=(10, 5))
#         angle_AIC_plot = 90
#     else:
#         fig, (ax_main, ax_density) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 4]}, figsize=(10, 5))
#         angle_AIC_plot = -90
#     #---
#     for line in ax_fits.get_lines():
#         x ,y = line.get_xdata(), line.get_ydata()
#         ax_main.plot(x, y, label=line.get_label(), color=line.get_color())
#     for line in ax_AIC.get_lines():
#         x ,y = line.get_xdata(), line.get_ydata()
#         theta = np.radians(angle_AIC_plot)

#         # Create a rotation matrix
#         rotation_matrix = np.array([
#             [np.cos(theta), -np.sin(theta)],
#             [np.sin(theta), np.cos(theta)]
#         ])

#         # Stack x and y as coordinates
#         coords = np.vstack((x, y))

#         # Apply the rotation
#         rotated_coords = rotation_matrix @ coords
#         x_rotated, y_rotated = rotated_coords[0, :], rotated_coords[1, :]
#         ax_density.plot(x, y, label=line.get_label(), color=line.get_color())
    
        
#     return fig
        
    

# quit()



# def fits_and_AIC_side(data_list, func_list, x_values, param_list):
#     """
#     Plot scatter plot of data with model fits and side probability density from model averaging.
    
#     Parameters:
#     - data_list: List of np.array, each array containing data points (y-values) to fit.
#     - func_list: List of functions, each representing a model fit f(x, p).
#     - x_values: np.array, x-values for plotting and fitting.
#     - param_list: List of parameter lists, each set of parameters for corresponding fit function.
#     """
#     # Main plot setup with scatter plot and fit traces
#     fig, (ax_density, ax_main) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 4]}, figsize=(10, 5))
    
#     # Ensure space between main and density plot
#     fig.subplots_adjust(wspace=0.3)

#     # Scatter plot of original data on main axis
#     for i, data in enumerate(data_list):
#         ax_main.scatter(x_values, data, label=f'Data {i+1}', alpha=0.7)

#     # Generate fit traces and model predictions
#     model_predictions = []
#     for i, (func, params) in enumerate(zip(func_list, param_list)):
#         y_fit = func(x_values, *params)
#         ax_main.plot(x_values, y_fit, label=f'Model {i+1}', linestyle='--')
#         model_predictions.append(y_fit)
    
#     # Model averaging for probability density estimation
#     model_predictions = np.array(model_predictions)
#     averaged_model = np.mean(model_predictions, axis=0)
    
#     # Kernel Density Estimation on the averaged model predictions
#     density = gaussian_kde(averaged_model)
#     y_density = np.linspace(min(averaged_model), max(averaged_model), 100)
#     density_values = density(y_density)
    
#     # Plot the density on the vertical axis
#     ax_density.plot(density_values, y_density, color='purple')
#     ax_density.set_xlabel("Density")
#     ax_density.set_ylabel("Model Prediction")
#     ax_density.set_title("Probability Density")

#     # Main axis labels and legend
#     ax_main.set_xlabel("X")
#     ax_main.set_ylabel("Y")
#     ax_main.set_title("Fit Traces with Model Averaging")
#     ax_main.legend()

#     plt.show()

# # Example Usage
# # Define some example data, functions, and parameters
# x_values = np.linspace(0, 10, 50)
# data_list = [np.sin(x_values) + np.random.normal(0, 0.2, len(x_values)),
#              np.cos(x_values) + np.random.normal(0, 0.2, len(x_values))]

# # Define model functions
# def model1(x, a, b):
#     return a * np.sin(b * x)

# def model2(x, a, b):
#     return a * np.cos(b * x)

# # Define parameter lists for the functions
# param_list = [
#     [1, 1],  # parameters for model1
#     [1, 1]   # parameters for model2
# ]

# # Call the plotting function
# fits_and_AIC_side(data_list, [model1, model2], x_values, param_list)
