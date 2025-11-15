# import matplotlib.pyplot as plt
# import numpy as np
# from mpl_toolkits.mplot3d import Axes3D

# # Assuming your data is in arrays: x_coords, y_coords, z_mz, intensities
# # Example data generation (replace with your actual data)
# np.random.seed(42)
# n_points = 1000
# x_coords = np.random.uniform(0, 100, n_points)
# y_coords = np.random.uniform(0, 100, n_points)
# z_mz = np.random.uniform(50, 2000, n_points)
# intensities = np.random.exponential(5, n_points)

# # Create the 3D scatter plot
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Scatter plot colored by intensity
# scatter = ax.scatter(x_coords,z_mz, y_coords, 
#                     c=intensities, 
#                     cmap='viridis', 
#                     alpha=0.7,
#                     s=10)

# # Add colorbar
# cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
# cbar.set_label('Intensity')

# # Labels and title
# ax.set_xlabel('X Coordinate')
# ax.set_ylabel('m/z')
# ax.set_zlabel('Y Coordinate')
# ax.set_title('MSI Data - 3D Visualization')

# plt.tight_layout()
# plt.show()

