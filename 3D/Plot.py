import numpy as np
import matplotlib.pyplot as plt
import os 
import argparse


def plot_msi_slices_3d(intensity_data, mz_channels, mz_values_to_plot, spatial_dims=(400, 400)):
    """
    Plot multiple 2D MSI images as slices in 3D space
    """
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create coordinate grids
    x = np.arange(spatial_dims[0])
    y = np.arange(spatial_dims[1])
    X, Y = np.meshgrid(x, y, indexing='ij')
    print(X.shape)
    
    # Plot each m/z slice
    for mz_target in mz_values_to_plot:
        mz_index = np.argmin(np.abs(mz_channels - mz_target))
        slice_data = intensity_data[:, mz_index]
        
        # Reshape the slice data to the spatial dimensions of the MSI data
        slice_2d = slice_data.reshape(spatial_dims)
        
        # Normalize for better visualization
        slice_norm = slice_2d / np.max(slice_2d) if np.max(slice_2d) > 0 else slice_2d
        
        # Create surface at the m/z level
        Z = np.ones_like(X) * mz_channels[mz_index]
        
        # Use surface plot with color mapping
        surf = ax.plot_surface(X, Z, Y, 
                              facecolors=plt.cm.viridis(slice_norm),
                              rstride=5, cstride=5,  # Subsample for performance
                              alpha=0.7,
                              shade=False)
        # Add colorbar for the last surface
        if mz_target == mz_values_to_plot[-1]:
            m = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
            m.set_array([0, 1])
            plt.colorbar(m, ax=ax, shrink=0.5, aspect=5, label='Normalized Intensity')
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('m/z')
    ax.set_zlabel('Y Coordinate')
    ax.set_title('MSI Data - Multiple m/z Slices in 3D')
    
    return fig, ax


parser = argparse.ArgumentParser(description="3D Slice plot")
parser.add_argument("--X", required=True, help="Path to the X data.")
parser.add_argument("--mzs", required=True, help="Path to the mzs file.")
parser.add_argument("--output", required=True, help="Directory to save the plot.")
parser.add_argument("--name", required=True, help="Name of plots.")

args = parser.parse_args()

intensity_data = np.load(args.X,mmap_mode='r')
mzs = np.load(args.mzs,mmap_mode='r')


spatial_dims = (400, 400)
# num_spectra = spatial_dims[0] * spatial_dims[1]
# num_mz_channels = len(mzs)

# x = np.linspace(0, spatial_dims[0], num=spatial_dims[0], endpoint=True)
# y = np.linspace(0, spatial_dims[1], num=spatial_dims[1], endpoint=True)
# X, Y = np.meshgrid(x, y, indexing='ij')

# # Create intensity patterns using sin/cos of coordinates
# intensity_data = np.zeros((num_spectra, num_mz_channels))
# for i in range(num_mz_channels):
#     # Vary frequency based on m/z channel
#     freq = (i + 1) / num_mz_channels * 2 * np.pi
#     pattern = np.sin(X * freq / 50) * np.cos(Y * freq / 50)
#     intensity_data[:, i] = pattern.flatten()

# mz_channels = np.linspace(100, 1500, num_mz_channels)
mz_targets = [100, 300, 500, 800, 1200]
fig, ax = plot_msi_slices_3d(intensity_data, mzs, mz_targets, spatial_dims=spatial_dims)

# Set the viewing angle (elevation, azimuth)
ax.view_init(elev=30, azim=45)

fig.savefig('msi_slices_3d.png', dpi=300, bbox_inches='tight')