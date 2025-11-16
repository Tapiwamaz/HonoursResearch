import matplotlib.pyplot as plt
import math
import os
import h5py
import hdf5plugin
import numpy as np
import argparse


def get_image_data(file: h5py.File, sorted_keys: list[int], mz: float, coords: np.ndarray, tolerance: float = 50) -> np.ndarray:
    """
    Generates a 2D image array for a single m/z value.
    """
    max_x = int(np.max(coords[:, 0]))
    max_y = int(np.max(coords[:, 1]))
    shape = (max_x + 1, max_y + 1)
    img = np.zeros(shape)
    for index, key in enumerate(sorted_keys):
        mass_to_charges = file[str(key)]["x"][:]
        intensities = file[str(key)]["y"][:]
        mask = np.abs(mass_to_charges - mz) <= tolerance
        val = np.sum(intensities[mask]) if np.any(mask) else 0
        row, col = coords[index]
        img[row, col] = val
    non_zero_pixels = np.count_nonzero(img)
    print(f"m/z {mz}: {non_zero_pixels} non-zero pixels")
    return img


def plot_3d_slices(file: h5py.File, sorted_keys: list[int], mz_values: list[float], name: str, coords: np.ndarray, output_dir: str, tolerance: float = 50):
    """
    Plots multiple 2D MSI images as slices in a 3D space.
    """
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    max_x = int(np.max(coords[:, 0]))
    max_y = int(np.max(coords[:, 1]))
    shape = (max_x + 1, max_y + 1)

    # Create coordinate grids
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Plot each m/z slice
    for mz_target in mz_values:
        # Get the 2D image data for the current m/z
        slice_2d = get_image_data(file, sorted_keys, mz_target, coords, tolerance)

        # Normalize for better visualization
        slice_max = np.max(slice_2d)
        slice_norm = slice_2d / slice_max if slice_max > 0 else slice_2d

        # Create a constant Z plane at the m/z level
        Z = np.full(X.shape, mz_target)

        # Plot the surface with colors based on intensity
        ax.plot_surface(X, Z, Y,
                        facecolors=plt.cm.magma(slice_norm),
                        rstride=5, cstride=5,  # Subsample for performance
                        alpha=1, shade=False)

    # Add a colorbar
    m = plt.cm.ScalarMappable(cmap=plt.cm.magma)
    m.set_array([0, 1])
    fig.colorbar(m, ax=ax, shrink=0.25, aspect=10, label='Normalized Intensity')

    ax.set_xlabel('X Position')
    ax.set_zlabel('Y Position')
    ax.set_ylabel('m/z')
    ax.set_title(f'3D Slices for {name}')

    views = [
    (30, 45, 'view_1'),
    (30, 135, 'view_2'),
    (30, 225, 'view_3'),
    (30, 315, 'view_4'),
    ]

    for elev, azim, view_name in views:
        ax.view_init(elev=elev, azim=azim)
        output_path = os.path.join(output_dir, f"{name}_{view_name}.png")
        fig.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f'Saved: {output_path}')

parser = argparse.ArgumentParser(description="Generate ion image plot.")
parser.add_argument("--input", required=True, help="Path to the input HDF5 file.")
parser.add_argument("--output", required=True, help="Directory to save the plot.")
parser.add_argument("--name", required=True, help="Name of plots.")
parser.add_argument("--coords", required=True, help="Coords of plots.")



args = parser.parse_args()

# Load data
f = h5py.File(args.input, 'r')
print(f"File loaded")
sorted_keys = sorted([int(key) for key in f.keys()])

mzs = [100,200,350]
coords = np.load(args.coords)

plot_3d_slices(f,sorted_keys=sorted_keys,mz_values=mzs,name=args.name,coords=coords,output_dir=args.output,tolerance=50)
