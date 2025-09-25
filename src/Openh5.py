import h5py
import hdf5plugin
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt


def plot_image( file: h5py.File ,sorted_keys: list[str],mz : float ,shape ,output_dir: str,tolerance: float=0.02):
    img = np.zeros(shape)
    for index, key in enumerate(sorted_keys):
        mass_to_charges = file[str(key)]["x"][:]
        intensities = file[str(key)]["y"][:]
        mask = np.abs(mass_to_charges - mz) <= tolerance
        val = np.sum(intensities[mask]) if np.any(mask) else 0
        
        row,col = divmod(index,shape[1])
        img[row,col] = val

    plt.imshow(img, cmap='viridis', origin='lower')
    plt.colorbar(label='Intensity')
    plt.title(f"Ion Image for m/z = {mz}")
    plt.xlabel("X intensity")
    plt.ylabel("Y intensity")

    output_path = os.path.join(output_dir, f"image_{mz}.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}") 

parser = argparse.ArgumentParser(description="Generate ion image plot.")
parser.add_argument("--input", required=True, help="Path to the input HDF5 file.")
parser.add_argument("--output", required=True, help="Directory to save the plot.")
parser.add_argument("--name", required=True, help="Name of plots.")


args = parser.parse_args()

# Load data
f = h5py.File(args.input, 'r')
print(f"File loaded")
sorted_keys = sorted(list(f.keys()))

target_mzs = [120,150,200,250,271]
tolerance = 0.02
width = 400
height = 400

output_path = f"{args.output}{args.name}"
for mz in target_mzs:
    plot_image(file=f,sorted_keys=sorted_keys,shape=(width,height),output_dir=output_path,mz=mz)

print("Done")
