import h5py
import hdf5plugin
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Generate ion image plot.")
parser.add_argument("--input", required=True, help="Path to the input HDF5 file.")
parser.add_argument("--coordinates", required=True, help="Path to the coordinates file.")
parser.add_argument("--output", required=True, help="Directory to save the plot.")
args = parser.parse_args()

# Load data
f = h5py.File(args.input, 'r')
coordinates = np.load(args.coordinates)

my_spectra = []
keys = list(f.keys())
for index in range(len(keys)):
    key = keys[index]
    my_spectra.append([f.get(key)["x"][:],f.get(key)["y"][:],coordinates[index]])
    
  

print("Done adding to array!")    


import numpy as np
import matplotlib.pyplot as plt
# Choose the m/z you want to plot  and tolerance
target_mz = 165
tolerance = 0.02

# Get image dimensions
all_coords = [coord for _, _, coord in my_spectra]
xs, ys, _ = zip(*all_coords)
width = max(xs) + 1
height = max(ys) + 1
print(width,height)



# Create empty image
ion_image = np.zeros((height, width))

# Fill in the ion image with intensities for target m/z
times = 0 
for mzs, intensities, (x, y, _) in my_spectra:
    # Get mask of indices where mz is within target window
    mz_mask = (mzs >= target_mz - tolerance) & (mzs <= target_mz + tolerance)
    if np.any(mz_mask):
        times+=1
        ion_intensity = np.mean(intensities[mz_mask])
        ion_image[y, x] = ion_intensity
print(f"Points found", times)

output_path = os.path.join(args.output, "ion_image.png")
plt.savefig(output_path)
print(f"Plot saved to {output_path}")