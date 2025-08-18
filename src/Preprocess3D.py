import argparse
import os
from pyimzml.ImzMLParser import ImzMLParser
import numpy as np
import random
import matplotlib.pyplot as plt
import math

parser = argparse.ArgumentParser(description="Generate ion image plot.")
parser.add_argument("--input", required=True, help="Path to the input imzml and ibd files.")
parser.add_argument("--output", required=True, help="Directory to save the plot.")
parser.add_argument("--name", required=True, help="Name to save output")

args = parser.parse_args()



print(f"####################################\nLoading spectra...")
p = ImzMLParser(args.input)
my_spectra = []
for idx, (x,y,_) in enumerate(p.coordinates):
    mzs, intensities = p.getspectrum(idx)
    my_spectra.append([mzs, intensities, (x, y)])

print(f"Total spectra loaded: {len(my_spectra)}")


max_mz = -float('inf')
min_mz = float('inf')
for spectrum in my_spectra:
    max_mz = math.ceil(max(max(spectrum[0]),max_mz))
    min_mz = math.floor(min(min(spectrum[0]),min_mz))
print(f'Range of mz values:',(min_mz,max_mz))


# Assuming `my_spectra` contains tuples of (mzs, intensities, (x, y)) where (x, y) are the coordinates of the spectrum.

# Define the range of x and y coordinates
x_coords = sorted(set(coord[0] for _, _, coord in my_spectra))
y_coords = sorted(set(coord[1] for _, _, coord in my_spectra))

# Create mappings from coordinates to indices
x_to_index = {x: i for i, x in enumerate(x_coords)}
y_to_index = {y: i for i, y in enumerate(y_coords)}

# Initialize the 3D array: k x p x m
common_mzs = np.arange(min_mz, max_mz, 0.02)
binned = np.zeros((len(y_coords), len(x_coords),len(common_mzs)), dtype=np.float32)

# Populate the 3D array
for mzs, intensities, (x, y) in my_spectra:
    x_idx = x_to_index[x]
    y_idx = y_to_index[y]
    indices = np.digitize(mzs, common_mzs) - 1
    for k, val in zip(indices, intensities):
        if 0 <= k < binned.shape[2]:
            binned[y_idx, x_idx, k] += val

# Normalize the intensities along the m-axis (summing over k)
tic = binned.sum(axis=2, keepdims=True)
tic[tic == 0] = 1

X = binned / tic

# Convert to float16 AFTER normalization (values are now 0-1 range)
X = X.astype(np.float16)

print(f"3D Matrix created!")
print(f"Matrix has dimensions of {X.shape}")

name = f"{args.name}_x.npy"
output = os.path.join(args.output, name)
np.save(output, X)


print("Done creating data")