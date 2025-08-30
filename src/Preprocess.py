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
for idx, (x,y,z) in enumerate(p.coordinates):
    mzs, intensities = p.getspectrum(idx)
    my_spectra.append([mzs, intensities, (x, y, z)])

print(f"Total spectra loaded: {len(my_spectra)}")


max_mz = -float('inf')
min_mz = float('inf')
for spectrum in my_spectra:
    max_mz = math.ceil(max(max(spectrum[0]),max_mz))
    min_mz = math.floor(min(min(spectrum[0]),min_mz))
print(f'Range of mz values:',(min_mz,max_mz))


common_mzs = np.arange(min_mz,max_mz,0.02)
binned = np.zeros((len(my_spectra), len(common_mzs)), dtype=np.float32)

for i, (mzs, intensities,_) in enumerate(my_spectra):
    indices = np.digitize(mzs, common_mzs) - 1
    for k, val in zip(indices, intensities):
        if 0 <= k < binned.shape[1]:
            binned[i, k] += val


tic = binned.sum(axis=1, keepdims=True)
X = binned / tic

# Min-max normalization to [0, 1] range per pixel
X_min = X.min(axis=1, keepdims=True)
X_max = X.max(axis=1, keepdims=True)
X = (X - X_min) / (X_max - X_min + 1e-8)  # Adding a small epsilon to avoid division by zero

X = X.astype(np.float16)
print(f"Matrix created!")
print(f"Matrix has dimensions of",X.shape)

name = f"{args.name}_x.npy"
output = os.path.join(args.output, name)
np.save(output, X)

name = f"{args.name}_mzs.npy"
output = os.path.join(args.output, name)
np.save(output, common_mzs)

name = f"{args.name}_coords.npy"
output = os.path.join(args.output, name)
np.save(output, [coord for _, _, coord in my_spectra])


print("Done creating data")