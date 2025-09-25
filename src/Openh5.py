import h5py
import hdf5plugin
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import math


def plot_image( file: h5py.File ,sorted_keys: list[int],mz : float,name: str ,shape ,output_dir: str,tolerance: float=100.02):
    img = np.zeros(shape)
    count = 0
    for index, key in enumerate(sorted_keys):
        if index >= shape[0] * shape[1]:
            print(f"Warning: More data points ({len(sorted_keys)}) than matrix elements ({shape[0] * shape[1]}). Truncating.")
            break
        mass_to_charges = file[str(key)]["x"][:]
        intensities = file[str(key)]["y"][:]
        mask = np.abs(mass_to_charges - mz) <= tolerance
        val = np.sum(intensities[mask]) if np.any(mask) else 0
        if np.any(mask):
            count +=1
        
        row,col = divmod(index,shape[1])
        img[row,col] = val

    plt.figure(figsize=(12, 8))
    plt.imshow(img, cmap='hot', origin='lower', aspect='auto')
    plt.colorbar(label='Intensity')
    plt.title(f"Ion Image for m/z = {mz}")
    plt.xlabel("X position")
    plt.ylabel("Y position")

    output_path = os.path.join(output_dir, f"{name}_image_{mz}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")
    print(f"Found: {count}") 


def prepare_data(file: h5py.File, name :str , output_dir: str,shape,sorted_keys: list[str]) -> None:
    my_spectra = []
    coords = []
    for idx, key in enumerate(sorted_keys):
        mzs = file[str(key)]["x"][:]
        intensities =file[str(key)]["y"][:]
        my_spectra.append([mzs, intensities])
        row,col = divmod(idx,shape[1])
        coords.append((row,col))

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

    del binned
    del common_mzs

    X_min = X.min(axis=1, keepdims=True)
    X_max = X.max(axis=1, keepdims=True)
    X = (X - X_min) / (X_max - X_min + 1e-6)  # Adding a small epsilon to avoid division by zero

    # Filter to only include m/z values between 150 and 1500
    mz_mask = (common_mzs >= 150) & (common_mzs <= 1500)
    X = X[:, mz_mask]
    X = X[:,:len(X[0])-1]
    common_mzs = common_mzs[mz_mask]

    X = X.astype(np.float32)
    print(f"Matrix created!")
    print(f"Matrix has dimensions of", X.shape)

    name = f"{name}_x.npy"
    output = os.path.join(output_dir, name)
    np.save(output, X)

    name = f"{name}_coords.npy"
    output = os.path.join(output_dir, name)
    np.save(output, coords)

parser = argparse.ArgumentParser(description="Generate ion image plot.")
parser.add_argument("--input", required=True, help="Path to the input HDF5 file.")
parser.add_argument("--output", required=True, help="Directory to save the plot.")
parser.add_argument("--name", required=True, help="Name of plots.")


args = parser.parse_args()

# Load data
f = h5py.File(args.input, 'r')
print(f"File loaded")
sorted_keys = sorted([int(key) for key in f.keys()])

# print(sorted_keys)
# print("\n")

target_mzs = [271,390]
tolerance = 100
width = 400
height = 400

# for mz in target_mzs:
#     plot_image(file=f,sorted_keys=sorted_keys,shape=(width,height),output_dir=args.output,mz=mz,name=args.name)

# print("Done")



prepare_data(shape=(width,height),sorted_keys=sorted_keys,output_dir=args.output,name=args.name,file=f)

print("Done creating data")



