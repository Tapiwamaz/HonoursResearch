import matplotlib.pyplot as plt
import h5py
import hdf5plugin
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import math


def prepare_data(file: h5py.File, name :str , output_dir: str,sorted_keys: list[str]) -> None:
    my_spectra = []
    max_mz = -float('inf')
    min_mz = float('inf')
    for idx, key in enumerate(sorted_keys):
        mzs = file[str(key)]["x"][:]
        max_mz = math.ceil(max(max(mzs),max_mz))
        min_mz = math.floor(min(min(mzs),min_mz))
        intensities =file[str(key)]["y"][:]
        my_spectra.append([mzs, intensities])

    print(f'Range of mz values:',(min_mz,max_mz))


    common_mzs = np.arange(min_mz,max_mz,1)
    binned = np.zeros((len(my_spectra), len(common_mzs)), dtype=np.float32)

    for i, (mzs, intensities) in enumerate(my_spectra):
        indices = np.digitize(mzs, common_mzs) - 1
        for k, val in zip(indices, intensities):
            if 0 <= k < binned.shape[1]:
                binned[i, k] += val


    X = binned 

    del binned


    # X = X.astype(np.float32)
    print(f"Matrix created!")
    print(f"Matrix has dimensions of", X.shape)

    data_name = f"3d-{name}_x.npy"
    output = os.path.join(output_dir, data_name)
    np.save(output, X)
    data_name = f"3d-{name}_mzs.npy"
    output = os.path.join(output_dir, data_name)
    np.save(output, common_mzs)



parser = argparse.ArgumentParser(description="Generate ion image plot.")
parser.add_argument("--input", required=True, help="Path to the input HDF5 file.")
parser.add_argument("--output", required=True, help="Directory to save the plot.")
parser.add_argument("--name", required=True, help="Name of plots.")


args = parser.parse_args()

# Load data
f = h5py.File(args.input, 'r')
print(f"File loaded")
sorted_keys = sorted([int(key) for key in f.keys()])



prepare_data(sorted_keys=sorted_keys,output_dir=args.output,name=args.name,file=f)
