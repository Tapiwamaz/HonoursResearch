import h5py
import hdf5plugin
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Data Analysis.")
parser.add_argument("--input", required=True, help="Path to the input HDF5 file.")
parser.add_argument("--output", required=True, help="Directory to save the plot.")
parser.add_argument("--coordinates", required=True, help="Path to the coordinates file.")
parser.add_argument("--job_id",required=True,help="Job id")
args = parser.parse_args()

# Load data
f = h5py.File(args.input, 'r')
coordinates = np.load(args.coordinates)
job= args.job_id

my_spectra = []
keys = list(f.keys())
print(f"Number of keys:",len(keys))
for index in range(len(keys)):
    if index < len(coordinates):
        key = keys[index]
        my_spectra.append([f.get(key)["x"][:],f.get(key)["y"][:],coordinates[index]])
    else:
        my_spectra.append([f.get(key)["x"][:],f.get(key)["y"][:],[0,0]])   
print("Done adding to array!")

# ==================================================================================================
# Min of max  recorded mzs and intensities
max_mz = -float('inf')
max_intensity = -float('inf')
min_mz = float('inf')
min_intensity = float('inf')
for spectrum in my_spectra:
    max_mz = max(max(spectrum[0]),max_mz)
    min_mz = min(min(spectrum[0]),min_mz)
    max_intensity = max(max(spectrum[1]),max_intensity)
    min_intensity = min(min(spectrum[1]),min_intensity)

print(f'Range of mz values:',(min_mz,max_mz))
print(f'Range of intensities values:',(min_intensity,max_intensity))
print(f'Number of spectra:',len(my_spectra))
# ==================================================================================================


# ==================================================================================================
import numpy as np
import matplotlib.pyplot as plt

# Creating histogram to see the distribution of recorded m/z values
# For a given tolerance
tolerance = 0.02

histogram_data = np.array([[0,0]])
most_commmon_mz = [0,0]

# I will target the mz values between the min and max mz found incrementing by 10
for mz in range(100,1500,1):
    count = 0
    current_locations = []
    for mzs,_,coord in my_spectra:
        mz_mask = (mzs >= mz - tolerance) & (mzs <= mz + tolerance)
        # mz_mask is an array of the same size as mzs and has a true where there is a an mz in mzs that is in the range
        # If a single true is found we add the spectrum(pixel) to the count for the range we are looking for 
        if np.any(mz_mask):
            count += 1
            current_locations.append(coord) 
            # go to next spectrum
            continue
 
    data = np.array([mz, count])
    if  count > most_commmon_mz[1]:
        most_commmon_mz = data
        most_common_locations = current_locations   
    histogram_data = np.vstack([histogram_data, data])



print(f'The most frequently occuring mz range is mz={most_commmon_mz[0]} with {most_commmon_mz[1]} spectra')
print(f'This constitutes {(most_commmon_mz[1]/len(my_spectra))*100} % of the pixels')
print(f'The coordinates of the spectra with the most recorded mz range:\n {most_common_locations}')
# ==================================================================================================


plt.bar(histogram_data[1:,0], histogram_data[1:,1],color="g")
plt.xlabel('m/z')
plt.ylabel('Frequency (Pixels found within region')
plt.title(f'Distribution of m/z values across spectra (tolerance={tolerance})')

name = f"histogram_graph_cancer_{job}.png"
output_path = os.path.join(args.output, name)
plt.savefig(output_path)
print(f"Plot saved to {output_path}")

# Scatter plot of the pixels in the most_common_locations array
if most_common_locations:
    # Extract x and y coordinates from the most_common_locations
    x_coords = [coord[0] for coord in most_common_locations]
    y_coords = [coord[1] for coord in most_common_locations]

    plt.figure()  # Create a new figure for the scatter plot
    plt.scatter(x_coords, y_coords, c='blue', marker='o', label='Most Common Spectra')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Scatter Plot of Most Common Spectra (mz={most_commmon_mz[0]})')
    plt.legend()
    name = f"most_mommon_spectra_cancer_{job}.png"
    scatter_output_path = os.path.join(args.output, name)
    plt.savefig(scatter_output_path)
    print(f"Scatter plot saved to {scatter_output_path}")
else:
    print("No most common locations found to plot.")
