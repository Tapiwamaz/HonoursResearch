from pyimzml.ImzMLParser import ImzMLParser
import math
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


parser = argparse.ArgumentParser(description="Data Analysis.")
parser.add_argument("--input", required=True, help="Path to the input imzml file.")
parser.add_argument("--output", required=True, help="Directory to save the plot.")
parser.add_argument("--job_id",required=True,help="Job id")
parser.add_argument("--job_type",required=True,help="Dataset")
args = parser.parse_args()

job  = args.job_id
job_type = args.job_type



p = ImzMLParser(args.input)
my_spectra = []
number_of_mzs_hist = []

for idx, (x,y,z) in enumerate(p.coordinates):
    mzs, intensities = p.getspectrum(idx)
    number_of_mzs_hist.append([idx, len(mzs)])
    my_spectra.append([mzs, intensities, (x, y, z)])

number_of_mzs_hist = np.array(number_of_mzs_hist) 
# Plot the number of m/z values for each spectrum
plt.figure(figsize=(15, 6))
plt.xlabel("Spectrum Index")
plt.ylabel("Number of Recorded m/z Values")
plt.title("Number of m/z Values for Each Spectrum")
plt.bar(number_of_mzs_hist[:, 0], number_of_mzs_hist[:, 1], width=1, color="b")

# Save the plot
name = f"mzs_spread_{job_type}_{job}.png"
output_path = os.path.join(args.output, name)
plt.savefig(output_path)
print(f"Plot saved to {output_path}")


max_mz = -float('inf')
max_intensity = -float('inf')
min_mz = float('inf')
min_intensity = float('inf')

for spectrum in my_spectra:
    max_mz = math.ceil(max(max(spectrum[0]),max_mz))
    min_mz = math.floor(min(min(spectrum[0]),min_mz))
    max_intensity = max(max(spectrum[1]),max_intensity)
    min_intensity = min(min(spectrum[1]),min_intensity)

print(f'Range of mz values:',(min_mz,max_mz))
print(f'Range of intensities values:',(min_intensity,max_intensity))
print(f'Number of spectrum:',len(my_spectra))


# Creating histogram to see the distribution of recorded m/z values
# For a given tolerance
tolerance = 0.02

histogram_data = np.array([[0,0]])
most_commmon_mz = [0,0]

# I will target the mz values between the min and max mz found incrementing by 10
for mz in range(math.floor(min_mz),math.ceil(max_mz),0.02):
    count = 0
    for mzs,_,_ in my_spectra:
        mz_mask = (mzs >= mz - tolerance) & (mzs <= mz + tolerance)
        if np.any(mz_mask):
            count += 1
            continue
 
    data = np.array([mz, count])
    if  count > most_commmon_mz[1]:
        most_commmon_mz = data  
    histogram_data = np.vstack([histogram_data, data])



print(f'The most frequently occuring mz range is mz={most_commmon_mz[0]} with {most_commmon_mz[1]} spectra')
print(f'This constitutes {(most_commmon_mz[1]/len(my_spectra))*100} % of the pixels')
plt.figure(figsize=(15,6))
plt.bar(histogram_data[1:,0], histogram_data[1:,1],color="g",width=2)
plt.xlabel('m/z')
plt.ylabel('Frequency (Pixels found within region')
plt.title(f'Distribution of m/z values across spectra (tolerance={tolerance})')

name = f"histogram_{job_type}_{job}.png"
output_path = os.path.join(args.output, name)
plt.savefig(output_path)
print(f"Plot saved to {output_path}")