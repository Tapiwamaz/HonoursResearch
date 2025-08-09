import os
import numpy as np
from pyimzml.ImzMLParser import ImzMLParser
from scipy.interpolate import interp1d
import argparse

def preprocess_spectra(spectra_list, target_length=20000):
    processed_intensities = []
    all_mzs = []
    
    for spectrum in spectra_list:
        mzs, intensities, _ = spectrum
        all_mzs.extend(mzs)
    
    min_mz, max_mz = min(all_mzs), max(all_mzs)
    common_mzs = np.linspace(min_mz, max_mz, target_length)
    
    for spectrum in spectra_list:
        mzs, intensities, _ = spectrum
        interpolator = interp1d(mzs, intensities, kind='linear', 
                    bounds_error=False, fill_value=0)
        interpolated_intensities = interpolator(common_mzs)
        processed_intensities.append(interpolated_intensities)
    
    X = np.array(processed_intensities)    
    X_min = X.min()
    X_max = X.max()
    X = (X - X_min) / (X_max - X_min)
    
    return X, common_mzs

def main():
    parser = argparse.ArgumentParser(description='Process MALDI-TOF data')
    parser.add_argument('--input', required=True, help='Path to input imzML file')
    parser.add_argument('--output', required=True, help='Output directory')
    args = parser.parse_args()

    print(f"Starting processing of {args.input}")
    p = ImzMLParser(args.input)
    my_spectra = []
    
    for idx, (x,y,z) in enumerate(p.coordinates):
        mzs, intensities = p.getspectrum(idx)
        my_spectra.append([mzs, intensities, (x, y, z)])

    print(f"Loaded {len(my_spectra)} spectra")
    print("Preprocessing spectra...")
    
    X_processed, common_mzs = preprocess_spectra(my_spectra)

    # Save outputs
    os.makedirs(args.output, exist_ok=True)
    np.save(os.path.join(args.output, "HIV_Common_mzs.npy"), common_mzs)
    np.save(os.path.join(args.output, "HIV_OG_Normalized.npy"), X_processed)
    
    print(f"Processing complete! Output saved to {args.output}")
    print("Praise the LORD ALMIGHTY!!!")

if __name__ == "__main__":
    main()
