from pyimzml.ImzMLParser import ImzMLParser
import os
import numpy as np
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt

class SpectrumAutoencoder(Model):
    def __init__(self, latent_dim, n_peaks):
        super(SpectrumAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.n_peaks = n_peaks
        
        self.encoder = tf.keras.Sequential([
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(latent_dim, activation='relu'),
        ])
        
        self.decoder = tf.keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(n_peaks, activation='linear'),  
        ])

    def call(self, intensities):
        encoded = self.encoder(intensities)
        decoded_intensities = self.decoder(encoded)
        return decoded_intensities


imzml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../HIV.imzml"))
if not os.path.exists(imzml_path):
    raise FileNotFoundError(f"imzML file not found at: {imzml_path}")

print("File found")
print("Starting to parse through file....")
p = ImzMLParser(imzml_path)
my_spectra = []
for idx, (x,y,z) in enumerate(p.coordinates):
    mzs, intensities = p.getspectrum(idx)
    my_spectra.append([mzs, intensities, (x, y, z)])
print("my_spectra has been populated")
 # Data preprocessing - convert all spectra to same length

def preprocess_spectra(spectra_list, target_length=20000):
    processed_intensities = []
    all_mzs = []
    
    # Collect all mz values to find common range
    for spectrum in spectra_list:
        mzs, intensities, _ = spectrum
        all_mzs.extend(mzs)
    
    # Create common mz grid
    min_mz, max_mz = min(all_mzs), max(all_mzs)
    common_mzs = np.linspace(min_mz, max_mz, target_length)
    
 
    for spectrum in spectra_list:
        mzs, intensities, _ = spectrum
        # interpolator is a topic of discussion
        interpolator = interp1d(mzs, intensities, kind='linear', 
                    bounds_error=False, fill_value=0)
        
        interpolated_intensities = interpolator(common_mzs)
        processed_intensities.append(interpolated_intensities)
    

    X = np.array(processed_intensities)    
    #  normalization to [0, 1] range 
    X_min = X.min()
    X_max = X.max()
    X = (X - X_min) / (X_max - X_min)
    
    return X, common_mzs

# Preprocess the data
print("Starting preprocessing....")
print(f"Total spectra: {len(my_spectra)}")
X_processed, common_mzs = preprocess_spectra(my_spectra)
print("Done preprocessing....")
print(f"Processed data shape: {X_processed.shape}")
print(f"Data range: [{X_processed.min():.4f}, {X_processed.max():.4f}]\n")   


with open("HIV_Common_mzs.npy",'wb') as file:
    np.save(file,common_mzs)

with open("HIV_OG_Normalized.npy",'wb') as file:
    np.save(file,X_processed)


print("Praise the LORD ALMIGHTY!!!")    