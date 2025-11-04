import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Decode data using a Keras decoder model and plot the resulting tissue sample.")
    parser.add_argument('--x', type=str, required=True, help='Path to X.npy file')
    parser.add_argument('--coords', type=str, required=True, help='Coordinates')
    parser.add_argument('--decoder', type=str, required=True, help='Path to decoder .keras model')
    parser.add_argument('--output', type=str, required=True, help='Directory to save output')
    args = parser.parse_args()

    X = np.load(args.x)
    print(f"Latent embeddings: {X.shape}")

    decoder = load_model(args.decoder)
    print(f"Decoder loaded")
    decoder.summary()

    print(f"Decoding data...")
    X = decoder(X)
    print(f"Decoded data: {X.shape}")

    del decoder
    coords = np.load(args.coords)

    # The data from X is the intentities per spectrum for a range of mz 150-1500
    # X.shape = (num_spectra,67499)
    # I want to plot the resultant image at the mz range 0.02 +- 390 which  would correspond to X[:,11999:12001]
    intensities = X[:, 11999:12001].sum(axis=1)  # Sum across the m/z window
    
    # Create the spatial map
    plt.figure(figsize=(10, 8))
    plt.scatter(coords[:, 0], coords[:, 1], c=intensities, cmap='cividis', origin='lower')
    plt.colorbar(label='Intensity')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Tissue Sample - m/z ~390 (±0.02)')
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, 'ae_390.png')
    plt.savefig(output_path, dpi=500, bbox_inches='tight')
    print(f"Image saved to: {output_path}")
    plt.close()
 

if __name__ == "__main__":
    main()