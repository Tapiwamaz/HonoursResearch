import argparse
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Decode data using a PCA model and plot the resulting tissue sample.")
    parser.add_argument('--x', type=str, required=True, help='Path to X.npy file (latent embeddings)')
    parser.add_argument('--coords', type=str, required=True, help='Coordinates')
    parser.add_argument('--decoder', type=str, required=True, help='Path to PCA model (.pkl or .joblib)')
    parser.add_argument('--scaler', type=str, required=True, help='Path to the scaler (.pkl or .joblib)')
    parser.add_argument('--output', type=str, required=True, help='Directory to save output')
    args = parser.parse_args()

    X = np.load(args.x)
    print(f"Latent embeddings: {X.shape}")

    pca = joblib.load(args.decoder)
    scaler = joblib.load(args.scaler)
    print(f"PCA model loaded")
    print(f"Components shape: {pca.components_.shape}")

    print(f"Decoding data...")
    # For PCA reconstruction: inverse_transform back to original space
    X_reconstructed = pca.inverse_transform(X)
    # Then inverse transform the scaler
    X_reconstructed = scaler.inverse_transform(X_reconstructed)
    print(f"Decoded data: {X_reconstructed.shape}")

    del pca, scaler
    coords = np.load(args.coords)

    # The data from X is the intensities per spectrum for a range of mz 150-1500
    # X.shape = (num_spectra,67499)
    # I want to plot the resultant image at the mz range 0.02 +- 390 which would correspond to X[:,11999:12001]
    intensities = X_reconstructed[:, 11999:12001].sum(axis=1)  # Sum across the m/z window
    
    # Create the spatial map
    plt.figure(figsize=(10, 8))
    plt.scatter(coords[:, 0], coords[:, 1], c=intensities, cmap='cividis',origin='lower')
    plt.colorbar(label='Intensity')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('PCA Reconstruction - m/z ~390 (±0.02)')
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, 'pca_390.png')
    plt.savefig(output_path, dpi=500, bbox_inches='tight')
    print(f"Image saved to: {output_path}")
    plt.close()
 

if __name__ == "__main__":
    main()
