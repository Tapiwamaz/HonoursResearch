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
    print(f"Scaler type: {type(scaler)}")
    print(f"PCA model loaded")
    print(f"Components shape: {pca.components_.shape}")

    print(f"Decoding data in batches...")
    num_samples = X.shape[0]
    batch_size = 1000
    
    # Initialize array to store only the m/z range we need (11999:12001)
    intensities_sum = np.zeros(num_samples)
    
    # Process in batches
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        batch = X[i:end_idx]
        
        X_reconstructed = pca.inverse_transform(batch)
        # Then inverse transform the scaler
        decoded_batch = scaler.inverse_transform(X_reconstructed)
        
        intensities_sum[i:end_idx] = decoded_batch[:, :].sum(axis=1)
        
        print(f"Processed {end_idx}/{num_samples} samples", end='\r')
    
    print(f"\nDecoding complete!")

    del pca ,scaler
    coords = np.load(args.coords)

    # Create the spatial map
    plt.figure(figsize=(10, 8))
    plt.scatter(coords[:, 0], coords[:, 1], c=intensities_sum, cmap='hot')
    plt.colorbar(label='Intensity')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('NMF Reconstruction - m/z ~390 (±0.02)')
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, 'pca_cancer.png')
    plt.savefig(output_path, dpi=500, bbox_inches='tight')
    print(f"Image saved to: {output_path}")
    plt.close()
 

if __name__ == "__main__":
    main()
