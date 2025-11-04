import argparse
import numpy as np
import joblib
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

    nmf = joblib.load(args.decoder)
    print(f"NMF model loaded")
    print(f"Components shape: {nmf.components_.shape}")


    print(f"Decoding data in batches...")
    num_samples = X.shape[0]
    batch_size = 1000
    
    # Initialize array to store only the m/z range we need (11999:12001)
    intensities_sum = np.zeros(num_samples)
    
    # Process in batches
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        batch = X[i:end_idx]
        
        # Decode batch: X_reconstructed = W @ H
        decoded_batch = np.dot(batch, nmf.components_)
        
        # Extract and sum intensities at m/z ~390 (indices 11999:12001)
        intensities_sum[i:end_idx] = decoded_batch[:, :].sum(axis=1)
        
        print(f"Processed {end_idx}/{num_samples} samples", end='\r')
    
    print(f"\nDecoding complete!")

    del nmf
    coords = np.load(args.coords)

    # Create the spatial map
    plt.figure(figsize=(10, 8))
    plt.scatter(coords[:, 0], coords[:, 1], c=intensities_sum, cmap='cividis')
    plt.colorbar(label='Intensity')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('NMF Reconstruction - m/z ~390 (±0.02)')
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, 'nmf_390.png')
    plt.savefig(output_path, dpi=500, bbox_inches='tight')
    print(f"Image saved to: {output_path}")
    plt.close()
 

if __name__ == "__main__":
    main()