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
    parser.add_argument('--cnn', type=bool,default=False, help='Set this flag if using a CNN decoder (outputs 3D tensors)')
    args = parser.parse_args()

    X = np.load(args.x)
    print(f"Latent embeddings: {X.shape}")

    decoder = load_model(args.decoder)
    print(f"Decoder loaded")
    decoder.summary()

    print(f"Decoding data in batches...")
    num_samples = X.shape[0]
    batch_size = 100
    
    # Initialize array to store only the m/z range we need (11999:12001)
    intensities_sum = np.zeros(num_samples)
    
    # Process in batches
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        batch = X[i:end_idx]
        
        # Decode batch
        decoded_batch = decoder.predict(batch, verbose=0)
        
        if args.cnn and len(decoded_batch.shape) == 3:
            decoded_batch = decoded_batch.squeeze(-1)  # Remove last dimension
        
        intensities_sum[i:end_idx] = decoded_batch[:, :].sum(axis=1)
        
        print(f"Processed {end_idx}/{num_samples} samples", end='\r')
        print(f"Intensity stats - min: {np.min(decoded_batch)}, max: {np.max(decoded_batch)}, mean: {np.mean(decoded_batch)}")
    
    print(f"\nDecoding complete!")

    del decoder
    coords = np.load(args.coords)
    print(f"Coordinates shape: {coords.shape}")

    print(f"Intensity stats - min: {intensities_sum.min()}, max: {intensities_sum.max()}, mean: {intensities_sum.mean()}")
    # Create the spatial map
    plt.figure(figsize=(10, 8))
    plt.scatter(coords[:, 0], coords[:, 1], c=intensities_sum, cmap='hot')
    plt.colorbar(label='Intensity')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('AE Reconstruction - m/z ~390 (±0.02)')
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, 'cnn_cancer.png')
    plt.savefig(output_path, dpi=500, bbox_inches='tight')
    print(f"Image saved to: {output_path}")
    plt.close()
    print(intensities_sum)
 

if __name__ == "__main__":
    main()
