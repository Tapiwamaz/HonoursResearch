import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

def main():
    parser = argparse.ArgumentParser(description="Decode centroids using a Keras decoder model.")
    parser.add_argument('--centroids', type=str, required=True, help='Path to centroids .npy file')
    parser.add_argument('--decoder', type=str, required=True, help='Path to decoder .keras model')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output .txt file')
    args = parser.parse_args()

    centroids = np.load(args.centroids)
    print(f"Centroids shape: {centroids.shape}")

    decoder = load_model(args.decoder)
    decoder.summary()
    Y = decoder.predict(centroids)
    print(Y)

    os.makedirs(args.output_dir, exist_ok=True)
    for i in range(len(Y)):
        fname = f"c{i}.txt"
        output_path = os.path.join(args.output_dir, fname)
        np.savetxt(output_path, Y[i], fmt='%.4f')
        print(f"Decoded {i+1}/{len(Y)} saved to {output_path}")

if __name__ == "__main__":
    main()