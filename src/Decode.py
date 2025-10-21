import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt

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
        fname = f"c{i+1}y.txt"
        output_path = os.path.join(args.output_dir, fname)
        np.savetxt(output_path, Y[i], fmt='%.4f')
        print(f"Decoded {i+1}/{len(Y)} saved to {output_path}")

    # plot the first three decoded centroids (blue, green, red) and save as graph.png
    
    num_to_plot = min(3, len(Y))
    colors = ['blue', 'green', 'red']
    fig, axs = plt.subplots(num_to_plot, 1, figsize=(8, 3 * num_to_plot), sharex=True)
    if num_to_plot == 1:
        axs = [axs]
    for idx in range(num_to_plot):
        vec = np.squeeze(Y[idx]).reshape(-1)
        x = np.linspace(150, 1500, len(vec))
        axs[idx].plot(x, vec, color=colors[idx], label=f'centroid {idx}')
        axs[idx].set_xlim(150, 1500)
        axs[idx].set_ylabel('Label intensity')
        axs[idx].legend()
        axs[idx].grid(True, linestyle='--', alpha=0.3)
    axs[-1].set_xlabel('Range')
    fig.suptitle('Decoded centroids')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_img = os.path.join(args.output_dir, 'graph.png')
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to {out_img}")

if __name__ == "__main__":
    main()