from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
import os

parser = argparse.ArgumentParser(description="T-SNE")
parser.add_argument("--input", required=True, help="X to the input preprocess npy.")
parser.add_argument("--output", required=True, help="Directory to save plot.")
parser.add_argument("--coords", required=True, help="Coordinates")
parser.add_argument("--encoder", required=True, help="Path to the existing encoder .keras file")
args = parser.parse_args()

print(f"Loading encoder from {args.encoder}")
encoder = load_model(args.encoder)
print("Encoder loaded.")

print(f"Loading input data from {args.input}")
X = np.load(args.input, mmap_mode="r")
print(f"Input data loaded. Shape: {X.shape}")

print(f"Loading coordinates from {args.coords}")
coords = np.load(args.coords, mmap_mode="r")
print(f"Coordinates loaded. Shape: {coords.shape}")

from sklearn.preprocessing import StandardScaler

print("Encoding input data...")
latent_vectors = encoder.predict(X, verbose=1)
print(f"Latent vectors shape: {latent_vectors.shape}")

print("Scaling latent vectors...")
scaler = StandardScaler()
latent_vectors = scaler.fit_transform(latent_vectors)
print("Latent vectors scaled.")

print("Running t-SNE...")
tsne = TSNE(n_components=2, perplexity=100, max_iter=2000, random_state=42)
latent_tsne = tsne.fit_transform(latent_vectors)
print("t-SNE completed.")

plt.figure(figsize=(10, 8))
scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1],
                      c=coords[:, 0], alpha=0.6, s=10, cmap='viridis')
plt.colorbar(scatter, label='Spatial X-coordinate')
plt.title('t-SNE of Latent Space Colored by Spatial Location')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')

output_path = os.path.join(args.output, "tnse_plot.png")
plt.savefig(output_path)
print(f"t-SNE plot saved to {output_path}")
plt.close()
