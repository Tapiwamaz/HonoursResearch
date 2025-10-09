from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
import argparse
import os

parser = argparse.ArgumentParser(description="T-SNE")
parser.add_argument("--input", required=True, help="X to the input preprocess npy.")
parser.add_argument("--output", required=True, help="Directory to save plot.")
# parser.add_argument("--coords", required=True, help="Coordinates")
parser.add_argument("--name", required=True, help="Name")
# parser.add_argument("--encoder", required=True, help="Path to the existing encoder .keras file")
parser.add_argument("--centroids", required=True, help="Path to the K-means centroids .npy file")

args = parser.parse_args()

# print(f"Loading encoder from {args.encoder}")
# encoder = load_model(args.encoder)
# print("Encoder loaded.")

print(f"Loading input data from {args.input}")
X = np.load(args.input, mmap_mode="r")
print(f"Input data loaded. Shape: {X.shape}")

# print(f"Loading coordinates from {args.coords}")
# coords = np.load(args.coords, mmap_mode="r")
# print(f"Coordinates loaded. Shape: {coords.shape}")

print(f"Loading K-means centroids from {args.centroids}")
centroids = np.load(args.centroids)
print(f"Centroids loaded. Shape: {centroids.shape}")

from sklearn.preprocessing import StandardScaler

# print("Encoding input data...")
# latent_vectors = encoder.predict(X, verbose=1)
# print(f"Latent vectors shape: {latent_vectors.shape}")
# del encoder

print("Scaling latent vectors...")
scaler = StandardScaler()
latent_vectors_scaled = scaler.fit_transform(X)
print("Latent vectors scaled.")

# Assign cluster labels based on centroids
print("Assigning cluster labels based on centroids...")
distances = euclidean_distances(X, centroids)
cluster_labels = np.argmin(distances, axis=1)
print(f"Cluster assignment completed. Found {len(np.unique(cluster_labels))} clusters.")

print("Running t-SNE...")
tsne = TSNE(n_components=2, perplexity=100, max_iter=2000, random_state=42)
latent_tsne = tsne.fit_transform(latent_vectors_scaled)
print("t-SNE completed.")

# Define colors for clusters
colors = [
    'blue', 'green', 'red', 'orange', 'purple', 
    'brown', 'pink', 'gray', 'olive', 'cyan',
    'magenta', 'yellow', 'black', 'navy', 'lime'
]

plt.figure(figsize=(12, 8))
unique_clusters = np.unique(cluster_labels)
for i, cluster_id in enumerate(unique_clusters):
    cluster_mask = cluster_labels == cluster_id
    plt.scatter(latent_tsne[cluster_mask, 0], latent_tsne[cluster_mask, 1],
                c=colors[i % len(colors)], alpha=0.6, s=10, 
                label=f'Cluster {cluster_id}')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title(f't-SNE of Latent Space Colored by K-means Clusters - {args.name}')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')

output_path = os.path.join(args.output, f"tsne_kmeans_clusters_{args.name}.png")
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"t-SNE plot with K-means clusters saved to {output_path}")
plt.close()

# Also create a version with continuous color mapping
plt.figure(figsize=(10, 8))
scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1],
                      c=cluster_labels, alpha=0.6, s=10, cmap='tab10')
plt.colorbar(scatter, label='Cluster ID')
plt.title(f't-SNE of Latent Space with K-means Cluster Colors - {args.name}')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')

output_path_continuous = os.path.join(args.output, f"tsne_kmeans_continuous_{args.name}.png")
plt.savefig(output_path_continuous, dpi=300)
print(f"t-SNE plot with continuous cluster colors saved to {output_path_continuous}")
plt.close()
