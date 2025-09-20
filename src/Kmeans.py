from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import argparse
import os

parser = argparse.ArgumentParser(description="Kmeans")
parser.add_argument("--input", required=True, help="X to the input preprocess npy.")
parser.add_argument("--output", required=True, help="Directory to save plot.")
parser.add_argument("--coords", required=True, help="Coordinates")
parser.add_argument("--encoder", required=True, help="Path to the existing encoder .keras file")
parser.add_argument("--name",required=True,help="Job name")
parser.add_argument("--k",required=True,help="number of clusters")

args = parser.parse_args()

encoder = load_model(args.encoder)
print(f"Loaded encoder")
X = np.load(args.input,mmap_mode="r")
print(f"Loaded data")


latent_vectors = encoder.predict(X,verbose=0)
print(f"Shape: {latent_vectors.shape}")

optimal_k = int(args.k)
kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
cluster_labels = kmeans.fit_predict(latent_vectors)

coords = np.load(args.coords)  
print(f"Coordinates shape: {coords.shape}")

# Handle different coordinate formats
if coords.shape[1] == 3:
    xs, ys, _ = coords[:, 0], coords[:, 1], coords[:, 2]
elif coords.shape[1] == 2:
    xs, ys = coords[:, 0], coords[:, 1]

width = int(max(xs)) + 1
height = int(max(ys)) + 1
print(f"Image dimensions: {width} x {height}")

cluster_map = np.full((height, width), -1, dtype=int)
for idx, (x, y) in enumerate(zip(xs, ys)):
    cluster_map[int(y), int(x)] = cluster_labels[idx]

# Print cluster statistics
unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
print(f"\nCluster Statistics:")
print(f"Total number of pixels: {len(cluster_labels)}")
for cluster_id, count in zip(unique_clusters, cluster_counts):
    percentage = (count / len(cluster_labels)) * 100
    print(f"Cluster {cluster_id}: {count} pixels ({percentage:.2f}%)")

# Create output directory if it doesn't exist
os.makedirs(args.output, exist_ok=True)

colors = [
    "blue", "green","red" , "orange", "purple", "brown", "pink", "gray", "olive", "cyan"
]
# Ensure there are enough colors for the number of clusters
if optimal_k > len(colors):
    raise ValueError(f"Not enough colors defined for {optimal_k} clusters. Add more colors to the array.")

# Joint plot: All clusters together
plt.figure(figsize=(12, 10))
colored_cluster_map = np.full((height, width, 3), 0, dtype=np.uint8)  # Default to black background
for cluster_id in range(optimal_k):
    cluster_color = plt.colors.to_rgb(colors[cluster_id % len(colors)])  # Convert to RGB
    cluster_color = [int(c * 255) for c in cluster_color]  # Scale to 0-255
    colored_cluster_map[cluster_map == cluster_id] = cluster_color

plt.imshow(colored_cluster_map, origin='lower')
plt.title(f"Tissue Segmentation via K-Means (k={optimal_k}) on Latent Space - {args.name}")
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

# Save joint plot
joint_plot_path = os.path.join(args.output, f"{args.name}_kmeans_k{optimal_k}_joint.png")
plt.savefig(joint_plot_path, dpi=300, bbox_inches='tight')
print(f"Saved joint plot to: {joint_plot_path}")
plt.close()

# Individual cluster plots as subplots
import math
cols = min(3, optimal_k)  # Maximum 3 columns
rows = math.ceil(optimal_k / cols)

fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
if optimal_k == 1:
    axes = [axes]
elif rows == 1:
    axes = axes.reshape(1, -1)

for cluster_id in range(optimal_k):
    row = cluster_id // cols
    col = cluster_id % cols
    
    cluster_only_map = np.where(cluster_map == cluster_id, 1, 0)
    
    # Use the static color array for the title or legend
    axes[row, col].imshow(cluster_only_map, cmap=colors[cluster_id % len(colors)], vmin=0, vmax=1, origin='lower')
    axes[row, col].set_title(f"Cluster {cluster_id} ({colors[cluster_id % len(colors)]})")
    axes[row, col].set_xlabel('X Coordinate')
    axes[row, col].set_ylabel('Y Coordinate')
    

# Hide empty subplots if any
for i in range(optimal_k, rows * cols):
    row = i // cols
    col = i % cols
    axes[row, col].set_visible(False)

plt.suptitle(f"Individual Clusters - {args.name} (K-Means k={optimal_k})", fontsize=16)
plt.tight_layout()

# Save subplot figure
subplot_path = os.path.join(args.output, f"{args.name}_kmeans_k{optimal_k}_individual_clusters.png")
plt.savefig(subplot_path, dpi=300, bbox_inches='tight')
print(f"Saved individual clusters subplot to: {subplot_path}")
plt.close()

print(f"\nAll plots saved to: {args.output}")
print(f"Job '{args.name}' completed successfully!")