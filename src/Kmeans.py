from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import argparse
import os

def get_optimal_k(data: np.ndarray, max_k: int = 10):
    """
    Function to get the optimal k using the elbow method.
    
    Parameters:
        data (np.ndarray): The input data for clustering.
        max_k (int): The maximum number of clusters to test.
        
    Returns:
        int: The optimal number of clusters.
    """
    inertias = []
    k_values = range(2, max_k + 1)

    silhouette_scores = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
        labels = kmeans.labels_
        silhouette_avg = silhouette_score(data, labels)
        silhouette_scores.append(silhouette_avg)

    
    # Plot elbow curve and silhouette scores
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Elbow plot
    ax1.plot(k_values, inertias, marker='o')
    ax1.set_title("Elbow Method for Optimal k")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia")
    ax1.set_xticks(k_values)
    ax1.grid(True)
    
    # Silhouette score plot
    ax2.plot(k_values, silhouette_scores, marker='o', color='red')
    ax2.set_title("Silhouette Score for Different k")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_xticks(k_values)
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(args.output, f"{args.name}_elbow_silhouette_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved elbow analysis plot to: {plot_path}")
    plt.close()

    print("Inspect the elbow plot to determine the optimal k.")
    return inertias


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

# inertias = get_optimal_k(data=latent_vectors,max_k=10)
# print(f"Inertias: {inertias}")

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
    (0, 0, 1),      # blue
    (0, 1, 0),      # green
    (1, 0, 0),      # red
    (1, 0.647, 0),  # orange
    (0.5, 0, 0.5),  # purple
    (0.647, 0.165, 0.165),  # brown
    (1, 0.753, 0.796),      # pink
    (0.5, 0.5, 0.5),        # gray
    (0.5, 0.5, 0),          # olive
    (0, 1, 1)       # cyan
]

# Joint plot: All clusters together
plt.figure(figsize=(12, 10))
colored_cluster_map = np.full((height, width, 3), 0, dtype=np.uint8)  # Default to black background
for cluster_id in range(optimal_k):
    cluster_color = colors[cluster_id]  # Convert to RGB
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
    
    cluster_only_map = np.zeros((height, width, 3), dtype=np.uint8)  # Default to black
    cluster_color = [int(c * 255) for c in colors[cluster_id % len(colors)]]  # Scale RGB to 0-255
    cluster_only_map[cluster_map == cluster_id] = cluster_color

    # Plot the cluster map
    axes[row, col].imshow(cluster_only_map, origin='lower')
    axes[row, col].set_title(f"Cluster {cluster_id}")
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

