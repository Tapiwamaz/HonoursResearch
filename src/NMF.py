import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error,root_mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description="Generate ion image plot.")
parser.add_argument("--input", required=True, help="Path to the input imzml and ibd files.")
parser.add_argument("--output", required=True, help="Directory to save the plot.")
parser.add_argument("--name", required=True, help="Name to save output")
parser.add_argument("--mzs", required=True, help="common mz channels")


args = parser.parse_args()


X = np.load(args.input)


n_components = 1500  # Choose number of components
<<<<<<< HEAD
nmf = NMF(n_components=n_components, random_state=42, max_iter=10)
W = nmf.fit_transform(X)  # Sample weights
H = nmf.components_  # Component spectra
=======
X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

# Fit NMF on training data only
nmf = NMF(n_components=n_components, random_state=42, max_iter=200)
W_train = nmf.fit_transform(X_train)
H = nmf.components_
>>>>>>> main

# Reconstruct train and test sets
X_train_reconstructed = np.dot(W_train, H)
W_test = nmf.transform(X_test)
X_test_reconstructed = np.dot(W_test, H)

# Calculate metrics for train and test sets
mae_train = mean_absolute_error(X_train, X_train_reconstructed)
rmse_train = root_mean_squared_error(X_train, X_train_reconstructed)
mse_train = mean_squared_error(X_train, X_train_reconstructed)

mae_test = mean_absolute_error(X_test, X_test_reconstructed)
rmse_test = root_mean_squared_error(X_test, X_test_reconstructed)
mse_test = mean_squared_error(X_test, X_test_reconstructed)

print(f"Train Metrics:")
print(f"MSE: {mse_train:.8f}")
print(f"MAE: {mae_train:.8f}")
print(f"RMSE: {rmse_train:.8f}")

print(f"Test Metrics:")
print(f"MSE: {mse_test:.8f}")
print(f"MAE: {mae_test:.8f}")
print(f"RMSE: {rmse_test:.8f}")
   

for i in range(min(3, n_components)):
    top_samples = np.argsort(W[:, i])[-5:] 
    print(f"Component {i+1} - Top sample indices: {top_samples}")
    print(f"Component {i+1} - Top weights: {W[top_samples, i]}")

# 5. Original vs Reconstructed Spectra Comparison
plt.figure(figsize=(15, 10))
sample_indices = [0, len(X)//4, len(X)//2, 3*len(X)//4, len(X)-1]

# Calculate metrics for each sample
sample_mae = []
sample_rmse = []
sample_mse = []

for i, idx in enumerate(sample_indices):
    # Calculate metrics for this specific sample
    mae = mean_absolute_error(X[idx], X_reconstructed[idx])
    rmse = root_mean_squared_error(X[idx], X_reconstructed[idx])
    mse = mean_squared_error(X[idx], X_reconstructed[idx])
    
    sample_mae.append(mae)
    sample_rmse.append(rmse)
    sample_mse.append(mse)
    
    plt.subplot(2, 3, i+1)
    plt.plot(X[idx], label='Original', alpha=0.8)
    plt.plot(X_reconstructed[idx], label='Reconstructed', alpha=0.8, linestyle='--')
    plt.title(f'Sample {idx}\nMAE: {mae:.2e}, RMSE: {rmse:.2e}')
    plt.xlabel('m/z channel')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(args.output, f'{args.name}_original_vs_reconstructed.png'), dpi=300)
plt.close()


mzs = np.load(args.mzs)  # Load mzs from npy file

plt.figure(figsize=(15, 8))
n_top_components = min(5, n_components)
for i in range(n_top_components):
    # Find top m/z channels for this component
    top_mz_indices = np.argsort(H[i])[-20:]  # Top 20 m/z channels
    top_mz_values = mzs[top_mz_indices]      # Get actual m/z values

    plt.subplot(2, 3, i+1)
    plt.bar(range(len(top_mz_indices)), H[i][top_mz_indices])
    plt.title(f'Component {i+1} - Top m/z Channels')
    plt.xlabel('m/z')
    plt.ylabel('Loading')
    plt.xticks(range(len(top_mz_indices)), [f"{mz:.2f}" for mz in top_mz_values], rotation=45)
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(args.output, f'{args.name}_component_loadings.png'), dpi=300)
plt.close()

