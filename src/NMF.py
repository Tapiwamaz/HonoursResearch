import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error,root_mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib


parser = argparse.ArgumentParser(description="Generate ion image plot.")
parser.add_argument("--input", required=True, help="Path to the input imzml and ibd files.")
parser.add_argument("--output", required=True, help="Directory to save the plot.")
parser.add_argument("--name", required=True, help="Name to save output")
parser.add_argument("--mzs", required=True, help="common mz channels")


args = parser.parse_args()


X = np.load(args.input)
X_temp, X_test = train_test_split(X, test_size=0.2, random_state=42)
X_train, X_val = train_test_split(X_temp, test_size=0.125, random_state=42)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_val: {X_val.shape}")
print(f"Shape of X_test: {X_test.shape}")



n_components = 200  
print(f"Number of components: {n_components}")
nmf = NMF(n_components=n_components, random_state=42, max_iter=50)
W = nmf.fit_transform(X)

W_train = nmf.fit_transform(X_train)
H = nmf.components_

# Save the NMF model
model_path = os.path.join(args.output, f"{args.name}_nmf_model.pkl")
joblib.dump(nmf, model_path)
print(f"NMF model saved to {model_path}")

# Reconstruct training set
X_train_reconstructed = np.dot(W_train, H)

# Calculate metrics for training set
mae_train = mean_absolute_error(X_train, X_train_reconstructed)
rmse_train = root_mean_squared_error(X_train, X_train_reconstructed)
mse_train = mean_squared_error(X_train, X_train_reconstructed)

print(f"Training Set Metrics:")
print(f"MSE: {mse_train:.8f}")
print(f"MAE: {mae_train:.8f}")
print(f"RMSE: {rmse_train:.8f}")


W_test = nmf.transform(X_test)  # Transform test set using the trained model
X_test_reconstructed = np.dot(W_test, H)

# Calculate metrics for test set
mae_test = mean_absolute_error(X_test, X_test_reconstructed)
rmse_test = root_mean_squared_error(X_test, X_test_reconstructed)
mse_test = mean_squared_error(X_test, X_test_reconstructed)

print(f"Test Set Metrics:")
print(f"MSE: {mse_test:.8f}")
print(f"MAE: {mae_test:.8f}")
print(f"RMSE: {rmse_test:.8f}")

# Combine reconstructed training and test sets for further analysis
X_reconstructed = np.vstack((X_train_reconstructed, X_test_reconstructed))



plt.figure(figsize=(15, 10))
sample_indices = [0,1,2]

# Calculate metrics for each sample
sample_mae = []
sample_rmse = []
sample_mse = []

for i in range(3):
    # Calculate metrics for this specific sample
    mae = mean_absolute_error(X[i], X_reconstructed[i])
    rmse = root_mean_squared_error(X[i], X_reconstructed[i])
    mse = mean_squared_error(X[i], X_reconstructed[i])
    
    sample_mae.append(mae)
    sample_rmse.append(rmse)
    sample_mse.append(mse)
    
    plt.subplot(2, 3, i+1)
    plt.plot(X[i], label='Original', alpha=0.8)
    plt.plot(X_reconstructed[i], label='Reconstructed', alpha=0.8, linestyle='--')
    plt.title(f'Sample {i}\nMAE: {mae:.2e}, RMSE: {rmse:.2e}')
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

