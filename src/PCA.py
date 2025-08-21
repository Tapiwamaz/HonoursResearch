import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,root_mean_squared_error,mean_absolute_error


parser = argparse.ArgumentParser(description="Generate ion image plot.")
parser.add_argument("--input", required=True, help="Path to the input imzml and ibd files.")
parser.add_argument("--output", required=True, help="Directory to save the plot.")
parser.add_argument("--name", required=True, help="Name to save output")

args = parser.parse_args()


X = np.load(args.input)



scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
n_components = 1500  # Choose number of components to keep
pca = PCA(n_components=n_components)


X_pca = pca.fit_transform(X_scaled)
X_reconstructed = pca.inverse_transform(X_pca)
X_reconstructed_original_scale = scaler.inverse_transform(X_reconstructed)

 
pca_loss_mse = mean_squared_error(X_scaled, X_reconstructed)   
pca_loss_mae = mean_absolute_error(X_scaled, X_reconstructed)   
pca_loss_rmse = root_mean_squared_error(X_scaled, X_reconstructed)   


# Print individual variance contribution of each PC
print("Individual Principal Component Analysis:")
print("============================================================")


var_ratio = pca.explained_variance_ratio_[0]
var_value = pca.explained_variance_[0]
print(f"PC{1}:")
print(f"  Explained Variance: {var_value:.6f}")
print(f"  Explained Variance Ratio: {var_ratio:.6f} ({var_ratio*100:.2f}%)")
print()

print(f"Total variance explained by {n_components} components: {pca.explained_variance_ratio_.sum():.6f} ({pca.explained_variance_ratio_.sum()*100:.2f}%)")

print(f'\nMSE: {pca_loss_mse}')
print(f'MAE: {pca_loss_mae}')
print(f'RMSE: {pca_loss_rmse}\n')


plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title(f'PCA of Spectral Data {args.name}')

output_path = os.path.join(args.output, f"PCA_{args.name}.png")
plt.savefig(output_path)
print(f"Plot saved to {output_path}")