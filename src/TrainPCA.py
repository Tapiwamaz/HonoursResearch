import numpy as np
import argparse
import os
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,root_mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description="Train NMF and save W/H matrices.")
parser.add_argument("--input", required=True, help="Path to input npy file.")
parser.add_argument("--classifier_data",required=True, help="Data to used in classifier")
parser.add_argument("--output", required=True, help="Directory to save results.")
parser.add_argument("--name", required=True, help="Base name for output files.")
parser.add_argument("--n_components", type=int, default=500, help="Number of NMF components.")
args = parser.parse_args()

X = np.load(args.input,mmap_mode='r')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_temp = train_test_split(X_scaled, test_size=0.3, random_state=42)
X_val, X_test = train_test_split(X_temp, test_size=2/3, random_state=42)

print(f'Train shape: {X_train.shape}')
print(f'Test shape: {X_test.shape}')
print(f'Val shape: {X_val.shape}')


pca = PCA(n_components=args.n_components,random_state=42)
print(f'Number of components: {args.n_components}')

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

X_train_reconstructed = pca.inverse_transform(X_train_pca)
X_test_reconstructed = pca.inverse_transform(X_test_pca)


X_train_reconstructed = pca.inverse_transform(X_train_pca)
X_test_reconstructed = pca.inverse_transform(X_test_pca)
# Reconstruct and calculate metrics for training data
X_reconstructed_train = pca.inverse_transform(X_train_pca)
X_reconstructed_train_original_scale = scaler.inverse_transform(X_reconstructed_train)

X_train_reconstructed_original = scaler.inverse_transform(X_train_reconstructed)
X_test_reconstructed_original = scaler.inverse_transform(X_test_reconstructed)

train_mse = mean_squared_error(X_train, X_train_reconstructed)
test_mse = mean_squared_error(X_test, X_test_reconstructed)
train_mae = mean_absolute_error(X_train, X_train_reconstructed)
test_mae = mean_absolute_error(X_test, X_test_reconstructed)
train_rmse = root_mean_squared_error(X_train, X_train_reconstructed)
test_rmse = root_mean_squared_error(X_test, X_test_reconstructed) 

print(f"Total variance explained by {args.n_components} components: {pca.explained_variance_ratio_.sum():.6f} ({pca.explained_variance_ratio_.sum()*100:.2f}%)")

print("\nTraining Data Metrics:")
print(f'MSE: {train_mse}')
print(f'MAE: {train_mae}')
print(f'RMSE: {train_rmse}')

print("Test Data Metrics:")
print(f'MSE: {test_mse}')
print(f'MAE: {test_mae}')
print(f'RMSE: {test_rmse}')

# Release memory for X and X_scaled
del X, X_scaled, X_train, X_val, X_test, X_train_pca, X_test_pca, X_train_reconstructed, X_test_reconstructed
del X_reconstructed_train, X_reconstructed_train_original_scale, X_train_reconstructed_original, X_test_reconstructed_original

# Load classifier data
classifier_data = np.load(args.classifier_data, mmap_mode='r')

# Scale classifier data
classifier_data_scaled = scaler.transform(classifier_data)

# Predict using PCA
classifier_data_pca = pca.transform(classifier_data_scaled)

# Release memory for classifier_data_scaled
del classifier_data_scaled

# Save the PCA-transformed classifier data
output_path = os.path.join(args.output, f"{args.name}_classifier_data_pca.npy")
np.save(output_path, classifier_data_pca)

print(f"Classifier data PCA transformation saved to {output_path}")

