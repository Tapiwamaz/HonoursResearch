import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,root_mean_squared_error,mean_absolute_error
import joblib  # Import joblib for saving/loading models
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description="PCA")
parser.add_argument("--input", required=True, help="Path to the input")
parser.add_argument("--output", required=True, help="Directory to save output")
parser.add_argument("--name", required=True, help="Name to save output")
parser.add_argument("--encode", required=True, help="Data to encode")



args = parser.parse_args()


X = np.load(args.input,mmap_mode="r")



scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
del X
X_temp, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)
X_train, X_val = train_test_split(X_temp, test_size=0.125, random_state=42)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_val: {X_val.shape}")
print(f"Shape of X_test: {X_test.shape}")



n_components = 200  
pca = PCA(n_components=n_components)
print(f'Number of components: {n_components}')

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

pca_model_path = os.path.join(args.output, f"{args.name}_pca_model.joblib")
scaler_model_path = os.path.join(args.output, f"{args.name}_scaler_model.joblib")



X_train_reconstructed = pca.inverse_transform(X_train_pca)
X_test_reconstructed = pca.inverse_transform(X_test_pca)


X_train_reconstructed = pca.inverse_transform(X_train_pca)
X_test_reconstructed = pca.inverse_transform(X_test_pca)

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

frobenius_norm_train = np.linalg.norm(X_train - X_reconstructed_train_original_scale, 'fro')



non_zero_count_test = np.count_nonzero(X_test_reconstructed)
total_elements_test = X_test_reconstructed.size
print(f"\nTest set reconstruction statistics:")
print(f"Non-zero elements: {non_zero_count_test} out of {total_elements_test} ({non_zero_count_test/total_elements_test*100:.2f}%)")


print(f"Total variance explained by {n_components} components: {pca.explained_variance_ratio_.sum():.6f} ({pca.explained_variance_ratio_.sum()*100:.2f}%)")

print("\nTraining Data Metrics:")
print(f'MSE: {train_mse}')
print(f'MAE: {train_mae}')
print(f'RMSE: {train_rmse}')

print("Test Data Metrics:")
print(f'MSE: {test_mse}')
print(f'MAE: {test_mae}')
print(f'RMSE: {test_rmse}')


joblib.dump(pca, pca_model_path)
joblib.dump(scaler, scaler_model_path)

print(f"PCA model saved to {pca_model_path}")
print(f"Scaler model saved to {scaler_model_path}")


del X_reconstructed_train, X_temp, X_scaled ,X_reconstructed_train_original_scale, X_train,X_test,X_val

encodeData = np.load(args.encode)

encodeData_scaled = scaler.transform(encodeData)
encodeData_pca = pca.transform(encodeData_scaled)
encoded_output_path = os.path.join(args.output, f"{args.name}_encoded_pca.npy")
np.save(encoded_output_path, encodeData_pca)

print(f"Encoded data saved to {encoded_output_path}")
print(f"Encoded data shape: {encodeData_pca.shape}")
