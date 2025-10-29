import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error,root_mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import joblib


parser = argparse.ArgumentParser(description="NMF full")
parser.add_argument("--input", required=True, help="Path to the input imzml and ibd files.")
parser.add_argument("--output", required=True, help="Directory to save the plot.")
parser.add_argument("--name", required=True, help="Name to save output")
parser.add_argument("--encode", required=True,help="Data to encode and return e.g cancer-h5")


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

X_train_reconstructed = np.dot(W_train, H)

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



print("Freeing some memory...")
del X,X_temp,X_test,X_train,X_train_reconstructed,X_val
print("Done\n")
print("Loading data to encode...")

X = np.load(args.encode)
print(f"Encoding data with shape: {X.shape}")

# Transform the data using the trained NMF model
W_encoded = nmf.transform(X)
print(f"Encoded data shape: {W_encoded.shape}")

# Save the encoded data
output_path = os.path.join(args.output, f'{args.name}_nmf.npy')
np.save(output_path, W_encoded)
print(f"Encoded data saved to: {output_path}")




