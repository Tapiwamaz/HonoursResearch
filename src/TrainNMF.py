import numpy as np
import argparse
import os
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib

parser = argparse.ArgumentParser(description="Train NMF and save W/H matrices.")
parser.add_argument("--input", required=True, help="Path to input npy file.")
parser.add_argument("--output", required=True, help="Directory to save results.")
parser.add_argument("--name", required=True, help="Base name for output files.")
parser.add_argument("--n-components", type=int, default=500, help="Number of NMF components.")
args = parser.parse_args()

X = np.load(args.input,mmap_mode='r')
X_train, X_temp = train_test_split(X, test_size=0.3, random_state=42)
X_val, X_test = train_test_split(X_temp, test_size=2/3, random_state=42)

nmf = NMF(n_components=args.n_components, random_state=42, max_iter=100)
W_train = nmf.fit_transform(X_train)
H = nmf.components_

# Save W and H
# np.save(os.path.join(args.output, f"{args.name}_W_train.npy"), W_train)
# np.save(os.path.join(args.output, f"{args.name}_H.npy"), H)

# Reconstruct and evaluate
X_train_recon = np.dot(W_train, H)
mae_train = mean_absolute_error(X_train, X_train_recon)
mse_train = mean_squared_error(X_train, X_train_recon)
print(f"Train MAE: {mae_train:.10f}, MSE: {mse_train:.10}")

# Transform val/test using trained NMF
W_val = nmf.transform(X_val)
X_val_recon = np.dot(W_val, H)
mae_val = mean_absolute_error(X_val, X_val_recon)
print(f"Val MAE: {mae_val:.10f}")

W_test = nmf.transform(X_test)
X_test_recon = np.dot(W_test, H)
mae_test = mean_absolute_error(X_test, X_test_recon)
print(f"Test MAE: {mae_test:.10f}")

# Save W for val/test if needed
# np.save(os.path.join(args.output, f"{args.name}_W_val.npy"), W_val)
# np.save(os.path.join(args.output, f"{args.name}_W_test.npy"), W_test)


