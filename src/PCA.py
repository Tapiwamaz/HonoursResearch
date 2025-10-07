import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,root_mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description="PCA")
parser.add_argument("--input", required=True, help="Path to the input")
parser.add_argument("--output", required=True, help="Directory to save output")
parser.add_argument("--name", required=True, help="Name to save output")
parser.add_argument("--mzs", required=True, help="common mz channels")


args = parser.parse_args()


X = np.load(args.input)



from sklearn.externals import joblib  # Import joblib for saving/loading models

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
del X
X_temp, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)
X_train, X_val = train_test_split(X_temp, test_size=0.125, random_state=42)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_val: {X_val.shape}")
print(f"Shape of X_test: {X_test.shape}")


# Apply PCA
n_components = 200  
pca = PCA(n_components=n_components)
print(f'Number of components: {n_components}')

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Save the PCA model and scaler for future use
pca_model_path = os.path.join(args.output, f"{args.name}_pca_model.joblib")
scaler_model_path = os.path.join(args.output, f"{args.name}_scaler_model.joblib")

joblib.dump(pca, pca_model_path)
joblib.dump(scaler, scaler_model_path)

print(f"PCA model saved to {pca_model_path}")
print(f"Scaler model saved to {scaler_model_path}")

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

# Apply PCA transformation to test data
X_pca_test = pca.transform(X_test)

num_spectra = 3
fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True, sharey=True)

for i, ax in enumerate(axes.flat):
    orig = X_test[i]
    recon = X_test_reconstructed_original[i]
    mse = mean_squared_error(orig.reshape(1, -1), recon.reshape(1, -1))
    rmse = np.sqrt(mse)
    
    ax.plot(orig, label='Original Spectrum', linewidth=2)
    ax.plot(recon, label=f'PCA Reconstruction ({n_components} components)', linewidth=2, alpha=0.6)
    ax.set_title(f'Spectrum {i}\nMSE: {mse:.2e}, RMSE: {rmse:.2e}')
    ax.set_xlabel('m/z index')
    ax.set_ylabel('Intensity')
    ax.grid(True, alpha=0.7)
    if i == 0:
        ax.legend()

plt.tight_layout()
output_path = os.path.join(args.output, f"pca_spectra_reconstruction_{args.name}.png")
plt.savefig(output_path)
print(f"Plot saved to {output_path}")

# frobenius_norm_test = np.linalg.norm(X_test - X_reconstructed_test_original_scale, 'fro')


print(f"Total variance explained by {n_components} components: {pca.explained_variance_ratio_.sum():.6f} ({pca.explained_variance_ratio_.sum()*100:.2f}%)")

print("\nTraining Data Metrics:")
print(f'MSE: {train_mse}')
print(f'MAE: {train_mae}')
print(f'RMSE: {train_rmse}')
# print(f"Frobenius norm (PCA reconstruction error): {frobenius_norm_train}\n")

print("Test Data Metrics:")
print(f'MSE: {test_mse}')
print(f'MAE: {test_mae}')
print(f'RMSE: {test_rmse}')
# print(f"Frobenius norm (PCA reconstruction error): {frobenius_norm_test}\n")



# Load the mzs file to use as the x-axis
mzs = np.load(args.mzs)

fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, sharey=True)

for i in range(3):  # Plot the first 3 spectra
    axes[i].plot(mzs, X_test[i], label="Original", alpha=0.7)
    axes[i].plot(mzs, X_test_reconstructed_original[i], label="Reconstructed by PCA", alpha=0.7)
    axes[i].set_title(f"Spectrum {i + 1}")
    axes[i].legend()

fig.suptitle(f"Original vs Reconstructed Spectra ({args.name})", fontsize=16)
fig.supxlabel("m/z")
fig.supylabel("Intensity")

output_path = os.path.join(args.output, f"pca_spectra_reconstruction_{args.name}.png")
plt.savefig(output_path)
print(f"Plot saved to {output_path}")
