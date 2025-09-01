import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error,root_mean_squared_error,mean_absolute_error




parser = argparse.ArgumentParser(description="Don NMF.")
parser.add_argument("--input", required=True, help="Path to the input prerpocessed npy file.")
parser.add_argument("--output", required=True, help="Directory to save the plot.")
parser.add_argument("--name", required=True, help="Name to save output")
parser.add_argument("--mzs", required=True, help="common mz channels")


args = parser.parse_args()


X = np.load(args.input)
# intensities of each spectrum

class SpectrumAutoencoder(Model):
    def __init__(self, latent_dim, n_peaks):
        super(SpectrumAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.n_peaks = n_peaks
        
        self.encoder = tf.keras.Sequential([
            layers.Dense(5000, activation='relu'),
            layers.Dense(1000, activation='relu'),
            layers.Dense(latent_dim, activation='relu'),
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(1000, activation='relu'),
            layers.Dense(5000, activation='relu'),
            layers.Dense(n_peaks, activation='relu'),  
        ])

    def call(self, intensities):
        encoded = self.encoder(intensities)
        decoded_intensities = self.decoder(encoded)
        return decoded_intensities



# Split data into train and test sets
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

latent_dim = 500 
input_dim = X_train.shape[1]  


autoencoder = SpectrumAutoencoder(latent_dim=latent_dim, n_peaks=input_dim)

autoencoder.compile(
    optimizer='adam',
    loss='mse',  
    metrics=['mae','mse']  
)

print(f"Autoencoder created with latent_dim={latent_dim}, input_dim={input_dim}")


# Training the autoencoder
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

print(f"Training data stats:")
print(f"  Shape: {X_train.shape}")
print(f"  Min: {X_train.min():.6f}, Max: {X_train.max():.6f}")
print(f"  Mean: {X_train.mean():.6f}, Std: {X_train.std():.6f}")


print("Starting training...")
history = autoencoder.fit(
    X_train, X_train,  
    epochs=10,  
    batch_size=32,
    validation_data=(X_test, X_test),
    callbacks=[early_stopping],
    verbose=0

)

print("Training completed!")

# Evaluate the model
test_loss, test_mae,test_mse = autoencoder.evaluate(X_test, X_test, verbose=0)
reconstructed = autoencoder.predict(X_test)


# Compute the Frobenius norm of the test data and the reconstructed test data
reconstructed = autoencoder.predict(X_test)
frobenius_norm_original = np.linalg.norm(X_test, 'fro')
frobenius_norm_reconstructed = np.linalg.norm(reconstructed, 'fro')
frobenius_norm_difference = frobenius_norm_original - frobenius_norm_reconstructed

print(f"Frobenius Norm of Original Test Data: {frobenius_norm_original:.10f}")
print(f"Frobenius Norm of Reconstructed Test Data: {frobenius_norm_reconstructed:.10f}")
print(f"Difference in Frobenius Norms: {frobenius_norm_difference:.10f}")
print(f"Test Loss: {test_loss:.10f}")
print(f"Test Loss (MAE): {test_mae:.10f}")
print(f"Test loss 2 (MSE): {test_mse:.10f}")


mae_test = mean_absolute_error(X_test, reconstructed)
rmse_test = root_mean_squared_error(X_test, reconstructed)
mse_test = mean_squared_error(X_test, reconstructed)

print(f"Mean Absolute Error (MAE) on Test Data: {mae_test:.10f}")
print(f"Root Mean Squared Error (RMSE) on Test Data: {rmse_test:.10f}")
print(f"Mean Squared Error (MSE) on Test Data: {mse_test:.10f}")

# Plot and save the reconstructed vs original spectrum for 5 random spectra

# Create a figure with 5 subplots
fig, axes = plt.subplots(5, 1, figsize=(10, 20))
fig.suptitle("Original vs Reconstructed Spectra", fontsize=16)


mzs = args.mzs
for i, ax in enumerate(axes.flat):
    orig = X_test[i]
    recon = reconstructed[i]
    mse = np.mean((orig - recon) ** 2)
    rmse = np.sqrt(mse)

    ax.plot(orig, label='Original Spectrum', linewidth=2)
    ax.plot(recon, label='AE Reconstruction', linewidth=2, alpha=0.6)
    ax.set_title(f'Spectrum {mzs[i]}\nMSE: {mse:.2e}, RMSE: {rmse:.2e}')
    ax.set_xlabel('Index')
    ax.set_ylabel('Intensity')
    ax.grid(True, alpha=0.7)
    if i == 0:
        ax.legend()

# Adjust layout and save the plot
plt.tight_layout(rect=[0, 0, 1, 0.96])
output_path = os.path.join(args.output, f"{args.name}_AE_reconstructed_vs_original.png")
plt.savefig(output_path)
print(f"Plot saved to {output_path}")
