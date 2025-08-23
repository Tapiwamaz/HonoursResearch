import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.models import Model


parser = argparse.ArgumentParser(description="Don NMF.")
parser.add_argument("--input", required=True, help="Path to the input prerpocessed npy file.")
# parser.add_argument("--output", required=True, help="Directory to save the plot.")
# parser.add_argument("--name", required=True, help="Name to save output")
# parser.add_argument("--mzs", required=True, help="common mz channels")


args = parser.parse_args()


X = np.load(args.input)
# intensities of each spectrum

class SpectrumAutoencoder(Model):
    def __init__(self, latent_dim, n_peaks):
        super(SpectrumAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.n_peaks = n_peaks
        
        self.encoder = tf.keras.Sequential([
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(latent_dim, activation='relu'),
        ])
        
        self.decoder = tf.keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(n_peaks, activation='linear'),  
        ])

    def call(self, intensities):
        encoded = self.encoder(intensities)
        decoded_intensities = self.decoder(encoded)
        return decoded_intensities


# Split data into train and test sets
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

latent_dim = 64  
input_dim = X_train.shape[1]  


autoencoder = SpectrumAutoencoder(latent_dim=latent_dim, n_peaks=input_dim)

# Compile the model
autoencoder.compile(
    optimizer='adam',
    loss='mse',  # Mean Squared Error for reconstruction
    metrics=['mae','mse']  # Mean Absolute Error as additional metric
)

print(f"Autoencoder created with latent_dim={latent_dim}, input_dim={input_dim}")


# Training the autoencoder
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Check data before training
print(f"Training data stats:")
print(f"  Shape: {X_train.shape}")
print(f"  Min: {X_train.min():.6f}, Max: {X_train.max():.6f}")
print(f"  Mean: {X_train.mean():.6f}, Std: {X_train.std():.6f}")


print("Starting training...")
history = autoencoder.fit(
    X_train, X_train,  
    epochs=2,  
    batch_size=32,
    validation_data=(X_test, X_test),
    callbacks=[early_stopping],
    verbose=0

)

print("Training completed!")

# Evaluate the model
test_loss, test_mae,test_mse = autoencoder.evaluate(X_test, X_test, verbose=0)
print(f"Test Loss (MAE): {test_loss:.6f}")
print(f"Test Loss (MAE 2): {test_mae:.6f}")
print(f"Test Loss (MSE): {test_mse:.6f}")

