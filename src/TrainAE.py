import numpy as np
import tensorflow as tf
import argparse
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error
import math

# GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

parser = argparse.ArgumentParser(description="Continue training AE encoder.")
parser.add_argument("--input", required=True, help="Path to the input preprocess npy.")
parser.add_argument("--output", required=True, help="Directory to save the encoder.")
parser.add_argument("--name", required=True, help="Name to save output")
parser.add_argument("--partitions", required=True, help="Number of partitions")
parser.add_argument("--partNum", required=True, help="The number of the partition we are on")
parser.add_argument("--encoder", required=True, help="Path to the existing encoder .keras file")

args = parser.parse_args()

part_num = int(args.partNum)
partitions = int(args.partitions)

X = np.load(args.input, mmap_mode='r')
start = math.floor(len(X)*((part_num-1)/partitions))
end = math.ceil(len(X)*(part_num/partitions))
print(f'Strat to end indices: ({start},{end})')
X_subset = X[math.floor(len(X)*((part_num-1)/partitions)):math.ceil(len(X)*(part_num/partitions))]
# print(f"Dataset partitioned into {partitions} number of chunks\nPartition: {part_num}")

# Load the existing encoder
encoder = load_model(args.encoder)
latent_dim = encoder.output_shape[-1]
input_dim = encoder.input_shape[-1]
print(f"Latent dimension: {latent_dim}")
print(f"Input dimension: {input_dim}")


# Build a new decoder
decoder = tf.keras.Sequential([
    layers.Dense(1000, activation='tanh'),
    layers.Dense(5000, activation='tanh'),
    layers.Dense(input_dim, activation='relu'),
])

# Build the full autoencoder model
class SpectrumAutoencoder(Model):
    def __init__(self, encoder, decoder):
        super(SpectrumAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, intensities):
        encoded = self.encoder(intensities)
        decoded_intensities = self.decoder(encoded)
        return decoded_intensities

autoencoder = SpectrumAutoencoder(encoder, decoder)
autoencoder.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae', 'mse']
)

X_temp, X_test = train_test_split(X_subset, test_size=0.2, random_state=42)
X_train, X_val = train_test_split(X_temp, test_size=0.125, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
print(f"Test set shape: {X_test.shape}")

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

print("Starting training...")
history = autoencoder.fit(
    X_train, X_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_val, X_val),
    callbacks=[early_stopping],
    verbose=0
)
print("Training completed!")

# Evaluate and print metrics
test_loss, test_mae, test_mse = autoencoder.evaluate(X_test, X_test, verbose=0)
reconstructed = autoencoder.predict(X_test)
mae_test = mean_absolute_error(X_test, reconstructed)
rmse_test = root_mean_squared_error(X_test, reconstructed)
mse_test = mean_squared_error(X_test, reconstructed)

print(f"Test Loss: {test_loss:.10f}")
print(f"Test Loss (MAE): {test_mae:.10f}")
print(f"Test loss 2 (MSE): {test_mse:.10f}")
print(f"Mean Absolute Error (MAE) on Test Data: {mae_test:.10f}")
print(f"Root Mean Squared Error (RMSE) on Test Data: {rmse_test:.10f}")
print(f"Mean Squared Error (MSE) on Test Data: {mse_test:.10f}")

# Overwrite the encoder file with the updated encoder
encoder_save_path = args.encoder
autoencoder.encoder.save(encoder_save_path)
print(f"Encoder updated and saved to {encoder_save_path}")
