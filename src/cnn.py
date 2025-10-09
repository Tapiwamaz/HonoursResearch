import numpy as np
import tensorflow as tf
import argparse
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error,root_mean_squared_error,mean_absolute_error
import math
import wandb
from wandb.integration.keras.callbacks import WandbMetricsLogger, WandbModelCheckpoint


# Add at the beginning of your script
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class SpectrumAutoencoder(Model):
    def __init__(self, latent_dim, input_shape):
        super(SpectrumAutoencoder, self).__init__()
        self.latent_dim = latent_dim

        # Encoder: CNN layers
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=input_shape),
            layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
            layers.MaxPooling1D(pool_size=2, padding='same'),
            layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            layers.MaxPooling1D(pool_size=2, padding='same'),
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu')  # Latent representation
        ])

        # Decoder: Reverse of the encoder
        self.decoder = tf.keras.Sequential([
            layers.Dense((input_shape[0] // 4) * 64, activation='relu'),  # Reverse of Flatten
            layers.Reshape((input_shape[0] // 4, 64)),  # Reshape to match the last Conv1D layer
            layers.UpSampling1D(size=2),  # Reverse of MaxPooling1D
            layers.Conv1DTranspose(64, kernel_size=3, activation='relu', padding='same'),
            layers.UpSampling1D(size=2),  # Reverse of MaxPooling1D
            layers.Conv1DTranspose(32, kernel_size=3, activation='relu', padding='same'),
            layers.Conv1DTranspose(input_shape[1], kernel_size=3, activation='sigmoid', padding='same'),  # Final layer
            layers.ZeroPadding1D(padding=(0, input_shape[0] - 67496))
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        # print(f"Encoded shape: {encoded.shape}")
        decoded = self.decoder(encoded)
        # print(f"Decoded shape: {decoded.shape}")
        return decoded

parser = argparse.ArgumentParser(description="Saving the AE encoder.")
parser.add_argument("--input", required=True, help="Path to the input preprocess npy.")
parser.add_argument("--output", required=True, help="Directory to save the encoder.")
parser.add_argument("--name", required=True, help="Name to save output")
parser.add_argument("--partitions",required=True,help="Number of paritions")
parser.add_argument("--partNum",required=True,help="The number of the partition we are on")

args = parser.parse_args()

part_num = int(args.partNum)
partitions = int(args.partitions)

X = np.load(args.input,mmap_mode='r')
start = 0
end = math.ceil(len(X)*(part_num/partitions))
print(f'Start to end indices: ({start},{end})')
X_subset = X[math.floor(len(X)*((part_num-1)/partitions)):math.ceil(len(X)*(part_num/partitions))]
del X
# intensities of each spectrum
print(f"Dataset partitioned into {partitions} number of chunks\nPartition: {part_num}")


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=900,
    decay_rate=0.96,
    staircase=True
)

X_temp, X_test = train_test_split(X_subset, test_size=0.2, random_state=42)
X_train, X_val = train_test_split(X_temp, test_size=0.125, random_state=42)  # 0.125 * 0.8 = 0.1
del X_subset
del X_temp

X_train = X_train.reshape((-1, X_train.shape[1], 1))  # Add channel dimension
X_val = X_val.reshape((-1, X_val.shape[1], 1))
X_test = X_test.reshape((-1, X_test.shape[1], 1))

print(f"Training set reshaped to: {X_train.shape}")
print(f"Validation set reshaped to: {X_val.shape}")
print(f"Test set reshaped to: {X_test.shape}")


wandb.init(
    project="CNN",
    config={
        "latent_dim": 200,
        "encoder_filters": [64, 128, 200],
        "decoder_filters": [200, 128, 64],
        "kernel_size": 3,
        "activation": "relu",
        "output_activation": "sigmoid",
        "optimizer": "adam",
        "learning_rate": lr_schedule,
        "loss": "mse",
        "metrics": ["mae", "mse"],
        "epochs": 25,
        "batch_size": 32,
        "early_stopping_patience": 5
    }
)

config = wandb.config

latent_dim = config.latent_dim
input_dim = X_train.shape[1:]

autoencoder = SpectrumAutoencoder(latent_dim=latent_dim, input_shape=input_dim)

autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=config.loss,
    metrics=config.metrics
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=config.early_stopping_patience,
    restore_best_weights=True
)


print("Starting training...")
history = autoencoder.fit(
    X_train, X_train,  
    epochs=config.epochs,  
    batch_size=config.batch_size,
    validation_data=(X_val, X_val),
    callbacks=[early_stopping,
               WandbMetricsLogger()],
    verbose=0
)



print("Training completed!\n")

autoencoder.summary()
print("\n")

test_loss, test_mae,test_mse = autoencoder.evaluate(X_test, X_test, verbose=0)
reconstructed = autoencoder.predict(X_test)

print(f"Test Loss: {test_loss:.10f}")
print(f"Test Loss (MAE): {test_mae:.10f}")
print(f"Test loss (MSE): {test_mse:.10f}")


mae_test = mean_absolute_error(X_test, reconstructed)
rmse_test = root_mean_squared_error(X_test, reconstructed)
mse_test = mean_squared_error(X_test, reconstructed)

print(f"Mean Absolute Error (MAE) on Test Data: {mae_test:.10f}")
print(f"Root Mean Squared Error (RMSE) on Test Data: {rmse_test:.10f}")
print(f"Mean Squared Error (MSE) on Test Data: {mse_test:.10f}")


encoder_save_path = os.path.join(args.output, f"{args.name}.keras")
autoencoder.encoder.save(encoder_save_path)
print(f"Encoder saved to {encoder_save_path}")

decoder_save_path = os.path.join(args.output, f"{args.name}_decoder.keras")
autoencoder.decoder.save(decoder_save_path)
print(f"Decoder saved")