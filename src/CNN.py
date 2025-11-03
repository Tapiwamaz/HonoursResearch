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
import wandb
from wandb.integration.keras.callbacks import WandbMetricsLogger, WandbModelCheckpoint

# GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

parser = argparse.ArgumentParser(description="Continue training CNN AE.")
parser.add_argument("--input", required=True, help="Path to the input preprocess npy.")
parser.add_argument("--output", required=True, help="Directory to save the encoder.")
parser.add_argument("--name", required=True, help="Name to save output")
parser.add_argument("--partitions", required=True, help="Number of partitions")
parser.add_argument("--partNum", required=True, help="The number of the partition we are on")
parser.add_argument("--encoder", required=False, default=None, help="Path to the existing encoder .keras file")
parser.add_argument("--decoder", required=False, default=None, help="Path to the existing decoder .keras file")


args = parser.parse_args()

part_num = int(args.partNum)
partitions = int(args.partitions)

X = np.load(args.input, mmap_mode='r')
start = math.floor(len(X)*((part_num-1)/partitions))
end = math.ceil(len(X)*(part_num/partitions))
print(f'Start to end indices: ({start},{end})')
X_subset = X[start:end]
del X

# Reshape for CNN
X_subset = np.expand_dims(X_subset, axis=-1)

encoder_path = args.encoder
decoder_path = args.decoder

class SpectrumCNNAutoencoder(Model):
    def __init__(self, latent_dim, input_shape):
        super(SpectrumCNNAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_shape_cnn = input_shape

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=self.input_shape_cnn),
            layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
            layers.MaxPooling1D(pool_size=2, padding='same'),
            layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            layers.MaxPooling1D(pool_size=2, padding='same'),
            layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            layers.MaxPooling1D(pool_size=2, padding='same'),
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu')
        ])

        pre_flatten_shape = self.encoder.layers[-3].output_shape[1:]

        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.Dense(np.prod(pre_flatten_shape), activation='relu'),
            layers.Reshape(pre_flatten_shape),
            layers.Conv1DTranspose(128, kernel_size=3, activation='relu', padding='same'),
            layers.UpSampling1D(size=2),
            layers.Conv1DTranspose(64, kernel_size=3, activation='relu', padding='same'),
            layers.UpSampling1D(size=2),
            layers.Conv1DTranspose(32, kernel_size=3, activation='relu', padding='same'),
            layers.UpSampling1D(size=2),
            layers.Conv1DTranspose(1, kernel_size=3, activation='relu', padding='same'),
        ])

    def call(self, intensities):
        encoded = self.encoder(intensities)
        decoded_intensities = self.decoder(encoded)
        return decoded_intensities

class PretrainedAE(Model):
    def __init__(self, encoder, decoder):
        super(PretrainedAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, intensities):
        encoded = self.encoder(intensities)
        decoded_intensities = self.decoder(encoded)
        return decoded_intensities

if encoder_path and decoder_path and os.path.exists(encoder_path) and os.path.exists(decoder_path):
    print("Loading existing encoder and decoder.")
    encoder = load_model(encoder_path)
    decoder = load_model(decoder_path)
    autoencoder = PretrainedAE(encoder, decoder)
    latent_dim = encoder.output_shape[-1]
    input_dim = encoder.input_shape[-2]
else:
    print("Creating a new SpectrumCNNAutoencoder.")
    latent_dim = 200
    input_dim = X_subset.shape[1]
    autoencoder = SpectrumCNNAutoencoder(latent_dim, (input_dim, 1))

print(f"Latent dimension: {latent_dim}")
print(f"Input dimension: {input_dim}")

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0015,
    decay_steps=900,
    decay_rate=0.97,
    staircase=True
)

def weighted_mse_loss(y_true, y_pred):
    weight = 1.0 + 9.0 * tf.cast(y_true > 0, tf.float32)
    squared_error = tf.square(y_true - y_pred)
    weighted_squared_error = weight * squared_error
    return tf.reduce_mean(weighted_squared_error)

wandb.init(
    project="CNN-AE",
    config={
        "latent_dim": latent_dim,
        "encoder_conv_1": 32,
        "encoder_conv_2": 64,
        "encoder_conv_3": 128,
        "decoder_conv_1": 128,
        "decoder_conv_2": 64,
        "decoder_conv_3": 32,
        "activation": "relu",
        "output_activation": "relu",
        "optimizer": "adam",
        "learning_rate": "ExponentialDecay",
        "loss": "weighted_mse_loss",
        "metrics": ["mae", "mse"],
        "epochs": 30,
        "batch_size": 32,
        "early_stopping_patience": 9
    }
)

autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=weighted_mse_loss,
    metrics=['mae', 'mse']
)

X_temp, X_test = train_test_split(X_subset, test_size=0.2, random_state=42)
X_train, X_val = train_test_split(X_temp, test_size=0.125, random_state=42)
del X_subset
del X_temp

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
print(f"Test set shape: {X_test.shape}")

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=9,
    restore_best_weights=True
)

print("Starting training...")
history = autoencoder.fit(
    X_train, X_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_val, X_val),
    callbacks=[early_stopping, WandbMetricsLogger()],
    verbose=1
)
print("Training completed!")
autoencoder.summary()
del X_train, X_val

# Evaluate and print metrics
test_loss, test_mae, test_mse = autoencoder.evaluate(X_test, X_test, verbose=0)
reconstructed = autoencoder.predict(X_test)
mae_test = mean_absolute_error(np.squeeze(X_test, axis=-1), np.squeeze(reconstructed, axis=-1))
rmse_test = root_mean_squared_error(np.squeeze(X_test, axis=-1), np.squeeze(reconstructed, axis=-1))
mse_test = mean_squared_error(np.squeeze(X_test, axis=-1), np.squeeze(reconstructed, axis=-1))

print(f"Test Loss: {test_loss:.10f}")
print(f"Test Loss (MAE): {test_mae:.10f}")
print(f"Test loss (MSE): {test_mse:.10f}")
print(f"Mean Absolute Error (MAE) on Test Data: {mae_test:.10f}")
print(f"Root Mean Squared Error (RMSE) on Test Data: {rmse_test:.10f}")
print(f"Mean Squared Error (MSE) on Test Data: {mse_test:.10f}")

# Save the updated encoder and decoder
encoder_save_path = os.path.join(args.output, f"{args.name}-encoder.keras")
autoencoder.encoder.save(encoder_save_path)
print(f"Encoder updated and saved to {encoder_save_path}")

decoder_save_path = os.path.join(args.output, f"{args.name}-decoder.keras")
autoencoder.decoder.save(decoder_save_path)
print(f"Decoder updated and saved to {decoder_save_path}")
