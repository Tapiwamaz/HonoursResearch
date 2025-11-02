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

parser = argparse.ArgumentParser(description="Continue training AE encoder.")
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
print(f'Strat to end indices: ({start},{end})')
X_subset = X[start:end]
# print(f"Dataset partitioned into {partitions} number of chunks\nPartition: {part_num}")
del X

encoder_path = args.encoder
decoder_path = args.decoder

class SpectrumAutoencoder(Model):
    def __init__(self, latent_dim, n_peaks):
        super(SpectrumAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.n_peaks = n_peaks
        
        self.encoder = tf.keras.Sequential([
            layers.Dense(10000, activation='tanh'),
            layers.Dropout(0.3),
            layers.Dense(1000, activation='tanh'),
            layers.Dropout(0.3),
            layers.Dense(latent_dim, activation='tanh'),
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(1000, activation='tanh'),
            layers.Dropout(0.3),
            layers.Dense(10000, activation='tanh'),
            layers.Dropout(0.3),
            layers.Dense(n_peaks, activation='relu'),
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
    input_dim = encoder.input_shape[-1]
else:
    print("Creating a new SpectrumAutoencoder.")
    latent_dim = 200 
    input_dim = X_subset.shape[1]
    autoencoder = SpectrumAutoencoder(latent_dim, input_dim)
print(f"Latent dimension: {latent_dim}")
print(f"Input dimension: {input_dim}")

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0015,
    decay_steps=900,
    decay_rate=0.97,
    staircase=True
)

def weighted_mse_loss(y_true, y_pred):
    """
    Custom loss function that assigns more weight to non-zero values.
    """
    # Create a weight tensor: 10.0 for non-zero values, 1.0 for zeros.
    # The factor (10 here) is a hyperparameter you can tune.
    weight = 1.0 + 9.0 * tf.cast(y_true > 0, tf.float32)
    
    # Calculate the squared error
    squared_error = tf.square(y_true - y_pred)
    
    # Apply the weights and compute the mean
    weighted_squared_error = weight * squared_error
    return tf.reduce_mean(weighted_squared_error)

cosine_similarity_loss = tf.keras.losses.CosineSimilarity(axis=-1)

def combined_loss(y_true, y_pred):
    """
    Combines Mean Squared Error with Cosine Similarity.
    The lambda hyperparameter balances the two loss components.
    """
    lambda_val = 0.9 # Hyperparameter to tune
    
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    cosine_loss = cosine_similarity_loss(y_true, y_pred)
    
    return mse + lambda_val * cosine_loss

def intensity_weighted_mse_loss(y_true, y_pred):
    """
    Custom loss function that assigns more weight based on the intensity of the true signal.
    The alpha hyperparameter controls how much to penalize errors on high-intensity peaks.
    """
    alpha = 10.0  # Hyperparameter to tune
    # Weight is scaled by the true intensity. Using log1p for stability.
    weight = 1.0 + alpha * tf.math.log1p(y_true * 100) # Scale y_true if intensities are small
    
    squared_error = tf.square(y_true - y_pred)
    weighted_squared_error = weight * squared_error
    return tf.reduce_mean(weighted_squared_error)

wandb.init(
    project="CorrectAE",
    # track hyperparameters and run metadata with wandb.config
    config={
        "latent_dim": 200,
        "encoder_layer_1": 2000,
        "encoder_layer_2": 1000,
        "decoder_layer_1": 1000,
        "decoder_layer_2": 2000,
        "activation": "sigmoid",
        "output_activation": "relu",
        "optimizer": "adam",
        "learning_rate": lr_schedule,
        "loss": "mse",
        "metrics": ["mae", "mse"],
        "epochs": 30,
        "batch_size": 32,
        "early_stopping_patience": 5
    }
)

autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss="mse",
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
    patience=5,
    restore_best_weights=True
)

print("Starting training...")
history = autoencoder.fit(
    X_train, X_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_val, X_val),
    callbacks=[early_stopping,WandbMetricsLogger()],
    verbose=0
)
print("Training completed!")
autoencoder.summary()
del X_train,X_val

# Evaluate and print metrics
test_loss, test_mae, test_mse = autoencoder.evaluate(X_test, X_test, verbose=0)
reconstructed = autoencoder.predict(X_test)
mae_test = mean_absolute_error(X_test, reconstructed)
rmse_test = root_mean_squared_error(X_test, reconstructed)
mse_test = mean_squared_error(X_test, reconstructed)

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
