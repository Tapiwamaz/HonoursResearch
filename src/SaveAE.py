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




class SpectrumAutoencoder(Model):
    def __init__(self, latent_dim, n_peaks):
        super(SpectrumAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.n_peaks = n_peaks
        
        #kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.001)
        #bias_initializer = tf.keras.initializers.Zeros()
        
        self.encoder = tf.keras.Sequential([
            layers.Dense(2000, activation='tanh'),
            layers.Dense(1000, activation='tanh'),
            layers.Dense(latent_dim, activation='tanh'),
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(1000, activation='tanh'),
            layers.Dense(2000, activation='tanh'),
            layers.Dense(n_peaks, activation='relu'),
        ])

    def call(self, intensities):
        encoded = self.encoder(intensities)
        decoded_intensities = self.decoder(encoded)
        return decoded_intensities


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0015,
    decay_steps=100,
    decay_rate=0.95,
    staircase=False
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

wandb.init(
    project="Autoencoder Training",
    # track hyperparameters and run metadata with wandb.config
    config={
        "latent_dim": 200,
        "encoder_layer_1": 2000,
        "encoder_layer_2": 1000,
        "decoder_layer_1": 1000,
        "decoder_layer_2": 2000,
        "activation": "tanh",
        "output_activation": "relu",
        "optimizer": "adam",
        "learning_rate": lr_schedule,
        "loss": "weighted_mse_loss",
        "metrics": ["mae", "mse"],
        "epochs": 20,
        "batch_size": 32,
        "early_stopping_patience": 5
    }
)

config = wandb.config

# Split data into train, validation, and test sets (70% train, 10% val, 20% test)
X_temp, X_test = train_test_split(X_subset, test_size=0.2, random_state=42)
X_train, X_val = train_test_split(X_temp, test_size=0.125, random_state=42)  # 0.125 * 0.8 = 0.1
del X_subset

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
print(f"Test set shape: {X_test.shape}")

latent_dim = 200
input_dim = X_train.shape[1]  

autoencoder = SpectrumAutoencoder(latent_dim=latent_dim, n_peaks=input_dim)

autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=weighted_mse_loss,  
    metrics=['mae','mse']  
)

print(f"Autoencoder created with latent_dim={latent_dim}, input_dim={input_dim}")


early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)


print("Starting training...")
history = autoencoder.fit(
    X_train, X_train,  
    epochs=25,  
    batch_size=32,
    validation_data=(X_val, X_val),
    callbacks=[early_stopping,
               WandbMetricsLogger()],
    verbose=0
)

print("Training completed!\n")

autoencoder.encoder.summary()
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
