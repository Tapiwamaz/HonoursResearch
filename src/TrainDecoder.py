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

parser = argparse.ArgumentParser(description="Train decoder on encoded data.")
parser.add_argument("--input", required=True, help="Path to the encoded data npy.")
parser.add_argument("--output", required=True, help="Directory to save the decoder.")
parser.add_argument("--name", required=True, help="Name to save output")
args = parser.parse_args()

X_encoded = np.load(args.input, mmap_mode='r')
print(f"Loaded encoded data with shape: {X_encoded.shape}")

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.002,
    decay_steps=500,
    decay_rate=0.96,
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

wandb.init(
    project="Pretraining",
    # track hyperparameters and run metadata with wandb.config
    config={
        "latent_dim": 250,
        "encoder_layer_1": 2000,
        "encoder_layer_2": 1000,
        "decoder_layer_1": 1000,
        "decoder_layer_2": 2000,
        "activation": "tanh",
        "output_activation": "relu",
        "optimizer": "adam",
        "learning_rate": lr_schedule,
        "loss": "wmse",
        "metrics": ["mae", "mse"],
        "epochs": 20,
        "batch_size": 32,
        "early_stopping_patience": 5
    }
)

# Build a new decoder
decoder = tf.keras.Sequential([
            layers.Dense(1000, activation='tanh'),
            layers.Dropout(0.3),
            layers.Dense(2000, activation='tanh'),
            layers.Dropout(0.3),
            layers.Dense(X_encoded.shape[-1], activation='relu'),  
])

decoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=weighted_mse_loss,
    metrics=['mae', 'mse']
)

X_temp, X_test = train_test_split(X_encoded, test_size=0.2, random_state=42)
X_train, X_val = train_test_split(X_temp, test_size=0.125, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
print(f"Test set shape: {X_test.shape}")

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

print("Starting training...")
history = decoder.fit(
    X_train, X_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, X_val),
    callbacks=[early_stopping, WandbMetricsLogger()],
    verbose=0
)
print("Training completed!")

# Evaluate and print metrics
test_loss, test_mae, test_mse = decoder.evaluate(X_test, X_test, verbose=0)
reconstructed = decoder.predict(X_test)
mae_test = mean_absolute_error(X_test, reconstructed)
rmse_test = root_mean_squared_error(X_test, reconstructed)
mse_test = mean_squared_error(X_test, reconstructed)

print(f"Test Loss: {test_loss:.10f}")
print(f"Test Loss (MAE): {test_mae:.10f}")
print(f"Test loss (MSE): {test_mse:.10f}")
print(f"Mean Absolute Error (MAE) on Test Data: {mae_test:.10f}")
print(f"Root Mean Squared Error (RMSE) on Test Data: {rmse_test:.10f}")
print(f"Mean Squared Error (MSE) on Test Data: {mse_test:.10f}")

# Save the trained decoder
decoder_save_path = os.path.join(args.output, f"{args.name}_decoder.keras")
decoder.save(decoder_save_path)
print(f"Decoder saved to {decoder_save_path}")
