import numpy as np
import tensorflow as tf
import argparse
import os
from tensorflow.keras.models import load_model

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


parser = argparse.ArgumentParser(description="Encode some data")
parser.add_argument("--input", required=True, help="Path to the input npy.")
parser.add_argument("--output", required=True, help="Directory to save the output.")
parser.add_argument("--name", required=True, help="Name to save output")
parser.add_argument("--encoder", required=True, help="Path to the existing encoder .keras file")

args = parser.parse_args()

X = np.load(args.input, mmap_mode='r')
encoder = load_model(args.encoder)


def encode_in_batches(encoder, X, batch_size=1000):
    latent_vectors = []
    num_samples = X.shape[0]
    
    for i in range(0, num_samples, batch_size):
        batch_end = min(i + batch_size, num_samples)
        batch = X[i:batch_end]
        batch_latent = encoder.predict(batch, verbose=0)
        latent_vectors.append(batch_latent)
    
    return np.concatenate(latent_vectors, axis=0)


latent_vectors = encode_in_batches(encoder, X, batch_size=500)
print(f"Shape: {latent_vectors.shape}")

del X
del encoder

ouput_path = os.path.join(args.output, f"{args.name}.npy")

np.save(ouput_path,latent_vectors)
print(f"Saved data to {ouput_path}")
