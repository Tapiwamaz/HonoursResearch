import numpy as np
import tensorflow as tf
import argparse
import os
from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser(description="Encode some data")
parser.add_argument("--input", required=True, help="Path to the input npy.")
parser.add_argument("--output", required=True, help="Directory to save the output.")
parser.add_argument("--name", required=True, help="Name to save output")
parser.add_argument("--encoder", required=True, help="Path to the existing encoder .keras file")

args = parser.parse_args()

X = np.load(args.input, mmap_mode='r')
encoder = load_model(args.encoder)
latent_vectors = encoder.predict(X,verbose=0)
print(f"Shape: {latent_vectors.shape}")

del X
del encoder

ouput_path = os.path.join(args.output, f"{args.name}.npy")

np.save(ouput_path,latent_vectors)
print(f"Saved data to {ouput_path}")
