import joblib
import numpy as np
import argparse
import os


parser = argparse.ArgumentParser(description="Train classifier on top of encoder")
parser.add_argument("--x", required=True, help="Path to the data.")
parser.add_argument("--output", required=True, help="Directory to save the classiier.")
parser.add_argument("--name", required=True, help="Name to save output")
parser.add_argument("--encoder", required=True, help="Path to the existing encoder .keras file")
args = parser.parse_args()

nmf = joblib.load(args.encoder)

X = np.load(args.x)

print(f"Data shape: {X.shape}")

X = nmf.predict(X)

print(f"X shape: {X.shape}")

ouput_path = os.path.join(args.output, f"{args.name}.npy")

np.save(ouput_path,X)
print(f"Saved data to {ouput_path}")

