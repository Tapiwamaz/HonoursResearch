import joblib
import numpy as np
import argparse
import os




parser = argparse.ArgumentParser(description="Train classifier on top of encoder")
parser.add_argument("--x", required=True, help="Path to the data.")
parser.add_argument("--output", required=True, help="Directory to save the classiier.")
parser.add_argument("--name", required=True, help="Name to save output")
parser.add_argument("--encoder", required=True, help="Path to the existing encoder .keras file")
parser.add_argument("--scaler", required=True, help="Path to the existing scaler")


args = parser.parse_args()


pca = joblib.load(args.encoder)
pca_scaler = joblib.load(args.scaler)
data = np.load(args.x)



# data = data.astype(np.float32)
print(f"Data before transform: {data.shape}")

X = pca_scaler.fit_transform(data)
X = pca.fit_transform(X)
print(X.shape)
print(f"Transformed  data: {X.shape}")

ouput_path = os.path.join(args.output, f"{args.name}.npy")

np.save(ouput_path,X)
print(f"Saved data to {ouput_path}")
