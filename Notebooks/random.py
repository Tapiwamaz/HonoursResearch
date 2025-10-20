import numpy as np
import tensorflow as tf

centroids = np.load("../kmeans/hiv-h5_centroids_k3.npy")
# centroids = np.array([np.round(mz, 4) for mz in centroids])
# mzs = np.load("../Data/HIV/mzs.npy")

decoder = tf.keras.models.load_model("../Models/Decoder/encoder_h5_wmse_200_decoder.keras")
out = decoder.predict(centroids)
del decoder
del centroids
print(f"Shape: {out.shape}")

np.savetxt("../c1y.txt", out[0], fmt="%.4f")
np.savetxt("../c2y.txt", out[1], fmt="%.4f")
np.savetxt("../c3y.txt", out[2], fmt="%.4f")
