import numpy as np
import tensorflow as tf

centroids = np.load("../Data/HIV/hiv-150-1500_x.npy")
# centroids = np.array([np.round(mz, 4) for mz in centroids])
# mzs = np.load("../Data/HIV/mzs.npy")

decoder = tf.keras.models.load_model("../Models/Decoder/decoder.keras")
out = decoder.predict(centroids)
del decoder
del centroids
print(f"0: {out[0]}")
print(f"1: {out[1]}")
print(f"2: {out[2]}")

np.savetxt("../c11y.txt", out[0], fmt="%.4f")
np.savetxt("../c22y.txt", out[1], fmt="%.4f")
np.savetxt("../c33y.txt", out[2], fmt="%.4f")
