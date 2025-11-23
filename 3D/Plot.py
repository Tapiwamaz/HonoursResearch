from pyimzml.ImzMLParser import ImzMLParser
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Generate ion image plot.")
parser.add_argument("--x", required=True, help="Path to the imzml file")
parser.add_argument("--name", required=True, help="Path to the imzml file")



args = parser.parse_args()

p = ImzMLParser(args.x)
X,Y,Z,C = [],[],[],[]
# Define key m/z channels and range
key_mz_channels = [100, 150, 200, 250, 350, 400, 800, 1500]
mz_range = 10

for idx, (x,y,z) in enumerate(p.coordinates):
    mzs, intensities = p.getspectrum(idx)
    # Normalize intensities for this spectrum
    if len(intensities) > 0 and np.max(intensities) > 0:
        normalized_intensities = intensities / np.max(intensities)
    else:
        normalized_intensities = intensities
    
    for id in range(len(mzs)):
        # Check if mz is within range of any key channel
        if any(abs(mzs[id] - key_mz) <= mz_range for key_mz in key_mz_channels):
            X.append(x)
            Y.append(y)
            Z.append(mzs[id])
            C.append(normalized_intensities[id])
print(len(X))
print(f"Min: {np.min(C)}")
print(f"Max: {np.max(C)}")
print(f"Mean: {np.mean(C)}")


# Create 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(X, Z, Y, c=C, cmap='viridis',alpha=0.6)

# Set labels and title
ax.set_xlabel('X')
ax.set_zlabel('Y')
ax.set_ylabel('mzs')
ax.set_title(f'{args.name}')

# Add a color bar
fig.colorbar(scatter, ax=ax, label='Intensities')

# Save the image from different view angles
view_angles = [
    (30, 45),   # default-ish view
    (0, 0),     # front view
    (0, 90),    # side view
    (0, 180),   # back view
    (0, 270),   # opposite side view
    (90, 0),    # top view
    (-90, 0),   # bottom view
    (45, 135),  # diagonal view
    (45, 225),  # diagonal view (opposite)
    (30, 0),    # slight angle front
    (30, 90),   # slight angle side
    (30, 180),  # slight angle back
    (30, 270),  # slight angle opposite side
    (60, 45),   # steep angle
    (15, 45),   # shallow angle
]

for elev, azim in view_angles:
    ax.view_init(elev=elev, azim=azim)
    plt.savefig(f'{args.name}_view_{elev}_{azim}.png', dpi=350, bbox_inches='tight')
    print(f"Saved view: elevation={elev}, azimuth={azim}")

# plt.show()
