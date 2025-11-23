from pyimzml.ImzMLParser import ImzMLParser
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Generate ion image plot.")
parser.add_argument("--x", required=True, help="Path to the imzml file")


args = parser.parse_args()

p = ImzMLParser(args.x)
X,Y,Z,C = [],[],[],[]
for idx, (x,y,z) in enumerate(p.coordinates):
    mzs, intensities = p.getspectrum(idx)
    # Normalize intensities for this spectrum
    if len(intensities) > 0 and np.max(intensities) > 0:
        normalized_intensities = intensities / np.max(intensities)
    else:
        normalized_intensities = intensities
    
    for id in range(len(mzs)):
        # if normalized_intensities[id] < 20000.1 and idx >= 2: continue
        X.append(x)
        Y.append(y)
        Z.append(mzs[id])
        C.append(normalized_intensities[id])
    # if idx >= 500:
        # break    
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
ax.set_title('Cancer')

# Add a color bar
fig.colorbar(scatter, ax=ax, label='Intensities')

# Save the image from different view angles
view_angles = [
    (30, 45),   # default-ish view
    (0, 0),     # front view
    (0, 90),    # side view
    (90, 0),    # top view
    (45, 135),  # diagonal view
]

for elev, azim in view_angles:
    ax.view_init(elev=elev, azim=azim)
    plt.savefig(f'HIV_view_{elev}_{azim}.png', dpi=350, bbox_inches='tight')
    print(f"Saved view: elevation={elev}, azimuth={azim}")

# plt.show()
