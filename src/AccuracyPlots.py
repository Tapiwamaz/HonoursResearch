import numpy as np
import matplotlib.pyplot as plt


species = ("AE", "CNN-AE", "PCA", "NMF")
penguin_means = {
    'Accuracy': (0.88, 0.81, 0.8,0.8),
    'AUC': (0.95, 0.88, 0.87,0.87),
    'F1-score': (0.88, 0.81, 0.8,0.79),
    'Precision': (0.89,0.80,0.81 ,0.83)
}

x = np.arange(len(species))  # the label locations
width = 0.2 
multiplier = 0

fig, ax = plt.subplots(layout='constrained',figsize=(8,6))
ax.grid(axis='both', alpha=0.5, linestyle='--', linewidth=0.7)
ax.set_ylim(0, 1.1)  # Extend y-axis to accommodate labels

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute,alpha=0.9)
    ax.bar_label(rects, padding=4)
    multiplier += 1

ax.set_ylabel('Measure')
ax.set_title('Classifier perfomance on  LPS data')
ax.set_xticks(x + width, species)
ax.legend(loc="lower right")

plt.show()
