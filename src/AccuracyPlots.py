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
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')
#plt.figure(figsize=(12, 10))
for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

ax.set_ylabel('Measure')
ax.set_title('Classifier perfomance on  LPS data')
ax.set_xticks(x + width, species)
ax.legend()
#ax.set_ylim(0, 1)

plt.savefig("ClassifierPerformances.png",dpi=500)
