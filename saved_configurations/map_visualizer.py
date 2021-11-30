import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting


def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)


configuration_name = "piwnica"

with open(os.path.join(configuration_name, 'map.pickle'), 'rb') as f:
    real_values = pickle.load(f)

cmap = get_cmap(len(real_values))

max_val = -float('inf')
min_val = float('inf')
for key in real_values:
    vals = real_values[key]
    max = np.max(vals)
    min = np.min(vals)
    if max > max_val:
        max_val = max
    if min < min_val:
        min_val = min

scale = (max_val - min_val) / 10

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for idx, key in enumerate(real_values):
    cords = real_values[key]
    cords = np.vstack((cords, cords[0]))
    ax.plot(cords[:, 0], cords[:, 1], cords[:, 2], color=cmap(idx))
    ax.scatter(cords[0][0], cords[0][1], cords[0][2], color=cmap(idx))
    ax.scatter(cords[1][0], cords[1][1], cords[1][2], color=cmap(idx))
    ax.scatter(cords[2][0], cords[2][1], cords[2][2], color=cmap(idx))
    ax.scatter(cords[3][0], cords[3][1], cords[3][2], color=cmap(idx))
    ax.text(cords[0][0] + scale, cords[0][1] - scale, cords[0][2], f'{key}', color='black')
    ax.text(cords[0][0], cords[0][1], cords[0][2], 'a', color='black')
    ax.text(cords[1][0], cords[1][1], cords[1][2], 'b', color='black')
    ax.text(cords[2][0], cords[2][1], cords[2][2], 'c', color='black')
    ax.text(cords[3][0], cords[3][1], cords[3][2], 'd', color='black')

plt.show()
