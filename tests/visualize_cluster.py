import numpy as np
from matplotlib import pyplot as plt
import corner
from utils import Clusters, estimate_density, feature_scaling, reverse_scaling

clusters = Clusters(features='pos')
X, cluster_name_tr = clusters.next_train(return_name=True)

sampling_method = 'RWMC'

# normalize data
y, means, stds = feature_scaling(estimate_density(X), return_mean_std=True)
model_path = './results/3D_baseline_march29_t1809'
x = np.load(f'{model_path}/RWMC_cluster.npy')
y_x = estimate_density(x)

# Define labels for each variable
labels = ['x', 'y', 'z']


print(x.shape, X.shape)
# Create the corner plot
figure = corner.corner(x, color='red', labels=labels, show_titles=True, bins=20)
corner.corner(X, color='blue', fig=figure, bins=40)
plt.plot([], [], label='simulation', color='blue')
plt.plot([], [], label=sampling_method, color='red')
plt.legend(loc='upper right')
plt.suptitle(f'{sampling_method} Corner Plot')
plt.show()

# Create the figure and axes objects
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

# Plot the data and the estimate density on the first axis
ax1.scatter(x[:, 0], x[:, 1], c=y_x, edgecolor="k", s=60, cmap="plasma")
ax1.set_title(sampling_method, fontsize=16)
ax1.set_xlabel('x (pc)', fontsize=12)
ax1.set_ylabel('y (pc)', fontsize=12)
ax1.set_xlim([np.min([np.min(X[:, 0]), np.min(x[:, 0])]), np.max([np.max(X[:, 0]), np.max(x[:, 0])])])
ax1.set_ylim([np.min([np.min(X[:, 1]), np.min(x[:, 1])]), np.max([np.max(X[:, 1]), np.max(x[:, 1])])])
fig.colorbar(ax1.collections[0], ax=ax1)

# Plot the data and the estimate density on the second axis
ax2.scatter(X[:, 0], X[:, 1], c=estimate_density(X), edgecolor="k", s=60, cmap="plasma")
ax2.set_title('Simulation Data', fontsize=16)
ax2.set_xlabel('x (pc)', fontsize=12)
ax2.set_ylabel('y (pc)', fontsize=12)
fig.colorbar(ax2.collections[0], ax=ax2)
plt.savefig(f'{model_path}/{sampling_method}.png')
# Show the plot
plt.show()
