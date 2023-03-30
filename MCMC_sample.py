import time
import numpy as np
from matplotlib import pyplot as plt
import torch
import gpytorch
from GP_model import ExactGPModel
from utils import Clusters, estimate_density, feature_scaling, reverse_scaling
from samplers import MCMCSampler


# data
clusters = Clusters(features='pos')
X, cluster_name_tr = clusters.next_train(return_name=True)

# normalize data
y, means, stds = feature_scaling(estimate_density(X), return_mean_std=True)
X = feature_scaling(X)
print(X)
# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
train_x = torch.from_numpy(X).float().cuda()
train_y = torch.from_numpy(y).float().cuda()
model = ExactGPModel(train_x, train_y, likelihood).cuda()

# Load saved model weights
model_path = './results/3D_baseline_march29_t1809'
checkpoint = torch.load(f'{model_path}/model.pth')
model.load_state_dict(checkpoint)

timestamp = time.strftime("march%d_t%H%M", time.gmtime())
sampler = MCMCSampler(model, likelihood, clusters, 3000, random_seed=40)
step_size = 0.1
samples = sampler.sample(step_size=step_size)
np.save(f'{model_path}/MCMC_{step_size}_cluster.npy', samples)
x = samples
y_x = reverse_scaling(feature_scaling(estimate_density(x)), means=means, stds=stds)

# Create the figure and axes objects
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

# Plot the data and the estimate density on the first axis
ax1.scatter(x[:, 0], x[:, 1], c=y_x, edgecolor="k", s=60, cmap="plasma")
ax1.set_title('MCMC', fontsize=16)
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
plt.savefig(f'{model_path}/MCMC.png')
# Show the plot
plt.show()
