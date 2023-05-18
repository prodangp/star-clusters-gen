import time
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import gpytorch
from matplotlib import pyplot as plt
from utils import Data
from GP_model import ExactGPModel
from samplers import MCMCSampler, RejectionSampler


def rosenbrock(x):
    x1 = x[0]
    x2 = x[1]
    return np.exp(-100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2)


# SAMPLING
# sampler = MCMCSampler(num_samples=10000, num_dimensions=2, random_seed=1565)
# step_size = 0.4
# samples = sampler.sample(rosenbrock, step_size=step_size, random_walk=True)
# # sampler = RejectionSampler(num_samples=1500, num_dimensions=2)
# # samples = sampler.sample(rosenbrock)
# x = samples
# np.save('rosenbrock', np.array(x))

x = np.load('results_rosenbrock/MCMC_50k_B1k_may10_t1056/rosenbrock.npy')

# Create a grid of points in the x1-x2 plane
x1_range = np.linspace(-5, 5, 1000)
x2_range = np.linspace(-1, 15, 1000)

# Sample 5000 points uniformly from the range
num_samples = 500
x1_samples = np.random.uniform(-5, 5, size=num_samples)
x2_samples = np.random.uniform(-1, 15, size=num_samples)

# Combine the samples into a 2D array
outliers = np.column_stack((x1_samples, x2_samples))

x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

x_grid = np.stack([x1_grid, x2_grid], axis=-1)

# Evaluate the Rosenbrock function at each point in the grid
rosenbrock_grid = np.zeros_like(x1_grid)
for i in range(x1_grid.shape[0]):
    for j in range(x1_grid.shape[1]):
        rosenbrock_grid[i, j] = rosenbrock(x_grid[i, j])


# Plot the Rosenbrock function using a 2D colormap
plt.contourf(x1_grid, x2_grid, np.log(rosenbrock_grid + 1e-16), cmap='viridis')
#x = np.concatenate((x, outliers))

plt.plot(x[:, 0], x[:, 1], linestyle='', marker='o', markersize=2, color='blue', alpha=0.3, label='sampled data')
#plt.plot(outliers[:, 0], outliers[:, 1], linestyle='', marker='o', markersize=2, color='red', alpha=0.3, label='outliers')
plt.xlim((-5, 5))
plt.ylim((-1, 15))
plt.colorbar(label='Probability Density')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Rosenbrock Distribution')
plt.show()


# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
dataset = Data(x)

# Create a DataLoader to create batches
batch_size = 1000
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
train_x, train_y = next(iter(dataloader))
model = ExactGPModel(train_x, train_y, likelihood).cuda()

timestamp = time.strftime("MCMC_10k_B1k_may%d_t%H%M", time.gmtime())
model = model.cuda()
likelihood = likelihood.cuda()

# # Load saved model weights
# checkpoint = torch.load('./results/3D_baseline_march29_t1757/model.pth')
#
# # Load weights into model
# model.load_state_dict(checkpoint)

# Find optimal model hyperparameters
model.train()
likelihood.train()

val_loss = []
train_loss = []
train_loss_ = []


# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
training_iter = 800
val_on = False
# norm_train = 1000 / clusters.count_stars_train()
for i in range(training_iter):
    epoch_loss = 0
    for train_x, train_y in dataloader:
        train_x = train_x.float().cuda()
        train_y = train_y.float().cuda()
        model.set_train_data(inputs=train_x, targets=train_y, strict=False)
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        #print(output.sample())
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        epoch_loss += loss.item() / len(dataloader)
        #norm_loss = loss.item() * norm_train
    print('Iter %d/%d - Loss: %.3f  -  lengthscale: %.3f - noise: %.3f' % (
        i + 1, training_iter, epoch_loss,
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    train_loss_.append(loss.item())
    # if loss.item() < 0.01:
    #   break
    optimizer.step()


plt.plot(train_loss_, label='train loss')
plt.legend()
os.mkdir(f'./results_rosenbrock/{timestamp}')
plt.savefig(f'./results_rosenbrock/{timestamp}/loss.png')
np.save(f'./results_rosenbrock/{timestamp}/x.npy', np.array(x))
plt.show()
torch.save(model.state_dict(), f'./results_rosenbrock/{timestamp}/model.pth')
