import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import gpytorch
from matplotlib import pyplot as plt
from utils import Data, feature_scaling
from GP_model import ExactGPModel
from samplers import MCMCSampler, RejectionSampler


def gp_pdf(proposed, model_, likelihood_):
    with torch.no_grad():
        proposed = torch.from_numpy(proposed).float().unsqueeze(0).cuda()
        y_preds = likelihood_(model_(proposed))
        return y_preds.mean[0].cpu().numpy()


# LOADING MODEL
# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
batch_size = 1000
model_path = './results_rosenbrock/MCMC_10k_B1k_may12_t1314'
x = np.load(f'{model_path}/x.npy')
dataset = Data(x)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
train_x, train_y = next(iter(dataloader))
train_x = train_x.float().cuda()
train_y = train_y.float().cuda()
model = ExactGPModel(train_x, train_y, likelihood).cuda()

# Load saved model weights
checkpoint = torch.load(f'{model_path}/model.pth')

# Load weights into model
model.load_state_dict(checkpoint)
model.eval()
likelihood.eval()

# SAMPLING
# sampler = MCMCSampler(num_samples=10000, num_dimensions=2, random_seed=1565)
# step_size = 0.4
# samples = sampler.sample(rosenbrock, step_size=step_size, random_walk=True)
# # sampler = RejectionSampler(num_samples=1500, num_dimensions=2)
# # samples = sampler.sample(rosenbrock)
# x = samples
# np.save('rosenbrock_sampled', np.array(x))
#x = np.load('rosenbrock_sampled.npy')

epdf = lambda z: gp_pdf(z, model, likelihood)

# Create a grid of points in the x1-x2 plane
x1_range = np.linspace(-4, 4, 200)
x2_range = np.linspace(-1, 10, 200)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

x_grid = np.stack([x1_grid, x2_grid], axis=-1)

# Evaluate the Rosenbrock function at each point in the grid
rosenbrock_grid = np.zeros_like(x1_grid)
for i in tqdm(range(x1_grid.shape[0])):
    for j in range(x1_grid.shape[1]):
        rosenbrock_grid[i, j] = epdf((x_grid[i, j] - dataset.means) / dataset.stds)

np.save(f'{model_path}/rosenbrock_grid.npy', rosenbrock_grid)
# rosenbrock_grid = np.load('results_rosenbrock/MCMC_10.5k_B1k_may12_t1225/model.pth')

# Plot the Rosenbrock function using a 2D colormap
plt.contourf(x1_grid, x2_grid, rosenbrock_grid, cmap='viridis')
plt.scatter(x[:, 0], x[:, 1], color='red', alpha=0.3)
plt.xlim((-4, 4))
plt.ylim((-1, 10))
plt.colorbar(label='Probability Density')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Rosenbrock Distribution')
plt.savefig(f'{model_path}/gp_dist.png')
plt.show()
