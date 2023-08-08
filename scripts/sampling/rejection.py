import time
import psutil
import os
import numpy as np
import torch
import gpytorch
from GP_model import ExactGPModel
from utils import estimate_density, normalize_density, gp_pdf
from samplers import RejectionSampler
from clusters import Clusters

# Get initial memory usage
process = psutil.Process(os.getpid())
mem_info = process.memory_info()
initial_memory = mem_info.rss

# data
clusters = Clusters(features='all')
X = clusters.next_train()
# normalize data
y = normalize_density(estimate_density(X, k=50))

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
train_x = torch.from_numpy(X).float().cuda()
train_y = torch.from_numpy(y).float().cuda()
model = ExactGPModel(train_x, train_y, likelihood).cuda()

# Load saved model weights
model_path = '../../results/7D_10c_aug01_t1649'
checkpoint = torch.load(f'{model_path}/model.pth')
model.load_state_dict(checkpoint)

timestamp = time.strftime("aug%d_t%H%M", time.gmtime())
model.eval()
likelihood.eval()
pdf = lambda z: gp_pdf(z, model, likelihood)


# Get memory usage before starting sampling
mem_info = process.memory_info()
memory_before_sampling = mem_info.rss
print('Memory used by GP model: ', (memory_before_sampling - initial_memory) / (1024 ** 3), 'GB')

sampler = RejectionSampler(num_samples=3000, num_dimensions=7, random_seed=42, verbose=True)
sampler.set_bounds(clusters.get_ave_bounds())
samples = sampler.sample(pdf)
np.save(f'{model_path}/REJECTION_{timestamp}_cluster.npy', samples)

# Get memory after sampling
mem_info = process.memory_info()
memory_final = mem_info.rss #
print('Memory used by sampling algorithm: ', (memory_final - memory_before_sampling) / (1024 ** 3), 'GB')
