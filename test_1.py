import torch
import gpytorch
import numpy as np
from matplotlib import pyplot as plt
from utils import estimate_density, feature_scaling, normalize_density
from clusters import Clusters
from GP_model import ExactGPModel


# data
clusters = Clusters(features='all')
# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
X, cluster_name_tr = clusters.next_train(return_name=True)
# normalize data
#y = normalize_density(feature_scaling(np.log(estimate_density(X, k=50)), method='MinMax'))
y = normalize_density(estimate_density(X, k=50))



train_x = torch.from_numpy(X).float().cuda()
train_y = torch.from_numpy(y).float().cuda()
model = ExactGPModel(train_x, train_y, likelihood).cuda()

# normalize data
#y = normalize_density(feature_scaling(np.log(estimate_density(X, k=50)), method='MinMax'))
#y = feature_scaling(np.log(estimate_density(X, k=50)), method='standardization')


# noise
noise_amount = 5000
noise_amplitude = 1e-3
extra = torch.from_numpy(np.random.normal(0, 1, (noise_amount - train_x.shape[0], 7))).float().cuda()
train_x = torch.concatenate((train_x, extra))
extra = torch.from_numpy(np.abs(np.random.normal(0, noise_amplitude, noise_amount - train_y.shape[0]))).float().cuda()
train_y = torch.concatenate((train_y, extra))

# # Generate a permutation of indices
# indices = torch.randperm(len(train_x))
#
# # Shuffle both tensors
# train_x = train_x[indices]
# train_y = train_y[indices]
#
# print(min(train_y))
# Load saved model weights
checkpoint = torch.load('./results/7D_10c_aug01_t1649/model.pth')
#checkpoint = torch.load('./results/sim_1c_april06_t1115/model.pth')
# Load weights into model
model.load_state_dict(checkpoint)
model.eval()
likelihood.eval()
#X = clusters.next_train()
#val_x = torch.from_numpy(X).float().cuda()
#val_y = torch.from_numpy(estimate_density(X)).float().cuda()

# train_x =  torch.from_numpy(np.random.normal(0, 1, (8000, 7))).float().cuda()
f_preds = model(train_x)
y_preds = likelihood(f_preds)
y_samples = y_preds.sample(sample_shape=torch.Size([1,]))


with torch.no_grad():
    plt.plot(y_preds.stddev.cpu().numpy(), label='stddev')
    plt.plot(y_samples[0].cpu().numpy(),'go', markersize=2, label='predicted')
    plt.plot(y_preds.mean.cpu().numpy(), 'o', markersize=4, alpha=0.8, label='pred mean')
    plt.plot(train_y.cpu(), 'ro', markersize=2, label='simulation', alpha=0.5)
    # plt.yscale('log')
    plt.legend()
    plt.show()

#print(min(y_preds2.mean.detach().cpu().numpy()))

# with torch.no_grad():
#     mean = y_preds.mean.cpu().numpy()[2]
#     std = y_preds.stddev.cpu().numpy()[2]
#     n_bins = 50
#     y_sample = y_samples[:, 2].cpu().numpy()
#     x = np.linspace(mean - 3 * std, mean + 3 * std, 100)
#     y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
#     print(y_sample.shape)
#     plt.hist(y_sample, bins=n_bins)
#     plt.axvline(mean, color='red')
#     plt.plot(x, y * 80, color='orange')
#     plt.show()
