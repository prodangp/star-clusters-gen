import torch
import gpytorch
import numpy as np
from matplotlib import pyplot as plt
from utils import estimate_density, feature_scaling, normalize_density
from clusters import Clusters
from GP_model import ExactGPModel

# data
clusters = Clusters(features='phase space')
# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
X, cluster_name_tr = clusters.next_train(return_name=True)
# normalize data
y, d_min, d_max = normalize_density(estimate_density(X))
X = feature_scaling(X)
zero_points = 5000
train_x = torch.from_numpy(X).float().cuda()
train_y = torch.from_numpy(y).float().cuda()

extra = torch.from_numpy(np.random.normal(0, 1, (zero_points, 6))).float().cuda()

train_x = torch.concatenate((train_x, extra))
extra = torch.from_numpy(np.abs(np.random.normal(0, 0.001, zero_points))).float().cuda()

train_y = torch.concatenate((train_y, extra))

model = ExactGPModel(train_x, train_y, likelihood).cuda()

# Load saved model weights
checkpoint = torch.load('./results/6D_1c_zp8000_jun13_t1056/model.pth')
# checkpoint = torch.load('./results/sim_1c_april06_t1115/model.pth')
# Load weights into model
model.load_state_dict(checkpoint)
model.eval()
likelihood.eval()
# X = clusters.next_train()
# val_x = torch.from_numpy(X).float().cuda()
# val_y = torch.from_numpy(estimate_density(X)).float().cuda()

# train_x =  torch.from_numpy(np.random.normal(0, 1, (8000, 7))).float().cuda()
f_preds = model(train_x[4000:5000])
y_preds = likelihood(f_preds)
y_samples = y_preds.sample(sample_shape=torch.Size([1, ]))

with torch.no_grad():
    plt.plot(y_preds.stddev.cpu().numpy(), label='stddev')
    plt.plot(y_preds.sample().cpu().numpy(), 'go', markersize=1, label='predicted')
    plt.plot(train_y.cpu(), 'ro', markersize=1, label='simulation', alpha=0.5)
    # plt.yscale('log')
    plt.legend()
    plt.show()

# print(min(y_preds2.mean.detach().cpu().numpy()))

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
