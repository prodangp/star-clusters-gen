import torch
import gpytorch
import numpy as np
from matplotlib import pyplot as plt
from utils import estimate_density, feature_scaling, normalize_density
from clusters import Clusters
from GP_model import ExactGPModel



# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()

# normalize data
#y = normalize_density(feature_scaling(np.log(estimate_density(X, k=50)), method='MinMax'))

data_path = 'C:/Users/georg/PycharmProjects/star-clusters-gen/data/chains'

pe = np.load(f'{data_path}/pe_m1.e4.dat.npy')
pe = np.log(pe[:-1])
ke = np.load(f'{data_path}/ke_m1.e4.dat.npy')
ke = np.log(ke[:-1])

pe, mu_pe, std_pe = feature_scaling(pe, return_mean_std=True)
ke, mu_ke, std_ke = feature_scaling(ke, return_mean_std=True)

X = np.zeros((pe.shape[0] - 1, 3))
X[:, 0] = pe[:-1].flatten()
X[:, 1] = ke[:-1].flatten()
X[:, 2] = ke[1:].flatten()

y = feature_scaling(estimate_density(X))
train_x = torch.from_numpy(X).float().cuda()
train_y = torch.from_numpy(y).float().cuda()
model = ExactGPModel(train_x, train_y, likelihood, linear=True).cuda()

# normalize data
#y = normalize_density(feature_scaling(np.log(estimate_density(X, k=50)), method='MinMax'))
#y = feature_scaling(np.log(estimate_density(X, k=50)), method='standardization')


# # Generate a permutation of indices
# indices = torch.randperm(len(train_x))
#
# # Shuffle both tensors
# train_x = train_x[indices]
# train_y = train_y[indices]
#
# print(min(train_y))
# Load saved model weights
checkpoint = torch.load('./results_chains/chains_GP_pe_ke_5000iter_jul05_t1403/model.pth')
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
