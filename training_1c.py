import time
import os
import numpy as np
import torch
import gpytorch
from matplotlib import pyplot as plt
from utils import estimate_density, normalize_density, feature_scaling
from GP_model import ExactGPModel
from clusters import Clusters

plt.rcParams["font.size"] = 16
from matplotlib import cycler
colors = cycler('color',
                ['#EE6666', '#3388BB', '#9988DD',
                 '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
       axisbelow=True, grid=True, prop_cycle=colors)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('xtick', direction='out', color='black', labelsize=16)
plt.rc('ytick', direction='out', color='black', labelsize=16)
font = {'size': 16}
plt.rc('font', **font)
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)


# data
clusters = Clusters(features='all')
X, cluster_name_tr = clusters.next_train(return_name=True)

# normalize data
y = normalize_density(estimate_density(X, k=50))
print(np.sum(y), np.mean(y), np.std(y), np.max(y), np.min(y))

# noise
noise_amount = 8000
noise_amplitude = 1e-3
# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()

train_x = torch.from_numpy(X).float().cuda()
train_y = torch.from_numpy(y).float().cuda()
extra = torch.from_numpy(np.random.normal(0, 1, (noise_amount, 7))).float().cuda()
train_x = torch.concatenate((train_x, extra))
extra = torch.from_numpy(np.abs(np.random.normal(0, noise_amplitude, noise_amount))).float().cuda()
train_y = torch.concatenate((train_y, extra))


model = ExactGPModel(train_x, train_y, likelihood, linear=True).cuda()


timestamp = time.strftime(f"7D_linear_1c_noise{noise_amount}_aug%d_t%H%M", time.gmtime())
model = model.cuda()
likelihood = likelihood.cuda()

# Find optimal model hyperparameters
model.train()
likelihood.train()

train_loss = []

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Includes GaussianLikelihood parameters


# Cross Validation
n_train = clusters.n_train
n_val = clusters.n_val

training_start = time.time()

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
training_iter = 5000

for i in range(training_iter):
    train_x = torch.from_numpy(X).float().cuda()
    train_y = torch.from_numpy(y).float().cuda()
    extra = torch.from_numpy(np.random.normal(0, 1, (noise_amount, 7))).float().cuda()
    train_x = torch.concatenate((train_x, extra))
    extra = torch.from_numpy(np.abs(np.random.normal(0, noise_amplitude, noise_amount))).float().cuda()
    train_y = torch.concatenate((train_y, extra))
    model.set_train_data(inputs=train_x, targets=train_y, strict=False)
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f  -  lengthscale: %.3f - noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    train_loss.append(loss.item())
    optimizer.step()


train_loss = np.array(train_loss)
print('TIME (s):', time.time() - training_start)
plt.plot(train_loss, label=cluster_name_tr)
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title(r"$\mathcal{G}\mathcal{P}(M, \vec{r}, \vec{v})$ " + cluster_name_tr)
os.mkdir(f'./results/{timestamp}')
plt.tight_layout()
plt.savefig(f'./results/{timestamp}/loss.png')
plt.show()
torch.save(model.state_dict(), f'./results/{timestamp}/model.pth')
