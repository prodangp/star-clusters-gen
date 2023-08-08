import time
import os
import numpy as np
import torch
import gpytorch
from matplotlib import pyplot as plt
from utils import estimate_density, normalize_density
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
clusters = Clusters(features='phase space')
X, cluster_name_tr = clusters.next_train(return_name=True)

# normalize pdf
y = normalize_density(estimate_density(X, k=50))
print(np.sum(y), np.mean(y), np.std(y), np.max(y), np.min(y))

# zero_points = 2000

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
train_x = torch.from_numpy(X).float().cuda()
train_y = torch.from_numpy(y).float().cuda()

# extra = torch.from_numpy(np.random.normal(0, 1, (zero_points, 6))).float().cuda()
# train_x = torch.concatenate((train_x, extra))
# extra = torch.from_numpy(np.abs(np.random.normal(0, 0.001, zero_points))).float().cuda()
# train_y = torch.concatenate((train_y, extra))

model = ExactGPModel(train_x, train_y, likelihood).cuda()

timestamp = time.strftime(f"6D_10c_aug%d_t%H%M", time.gmtime())
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
val_loss_std = []
train_loss = []
train_loss_std = []
train_loss_ = []

n_val = 3
n_test = 7

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
training_iter = 1000
val_on = True
training_start = time.time()
patience = 35
best_loss = float('inf')
best_epoch = 0
# norm_train = 1000 / clusters.count_stars_train()
for i in range(1, training_iter + 1):
    X = clusters.next_train()
    train_x = torch.from_numpy(X).float().cuda()
    train_y = torch.from_numpy(normalize_density(estimate_density(X, k=50))).float().cuda()
    model.set_train_data(inputs=train_x, targets=train_y, strict=False)
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # print(output.sample())
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    # norm_loss = loss.item() * norm_train
    print('Iter %d/%d - Loss: %.3f  -  lengthscale: %.3f - noise: %.3f' % (
        i, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    train_loss_.append(loss.item())
    # if loss.item() < 0.01:
    #   break
    optimizer.step()
    if i % n_test == 0 and val_on:
        train_loss.append(np.mean(train_loss_))
        train_loss_std.append(np.std(train_loss_))
        train_loss_ = []
        model.eval()
        likelihood.eval()
        val_loss_ = []
        for j in range(n_val):
            X = clusters.next_val(j)
            val_x = torch.from_numpy(X).float().cuda()
            val_y = torch.from_numpy(normalize_density(estimate_density(X, k=50))).float().cuda()
            output = model(val_x)
            loss = -mll(output, val_y)
            val_loss_.append(loss.item())
            print('*** VALIDATION  %d/%d - Validation Loss %d: %.3f - Normalized: %.3f   ' % (
                i, training_iter, j + 1, loss.item(), loss.item()
            ))
        validation_loss = np.mean(val_loss_)
        val_loss.append(validation_loss)
        val_loss_std.append(np.std(val_loss_))
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = i
        elif (i - best_epoch) >= patience:
            print(f"Early stopping on iter. {i}")
            break
        model.train()
        likelihood.train()

train_loss = np.array(train_loss)
train_loss_std = np.array(train_loss_std)
val_loss = np.array(val_loss)
val_loss_std = np.array(val_loss_std)
print('TIME (s):', time.time() - training_start)
plt.plot(train_loss, label='train loss')
plt.fill_between(range(len(train_loss)), train_loss - train_loss_std, train_loss + train_loss_std, alpha=0.2)
plt.plot(val_loss, label='validation loss')
plt.fill_between(range(len(val_loss)), val_loss - val_loss_std, val_loss + val_loss_std, alpha=0.2)
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title(r"$\mathcal{G}\mathcal{P}(\vec{r}, \vec{v})$ training")
os.mkdir(f'./results/{timestamp}')
plt.tight_layout()
plt.savefig(f'./results/{timestamp}/loss.png')
plt.show()
torch.save(model.state_dict(), f'./results/{timestamp}/model.pth')
