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

# normalize pdf
#
#y = normalize_density(feature_scaling())
y = normalize_density(estimate_density(X, k=50))
print(np.sum(y), np.mean(y), np.std(y), np.max(y), np.min(y))


# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
train_x = torch.from_numpy(X).float().cuda()
train_y = torch.from_numpy(y).float().cuda()
# noise
# noise_amount = 5000
# noise_amplitude = 1e-3
# extra = torch.from_numpy(np.random.normal(0, 1, (noise_amount - train_x.shape[0], 7))).float().cuda()
# train_x = torch.concatenate((train_x, extra))
# extra = torch.from_numpy(np.abs(np.random.normal(0, noise_amplitude, noise_amount - train_y.shape[0]))).float().cuda()
# train_y = torch.concatenate((train_y, extra))

model = ExactGPModel(train_x, train_y, likelihood).cuda()

timestamp = time.strftime(f"7D_10c_aug%d_t%H%M", time.gmtime())
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

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Includes GaussianLikelihood parameters


# Cross Validation
n_train = clusters.n_train
n_val = clusters.n_val
data_x = []
data_y = []

for _ in range(n_train):
    x = clusters.next_train()
    y = normalize_density(estimate_density(x, k=50))
    # noise_x = np.random.normal(0, 1, (noise_amount - x.shape[0], 7))
    # x = np.concatenate((x, noise_x))
    # noise_y = np.abs(np.random.normal(0, noise_amplitude, noise_amount - y.shape[0]))
    # y = np.concatenate((y, noise_y))
    data_x.append(x)
    data_y.append(y)

for i in range(n_val):
    x = clusters.next_val(i)
    y = normalize_density(estimate_density(x, k=50))
    # noise_x = np.random.normal(0, 1, (noise_amount - x.shape[0], 7))
    # x = np.concatenate((x, noise_x))
    # noise_y = np.abs(np.random.normal(0, noise_amplitude, noise_amount - y.shape[0]))
    # y = np.concatenate((y, noise_y))
    data_x.append(x)
    data_y.append(y)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
training_iter = 1000
val_on = True
training_start = time.time()
patience = 70
best_loss = float('inf')
best_epoch = 0

# data_x = np.array(data_x)
# data_y = np.array(data_y)

perm = np.random.permutation(10)
perm_train = perm[:n_train]
perm_val = perm[n_train:]
train_x = [data_x[i] for i in perm_train]
train_y = [data_y[i] for i in perm_train]
val_x = [data_x[i] for i in perm_val]
val_y = [data_y[i] for i in perm_val]

# train_x = data_x[:n_train]
# train_y = data_y[:n_train]
# val_x = data_x[n_train:]
# val_y = data_y[n_train:]

# norm_train = 1000 / clusters.count_stars_train()
for i in range(1, training_iter + 1):
    x = torch.from_numpy(train_x[(i - 1) % n_train]).float().cuda()
    y = torch.from_numpy(train_y[(i - 1) % n_train]).float().cuda()

    model.set_train_data(inputs=x, targets=y, strict=False)
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(x)
    # print(output.sample())
    # Calc loss and backprop gradients
    loss = -mll(output, y)
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
    if i % n_train == 0 and val_on:
        train_loss.append(np.mean(train_loss_))
        train_loss_std.append(np.std(train_loss_))
        train_loss_ = []
        model.eval()
        likelihood.eval()
        val_loss_ = []
        for j in range(n_val):
            with torch.no_grad():
                X = clusters.next_val(j)
                x = torch.from_numpy(val_x[j]).float().cuda()
                y = torch.from_numpy(val_y[j]).float().cuda()
                output = model(x)
                loss = -mll(output, y)
                val_loss_.append(loss.item())
                print('*** VALIDATION  %d/%d - Validation Loss %d: %.3f - Normalized: %.3f   ' % (
                    i, training_iter, j + 1, loss.item(), loss.item()
                ))
                # with torch.no_grad():
                #     y_preds = likelihood(output)
                #     plt.plot(y_preds.stddev.cpu().numpy(), label='stddev')
                #     plt.plot(y_preds.mean.cpu().numpy(), 'o', markersize=4, alpha=0.8, label='pred mean')
                #     plt.plot(y.cpu(), 'ro', markersize=2, label='simulation', alpha=0.5)
                #     # plt.yscale('log')
                #     plt.legend()
                #     plt.show()
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
        perm = np.random.permutation(10)
        perm_train = perm[:n_train]
        perm_val = perm[n_train:]
        train_x = [data_x[i] for i in perm_train]
        train_y = [data_y[i] for i in perm_train]
        val_x = [data_x[i] for i in perm_val]
        val_y = [data_y[i] for i in perm_val]

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
plt.title(r"$\mathcal{G}\mathcal{P}(M, \vec{r}, \vec{v})$ training")
os.mkdir(f'./results/{timestamp}')
plt.savefig(f'./results/{timestamp}/loss.png')
plt.show()
torch.save(model.state_dict(), f'./results/{timestamp}/model.pth')
