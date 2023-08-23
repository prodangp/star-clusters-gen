import time
import os
import numpy as np
import torch
import gpytorch
from matplotlib import pyplot as plt
from matplotlib import cycler
from sklearn.neighbors import KernelDensity
from GP_model import ExactGPModel
from clusters import Clusters

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

train_x = []
train_y = []
for _ in range(clusters.n_train):
    x = clusters.next_train()[:, 0]
    train_x.append(x)
    # Use KernelDensity to estimate the PDF
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(x.reshape(-1, 1))

    train_y.append(np.exp(kde.score_samples(x.reshape(-1, 1))))

val_x = []
val_y = []
for j in range(clusters.n_val):
    x = clusters.next_val(j)[:, 0]
    val_x.append(x)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(x.reshape(-1, 1))
    val_y.append(np.exp(kde.score_samples(x.reshape(-1, 1))))

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
x = torch.from_numpy(train_x[0]).float().cuda()
y = torch.from_numpy(train_y[0]).float().cuda()
timestamp = time.strftime(f"GP_mass_all_clusters_aug%d_t%H%M", time.gmtime())
model = ExactGPModel(x, y, likelihood).cuda()
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

n_val = clusters.n_val
n_train = clusters.n_train

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

for i in range(1, training_iter + 1):
    x = torch.from_numpy(train_x[(i - 1) % n_train]).float().cuda()
    y = torch.from_numpy(train_y[(i - 1) % n_train]).float().cuda()
    model.set_train_data(inputs=x, targets=y, strict=False)
    optimizer.zero_grad()
    # Output from model
    output = model(x)
    # Calc loss and backprop gradients
    loss = -mll(output, y)
    loss.backward()

    print('Iter %d/%d - Loss: %.3f  -  lengthscale: %.3f - noise: %.3f' % (
        i, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    train_loss_.append(loss.item())
    optimizer.step()
    if (i - 1) % n_train == 0 and val_on:
        train_loss.append(np.mean(train_loss_))
        train_loss_std.append(np.std(train_loss_))
        train_loss_ = []
        model.eval()
        likelihood.eval()
        val_loss_ = []
        for j in range(n_val):
            x = torch.from_numpy(val_x[j]).float().cuda()
            y = torch.from_numpy(val_y[j]).float().cuda()
            output = model(x)
            loss = -mll(output, y)
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
            print(f"Early stopping on iter: {i - 1}. Best iter: {i - 1 - patience}.")
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
plt.title(r"$\mathcal{G}\mathcal{P}(M)$ training")
os.mkdir(f'./results/{timestamp}')
plt.tight_layout()
plt.savefig(f'./results/{timestamp}/loss.png')
plt.show()
torch.save(model.state_dict(), f'./results/{timestamp}/model.pth')
