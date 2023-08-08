import time
import os
import numpy as np
import torch
import gpytorch
from matplotlib import pyplot as plt
from matplotlib import cycler
import matplotlib.ticker as ticker
from GP_model import ExactGPModel
from utils import estimate_density, feature_scaling, normalize_density

plt.rcParams["font.size"] = 16

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

data_path = './data/chains'
train = ["m1.e4", "m1.e5", "m2.e4", "m3.e4", "m4.e4", "m5.e4", "m6.e4"]
val = ["m7.e4", "m8.e4", "m9.e4"]

train_x = []
train_y = []
val_x = []
val_y = []


for file in train:
    pe = np.load(f'{data_path}/pe_{file}.dat.npy')
    pe = np.log(pe[:-1])
    ke = np.load(f'{data_path}/ke_{file}.dat.npy')
    ke = np.log(ke[:-1])

    pe = feature_scaling(pe)
    ke = feature_scaling(ke)

    X = np.zeros((pe.shape[0] - 1, 3))
    X[:, 0] = pe[:-1].flatten()
    X[:, 1] = ke[:-1].flatten()
    X[:, 2] = ke[1:].flatten()

    # compute and scale density
    y = feature_scaling(estimate_density(X))

    train_x.append(X)
    train_y.append(y)


for file in val:
    pe = np.load(f'{data_path}/pe_{file}.dat.npy')
    pe = np.log(pe[:-1])
    ke = np.load(f'{data_path}/ke_{file}.dat.npy')
    ke = np.log(ke[:-1])

    pe = feature_scaling(pe)
    ke = feature_scaling(ke)

    X = np.zeros((pe.shape[0] - 1, 3))
    X[:, 0] = pe[:-1].flatten()
    X[:, 1] = ke[:-1].flatten()
    X[:, 2] = ke[1:].flatten()

    # compute and scale density
    y = feature_scaling(estimate_density(X))

    val_x.append(X)
    val_y.append(y)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()


# # Create a DataLoader to create batches
# batch_size = 1000
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# train_x, train_y = next(iter(dataloader))

x = torch.from_numpy(train_x[0]).float().cuda()
y = torch.from_numpy(train_y[0]).float().cuda()
print(np.mean(train_y[0]), np.std(train_y[0]), np.min(train_y[0]), np.max(train_y[0]))

model = ExactGPModel(x, y, likelihood).cuda()

training_iter = 5000
timestamp = time.strftime(f"chains_linear_10c_GP_pe_ke_{training_iter}iter_aug%d_t%H%M", time.gmtime())
model = model.cuda()
likelihood = likelihood.cuda()

# # Load saved model weights
# checkpoint = torch.load('./results_chains/chains_GP_pos_50iter_may31_t1447/model.pth')
#
# # Load weights into model
# model.load_state_dict(checkpoint)

# Find optimal model hyperparameters
model.train()
likelihood.train()
#
val_loss = []
val_loss_std = []
train_loss = []
train_loss_std = []
train_loss_ = []
#
#
# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
training_iter = 5000
val_on = True
training_start = time.time()
patience = 105
best_loss = float('inf')
best_epoch = 0
n_train= len(train)
n_val = len(val)
data_x = train_x + val_x
data_y = train_y + val_y
perm = np.random.permutation(10)
perm_train = perm[:n_train]
perm_val = perm[n_train:]
train_x = [data_x[i] for i in perm_train]
train_y = [data_y[i] for i in perm_train]
val_x = [data_x[i] for i in perm_val]
val_y = [data_y[i] for i in perm_val]
for i in range(1, training_iter + 1):
    x = torch.from_numpy(train_x[(i - 1) % n_train]).float().cuda()
    y = torch.from_numpy(train_y[(i - 1) % n_train]).float().cuda()
    model.set_train_data(inputs=x, targets=y, strict=False)
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(x)
    # Calc loss and backprop gradients
    loss = -mll(output, y)
    loss.backward()

    train_loss_.append(loss.item())
    print('Epoch %d/%d - - Loss: %.3f  -  lengthscale: %.3f - noise: %.3f' % (
        i, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()
    if i % n_train == 0 and val_on:
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
            print(f"Early stopping on iter. {i}. Best epoch: {i - patience}.")
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
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title(r"$\mathcal{G}\mathcal{P}(U_{ij},K_i,K_j)$ training")
os.mkdir(f'./results_chains/{timestamp}')
plt.tight_layout()
plt.savefig(f'./results_chains/{timestamp}/loss.png')
plt.show()
torch.save(model.state_dict(), f'./results_chains/{timestamp}/model.pth')
