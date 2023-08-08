import os
import time as time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import gpytorch
from clusters import Clusters
from SDE_networks import DNN3L
from GP_model import ExactGPModel
from utils import sigma, DEVICE, normalize_density, estimate_density, feature_scaling


clusters = Clusters(features='all', mass='log', rescale=False, path='./data/')
# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
X, cluster_name_tr = clusters.next_train(return_name=True)
# normalize data
y, d_min, d_max = normalize_density(estimate_density(X))
X, mean, std = feature_scaling(X, return_mean_std=True)
X = torch.from_numpy(X).float().cuda()
zero_points = 4000
extra = torch.from_numpy(np.random.normal(0, 1, (zero_points, 7))).float().cuda()
train_x = torch.concatenate((X, extra))
extra = torch.from_numpy(np.abs(np.random.normal(0, 0.001, zero_points))).float().cuda()
train_y = torch.from_numpy(y).float().cuda()
train_y = torch.concatenate((train_y, extra))
gp_model = ExactGPModel(train_x, train_y, likelihood).cuda()

# Load saved model weights
checkpoint = torch.load('./results/sim_1c_zp_jun17_t1843/model.pth')

# Load weights into model
gp_model.load_state_dict(checkpoint)
gp_model.eval()
likelihood.eval()

y_preds = likelihood(gp_model(train_x))


model = DNN3L(units=1000)
model = model.to(DEVICE)
iterations = 100
learning_rate = 5e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
timestamp = time.strftime(f"SDE15_GP_{model.name}_{iterations}it_{learning_rate}LR_jun%d_t%H%M", time.gmtime())
losses = []
val_losses = []
train_counter = 0
losses_iter = []
for i in tqdm(range(iterations)):
    train_counter += 1
    optimizer.zero_grad()
    extra = torch.from_numpy(np.random.normal(0, 1, (zero_points, 7))).float().cuda()
    x = torch.concatenate((X, extra))
    f_preds = gp_model(x)
    y_preds = likelihood(f_preds)
    y = y_preds.sample(sample_shape=torch.Size([1, ])).squeeze(0)
    mask = y > 0.3
    x = x[mask]
    batch_size = x.shape[0]
    t = torch.rand(batch_size).to(DEVICE)
    z = torch.randn_like(x).to(DEVICE)
    lambda_t = (sigma(t)**2).view(-1, 1).to(DEVICE) # we define the weights here
    loss = torch.sum(lambda_t * (model(x + sigma(t)[:, None] * z, t) + z / sigma(t)[:, None])**2) / batch_size
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    # if train_counter == clusters.n_train:
    #     train_counter = 0
    #     model.eval()
    #     val_losses_iter = []
    #     for j in range(clusters.n_val):
    #       x = torch.from_numpy(clusters.next_val(j)).float().cuda()
    #       batch_size = x.shape[0]
    #       t = torch.rand(batch_size).to(DEVICE)
    #       z = torch.randn_like(x).to(DEVICE)
    #       lambda_t = (sigma(t)**2).view(-1, 1).to(DEVICE) # we define the weights here
    #       loss = torch.sum(lambda_t * (model(x + sigma(t)[:, None] * z, t) + z / sigma(t)[:, None])**2) / batch_size
    #       val_losses_iter.append(loss.item())
    #     val_losses.append(val_losses_iter)
    #     model.train()

val_losses = np.array(val_losses)
idx_val = np.array(range(val_losses.shape[0])) * 7
plt.plot(losses[5:], label='training loss')
# plt.plot(idx_val, val_losses[:, 0], label='val loss 1')
# plt.plot(idx_val + 1, val_losses[:, 1], label='val loss 2')
# plt.plot(idx_val + 2, val_losses[:, 2], label='val loss 3')
plt.ylabel("Score matching loss")
plt.xlabel("Iteration")
plt.show()
os.mkdir(f'./results_SDE/{timestamp}')
plt.savefig(f'./results_SDE/{timestamp}/loss.png')
# np.save(f'./results_SDE/{timestamp}/x.npy', np.array(x))
torch.save(model, f'./results_SDE/{timestamp}/model.pth')