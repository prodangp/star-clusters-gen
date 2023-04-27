import time
import os
import numpy as np
import torch
import gpytorch
from matplotlib import pyplot as plt
from utils import Clusters, estimate_density, feature_scaling, normalize_density
from GP_model import ExactGPModel

# data
clusters = Clusters(features='pos')
X, cluster_name_tr = clusters.next_train(return_name=True)

# normalize data
y, d_min, d_max = normalize_density(estimate_density(X))
X = feature_scaling(X)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
train_x = torch.from_numpy(X).float().cuda()
train_y = torch.from_numpy(y).float().cuda()
model = ExactGPModel(train_x, train_y, likelihood).cuda()


timestamp = time.strftime("3D_baseline_april%d_t%H%M", time.gmtime())
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
train_loss = []
train_loss_ = []

n_val = 3
n_test = 7

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Includes GaussianLikelihood parameters


# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
training_iter = 500
val_on = False
# norm_train = 1000 / clusters.count_stars_train()
for i in range(training_iter):
    # X = clusters.next_train()
    # train_x = torch.from_numpy(X).float().cuda()
    # train_y = torch.from_numpy(estimate_density(X)).float().cuda()
    # model.set_train_data(inputs=train_x, targets=train_y, strict=False)
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    #print(output.sample())
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    #norm_loss = loss.item() * norm_train
    print('Iter %d/%d - Loss: %.3f  -  lengthscale: %.3f - noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    train_loss_.append(loss.item())
    if loss.item() < 0.01:
      break
    optimizer.step()
    if i % n_test == 0 and val_on:
      train_loss.append(np.sum(train_loss_))
      train_loss_ = []
      model.eval()
      likelihood.eval()
      val_loss_ = []
      for j in range(n_val):
        X = clusters.next_val(j)
        norm_val = 1000 / clusters.count_stars_val(j)
        val_x = torch.from_numpy(X).float().cuda()
        val_y = torch.from_numpy(estimate_density(X)).float().cuda()
        output = model(val_x)
        loss = -mll(output, val_y)
        val_loss_.append(loss.item() * norm_val)
        print('*** VALIDATION  %d/%d - Validation Loss %d: %.3f - Normalized: %.3f   ' % (
          i + 1, training_iter, j + 1, loss.item(), loss.item() * norm_val
        ))
      val_loss.append(val_loss_)
      model.train()
      likelihood.train()


plt.plot(train_loss_, label='train loss')
plt.legend()
os.mkdir(f'./results/{timestamp}')
plt.savefig(f'./results/{timestamp}/loss.png')
plt.show()
torch.save(model.state_dict(), f'./results/{timestamp}/model.pth')
