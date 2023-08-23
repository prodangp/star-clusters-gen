from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as st
import torch
import gpytorch
from utils import Clusters, estimate_density, normalize_density, feature_scaling
from GP_model import ExactGPModel

clusters = Clusters(features='pos')
X, cluster_name_tr = clusters.next_train(return_name=True)

# normalize data
y, d_min, d_max = normalize_density(estimate_density(X))
X = feature_scaling(X)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
train_x = torch.from_numpy(X).float().cpu()
print(train_x.shape)
train_y = torch.from_numpy(y).float().cpu()
model = ExactGPModel(train_x, train_y, likelihood).cpu()

# Load saved model weights
model_path = '../results/3D_baseline_april18_t0716'
checkpoint = torch.load(f'{model_path}/model.pth')
model.load_state_dict(checkpoint)
n_grid = 25
dims = 3
reshape_arr = tuple([n_grid ] * dims)
grid_elements = [np.linspace(X[:, j].min(), X[:, j].max(), num=n_grid) for j in range(len(X[0]))]
grid = torch.from_numpy(np.array(list(product(*grid_elements)))).float().cpu()
print(grid.shape)
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    model.eval()
    likelihood.eval()
    f_preds = model(grid)
    y_preds = likelihood(f_preds)

f_mean = f_preds.mean
f_var = f_preds.variance
cov_matrix = f_preds.covariance_matrix
density_pred = f_preds.sample(sample_shape=torch.Size(3000,))
std = np.diag(cov_matrix)

posterior_num = 3
posteriors = st.multivariate_normal.rvs(mean=density_pred,
                                        cov=cov_matrix,
                                        size=posterior_num)
#posteriors[posteriors < 0] = 0

if posterior_num == 1:
    posteriors = [posteriors]

Xp = []
for i in range(grid.shape[1]):
    Xp.append(grid[:, i].reshape(reshape_arr))
fig = plt.figure(figsize=(posterior_num * 6, 4))
ax = [None] * posterior_num
im = [None] * posterior_num
for i, posterior in enumerate(posteriors):
    Z = np.reshape(posterior, reshape_arr)
    if dims == 2:
        ax[i] = fig.add_subplot(100 + posterior_num * 10 + i + 1)
        im[i] = ax[i].pcolormesh(Xp[0], Xp[1], Z, cmap="plasma")
    elif dims == 3:
        ax[i] = fig.add_subplot(100 + posterior_num * 10 + i + 1, projection='3d')
        im[i] = ax[i].scatter(Xp[0], Xp[1], Xp[2], marker='s', s=200, c=Z, alpha=0.15)
    fig.colorbar(im[i], ax=ax[i])
    ax[i].set_title(f"Generated density map #{i + 1}")
plt.show()
