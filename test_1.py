import torch
import gpytorch
from utils import Clusters, estimate_density
from GP_model import ExactGPModel
from matplotlib import pyplot as plt

# data
clusters = Clusters(features='pos')
# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
X, cluster_name_tr = clusters.next_train(return_name=True)
train_x = torch.from_numpy(X).float().cuda()
train_y = torch.from_numpy(estimate_density(X)).float().cuda()
model = ExactGPModel(train_x, train_y, likelihood).cuda()

# Load saved model weights
checkpoint = torch.load('./results/3D_baseline_march29_t1809/model.pth')

# Load weights into model
model.load_state_dict(checkpoint)
model.eval()
likelihood.eval()
#X = clusters.next_train()
#val_x = torch.from_numpy(X).float().cuda()
#val_y = torch.from_numpy(estimate_density(X)).float().cuda()
f_preds = model(train_x)
y_preds = likelihood(f_preds)

f_mean = f_preds.mean
f_var = f_preds.variance
f_covar = f_preds.covariance_matrix
f_samples = f_preds.sample(sample_shape=torch.Size([1,]))
with torch.no_grad():
    # plt.plot(y_preds.variance.cpu().numpy(), label='predicted')

    plt.plot(y_preds.mean.cpu().numpy()[:200], label='predicted')
    plt.plot(train_y.cpu()[:200], label='simulation', alpha=0.5)

    plt.yscale('log')

    plt.legend()
    plt.show()