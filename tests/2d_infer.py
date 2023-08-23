import torch
from torch.utils.data import Dataset, DataLoader
import gpytorch
import numpy as np
from GP_model import ExactGPModel
from matplotlib import pyplot as plt
from utils import Data, feature_scaling, normalize_density, inverse_rescaled_density


def rosenbrock(x):
    x1 = x[0]
    x2 = x[1]
    return np.exp(-100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2)



model_path = '../results_rosenbrock/MCMC_10.5k_B1k_may12_t1255'


# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
batch_size = 1000
x = np.load(f'{model_path}/x.npy')
dataset = Data(x)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
train_x, train_y = next(iter(dataloader))
train_x = train_x.float().cuda()
train_y = train_y.float().cuda()
model = ExactGPModel(train_x, train_y, likelihood).cuda()

# Load saved model weights
checkpoint = torch.load(f'{model_path}/model.pth')

# Load weights into model
model.load_state_dict(checkpoint)
model.eval()
likelihood.eval()
#X = clusters.next_train()
#val_x = torch.from_numpy(X).float().cuda()
#val_y = torch.from_numpy(estimate_density(X)).float().cuda()

print(train_x.shape)
f_preds = model(train_x)
random_x = (np.random.uniform(-1, 4, (1000, 2)) - dataset.means) / dataset.stds
y_random_x, _, _ = normalize_density(np.array([rosenbrock(x) for x in random_x]))
print(y_random_x.shape)
f_preds_2 = model(torch.from_numpy(random_x).float().cuda())
y_preds = likelihood(f_preds)
y_preds_2 = likelihood(f_preds_2)

# f_mean = f_preds.mean
# f_var = f_preds.variance
# f_covar = f_preds.covariance_matrix
# f_samples = f_preds.sample(sample_shape=torch.Size([1,]))
with torch.no_grad():
    # plt.plot(y_preds.variance.cpu().numpy()[:1000], label='predicted var')
    plt.plot(y_preds.mean.cpu().numpy(), label='predicted')
    plt.plot(train_y.cpu(), label='rosenbrock', alpha=0.5, color='green')
    # plt.plot(y_random_x, label='rosenbrock')
    # plt.yscale('log')
    plt.xlabel('Test sample ID')
    plt.ylabel('Normalized Density')
    plt.savefig(f'{model_path}/inference_test_1.png')
    plt.legend()
    plt.show()

    # plt.plot(y_preds.variance.cpu().numpy()[:1000], label='predicted var')
    plt.plot(y_preds_2.mean.cpu().numpy(), label='predicted')
    plt.plot(y_random_x, alpha=0.5, label='rosenbrock')
    plt.xlabel('Test sample ID')
    plt.ylabel('Normalized Density')
    plt.savefig(f'{model_path}/inference_test_2.png')
    plt.legend()
    plt.show()