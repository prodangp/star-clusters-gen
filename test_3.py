import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import scipy.stats as st
import gpytorch
import torch
from clusters import Clusters
from GP_model import ExactGPModel
plt.rcParams["font.size"] = 16




# data
clusters = Clusters(features='all', rescale=False)


def sample_walkers():
    """
    Generate e initial sample with L points.

    Returns
    -------
    samples: array
        An array of shape (L, dim) containing the generated samples.
    """
    # initialize the samples array
    samples = np.zeros((320, 3))
    cov_p = 0.5
    cov_v = 0.5
    cov_m = 1.0
    cov = [cov_p] * 3 + [cov_v] * 3 + [cov_m]
    bounds = [[-10, 10], [-10, 10], [-10, 10]]
    # Generate samples from the truncated normal distribution in each dimension
    for d in range(3):
        a, b = bounds[d]
        mu = 0
        sigma = cov[d]
        samples[:, d] = st.truncnorm((a - mu) / sigma, (b - mu) / sigma, mu, sigma).rvs(size=320)
    return samples


# Assuming x is your 1D array

x = clusters.next_train()[:, 0]
x = clusters.next_train()[:, 0]
x = clusters.next_train()[:, 0]
x = clusters.next_train()[:, 0]
x = clusters.next_train()[:, 0]

# Use KernelDensity to estimate the PDF
kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(x.reshape(-1, 1))

# Generate a range of values (the points where the density will be evaluated)
x_range = np.linspace(min(x) - 1, max(x) + 1, 1000).reshape(-1, 1)

# Get the log density for these values and exponentiate to get the density
log_density = kde.score_samples(x_range)
density = np.exp(log_density)
estimated_density = np.exp(kde.score_samples(x.reshape(-1, 1)))

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
x = torch.from_numpy(x).float().cuda()
y = torch.from_numpy(estimated_density).float().cuda()

model = ExactGPModel(x, y, likelihood).cuda()
model = model.cuda()

# Load saved model weights
checkpoint = torch.load('./results/GP_mass_all_clusters_jun30_t1812/model.pth')

# Load weights into model
model.load_state_dict(checkpoint)

model.eval()
likelihood.eval()

f_preds = model(x)
y_preds = likelihood(f_preds)
y_samples = y_preds.sample(sample_shape=torch.Size([3, ]))

with torch.no_grad():
    plt.fill_between(x_range.flatten(), density, alpha=0.2)
    plt.ylim(-0.02, max(density) + 0.02)
    plt.plot(x.cpu().numpy(), estimated_density, 'o', markersize=1, alpha=0.5, label='KDE')
    plt.plot(x.cpu().numpy(), y_samples[0].cpu().numpy(), 'o', markersize=1, alpha=0.5, label='GP1')
    plt.plot(x.cpu().numpy(), y_samples[1].cpu().numpy(), 'o', markersize=1, alpha=0.5, label='GP2')
    plt.plot(x.cpu().numpy(), y_samples[2].cpu().numpy(), 'o', markersize=1, alpha=0.5, label='GP3')
    plt.hist(x.cpu().numpy(), bins=50, density=True, alpha=0.3)
    plt.legend()
    plt.title(r"$\mathcal{G}\mathcal{P}(\vec{r}, \vec{v})$")
    plt.show()
