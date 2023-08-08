import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import scipy.stats as st
import gpytorch
import torch
from clusters import Clusters
from GP_model import ExactGPModel
from samplers import MetropolisSampler
import time

# data
clusters = Clusters(features='all')
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

model_path = "./results/GP_mass_all_clusters_aug06_t1007"
# Load saved model weights
checkpoint = torch.load(f'{model_path}/model.pth')

# Load weights into model
model.load_state_dict(checkpoint)

model.eval()
likelihood.eval()
print('noise:', model.likelihood.noise)
print('lengthscale:', model.covar_module.base_kernel.lengthscale)


def gp_pdf(proposed, model_, likelihood_):
    with torch.no_grad():
        proposed = torch.from_numpy(proposed).float().unsqueeze(0).cuda()
        y_preds = likelihood_(model_(proposed))
        return y_preds.sample(sample_shape=torch.Size([1, ]))[0][0].cpu().numpy()


e_pdf = lambda z: gp_pdf(z, model, likelihood)

sampler = MetropolisSampler(num_samples=3000, num_dimensions=1)
sampler.set_bounds([[0.5, 100]])
samples = sampler.sample(e_pdf, step_size=0.8, burn_in_max=100, random_walk=True)

timestamp = time.strftime(f"mass_distribution_{3000}stars_aug%d_t%H%M_M{int(np.sum(samples))}", time.gmtime())
np.save(f'{model_path}/{timestamp}', samples)

plt.hist(samples, bins=50, density=True, alpha=0.8)
plt.plot(x.cpu().numpy(), y.cpu().numpy(), 'o', markersize=1, alpha=0.5, label='KDE')

plt.legend()
plt.show()
