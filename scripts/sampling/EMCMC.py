import time
import psutil
import os
import torch
import gpytorch
import numpy as np
from GP_model import ExactGPModel
from utils import feature_scaling, estimate_density, gp_pdf, normalize_density
from samplers import EMCMCSampler
from clusters import Clusters

# Get initial memory usage
process = psutil.Process(os.getpid())
mem_info = process.memory_info()
initial_memory = mem_info.rss

model_path = '../../results_chains/chains_10c_GP_pe_ke_5000iter_aug06_t1616/'

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
data_path = '/data/chains'

train = ["m1.e4", "m1.e5", "m2.e4", "m3.e4", "m4.e4", "m5.e4", "m6.e4"]

params = {
    "pe": {
        "mu": [],
        "std": []
    },
    "ke":
        {
        "mu": [],
        "std": []
    }
}

for file in train:
    pe = np.load(f'{data_path}/pe_{file}.dat.npy')
    pe = np.log(pe[:-1])
    ke = np.load(f'{data_path}/ke_{file}.dat.npy')
    ke = np.log(ke[:-1])

    pe, mu_pe, std_pe = feature_scaling(pe, return_mean_std=True)
    ke, mu_ke, std_ke = feature_scaling(ke, return_mean_std=True)

    params['pe']['mu'].append(mu_pe)
    params['pe']['std'].append(std_pe)
    params['ke']['mu'].append(mu_ke)
    params['ke']['std'].append(std_ke)

standardization_params = {
    "pe": {
        "mu": np.mean(params['pe']['mu']),
        "std": np.sqrt(np.mean(np.array(params['pe']['std']) ** 2))
    },
    "ke":
        {
        "mu": np.mean(params['ke']['mu']),
        "std": np.sqrt(np.mean(np.array(params['ke']['std']) ** 2))
    }
}

X = np.zeros((pe.shape[0] - 1, 3))
X[:, 0] = pe[:-1].flatten()
X[:, 1] = ke[:-1].flatten()
X[:, 2] = ke[1:].flatten()

# compute and scale density
y = normalize_density(estimate_density(X))

train_x = torch.from_numpy(X).float().cuda()
train_y = torch.from_numpy(y).float().cuda()

model = ExactGPModel(train_x, train_y, likelihood).cuda()

# Load saved model weights
checkpoint = torch.load(f'{model_path}/model.pth')

# Load weights into model
model.load_state_dict(checkpoint)
model.eval()
likelihood.eval()

print('en noise:', model.likelihood.noise)
print('en lengthscale:', model.covar_module.base_kernel.lengthscale)

clusters = Clusters(features='all')
# initialize likelihood and model
likelihood_ps = gpytorch.likelihoods.GaussianLikelihood()
X, cluster_name_tr = clusters.next_train(return_name=True)
mass_mu = np.mean(X[:, 0])
std_mu = np.std(X[:, 0])
# normalize data
y = normalize_density(estimate_density(X[:, 1:]))

train_x = torch.from_numpy(X[:, 1:]).float().cuda()
train_y = torch.from_numpy(y).float().cuda()

model_ps = ExactGPModel(train_x, train_y, likelihood_ps).cuda()

# Load saved model weights
checkpoint = torch.load('../../results/6D_10c_aug06_t0929/model.pth')
# checkpoint = torch.load('./results/sim_1c_april06_t1115/model.pth')
# Load weights into model
model_ps.load_state_dict(checkpoint)
model_ps.eval()
likelihood_ps.eval()

print('6D noise:', model_ps.likelihood.noise)
print('6D lengthscale:', model_ps.covar_module.base_kernel.lengthscale)
#
N = 3000
rij_th = 1.5
step_size = 0.2
Nc = 5000
m_lower_bound = 0.5
# mass dist with M > 0.1
# m_dist = np.load(f'./results/GP_mass_all_clusters_aug06_t1007/mass_distribution_3000stars_aug06_t1106.npy')
# mass dist with M > 0.5
m_dist = np.load(f'../../results/GP_mass_all_clusters_aug06_t1007/mass_distribution_3000stars_aug08_t1157_M13540.npy')
e_pdf = lambda z: gp_pdf(z, model, likelihood)
ps_pdf = lambda z: gp_pdf(z, model_ps, likelihood_ps)

# Get memory usage before starting sampling
mem_info = process.memory_info()
memory_before_sampling = mem_info.rss
print('Memory used by GP model: ', (memory_before_sampling - initial_memory) / (1024 ** 3), ' GB')

sampler = EMCMCSampler(m_dist, e_pdf, ps_pdf, [mass_mu, std_mu], num_samples=N, rij_th=rij_th, iter_per_sample=Nc)
sampler.set_standardization_params(standardization_params)
sampler.set_bounds([[-2, 1.5], [-1.5, 2], [-1.5, 2]])
samples = sampler.sample(step_size=step_size, burn_in_max=500)

timestamp = time.strftime(f"gen_cluster_EMCMC_{N}stars_s{step_size}_r{rij_th}_Nc{Nc}_minM{m_lower_bound}_aug%d_t%H%M", time.gmtime())

# Get memory after sampling
mem_info = process.memory_info()
memory_final = mem_info.rss #
print('Memory used by sampling algorithm: ', (memory_final - memory_before_sampling) / (1024 ** 3), ' GB')

np.save(f'{model_path}/{timestamp}', samples)
# plt.plot(samples[:, 0], samples[:, 1], 'ro', markersize=1)
# new_cluster = np.zeros((n_stars, 4))
# stars = [None] * n_stars
# next_star = torch.cat((y_preds.mean.unsqueeze(-1)[0], y_preds.mean[1:] + train_x[0].squeeze(-1)[1:]))
#
#
#
# for i in tqdm(range(n_stars)):
#     stars[i] = next_star
#     sample = [0, 0, 0, 0]  # reset sample such that it can enter the while loop to find another
#     while not check_limits(sample):
#         current_star = stars[random.randint(0, i)]
#         f_preds = model(current_star)
#         y_preds = likelihood(f_preds)
#         y = y_preds.sample()
#         next_star = torch.cat((y.unsqueeze(-1)[0], y[1:] + current_star.squeeze(-1)[1:]))
#         sample = next_star.detach().cpu().numpy()
#     new_cluster[i] = sample
#
# print(new_cluster)
# timestamp = time.strftime(f"gen_cluster_GP_pos_{n_stars}stars_may%d_t%H%M", time.gmtime())
# np.save(f'{model_path}/{timestamp}', new_cluster)



# def check_limits(x):
#     limits = [[0.1, 20], [-8, 8], [-6, 6], [-5, 5]]
#     for i in range(len(x)):
#         if not (limits[i][0] <= x[i] <= limits[i][1]):
#             return False
#     return True
#
#
# for i in tqdm(range(n_stars)):
#     stars[i] = next_star
#     sample = [0, 0, 0, 0]  # reset sample such that it can enter the while loop to find another
#     while not check_limits(sample):
#         current_star = stars[random.randint(0, i)]
#         f_preds = model(current_star)
#         y_preds = likelihood(f_preds)
#         y = y_preds.sample()
#         next_star = torch.cat((y.unsqueeze(-1)[0], y[1:] + current_star.squeeze(-1)[1:]))
#         sample = next_star.detach().cpu().numpy()
#     new_cluster[i] = sample
#
# print(new_cluster)
# timestamp = time.strftime(f"gen_cluster_GP_pos_{n_stars}stars_may%d_t%H%M", time.gmtime())
# np.save(f'{model_path}/{timestamp}', new_cluster)
#
