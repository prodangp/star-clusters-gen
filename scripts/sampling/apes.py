import time
import psutil
import os
import numpy as np
import multiprocessing
import scipy.stats as st
from scipy.optimize import nnls
from scipy.special import gamma

from GP_model import ExactGPModel
from utils import estimate_density, normalize_density
from clusters import Clusters
from samplers import Sampler


class APESSampler(Sampler):
    def __init__(self, model, likelihood, num_samples, random_seed=42, walkers_nb=320, kernel='student', burn_in=5000, random_walk=False, verbose=True):
        super().__init__(num_samples=num_samples, random_seed=random_seed, verbose=verbose)
        self.walkers_nb = walkers_nb
        self.kernel = kernel
        self.burn_in = burn_in
        self.num_dimensions = 7
        self.bounds = [[-15, 15]] * self.num_dimensions
        # mass bounds
        self.bounds[0] = [0.1, 100]
        self.random_walk = random_walk
        self.L = [list(range(self.walkers_nb // 2)), list(range(self.walkers_nb // 2, self.walkers_nb))]
        self.walkers_per_block = len(self.L[0])
        if len(self.L[0]) != len(self.L[1]):
            print('[WARNING] The blocks have a different number of walkers!')
        self.realizations = None
        self.x = None
        self.density = None
        self.x_new = None
        self.density_new = None
        self.i = None
        self.h = (4 / (self.walkers_per_block * (self.num_dimensions + 2))) ** (1 / (self.num_dimensions + 4))
        self.w = np.ones((self.walkers_nb, 1)) / self.walkers_per_block
        self.cov_matrix = None
        self.model = model
        self.likelihood = likelihood

    def update_cov_matrix(self, i):
        # a single covariance matrix, Ck = C(xLi) is used for all distance computations of a new sample x
        cov_matrix = 0
        xk_ave = np.mean(self.realizations[self.L[i]], axis=0)
        for k in self.L[i]:
            xk = self.realizations[k]
            cov_matrix += np.outer(xk - xk_ave, xk - xk_ave)
        self.cov_matrix = cov_matrix / (self.walkers_per_block - 1)

    def update_weights(self):
        x_ = self.realizations[self.L[1 - self.i]]
        k_ = np.zeros((self.walkers_per_block, self.walkers_per_block))
        # Define the matrix K and the vector pi
        for i in range(self.walkers_per_block):
            for j in range(self.walkers_per_block):
                k_[i][j] = self.kernel_func(np.dot(np.dot(x_[i] - x_[j], np.linalg.inv(self.cov_matrix)),
                                                   x_[i] - x_[j]) / self.h, self.cov_matrix) \
                           / (self.h ** self.num_dimensions)

        # Solve the NNLS problem min||Kw - pi||^2 subject to w >= 0
        self.w, _ = nnls(k_, self.density[self.L[1 - self.i]])

    def kernel_func(self, d2, cov_matrix):
        # Gaussian kernel
        if self.kernel == 'gauss':
            return np.exp(-d2 / 2) / np.sqrt((2 * np.pi) ** self.num_dimensions * np.linalg.det(cov_matrix))
        elif self.kernel == 'student':
            d = np.sqrt(d2)
            nu = 3
            return gamma((nu + self.num_dimensions) / 2) * (1 + (d2 / nu)) ** ((nu + self.num_dimensions) / 2)\
                / (gamma(nu / 2) * np.sqrt(np.linalg.det(self.cov_matrix) * (nu * np.pi) ** d))

    def approx_density(self, x, i):
        p = 0
        for k in range(self.walkers_per_block):
            xk = self.realizations[self.L[i][k]]
            mahalanobis_distance = np.dot(np.dot(x - xk, np.linalg.inv(self.cov_matrix)), x - xk)
            p += self.w[k] * self.kernel_func(mahalanobis_distance / self.h, self.cov_matrix) / (
                        self.h ** self.num_dimensions)
        return p

    def compute_acceptance_probs(self, k):
        return np.min([1, (self.approx_density(self.x[k], 1 - self.i)
                      / self.approx_density(self.x_new[k], 1 - self.i))
                      * (self.density_new[k] / self.density[self.L[self.i][k]])])

    def sample_walkers(self):
        """
        Generate e initial sample with L points.

        Returns
        -------
        samples: array
            An array of shape (L, dim) containing the generated samples.
        """
        # initialize the samples array
        samples = np.zeros((self.walkers_nb, self.num_dimensions))
        cov_p = 0.5
        cov_v = 0.5
        cov_m = 1.0
        cov = [cov_p] * 3 + [cov_v] * 3 + [cov_m]
        # Generate samples from the truncated normal distribution in each dimension
        for d in range(self.num_dimensions):
            a, b = self.bounds[d]
            mu = 0
            sigma = cov[d]
            samples[:, d] = st.truncnorm((a - mu) / sigma, (b - mu) / sigma, mu, sigma).rvs(size=self.walkers_nb)
        self.density = self.predict_density(samples)
        return samples

    def predict_density(self, grid_point):
        with torch.no_grad():
            grid_point = torch.from_numpy(grid_point).float().unsqueeze(0).cpu()
            y_preds = self.likelihood(model(grid_point))
            return y_preds.mean[0].cpu().numpy()

    def prepare_new_draw(self, block):
        self.i = block - 1
        self.update_cov_matrix(1 - self.i)
        self.density = self.predict_density(self.realizations)
        self.update_weights()
        if self.random_walk:
            cov_p = np.abs(np.random.normal(0.5, 0.5, 3))
            cov_v = np.abs(np.random.normal(0.5, 0.5, 3))
            cov_m = np.abs(np.random.normal(2, 1))
            cov = list(cov_p) + list(cov_v) + [cov_m]
        else:
            cov = [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        self.x = self.realizations[self.L[self.i]]
        self.x_new = np.zeros((self.walkers_per_block, self.num_dimensions))
        for k in range(self.walkers_per_block):
            self.x_new[k] = self.generate_new_samples(mean=self.x[k], cov=cov, size=1, truncated=True)
        self.density_new = self.predict_density(self.x_new)



if __name__ == '__main__':
    import torch
    import gpytorch


    # Get initial memory usage
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    initial_memory = mem_info.rss

    # data
    clusters = Clusters(features='all')
    X, cluster_name_tr = clusters.next_train(return_name=True)

    # normalize data
    y = normalize_density(estimate_density(X, k=50))
    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    train_x = torch.from_numpy(X).float().cpu()
    train_y = torch.from_numpy(y).float().cpu()
    model = ExactGPModel(train_x, train_y, likelihood).cpu()
    # Load saved model weights
    model_path = '../../results/7D_10c_aug07_t0753'
    checkpoint = torch.load(f'{model_path}/model.pth')
    model.load_state_dict(checkpoint)
    model.eval()
    likelihood.eval()
    timestamp = time.strftime("aug%d_t%H%M", time.gmtime())
    walkers_nb = 50
    kernel = 'gauss'
    random_walk = False
    sampler = APESSampler(model, likelihood, 3000, walkers_nb=walkers_nb, kernel=kernel, random_seed=40,
                          random_walk=random_walk)
    samples = []
    # sampler.i = 0
    # sampler.realizations = sampler.sample_walkers()
    # sampler.cov_matrix = sampler.compute_cov_matrix(1 - sampler.i)
    # sampler.prepare_new_draw()
    # sampler.density = sampler.predict_density(sampler.realizations)
    # print(sampler.density)
    # print(sampler.compute_acceptance_probs(0))

    max_iter = 10
    burn_in = 10
    rejected = 0
    sampler.realizations = sampler.sample_walkers()
    np.random.seed(44)
    t0 = time.time()

    # run the APES sampler
    iter_count = 0
    discard = True

    # Get memory usage before starting sampling
    mem_info = process.memory_info()
    memory_before_sampling = mem_info.rss
    print('Memory used by GP model: ', (memory_before_sampling - initial_memory) / (1024 ** 3), ' GB')

    while iter_count < max_iter:
        # FIRST BLOCK
        # sample new block
        sampler.prepare_new_draw(block=1)

        pool = multiprocessing.Pool(processes=5)  # create a pool of processes
        # compute the acceptance probabilities in parallel
        acc_probs = pool.map(sampler.compute_acceptance_probs, list(range(sampler.walkers_per_block)))
        pool.close()
        pool.join()


        # update first block
        for k in range(sampler.walkers_per_block):
            if np.random.rand() < acc_probs[k]:
                sampler.realizations[sampler.L[0][k]] = sampler.x_new[k]
                if not discard:
                    samples.append(sampler.x_new[k])
            else:
                rejected += 1

        # SECOND BLOCK
        # sample new block
        sampler.prepare_new_draw(block=2)

        pool = multiprocessing.Pool(processes=5)  # create a pool of processes

        # compute the acceptance probabilities in parallel
        acc_probs = pool.map(sampler.compute_acceptance_probs, list(range(sampler.walkers_per_block)))
        pool.close()
        pool.join()

        # update second block
        for k in range(sampler.walkers_per_block):
            if np.random.rand() < acc_probs[k]:
                sampler.realizations[sampler.L[1][k]] = sampler.x_new[k]
                if not discard:
                    samples.append(sampler.x_new[k])
            else:
                rejected += 1

        iter_count += 1

        if sampler.verbose and iter_count % 5 == 0:
            t1 = time.time()
            print('%d/%d iterations --- rejection rate = %.4f [time: %.3f minutes]' % (iter_count, max_iter,
                  rejected / (iter_count * sampler.walkers_nb), (t1 - t0) / 60))

        if iter_count >= burn_in and discard:
            discard = False
            print(f"--- BURN IN ENDED ---")
            iter_count -= burn_in
            rejected = 0

    # Get memory after sampling
    mem_info = process.memory_info()
    memory_final = mem_info.rss  #
    print('Memory used by sampling algorithm: ', (memory_final - memory_before_sampling) / (1024 ** 3), ' GB')

    N = int((1 - (rejected / (iter_count * sampler.walkers_nb))) * max_iter * walkers_nb)
    np.save(f'{model_path}/APES_{timestamp}_{N}_{walkers_nb}_{kernel}_RW={random_walk}.npy', samples)

