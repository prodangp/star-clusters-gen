import time
import numpy as np
import multiprocessing
from matplotlib import pyplot as plt
import scipy.stats as st
from scipy.optimize import nnls
from scipy.special import gamma

from GP_model import ExactGPModel
from utils import estimate_density, feature_scaling, reverse_scaling, normalize_density
from clusters import Clusters
from samplers import Sampler


class APESSampler(Sampler):
    def __init__(self, model, likelihood, clusters, num_samples, random_seed, walkers_nb=320, burn_in=5000, random_walk=False, verbose=True):
        super().__init__(model, likelihood, clusters, num_samples, random_seed, verbose)
        self.walkers_nb = walkers_nb
        self.burn_in = burn_in
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

    def predict_density(self, grid_point):

        with torch.no_grad():
            grid_point = torch.from_numpy(grid_point).float().unsqueeze(0).cpu()
            y_preds = self.likelihood(self.model(grid_point))
            return y_preds.mean[0].numpy()

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

    def kernel_func(self, d2, cov_matrix, kernel='student'):
        # Gaussian kernel
        if kernel == 'gauss':
            return np.exp(-d2 / 2) / np.sqrt((2 * np.pi) ** self.num_dimensions * np.linalg.det(cov_matrix))
        elif kernel == 'student':
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
        bounds = [[-10, 10], [-10, 10], [-10, 10]]
        # Generate samples from the truncated normal distribution in each dimension
        for d in range(self.num_dimensions):
            a, b = bounds[d]
            mu = 0
            sigma = cov[d]
            samples[:, d] = st.truncnorm((a - mu) / sigma, (b - mu) / sigma, mu, sigma).rvs(size=self.walkers_nb)
        self.density = self.predict_density(samples)
        return samples

    def prepare_new_draw(self, block):
        self.i = block - 1
        self.update_cov_matrix(1 - self.i)
        self.density = self.predict_density(self.realizations)
        self.update_weights()
        if self.random_walk:
            cov_p = np.random.uniform(0.001, 0.1)
            cov_v = np.random.uniform(0.001, 0.01, 3)
            cov_m = np.random.uniform(0.01, 0.1)
            cov = [cov_p] * 3 + list(cov_v) + [cov_m]
        else:
            cov = [0.1, 0.1, 0.1]
        self.x = self.realizations[self.L[self.i]]
        self.x_new = np.zeros((self.walkers_per_block, self.num_dimensions))
        for k in range(self.walkers_per_block):
            self.x_new[k] = self.truncated_normal(mean=self.x[k], cov=cov, size=1)
        self.density_new = self.predict_density(self.x_new)


if __name__ == '__main__':
    import torch
    import gpytorch
    # data
    clusters = Clusters(features='pos')
    X, cluster_name_tr = clusters.next_train(return_name=True)

    # normalize data
    y, d_min, d_max = normalize_density(estimate_density(X))
    X = feature_scaling(X)
    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    train_x = torch.from_numpy(X).float().cpu()
    train_y = torch.from_numpy(y).float().cpu()
    model = ExactGPModel(train_x, train_y, likelihood).cpu()

    # Load saved model weights
    model_path = './results/3D_baseline_april18_t0716'
    checkpoint = torch.load(f'{model_path}/model.pth')
    model.load_state_dict(checkpoint)

    timestamp = time.strftime("march%d_t%H%M", time.gmtime())
    sampler = APESSampler(model, likelihood, clusters, 5000, walkers_nb=50, random_seed=40, random_walk=True)
    samples = []
    # sampler.i = 0
    # sampler.realizations = sampler.sample_walkers()
    # sampler.cov_matrix = sampler.compute_cov_matrix(1 - sampler.i)
    # sampler.prepare_new_draw()
    # sampler.density = sampler.predict_density(sampler.realizations)
    # print(sampler.density)
    # print(sampler.compute_acceptance_probs(0))

    max_iter = 80
    burn_in = 30
    rejected = 0
    sampler.realizations = sampler.sample_walkers()
    np.random.seed(44)
    t0 = time.time()

    # run the APES sampler
    iter_count = 0
    discard = True
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
            print('%d/%d iterations --- rejection rate = %.3f [time: %.3f minutes]' % (iter_count, max_iter,
                  rejected / (iter_count * sampler.walkers_nb), (t1 - t0) / 60))

        if iter_count >= burn_in and discard:
            discard = False
            print(f"--- BURN IN ENDED ---")
            iter_count -= burn_in
            rejected = 0

    np.save(f'{model_path}/APES_4000_50_student_RW.npy', samples)
    x = np.array(samples)
    #y_x = reverse_scaling(feature_scaling(estimate_density(x)), means=mean, stds=std)

    # Create the figure and axes objects
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    # Plot the data and the estimate density on the first axis
    ax1.scatter(x[:, 0], x[:, 1], c=estimate_density(x), edgecolor="k", s=60, cmap="plasma")
    ax1.set_title('APES', fontsize=16)
    ax1.set_xlabel('x (pc)', fontsize=12)
    ax1.set_ylabel('y (pc)', fontsize=12)
    ax1.set_xlim([np.min([np.min(X[:, 0]), np.min(x[:, 0])]), np.max([np.max(X[:, 0]), np.max(x[:, 0])])])
    ax1.set_ylim([np.min([np.min(X[:, 1]), np.min(x[:, 1])]), np.max([np.max(X[:, 1]), np.max(x[:, 1])])])
    fig.colorbar(ax1.collections[0], ax=ax1)

    # Plot the data and the estimate density on the second axis
    ax2.scatter(X[:, 0], X[:, 1], c=estimate_density(X), edgecolor="k", s=60, cmap="plasma")
    ax2.set_title('Simulation Data', fontsize=16)
    ax2.set_xlabel('x (pc)', fontsize=12)
    ax2.set_ylabel('y (pc)', fontsize=12)
    fig.colorbar(ax2.collections[0], ax=ax2)
    plt.savefig(f'{model_path}/APES_RW.png')
    # Show the plot
    plt.show()
