import multiprocessing
import numpy as np
import time
import torch
import scipy.stats as st


def predict_density(model, likelihood, grid_point):
    grid_point = torch.from_numpy(grid_point).float().unsqueeze(0).cuda()
    # f_preds = self.model(grid_point)
    y_preds = likelihood(model(grid_point))
    with torch.no_grad():
        return y_preds.mean[0].cpu().numpy()
    # return y_preds.sample()[0].cpu().numpy()


class Sampler:
    def __init__(self, num_samples=1000, num_dimensions=3, random_seed=42, verbose=True):
        """
        Initialize the sampler.

        Parameters
        ----------
        num_samples: int
            The number of samples to generate.
        random_seed: int
            The random seed used to initialize the random number generator.
        verbose: bool
            True if we show progress messages while the algorithm is running
        """

        self.num_samples = num_samples
        self.num_dimensions = num_dimensions
        self.random_seed = random_seed
        self.verbose = verbose
        self.bounds = [[-15, 15]] * self.num_dimensions

    def _interpolate(self, point):
        """
        TODO? - i am not sure if it is needed - it depends on the grid step and
        the number of stars in the sample

        Use linear interpolation to estimate the density at a point.

        Parameters
        ----------
        point: array
            An array of shape (D,) containing the point at which to estimate the density.

        Returns
        -------
        density: float
            The estimated density at the point.
        """
        return

    def generate_new_samples(self, mean, cov, size, truncated=False):
        """
        Generate samples from  normal / truncated normal distribution.

        Parameters
        ----------
        mean: array
            An array of shape (D,) specifying the mean of the normal distribution.
        cov: array
            The covariances of the normal distributions for each feature.
        size: int
            The number of new samples to generate.
        truncated: bool
            If True, use the truncated normal distribution to generate new samples.

        Returns
        -------
        samples: array
            An array of shape (size, D) containing the generated samples.
        """
        # initialize the samples array
        samples = np.zeros((size, len(mean)))

        # Generate samples from the truncated normal distribution in each dimension
        for d in range(len(mean)):
            mu = mean[d]
            sigma = cov[d]
            if truncated:
                a, b = self.bounds[d]
                samples[:, d] = st.truncnorm((a - mu) / sigma, (b - mu) / sigma, mu, sigma).rvs(size=size)
            else:
                samples[:, d] = np.random.normal(mu, sigma)
        return samples


class APESSampler(Sampler):
    def __init__(self, model, likelihood, clusters, num_samples, random_seed, walkers_nb=320, random_walk=False,
                 verbose=True):
        super().__init__(model, likelihood, clusters, num_samples, random_seed, verbose)
        self.walkers_nb = walkers_nb
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
        self.w = np.ones((self.walkers_nb, self.num_dimensions)) / self.walkers_per_block
        self.cov_matrix = None

    def predict_density(self, grid_point):

        with torch.no_grad():
            grid_point = torch.from_numpy(grid_point).float().unsqueeze(0).cpu()
            y_preds = self.likelihood(self.model(grid_point))
            return y_preds.mean[0].numpy()

    def compute_cov_matrix(self, i):
        # a single covariance matrix, Ck = C(xLi) is used for all distance computations of a new sample x
        cov_matrix = 0
        xk_ave = np.mean(self.realizations[self.L[i]], axis=0)
        for k in self.L[i]:
            xk = self.realizations[k]
            cov_matrix += np.outer(xk - xk_ave, xk - xk_ave)
        return cov_matrix / (self.walkers_per_block - 1)

    def kernel_func(self, d2, cov_matrix):
        # Gaussian kernel
        return np.exp(-d2 / 2) / np.sqrt((2 * np.pi) ** self.num_dimensions * np.linalg.det(cov_matrix))

    def approx_density(self, x, i):
        p = 0
        for k in self.L[i]:
            xk = self.realizations[k]
            mahalanobis_distance = np.dot(np.dot(x - xk, np.linalg.inv(self.cov_matrix)), x - xk)
            p += self.w[k] * self.kernel_func(mahalanobis_distance / self.h, self.cov_matrix) / (
                    self.h ** self.num_dimensions)
        return p[0]

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
        return samples

    def prepare_new_draw(self):
        if self.random_walk:
            cov_p = np.random.uniform(0.0001, 0.01)
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

    def sample(self, max_iter=1000):
        """
        Generate samples from the density map using APES sampling
        https://arxiv.org/pdf/2303.13667.pdf

        Returns
        -------
        samples: array
            An array of shape (num_samples, num_dimensions) containing the generated samples.
        """

        rejected = 0
        self.realizations = self.sample_walkers()
        self.density = self.predict_density(self.realizations)
        np.random.seed(self.random_seed)
        t0 = time.time()
        # run the APES sampler
        iter_count = 1

        while iter_count < max_iter:

            # if goback:
            #     return_point_idx = np.random.randint(0, count)

            # FIRST BLOCK
            # sample new block
            self.i = 0
            self.cov_matrix = self.compute_cov_matrix(1 - self.i)
            self.prepare_new_draw()

            pool = multiprocessing.Pool(processes=5)  # create a pool of processes

            # Process the items in parallel
            acc_probs = pool.map(self.compute_acceptance_probs, list(range(self.walkers_per_block)))
            pool.close()
            pool.join()

            for k in range(self.walkers_per_block):
                if np.random.rand() < acc_probs[k]:
                    self.realizations[self.L[0][k]] = self.x_new[k]
                else:
                    rejected += 1

            # SECOND BLOCK
            # sample new block
            self.i = 1
            self.cov_matrix = self.compute_cov_matrix(1 - self.i)
            self.prepare_new_draw()

            pool = multiprocessing.Pool(processes=5)  # create a pool of processes

            # Process the items in parallel
            acc_probs = pool.map(self.compute_acceptance_probs, list(range(self.walkers_per_block)))
            pool.close()
            pool.join()

            for k in range(self.walkers_per_block):
                if np.random.rand() < acc_probs[k]:
                    self.realizations[self.L[1][k]] = self.x_new[k]
                else:
                    rejected += 1

            if self.verbose and iter_count % 100 == 0:
                t1 = time.time()
                print('%d/%d iterations [time: %.3f minutes]' % (iter_count, max_iter, (t1 - t0) / 60))

        return self.realizations


class MCMCSampler(Sampler):
    def sample(self, pdf, step_size=0.1, random_walk=False):
        """
        Generate samples from the density map using MCMC sampling*.
        * this is a modified version in which the previous value is not kept
        when a sample is rejected - the sampler iterates again until the counter
        reaches the number of desired samples.

        Returns
        -------
        samples: array
            An array of shape (num_samples, num_dimensions) containing the generated samples.
        """
        # initialize the samples array and set the initial sample, declare const
        samples = np.zeros((self.num_samples, self.num_dimensions))
        samples[0] = np.zeros(self.num_dimensions)
        density = pdf(samples[0])
        prev_density = density.copy()
        np.random.seed(self.random_seed)
        t0 = time.time()
        # run the MCMC sampler
        count = 1
        rejected = 0
        step_size = [step_size] * self.num_dimensions
        while count < self.num_samples:
            sampling_center = samples[count - 1] if not random_walk else samples[np.random.randint(0, count)]

            # sample a new point from a truncated normal distribution centered at the current point
            new_point = self.generate_new_samples(mean=sampling_center, cov=step_size, size=1)[0]
            # the density at the new point
            density = pdf(new_point)

            # the acceptance probability
            acceptance_prob = np.min([1.0, density / prev_density])

            # apply Metropolis algorithm to decide if accept the new point

            if self.verbose and count % 100 == 0:
                t1 = time.time()
                print('%d/%d generated samples [time: %.3f minutes]' % (count, self.num_samples, (t1 - t0) / 60))
                print('Acceptance rate: %.4f ' % (count / (count + rejected)))

            if np.random.rand() < acceptance_prob:
                samples[count] = new_point
                count += 1
                prev_density = density.copy()
                self.verbose = True
            else:
                rejected += 1
                self.verbose = False

        return samples


class MCMCSamplerF(Sampler):
    def sample(self, step_size=0.1, goback=False, random_walk=False, ref=None):
        """
        Generate samples from the density map using MCMC sampling*.
        * this is a modified version in which the previous value is not kept
        when a sample is rejected - the sampler iterates again until the counter
        reaches the number of desired samples.

        Returns
        -------
        samples: array
            An array of shape (num_samples, num_dimensions) containing the generated samples.
        """
        # initialize the samples array and set the initial sample, declare const
        samples = np.zeros((self.num_samples, self.num_dimensions))
        samples[0] = np.zeros(self.num_dimensions)
        attempts = 0
        if not random_walk:
            step_size = [step_size] * self.clusters.dims
        density = self.predict_density(samples[0])
        prev_density = density.copy()
        np.random.seed(self.random_seed)
        t0 = time.time()
        # run the MCMC sampler
        count = 1 if ref is None else 0
        while count < self.num_samples:
            attempts += 1
            if goback:
                return_point_idx = np.random.randint(0, count)
            if random_walk:
                if ref is None:
                    step_size_p = np.random.uniform(0.0001, 0.5)
                    step_size_v = np.random.uniform(0.0001, 0.01, 3)
                    step_size_m = np.random.uniform(10, 50)
                else:
                    step_size_p = np.random.uniform(0.0001, 0.01)
                    step_size_v = np.random.uniform(0.001, 0.01, 3)
                    step_size_m = np.random.uniform(0.01, 0.1)

                step_size = [step_size_p] * 3 + list(step_size_v) + [step_size_m]

            if attempts > 20:
                for idx, s in enumerate(step_size):
                    step_size[idx] = s * 10

            if ref is None:
                sampling_center = samples[return_point_idx] if goback else samples[count - 1]
            else:
                sampling_center = ref[count]

            # sample a new point from a truncated normal distribution centered at the current point
            new_point = self.truncated_normal(mean=sampling_center, cov=step_size, size=1)

            # the density at the new point
            density = self.predict_density(new_point)[0]

            # the acceptance probability
            acceptance_prob = np.min([1.0, density / prev_density])

            # apply Metropolis algorithm to decide if accept the new point

            if self.verbose and count % 100 == 0:
                t1 = time.time()
                print('%d/%d generated samples [time: %.3f minutes]' % (count, self.num_samples, (t1 - t0) / 60))

            if np.random.rand() < acceptance_prob:
                samples[count] = new_point
                count += 1
                attempts = 0
                if ref is None:
                    prev_density = density.copy()
                else:
                    prev_density = self.predict_density(ref[count])
            else:
                pass

        return samples


class RejectionSampler(Sampler):
    def sample(self, pdf):
        """
        Generate samples from the density map using rejection sampling.

        Returns
        -------
        samples: array
            An array of shape (num_samples, num_dimensions) containing the generated samples.
        """

        # initialize, declare the sampler constants

        max_density = 1

        samples = np.zeros((self.num_samples, self.num_dimensions))
        samples[0] = np.zeros(self.num_dimensions)

        np.random.seed(self.random_seed)

        t0 = time.time()
        # run the rejection sampler
        count = 1
        while count < self.num_samples:

            # sample a point uniformly at random from the sample space
            sample = []
            for d in range(self.num_dimensions):
                lower, upper = self.bounds[d]
                sample.append(np.random.uniform(lower, upper))

            # accept the sample with probability proportional to its density
            density = pdf(np.array(sample))
            if density > max_density:
                max_density = density
            if np.random.rand() < density / max_density:
                samples[count] = sample
                count += 1
                if self.verbose and count % 100 == 0:
                    t1 = time.time()
                    print('%d/%d generated samples [time: %.3f minutes]' % (count, self.num_samples, (t1 - t0) / 60))

        return samples
