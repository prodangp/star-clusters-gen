import numpy as np
import time
import torch
import scipy.stats as st


class Sampler:
    def __init__(self, model, likelihood, clusters, num_samples, random_seed, verbose=True):
        """
        Initialize the sampler.

        Parameters
        ----------
        model: ExactGP model
            The model we trained for generating new realizations.
        likelihood: Likelihood function
            The likelihood function we use to predict the density
        clusters: Clusters object
            Object containing information about the simulation clusters
        num_samples: int
            The number of samples to generate.
        random_seed: int
            The random seed used to initialize the random number generator.
        """
        model.eval()
        self.model = model
        likelihood.eval()
        self.likelihood = likelihood
        self.clusters = clusters
        self.num_samples = num_samples
        self.random_seed = random_seed
        self.num_dimensions = len(clusters.mask)
        self.verbose = verbose

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

    def truncated_normal(self, mean, cov, size):
        """
        Generate samples from a truncated normal distribution.

        Parameters
        ----------
        mean: array
            An array of shape (D,) specifying the mean of the normal distribution.
        cov: float
            The covariance of the normal distribution.
        size: int
            The number of samples to generate.

        Returns
        -------
        samples: array
            An array of shape (size, D) containing the generated samples.
        """
        # initialize the samples array
        samples = np.zeros((size, len(mean)))

        #  specifying the bounds of the truncated normal distribution in each dimension.
        bounds = self.clusters.get_ave_bounds()

        # Generate samples from the truncated normal distribution in each dimension
        for d in range(len(mean)):
            a, b = bounds[d]
            mu = mean[d]
            sigma = cov[d]  # if d else cov * 10
            samples[:, d] = st.truncnorm((a - mu) / sigma, (b - mu) / sigma, mu, sigma).rvs(size=size)

        return samples

    def predict_density(self, grid_point):
        grid_point = torch.from_numpy(grid_point).float().unsqueeze(0).cuda()
        f_preds = self.model(grid_point)
        y_preds = self.likelihood(self.model(grid_point))
        with torch.no_grad():
            return y_preds.mean[0].cpu().numpy()

        # return y_preds.sample()[0].cpu().numpy()


class MCMCSampler(Sampler):
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
                # samples[count] = samples[count - 1].copy()
                # p = np.random.choice(3, 1)[0]
                # samples[count][p + 1] = -samples[count][p + 1]

        return samples


class RejectionSampler(Sampler):
    def sample(self):
        """
        Generate samples from the density map using rejection sampling.

        Returns
        -------
        samples: array
            An array of shape (num_samples, num_dimensions) containing the generated samples.
        """

        # initialize, declare the sampler constants

        max_density = 1
        bounds = self.clusters.get_ave_bounds()

        samples = np.zeros((self.num_samples, self.num_dimensions))
        samples[0] = np.zeros(self.num_dimensions)

        density = self.predict_density(samples[0])
        prev_density = density.copy()
        np.random.seed(self.random_seed)

        t0 = time.time()
        # run the rejection sampler
        count = 1
        while count < self.num_samples:

            # sample a point uniformly at random from the sample space
            sample = []
            for d in range(self.num_dimensions):
                lower, upper = bounds[d]
                sample.append(np.random.uniform(lower, upper))

            # accept the sample with probability proportional to its density
            density = self.predict_density(np.array(sample))
            if density > max_density:
                max_density = density
            if np.random.rand() < density / max_density:
                samples[count] = sample
                count += 1
                if self.verbose and count % 100 == 0:
                    t1 = time.time()
                    print('%d/%d generated samples [time: %.3f minutes]' % (count, self.num_samples, (t1 - t0) / 60))

        return samples