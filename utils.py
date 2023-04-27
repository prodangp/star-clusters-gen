import math
import glob
import numpy as np
from matplotlib import pyplot as plt


def estimate_density(x, k=5):
    """Estimate the density of a dataset using the k-nearest neighbors method.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The dataset for which to estimate the density.
    k : int, optional (default=5)
        The number of nearest neighbors to use for density estimation.

    Returns
    -------
    densities : array, shape (n_samples,)
        The estimated densities for each sample in the dataset.
    """
    dens = np.zeros(len(x))

    # The volume of a unit d-dimensional ball is given by the formula
    d = len(x[0])
    volume = math.pi ** (d / 2) / math.gamma(d / 2 + 1)

    for i in range(len(x)):
        r = np.sort(
            np.sqrt(
                np.sum([(x[:, j] - x[i][j]) ** 2 for j in range(len(x[0]))], axis=0)
            )
        )[k]

        dens[i] = k / (volume * (r ** d))

    return dens


def normalize_density(density):
    log_density = np.log(density + 1)
    # Shift and rescale density to be in the range [0, 1]
    density_min = np.min(log_density)
    density_shifted = log_density - density_min
    density_max = np.max(density_shifted)
    density_rescaled = density_shifted / density_max
    return density_rescaled, density_min, density_max


def inverse_rescaled_density(rescaled_density, density_min, density_max):
    density_shifted = rescaled_density * density_max
    density = density_shifted + density_min
    density = np.exp(density) - 1
    return density


def feature_scaling(inputs, return_mean_std=False):
    # Compute the mean and standard deviation of each feature
    means = np.mean(inputs, axis=0)
    stds = np.std(inputs, axis=0)

    # Scale the inputs
    scaled_inputs = (inputs - means) / stds

    # Return the scaled inputs and optionally the mean and standard deviation of each feature
    if return_mean_std:
        return scaled_inputs, means, stds
    else:
        return scaled_inputs


def reverse_scaling(rescaled_samples, means, stds):
    if len(rescaled_samples.shape) == 1:
        # Reverse scaling for 1D array with mean and std as floats
        original_samples = rescaled_samples * stds + means
    else:
        # Reverse scaling of each feature
        original_samples = rescaled_samples.copy()
        for i, (mean, std) in enumerate(zip(means, stds)):
            original_samples[:, i] = rescaled_samples[:, i] * std + mean

    return original_samples


class Clusters:
    def __init__(self, features='all', n=5000, path="./data/", generate=False,
                 mean=None, cov=None):
        self.N = n
        if features == 'all':
            self.mask = range(0, 7)
        elif features == 'pos':
            self.mask = range(1, 4)
        elif features == 'vel':
            self.mask = range(4, 7)
        else:
            print(f'[WARNING] {features} features not recognized! Full features are used.')
            self.mask = range(0, 7)
        self.dims = len(self.mask)
        self.iter_train = 0
        self.iter_val = 0
        self.n_train = 7
        self.n_val = 3
        if generate:
            self._mean = [0] * self.dims if mean == None else mean
            self._cov = self.generate_random_spd_matrix() if cov == None else cov
            self.data = np.random.multivariate_normal(self._mean, self._cov, self.N)
        else:
            self.path = path
            files = glob.glob(path + 'sink*')
            global N_CLUSTERS
            N_CLUSTERS = len(files)
            self.data_train = np.empty((self.n_train,), dtype=np.ndarray)
            self.data_val = np.empty((self.n_val,), dtype=np.ndarray)
            self.names_train = []
            self.names_val = []
            for i in range(N_CLUSTERS):
                if i < self.n_train:
                    self.data_train[i] = np.loadtxt(files[i], skiprows=1)[:, self.mask]
                    self.names_train.append(files[i][-9:])
                else:
                    self.data_val[i - self.n_train] = np.loadtxt(files[i], skiprows=1)[:, self.mask]
                    self.names_val.append(files[i][-9:])

    def next_train(self, return_name=False):
        self.iter_train += 1
        if return_name:
            return self.data_train[(self.iter_train - 1) % self.n_train], self.names_train[
                (self.iter_train - 1) % self.n_train]
        else:
            return self.data_train[(self.iter_train - 1) % self.n_train]

    def next_val(self, idx, return_name=False):
        if return_name:
            return self.data_val[idx], self.names_val[idx]
        else:
            return self.data_val[idx]

    def count_stars_train(self):
        return np.sum([self.clusters.data_train[idx].shape[0] for idx in range(self.n_train)])

    def count_stars_val(self, idx):
        return self.clusters.data_val[idx].shape[0]

    def generate_random_spd_matrix(self):
        # generate a random symmetric matrix
        A = np.random.rand(self.dims, self.dims)
        A = (A + A.T) / 2
        # ensure that the matrix is semi-positive definite by adding a multiple of the identity matrix
        I = np.eye(self.dims)
        A += 5 * I
        return A

    def subsample(self, n=300, plot=True):
        # select a random subsample
        idx = np.random.choice(np.arange(len(self.data)), n, False)
        x = self.data[idx]
        if plot and x.shape[1] == 2:
            plt.scatter(self.data[:, 0], self.data[:, 1], s=5)
            plt.scatter(x[:, 0], x[:, 1], c=estimate_density(x), edgecolor="k", s=60, cmap="plasma")
            plt.colorbar()
            plt.show()
        return x

    def get_bounds(self, X):
        # Getter for the bounds of the density map in each dimension
        bounds = []
        for d in range(self.dims):
            bounds.append((min(X[:, d]), max(X[:, d])))
        return bounds

    def get_ave_bounds(self):
        add = False
        for j in range(7):
            c = self.next_train()
            if add:
                ave_bounds += np.array(self.get_bounds(c))
            else:
                ave_bounds = np.array(self.get_bounds(c))
                add = True
        return ave_bounds / 7
