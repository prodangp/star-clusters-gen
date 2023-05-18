import math
import glob
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset


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


class Data(Dataset):
    def __init__(self, data):
        self.data, self.means, self.stds = feature_scaling(data, return_mean_std=True)
        # normalize data
        self.y, self.d_min, self.d_max = normalize_density(estimate_density(data))

    def rescale(self, x):
        x = np.log(x + 1e-16)
        x -= self.d_min
        x /= self.d_max
        return x

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):
        train_x = self.data[idx]
        train_y = self.y[idx]
        return train_x, train_y


def normalize_density(density):
    log_density = np.log(density + 1e-16)
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



