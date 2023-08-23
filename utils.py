import math
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
import torch
from torch.utils.data import Dataset

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

M_SOLAR = 1.989e30
GRAV_CONST = 6.6743e-11
PC = 3.08567758e16
KM = 1000


def virial_ratio(c):
    M = c[:, 0]

    distances = squareform(pdist(np.column_stack((c[:, 1], c[:, 2], c[:, 3]))))
    np.fill_diagonal(distances, np.inf)
    potential = np.sum(M[None, :] * M[:, None] * (M_SOLAR ** 2) * GRAV_CONST / (distances * PC)) / 2

    # Compute the kinetic energy
    velocities = np.column_stack((c[:, 4], c[:, 5], c[:, 6]))
    speeds = np.linalg.norm(velocities, axis=1)
    kinetic = 0.5 * np.sum(M * M_SOLAR * (speeds * KM) ** 2)

    # Compute the virial ratio
    virial_ratio = 2 * kinetic / potential
    return virial_ratio


def get_star_couples(c, dims=3):
    n = c.shape[0]  # Number of stars
    num_couples = int(n * (n - 1) / 2)  # Number of star couples

    couples = np.zeros((num_couples, dims * 2))  # Array to store couples

    index = 0
    for i in range(n):
        for j in range(i + 1, n):
            # Extract coordinates of star i and star j
            if dims == 3:
                xi, yi, zi = c[i, 1:4]
                xj, yj, zj = c[j, 1:4]
                couples[index] = [xi, yi, zi, xj, yj, zj]
            elif dims == 2:
                xi, yi = c[i, 1:3]
                xj, yj = c[j, 1:3]
                couples[index] = [xi, yi, xj, yj]
            index += 1

    return couples


def get_binaries(c):
    n = c.shape[0]  # Number of stars
    # num_couples = int(n * (n - 1) / 2)  # Number of star couples
    binary_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(c[i, 1:4] - c[j, 1:4])
            speed_cm = (c[i, 0] * c[i, 4:] + c[j, 0] * c[j, 4:]) / (c[i, 0] + c[j, 0])
            speed_i = np.linalg.norm(c[i, 4:] - speed_cm)
            speed_j = np.linalg.norm(c[j, 4:] - speed_cm)
            potential = c[i, 0] * c[j, 0] * (M_SOLAR ** 2) * GRAV_CONST / (distance * PC)
            kinetic = 0.5 * M_SOLAR * (c[i, 0] * speed_i ** 2 + c[j, 0] * speed_j ** 2) * KM ** 2
            total_energy = kinetic - potential
            if total_energy < 0:
                binary_count += 1
                break
    return binary_count


def estimate_density(x, k=50, verbose=False):
    """Estimate the density of a dataset using the k-nearest neighbors method.

    Parameters
    ----------

    x : array-like, shape (n_samples, n_features)
        The dataset for which to estimate the density.
    k : int, optional (default=50)
        The number of nearest neighbors to use for density estimation.
    verbose: boolen, optional
             Show tqdm bar if true
    Returns
    -------
    densities : array, shape (n_samples,)
        The estimated densities for each sample in the dataset.
    """
    dens = np.zeros(len(x))

    # The volume of a unit d-dimensional ball is given by the formula
    d = len(x[0])
    volume = math.pi ** (d / 2) / math.gamma(d / 2 + 1)

    if verbose:
        for i in tqdm(range(len(x))):
            r = np.sort(
                np.sqrt(
                    np.sqrt(np.sum((x - x[i]) ** 2, axis=1))
                )
            )[k]

            dens[i] = k / (volume * (r ** d))
    else:
        for i in range(len(x)):
            r = np.sort(
                np.sqrt(np.sum((x - x[i]) ** 2, axis=1))
            )[k]

            dens[i] = k / (volume * (r ** d))

    return dens


class Data(Dataset):
    def __init__(self, data, y=None):

        self.data, self.means, self.stds = feature_scaling(data, return_mean_std=True)
        # normalize data
        if y is None:
            self.y, self.d_min, self.d_max = normalize_density(estimate_density(data))
        else:
            self.y, self.d_min, self.d_max = normalize_density(y)

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


def gp_pdf(proposed, model_, likelihood_, mean=False, cuda=True):
    with torch.no_grad():
        if cuda:
            proposed = torch.from_numpy(proposed).float().unsqueeze(0).cuda()
        else:
            proposed = torch.from_numpy(proposed).float().unsqueeze(0).cpu()
        y_preds = likelihood_(model_(proposed))
        if mean:
            return y_preds.mean[0].cpu().numpy()
        else:
            return y_preds.sample()[0].cpu().numpy()


def normalize_density(density):
    ct = np.sum(density)
    density /= ct
    return density * density.shape[0]


def feature_scaling(inputs, method="standardization", return_mean_std=False):
    if method == "standardization":
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
    elif method == "MinMax":
        return (inputs - np.min(inputs)) / (np.max(inputs) - np.min(inputs))


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


