import math
import numpy as np
import colorsys
from tqdm import tqdm
from scipy.interpolate import interpn
from scipy.spatial.distance import pdist, squareform
from matplotlib.colors import Normalize
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import torch
from torch.utils.data import Dataset
# We can get access to the score using automatic differentiation
# Since grad expect a scalar, we use grad and vmap together for a batched gradient
from functorch import grad, vmap

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# First define a temperature schedule. We use some foresight and use a geometric series
sigma_x0 = 0.1 # the original width of our modes
sigma_min = 1e-2 # minimum temperature
sigma_max = 15 # maximum temperature

M_SOLAR = 1.989e30
GRAV_CONST = 6.6743e-11
PC = 3.08567758e16
KM = 1000

def g(t):
    return torch.sqrt(vmap(grad(lambda t: sigma(t)**2))(t))


def sigma(t):
    return sigma_min * (sigma_max / sigma_min)**t


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



def generate_color_array(start_hue, end_hue, num_colors):
    return [
        colorsys.hls_to_rgb(
            start_hue + i * (end_hue - start_hue) / (num_colors - 1),
            0.5,
            1
        )
        for i in range(num_colors)
    ]

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


def estimate_density(x, k=5, verbose=False):
    """Estimate the density of a dataset using the k-nearest neighbors method.

    Parameters
    ----------

    x : array-like, shape (n_samples, n_features)
        The dataset for which to estimate the density.
    k : int, optional (default=5)
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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def normalize_density(density):
    # log_density = np.log(density + 1e-16)
    ct = np.sum(density)
    density /= ct
    # # Shift and rescale density to be in the range [0, 1]
    # density_min = np.min(log_density)
    # density_shifted = log_density - density_min
    # density_max = np.max(density_shifted)
    # density_rescaled = density_shifted / density_max
    return density * density.shape[0]


def inverse_rescaled_density(rescaled_density, density_min, density_max):
    density_shifted = rescaled_density * density_max
    density = density_shifted + density_min
    density = np.exp(density) - 1
    return density


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


def density_scatter(points, fig=None, ax=None, sort=True, bins=20, cmap="magma", norm=None, ticks=None, colorbar=False,
                    **kwargs):
    """
    Scatter plot colored by 2d histogram
    """
    x = points[:, 0]
    y = points[:, 1]
    if ax is None:
        fig, ax = plt.subplots()
    data, x_edges, y_edges = np.histogram2d(x, y, bins=bins, density=True)
    x_bins = 0.5 * (x_edges[1:] + x_edges[:-1])
    y_bins = 0.5 * (y_edges[1:] + y_edges[:-1])
    z = interpn((x_bins, y_bins), data, np.vstack([x, y]).T, method="splinef2d", bounds_error=False)
    z[np.where(np.isnan(z))] = 0.0
    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    if norm is None:
        norm = Normalize(vmin=z.min(), vmax=z.max())
    ax.scatter(x, y, c=z, cmap=cmap, norm=norm, **kwargs)
    if fig is not None:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        box = ax.get_position()
        cax = plt.axes([box.x0 * 1.01 + box.width * 1.05, box.y0, 0.02, box.height])
        fig.colorbar(sm, cax=cax, ticks=ticks)
        cax.set_ylabel('Density')
    return ax


def vector_field(gradient_function, xmin, xmax, ymin, ymax, n=20, dx=0.05, scale=1.5e3, width=0.007, fig=None, ax=None):
    x = np.linspace(xmin, xmax, n)
    y = np.linspace(ymin, ymax, n)
    points = np.stack(np.meshgrid(x, y), axis=-1).reshape((-1, 2))
    g = gradient_function(torch.tensor(points).to(DEVICE).float()).detach().numpy().reshape([n, n, 2])
    colors = np.sqrt(g[..., 0] ** 2 + g[..., 1] ** 2).ravel()
    if ax is None:
        fig, ax = plt.subplots()
    norm = Normalize()
    colormap = cm.magma
    ax.quiver(x, y, g[..., 0], g[..., 1], color=colormap(norm(colors)), scale=scale, width=width)
    if fig is not None:
        sm = cm.ScalarMappable(norm=norm, cmap=colormap)
        sm.set_array([])
        box = ax.get_position()
        cax = plt.axes([box.x0 * 1.01 + box.width * 1.05, box.y0, 0.02, box.height])
        fig.colorbar(sm, cax=cax)
        cax.set_ylabel(r"$|| \nabla_{\mathbf{x}} \log p(\mathbf{x})||$")
    return ax


def density_contours(density_function, xmin, xmax, ymin, ymax, confidence_intervals, dx=0.01, dy=0.01, fig=None,
                     ax=None):
    x = np.arange(xmin, xmax, dx)
    y = np.arange(ymin, ymax, dy)
    n = x.size
    m = y.size
    points = np.stack(np.meshgrid(x, y), axis=-1).reshape((-1, 2))
    p = density_function(torch.tensor(points).to(DEVICE)).detach().numpy().reshape([m, n])
    if ax is None:
        fig, ax = plt.subplots()

    cumul = np.sort(p.ravel() * dx * dy)[::-1].cumsum()
    ps = []
    len_ci = len(confidence_intervals)
    for ci in confidence_intervals:
        p_at_ci = np.sort(p.ravel())[::-1][np.argmin((cumul - ci) ** 2)]
        ps.append(p_at_ci)
    cs = ax.contour(x, y, p, levels=ps[::-1], colors=[plt.cm.cool(i / (len_ci - 1)) for i in range(len_ci)],
                    linewidths=2, linestyles="--")

    def fmt(x):
        ci = cumul[np.argmin((x - np.sort(p.ravel())[::-1]) ** 2)]
        s = f"{ci * 100:.1f}"
        if s.endswith("0"):
            s = f"{ci * 100:.0f}"
        return rf"{s} \%" if plt.rcParams["text.usetex"] else f"{s} %"

    ax.clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=10)

    return ax


