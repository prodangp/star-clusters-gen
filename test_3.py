import numpy as np
import scipy.stats as st


def sample_walkers():
    """
    Generate e initial sample with L points.

    Returns
    -------
    samples: array
        An array of shape (L, dim) containing the generated samples.
    """
    # initialize the samples array
    samples = np.zeros((320, 3))
    cov_p = 0.5
    cov_v = 0.5
    cov_m = 1.0
    cov = [cov_p] * 3 + [cov_v] * 3 + [cov_m]
    bounds = [[-10, 10], [-10, 10], [-10, 10]]
    # Generate samples from the truncated normal distribution in each dimension
    for d in range(3):
        a, b = bounds[d]
        mu = 0
        sigma = cov[d]
        samples[:, d] = st.truncnorm((a - mu) / sigma, (b - mu) / sigma, mu, sigma).rvs(size=320)
    return samples


print(sample_walkers())