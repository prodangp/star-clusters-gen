import os
import numpy as np
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
from clusters import Clusters


def plot_interparticle_dist(c):
    n = c.shape[0]
    for i in range(200):
        d = np.zeros(n - i - 1)
        for j in range(i + 1, n):
            d_ = np.sqrt(np.sum((c[i, 1:4] - c[j, 1:4]) ** 2)) * 365
            d[j - i - 1] = d_
        d.sort()
        plt.plot(d[:100])
    plt.ylim(0.0, 20)
    plt.show()


def process_cluster(cluster):

    N, _ = cluster.shape
    coordinates = cluster[:, 1:4]
    velocities = np.sqrt(cluster[:, 4] ** 2 + cluster[:, 5] ** 2 + cluster[:, 6] ** 2)
    masses = cluster[:, 0]

    # compute the distance matrix
    dist_matrix = distance_matrix(coordinates, coordinates)

    # to avoid finding a star as its own nearest neighbour
    np.fill_diagonal(dist_matrix, np.inf)

    sorted_cluster = np.zeros_like(cluster)
    mj_dx_dy_dz = np.zeros((N, 4))
    pe_array = np.zeros((N, 1))
    ke1_array = np.zeros((N, 1))
    ke2_array = np.zeros((N, 1))

    # Let's choose the star with min x, y, z as our starting point.
    current_star_idx = np.argmin(np.sum(coordinates, axis=1))

    visited = np.zeros(N, dtype=bool)

    for i in range(N):
        # Mark current star as visited
        visited[current_star_idx] = True

        # Add current star to sorted cluster
        sorted_cluster[i] = cluster[current_star_idx]

        # Compute and store mj, dx, dy, dz for nearest neighbor
        if i < N - 1:  # except for the last star
            # Get nearest unvisited neighbor
            nn_distances = dist_matrix[current_star_idx]
            nn_distances[visited] = np.inf  # Set distances to visited stars as inf
            nn_idx = np.argmin(nn_distances)

            dx, dy, dz = coordinates[nn_idx] - coordinates[current_star_idx]
            mj_dx_dy_dz[i] = [masses[nn_idx], dx, dy, dz]
            dr = nn_distances[nn_idx]
            pe_array[i] = masses[nn_idx] * masses[current_star_idx] / dr
            ke1_array[i] = masses[current_star_idx] * velocities[current_star_idx] ** 2 / 2
            ke2_array[i] = masses[nn_idx] * velocities[nn_idx] ** 2 / 2

            # Update current star to the nearest unvisited neighbor
            current_star_idx = nn_idx

    plt.plot(pe_array, 'bo', markersize=1)
    plt.yscale('log')
    plt.show()

    plt.plot(ke2_array, 'ro', markersize=1)
    plt.yscale('log')
    plt.show()
    plt.hist(pe_array, bins=50)
    plt.yscale('log')
    plt.show()
    return sorted_cluster, mj_dx_dy_dz, pe_array, ke1_array


clusters = Clusters()

save_path = '../../data/chains/'
try:
    os.mkdir(save_path)
except:
    pass

for i, c in enumerate(clusters.data_train):
    cluster, mass_coo, pe, ke = process_cluster(c)
    np.save(save_path + f"cluster_{clusters.names_train[i]}", cluster)
    np.save(save_path + f"mass_coo_{clusters.names_train[i]}", mass_coo)
    np.save(save_path + f"pe_{clusters.names_train[i]}", pe)
    np.save(save_path + f"ke_{clusters.names_train[i]}", ke)

for i, c in enumerate(clusters.data_val):
    cluster, mass_coo, pe, ke = process_cluster(c)
    np.save(save_path + f"cluster_{clusters.names_val[i]}", cluster)
    np.save(save_path + f"mas_coo_{clusters.names_val[i]}", mass_coo)
    np.save(save_path + f"pe_{clusters.names_val[i]}", pe)
    np.save(save_path + f"ke_{clusters.names_val[i]}", ke)

dr = np.sum(mass_coo[:, 1:] ** 2, axis=1) ** (1 / 2)

plt.plot(dr, 'go', markersize=1)
plt.ylim(0, 0.01)
plt.show()
plt.plot(mass_coo[:, 0], 'ro', markersize=1)
plt.show()
plt.hist(cluster[:, 3], bins=50)

plt.show()
# reduce_binaries(x)
