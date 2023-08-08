import numpy as np
from clusters import Clusters
from utils import get_star_couples, estimate_density
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt

def estimate_density_2(x, n_neighbors=10):
    # Create an instance of the NearestNeighbors class
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
    # Fit the model to the data
    nbrs.fit(x)
    # Find the approximate nearest neighbors
    distances, _ = nbrs.kneighbors(x)
    # Calculate the average distance
    average_distance = np.mean(distances[:, 1:], axis=1)  # Exclude the first column (self-distance)
    # Compute the density estimate
    d = x.shape[1]  # Dimensionality of the data
    volume = np.prod(np.max(x, axis=0) - np.min(x, axis=0))  # Volume of the space
    density = n_neighbors / (volume * (average_distance ** d))
    return density


clusters = Clusters(features='all', mass='log', path='./data/')
c, name = clusters.next_train(return_name=True)
# x = get_star_couples(c, dims=3)
# y = estimate_density_2(c, n_neighbors=10)
y = estimate_density(c)
print(y[y<1].shape)

#y = y / max(y)
y = np.log(y)
y -= np.min(y)
y /= np.max(y)
plt.hist(y, bins=100)

plt.show()
# np.save(f'./results_SCS/{name[:-4]}.d', np.array(y))