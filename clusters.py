import glob
import numpy as np
from matplotlib import pyplot as plt
from utils import feature_scaling


class Clusters:
    def __init__(self, features='all', mass=None, n=5000, path="C:/Users/georg/PycharmProjects/star-clusters-gen/data/",
                 generate=False, mean=None, cov=None):
        self.N = n
        self.features = features
        if features == 'all':
            self.mask = range(0, 7)
        elif features == 'pos':
            self.mask = range(1, 4)
        elif features == 'vel':
            self.mask = range(4, 7)
        elif features == 'phase space':
            self.mask = range(1, 7)
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
                    if mass == 'log':
                        self.data_train[i][:, 0] = np.log(self.data_train[i][:, 0])
                    self.names_train.append(files[i][-9:])
                else:
                    self.data_val[i - self.n_train] = np.loadtxt(files[i], skiprows=1)[:, self.mask]
                    if mass == 'log':
                        self.data_val[i - self.n_train][:, 0] = np.log(self.data_val[i - self.n_train][:, 0])
                    self.names_val.append(files[i][-9:])
            if features == 'all':
                for i in range(N_CLUSTERS):
                    if i < self.n_train:
                        cm_position = np.sum(self.data_train[i][:, 0][:, np.newaxis] * self.data_train[i][:, 1:4],
                                              axis=0) / np.sum(self.data_train[i][:, 0])
                        cm_velocity = np.sum(self.data_train[i][:, 0][:, np.newaxis] *
                                              self.data_train[i][:, 4:], axis=0) / np.sum(self.data_train[i][:, 0])
                        self.data_train[i][:, 1:4] = self.data_train[i][:, 1:4] - cm_position
                        self.data_train[i][:, 4:] = self.data_train[i][:, 4:] - cm_velocity
                    else:
                        cm_position = np.sum(self.data_val[i - self.n_train][:, 0][:, np.newaxis] *
                                              self.data_val[i - self.n_train][:, 1:4], axis=0) / \
                                      np.sum(self.data_val[i - self.n_train][:, 0])
                        cm_velocity = np.sum(self.data_val[i - self.n_train][:, 0][:, np.newaxis] *
                                              self.data_val[i - self.n_train][:, 4:], axis=0) / \
                                      np.sum(self.data_val[i - self.n_train][:, 0])
                        self.data_val[i - self.n_train][:, 1:4] = self.data_val[i - self.n_train][:, 1:4] - cm_position
                        self.data_val[i - self.n_train][:, 4:] = self.data_val[i - self.n_train][:, 4:] - cm_velocity
            elif features == 'phase space':
                for i in range(N_CLUSTERS):
                    if i < self.n_train:
                        cm_position = np.sum(self.data_train[i][:, 0][:, np.newaxis] * self.data_train[i][:, :3],
                                              axis=0) / \
                                      np.sum(self.data_train[i][:, 0])
                        cm_velocity = np.sum(self.data_train[i][:, 0][:, np.newaxis] * self.data_train[i][:, 3:],
                                              axis=0) / \
                                      np.sum(self.data_train[i][:, 0])
                        self.data_train[i][:, :3] = self.data_train[i][:, :3] - cm_position
                        self.data_train[i][:, 3:] = self.data_train[i][:, 3:] - cm_velocity
                    else:
                        cm_position = np.sum(self.data_val[i - self.n_train][:, 0][:, np.newaxis] *
                                              self.data_val[i - self.n_train][:, :3], axis=0) / \
                                      np.sum(self.data_val[i - self.n_train][:, 0])
                        cm_velocity = np.sum(self.data_val[i - self.n_train][:, 0][:, np.newaxis] *
                                              self.data_val[i - self.n_train][:, 3:], axis=0) / \
                                      np.sum(self.data_val[i - self.n_train][:, 0])
                        self.data_val[i - self.n_train][:, :3] = self.data_val[i - self.n_train][:, :3] - cm_position
                        self.data_val[i - self.n_train][:, 3:] = self.data_val[i - self.n_train][:, 3:] - cm_velocity
        # if features == 'all' and rescale:
        #     _, self.means, self.stds = feature_scaling(np.concatenate(self.data_train), return_mean_std=True)
        #     for i in range(N_CLUSTERS):
        #         if i < self.n_train:
        #             self.data_train[i] = (self.data_train[i] - self.means) / self.stds
        #         else:
        #             self.data_val[i - self.n_train] = (self.data_val[i - self.n_train] - self.means) / self.stds

    def get_cluster(self, name):
        data = np.loadtxt(self.path + f'sink_{name}.dat', skiprows=1)[:, self.mask]
        cm_position = np.sum(data[:, 0][:, np.newaxis] * data[:, 1:4], axis=0) / np.sum(data[:, 0])
        cm_velocity = np.sum(data[:, 0][:, np.newaxis] * data[:, 4:], axis=0) / np.sum(data[:, 0][:, np.newaxis])
        data[:, 1:4] = data[:, 1:4] - cm_position
        data[:, 4:] = data[:, 4:] - cm_velocity
        return data

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
