import torch
import gpytorch
from utils import Clusters, estimate_density, feature_scaling, reverse_scaling
from GP_model import ExactGPModel
from matplotlib import pyplot as plt

# data
clusters = Clusters(features='pos')
# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
X, cluster_name_tr = clusters.next_train(return_name=True)
print(X)
X, mean, std = feature_scaling(estimate_density(X), return_mean_std=True)

X = reverse_scaling(X, means=mean, stds=std)