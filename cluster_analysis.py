import numpy as np

def stats_on_features(x):
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)
