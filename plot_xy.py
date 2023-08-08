import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist
import corner
from matplotlib import cycler
from utils import estimate_density, normalize_density, virial_ratio, get_binaries
from clusters import Clusters
import os

plt.rcParams["font.size"] = 16

colors = cycler('color',
                ['#EE6666', '#3388BB', '#9988DD',
                 '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
       axisbelow=True, grid=True, prop_cycle=colors)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('xtick', direction='out', color='black', labelsize=16)
plt.rc('ytick', direction='out', color='black', labelsize=16)
font = {'size': 16}
plt.rc('font', **font)
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)

clusters = Clusters(features='all')
sim1 = clusters.get_cluster('m1.e4')
sim2 = clusters.get_cluster('m4.e4')
sim3 = clusters.get_cluster('m9.e4')

emcmc1 = np.load("results_chains/chains_10c_GP_pe_ke_5000iter_aug06_t1616/gen_cluster_EMCMC_3000stars_s0.2_r1.5_aug08_t0931.npy")
emcmc2 = np.load("results_chains/chains_10c_GP_pe_ke_5000iter_aug06_t1616/gen_cluster_EMCMC_3000stars_s0.2_r1.5_Nc1000_minM0.5_aug08_t1224.npy")
emcmc3 = np.load("results_chains/chains_10c_GP_pe_ke_5000iter_aug06_t1616/gen_cluster_EMCMC_3000stars_s0.4_r1_Nc1000_minM0.5_aug08_t1331.npy")
nmmc = np.load("results/7D_10c_aug01_t1649_R/NMMC_aug05_t0957_1.0_cluster.npy")
apes = np.load("results/7D_10c_aug07_t0753/APES_aug07_t1022_2874_80_student_RW=False.npy")
mcmc = np.load("results/7D_10c_aug01_t1649_R/MCMC_aug04_t2148_0.25_cluster.npy")

for x in [emcmc1, emcmc2, emcmc3, nmmc, apes, mcmc]:
    cm_position = np.sum(x[:, 0][:, np.newaxis] * x[:, 1:4], axis=0) / np.sum(x[:, 0])
    cm_velocity = np.sum(x[:, 0][:, np.newaxis] * x[:, 4:], axis=0) / np.sum(x[:, 0])
    x[:, 1:4] = x[:, 1:4] - cm_position
    x[:, 4:] = x[:, 4:] - cm_velocity


# Generate random data for the sake of demonstration
data = [sim1, sim2, sim3, nmmc, apes, mcmc, emcmc1, emcmc2, emcmc3]
sampling_methods = ['m1.e4', 'm4.e4', 'm9.e4', 'NMMC s=1.0', 'APES (L=80, Student kernel)', 'MCMC (s=0.25)',
                    '$J=1.5$, $M=7.2M_{10^3\odot}$', '$J=1$, $M=13.5M_{10^3\odot}$', '$J=1$, $M=32.2M_{10^3\odot}$']
fig, axes = plt.subplots(3, 3, figsize=(10, 10))  # 3x3 grid of plots

for i, ax in enumerate(axes.ravel()):
    ax.scatter(data[i][:, 1], data[i][:, 2], c=normalize_density(estimate_density(data[i][:, 1:3], k=50)), edgecolor="k", s=20,
               cmap="plasma", alpha=0.2)
    ax.set_title(sampling_methods[i], y=1.05)
    ax.set_xlabel('x (pc)')
    ax.set_ylabel('y (pc)')

    if i == 6:
        ax.set_xlim((-2.5, 2.5))
        ax.set_ylim((-2.5, 2.5))
    else:
        ax.set_xlim((-5, 5))
        ax.set_ylim((-5, 5))

    fig.colorbar(ax.collections[0], ax=ax)

plt.tight_layout()  # Adjust spacing between plots for clarity
plt.savefig(f'./xy_grid.png')
plt.show()

