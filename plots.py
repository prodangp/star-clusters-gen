import numpy as np
from matplotlib import pyplot as plt
import corner
from scipy.spatial.distance import pdist
from clusters import Clusters

c_path = 'results_chains/chains_GP_pos_50iter_may31_t1507/gen_cluster_GP_pos_3000stars_may31_t1600.npy'
# Define labels for each variable
labels = ['x', 'y', 'z']
x = np.load(c_path)
cluster = Clusters(features='all', path='./data/', rescale=False)
Xr, name = cluster.next_train(return_name=True)
print(Xr)
figure = corner.corner(x[:, 1:], color='red', labels=labels, show_titles=True, bins=20)
corner.corner(Xr[:, 1:4], fig=figure)
plt.show()





distances_1 = pdist(np.column_stack((Xr[:, 1], Xr[:, 2], Xr[:, 3])))
#distances_2 = pdist(np.column_stack((Xr2[:, 1], Xr2[:, 2], Xr2[:, 3])))
distances_c1 = pdist(np.column_stack((x[:, 1], x[:, 2], x[:, 3])))

# Create the histogram
hist, bins = np.histogram(distances_1, bins=50, density=True)
#hist2, bins2 = np.histogram(distances_2, bins=50, density=True)
histc1, binsc1 = np.histogram(distances_c1, bins=50, density=True)
# histc2, binsc2 = np.histogram(distances_c2, bins=50, density=True)
# histc3, binsc3 = np.histogram(distances_c3, bins=50, density=True)
logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
#logbins2 = np.logspace(np.log10(bins2[0]), np.log10(bins2[-1]), len(bins2))
logbinsc1 = np.logspace(np.log10(binsc1[0]), np.log10(binsc1[-1]), len(binsc1))
# logbinsc2 = np.logspace(np.log10(binsc2[0]), np.log10(binsc2[-1]), len(binsc2))
# logbinsc3 = np.logspace(np.log10(binsc3[0]), np.log10(binsc3[-1]), len(binsc3))
plt.hist(distances_1, bins=logbins, alpha=0.3, density=True, label=name[:-4])
#plt.hist(distances_2, bins=logbins2, alpha=0.3, density=True, label=name2[:-4])
plt.hist(distances_c1, bins=logbinsc1, alpha=0.8, density=True, edgecolor='blue', linewidth=1.5, histtype='step',
         label='SDE')
# plt.hist(distances_c2, bins=logbinsc2, alpha=0.8, density=True, edgecolor='orange', linewidth=1.5, histtype='step', label='MCMC')
# plt.hist(distances_c3, bins=logbinsc3, alpha=0.8, density=True, edgecolor='darkred', linewidth=1.5, histtype='step', label='Random Walk MC + SIM.')
# Set the axis labels and title
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Inter-particle distance, $d$ (au)', fontsize=14)
# plt.axis(xmin=1e-4)
plt.ylabel(r'Distribution, $f(d)$', fontsize=14)
plt.title('Normalized Distribution of Inter-Particle Distances', fontsize=16)
plt.legend(loc='lower left')
#plt.savefig(f'{model_path}/ipd.png')
# Show the plot
plt.show()