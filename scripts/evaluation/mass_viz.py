import numpy as np
import matplotlib.pyplot as plt
from clusters import Clusters


# data
clusters = Clusters(features='all')
Xr, name = clusters.next_train(return_name=True)
Xr2, name2 = clusters.next_train(return_name=True)
mc1 = np.load('../../results/GP_mass_all_clusters_aug06_t1007/mass_distribution_3000stars_aug08_t1025.npy')
plt.hist(mc1, bins=50, density=True, alpha=0.8)
plt.legend()
plt.show()
m = Xr[:, 0]
m2 = Xr2[:, 0]


# Create the histogram
hist, bins = np.histogram(m, bins=50, density=True)
hist2, bins2 = np.histogram(m2, bins=50, density=True)

histc1, binsc1 = np.histogram(mc1, bins=50, density=True)
# histc2, binsc2 = np.histogram(mc2, bins=50, density=True)
# histc3, binsc3 = np.histogram(mc3, bins=50, density=True)
logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
logbins2 = np.logspace(np.log10(bins2[0]), np.log10(bins2[-1]), len(bins2))
logbinsc1 = np.logspace(np.log10(binsc1[0]), np.log10(binsc1[-1]), len(binsc1))
# logbinsc2 = np.logspace(np.log10(binsc2[0]), np.log10(binsc2[-1]), len(binsc2))
# logbinsc3 = np.logspace(np.log10(binsc3[0]), np.log10(binsc3[-1]), len(binsc3))


plt.hist(m, bins=logbins, alpha=0.5, density=True, label=name[:-4])
plt.hist(m2, bins=logbins2, alpha=0.5, density=True, label=name2[:-4])
plt.hist(mc1, bins=logbinsc1, alpha=0.8, density=True, edgecolor='blue', linewidth=1.5, histtype='step', label='EMCMC')
# plt.hist(mc2, bins=logbinsc2, alpha=0.8, density=True, edgecolor='orange', linewidth=1.5, histtype='step', label='MCMC')
# plt.hist(mc3, bins=logbinsc3, alpha=0.8, density=True, edgecolor='darkred', linewidth=1.5, histtype='step', label='Random Walk MC + SIM.')

# Set the axis labels and title
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Mass, $M$  ($M_\odot$) ', fontsize=14)
plt.ylabel(r'Distribution, $f(M)$', fontsize=14)
plt.title('Normalized Distribution of Masses', fontsize=16)
plt.legend()
# Show the plot
plt.show()