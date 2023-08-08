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


#model_path = "results_chains/chains_GP_pe_ke_5000iter_jul05_t1403/"

##############################
sampling_method = 'EMCMC'
##############################
# EMCMC
model_path = "results_chains/chains_10c_GP_pe_ke_5000iter_aug06_t1616/"
results_folder = "plots_gen_cluster_EMCMC_3000stars_s0.4_r1_Nc1000_minM0.5_aug08_t1331.npy"
x = np.load(model_path + "gen_cluster_EMCMC_3000stars_s0.4_r1_Nc1000_minM0.5_aug08_t1331.npy")
# 7D model
# model_path = "results/7D_10c_aug01_t1649/"
# model_path = "results/7D_10c_aug07_t0753/"
# # APES
# results_folder = "/plots_APES_aug07_t1043_2935_100_student_RW=False.npy"
# x = np.load(model_path + "APES_aug07_t1043_2935_100_student_RW=False.npy")
# # MCMC
# results_folder = "/plots_MCMC_aug05_t0846_1.0_cluster.npy"
# x = np.load(model_path + "MCMC_aug05_t0846_1.0_cluster.npy")
# RWMC
# results_folder = "/plots_rwMCMC_aug05_t0910_1.0_cluster.npy"
# x = np.load(model_path + "rwMCMC_aug05_t0910_1.0_cluster.npy")
# REJECTION
# results_folder = "plots_REJECTION_aug04_t1407_cluster.npy"
# x = np.load(model_path + "REJECTION_aug04_t1407_cluster.npy")
# rw NMMC
# results_folder = "plots_rwMC_aug05_t0952_1.0_cluster.npy"
# x = np.load(model_path + "rwMC_aug05_t0952_1.0_cluster.npy")
# NMMC
# results_folder = "plots_NMMC_aug05_t0957_1.0_cluster.npy"
# x = np.load(model_path + "NMMC_aug05_t0957_1.0_cluster.npy")
clusters = Clusters(features='all')

# x = clusters.get_cluster('m1.e4')
# results_folder = "/plots_m1e4"

cm_position = np.sum(x[:, 0][:, np.newaxis] * x[:, 1:4], axis=0) / np.sum(x[:, 0])
cm_velocity = np.sum(x[:, 0][:, np.newaxis] * x[:, 4:], axis=0) / np.sum(x[:, 0])
x[:, 1:4] = x[:, 1:4] - cm_position
x[:, 4:] = x[:, 4:] - cm_velocity
print(cm_position, cm_velocity)


model_path = model_path + results_folder
os.mkdir(model_path)
# Define labels
labels = ['x', 'y', 'z']
figure = corner.corner(x[:, 1:4], color='red', labels=labels, show_titles=True, bins=20)
# corner.corner(X, color='blue', fig=figure, bins=40)
plt.plot([], [], label='simulation', color='blue')
plt.plot([], [], label=f'{sampling_method} sampling', color='red')
plt.legend(loc='upper right')
plt.suptitle(f'Corner Plot')
plt.savefig(f'{model_path}/corner_xyz.png')
plt.show()


name = 'm1.e4'
name2 = 'm4.e4'
Xr = clusters.get_cluster(name)
Xr2 = clusters.get_cluster(name2)

# print(get_binaries(Xr), get_binaries(Xr2))
# Create the figure and axes objects
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

# Plot the data and the estimate density on the first axis
ax1.scatter(x[:, 1], x[:, 2], c=normalize_density(estimate_density(x, k=50)), edgecolor="k", s=20, cmap="plasma", alpha=0.2)
ax1.set_title(sampling_method)
ax1.set_xlabel('x (pc)')
ax1.set_ylabel('y (pc)')
fig.colorbar(ax1.collections[0], ax=ax1)

# Plot the data and the estimate density on the second axis
ax2.scatter(Xr[:, 1], Xr[:, 2], c=normalize_density(estimate_density(Xr, k=50)), edgecolor="k", s=20, cmap="plasma", alpha=0.2)
ax2.set_title('Simulation')
ax2.set_xlabel('x (pc)')
# ax2.set_ylabel('y (pc)', fontsize=12)
fig.colorbar(ax2.collections[0], ax=ax2)
plt.tight_layout()
plt.savefig(f'{model_path}/2d.png')
# Show the plot
plt.show()

# np.save(f'{model_path}/x.sample', x)'
VR = virial_ratio(x)
MASS = np.sum(x[:, 0])
BINARIES = get_binaries(x)
print('VR = ', VR)
print('Total Mass = ', MASS)
print('Binaries = ', BINARIES)

with open(f'{model_path}/data.txt', 'w') as file:
    file.write('CM SHIFT (r,v): ' + str(cm_position) + ' ' + str(cm_velocity) + '\n')
    file.write('VR = ' + str(VR) + '\n')
    file.write('Total Mass = ' + str(MASS) + '\n')
    file.write('Binaries = ' + str(BINARIES) + '\n')

distances_1 = pdist(np.column_stack((Xr[:, 1], Xr[:, 2], Xr[:, 3])))
distances_2 = pdist(np.column_stack((Xr2[:, 1], Xr2[:, 2], Xr2[:, 3])))
distances_c1 = pdist(np.column_stack((x[:, 1], x[:, 2], x[:, 3])))

# Create the histogram
hist, bins = np.histogram(distances_1, bins=50, density=True)
hist2, bins2 = np.histogram(distances_2, bins=50, density=True)
histc1, binsc1 = np.histogram(distances_c1, bins=50, density=True)
# histc2, binsc2 = np.histogram(distances_c2, bins=50, density=True)
# histc3, binsc3 = np.histogram(distances_c3, bins=50, density=True)
logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
logbins2 = np.logspace(np.log10(bins2[0]), np.log10(bins2[-1]), len(bins2))
logbinsc1 = np.logspace(np.log10(binsc1[0]), np.log10(binsc1[-1]), len(binsc1))
# logbinsc2 = np.logspace(np.log10(binsc2[0]), np.log10(binsc2[-1]), len(binsc2))
# logbinsc3 = np.logspace(np.log10(binsc3[0]), np.log10(binsc3[-1]), len(binsc3))
plt.hist(distances_1, bins=logbins, alpha=0.3, density=True, label=name)
plt.hist(distances_2, bins=logbins2, alpha=0.3, density=True, label=name2)
#plt.hist(distances_c1, bins=logbinsc1, alpha=0.3, density=True, label='m1e4')
plt.hist(distances_c1, bins=logbinsc1, alpha=0.8, density=True, edgecolor='blue', linewidth=1.5, histtype='step',
          label=sampling_method)
# plt.hist(distances_c2, bins=logbinsc2, alpha=0.8, density=True, edgecolor='orange', linewidth=1.5, histtype='step', label='MCMC')
# plt.hist(distances_c3, bins=logbinsc3, alpha=0.8, density=True, edgecolor='darkred', linewidth=1.5, histtype='step', label='Random Walk MC + SIM.')
# Set the axis labels and title
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Inter-particle distance, $d$ (pc)')
# plt.axis(xmin=1e-4)
plt.ylabel(r'Distribution, $f(d)$')
plt.title('Normalized Distribution of Inter-Particle Distances')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig(f'{model_path}/ipd.png')
# Show the plot
plt.show()

v = np.sqrt(Xr[:, 4] ** 2 + Xr[:, 5] ** 2 + Xr[:, 6] ** 2)
v2 = np.sqrt(Xr2[:, 4] ** 2 + Xr2[:, 5] ** 2 + Xr2[:, 6] ** 2)
vc1 = np.sqrt(x[:, 4] ** 2 + x[:, 5] ** 2 + x[:, 6] ** 2)
# vc2 = np.sqrt(c2[:, 4]**2 + c2[:, 5]**2 + c2[:, 6]**2)
# vc3 = np.sqrt(c3[:, 4]**2 + c3[:, 5]**2 + c3[:, 6]**2)
# Create the histogram
hist, bins = np.histogram(v, bins=50, density=True)
hist2, bins2 = np.histogram(v2, bins=50, density=True)
histc1, binsc1 = np.histogram(vc1, bins=50, density=True)
# histc2, binsc2 = np.histogram(vc2, bins=50, density=True)
# histc3, binsc3 = np.histogram(vc3, bins=50, density=True)
logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
logbins2 = np.logspace(np.log10(bins2[0]), np.log10(bins2[-1]), len(bins2))
logbinsc1 = np.logspace(np.log10(binsc1[0]), np.log10(binsc1[-1]), len(binsc1))
# logbinsc2 = np.logspace(np.log10(binsc2[0]), np.log10(binsc2[-1]), len(binsc2))
# logbinsc3 = np.logspace(np.log10(binsc3[0]), np.log10(binsc3[-1]), len(binsc3))
plt.hist(v, bins=logbins, alpha=0.5, density=True, label=name)
plt.hist(v2, bins=logbins2, alpha=0.5, density=True, label=name2)
#plt.hist(vc1, bins=logbinsc1, alpha=0.5, density=True, label='m1.e4')
plt.hist(vc1, bins=logbinsc1, alpha=0.8, density=True, edgecolor='blue', linewidth=1.5, histtype='step', label=sampling_method)
# plt.hist(vc2, bins=logbinsc2, alpha=0.8, density=True, edgecolor='orange', linewidth=1.5, histtype='step', label='MCMC')
# plt.hist(vc3, bins=logbinsc3, alpha=0.8, density=True, edgecolor='darkred', linewidth=1.5, histtype='step', label='Random Walk MC + SIM.')

# Set the axis labels and title
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Velocity, $v$ (km/s)')
plt.axis(xmin=0.1)
plt.ylabel(r'Distribution, $f(v)$')
plt.title('Normalized Distribution of Velocities')
plt.legend(loc='lower left')
plt.tight_layout()
plt.tight_layout()
plt.savefig(f'{model_path}/vd.png')
# Show the plot
plt.show()

m = Xr[:, 0]
m2 = Xr2[:, 0]
mc1 = x[:, 0]

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


plt.hist(m, bins=logbins, alpha=0.5, density=True, label=name)
plt.hist(m2, bins=logbins2, alpha=0.5, density=True, label=name2)
#plt.hist(mc1, bins=logbins2, alpha=0.5, density=True, label='m1.e4')
plt.hist(mc1, bins=logbinsc1, alpha=0.8, density=True, edgecolor='blue', linewidth=1.5, histtype='step', label=sampling_method)
# plt.hist(mc2, bins=logbinsc2, alpha=0.8, density=True, edgecolor='orange', linewidth=1.5, histtype='step', label='MCMC')
# plt.hist(mc3, bins=logbinsc3, alpha=0.8, density=True, edgecolor='darkred', linewidth=1.5, histtype='step', label='Random Walk MC + SIM.')

# Set the axis labels and title
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Mass, $M$  ($M_\odot$) ')
plt.ylabel(r'Distribution, $f(M)$')
plt.title('Normalized Distribution of Masses')
plt.legend()
# Show the plot
plt.tight_layout()
plt.savefig(f'{model_path}/md.png')
plt.show()
