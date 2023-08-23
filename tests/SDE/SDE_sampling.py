import numpy as np
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt
from tqdm import tqdm
import corner
import torch
import gpytorch
from GP_model import ExactGPModel
from SDE_networks import DNN3L
from clusters import Clusters
from utils import sigma, g, DEVICE, generate_color_array, virial_ratio, normalize_density, feature_scaling, estimate_density

clusters = Clusters(features='all', mass="log", rescale=False, path='../../data/')
X, cluster_name_tr = clusters.next_train(return_name=True)
x1 = X.copy()
# normalize data
y, d_min, d_max = normalize_density(estimate_density(X))
X, mean, std = feature_scaling(X, return_mean_std=True)
model_path = '../../results_SDE/SDE15_GP_DNN3L1000u_100it_0.005LR_jun17_t1852'
model = DNN3L()
model = torch.load(f'{model_path}/model.pth')
X = torch.from_numpy(X).float().cuda()
train_y = torch.from_numpy(y).float().cuda()
zero_points = 5000

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
extra = torch.from_numpy(np.random.normal(0, 1, (zero_points, 7))).float().cuda()
train_x = torch.concatenate((X, extra))
extra = torch.from_numpy(np.abs(np.random.normal(0, 0.001, zero_points))).float().cuda()
train_y = torch.concatenate((train_y, extra))
gp_model = ExactGPModel(train_x, train_y, likelihood).cuda()

# Load saved model weights
checkpoint = torch.load('./results/sim_1c_zp_jun17_t1843/model.pth')

# Load weights into model
gp_model.load_state_dict(checkpoint)
gp_model.eval()
likelihood.eval()


# Check if the model is properly configured
# train_x =  torch.from_numpy(np.random.normal(0, 1, (8000, 7))).float().cuda()


with torch.no_grad():
    f_preds = gp_model(train_x)
    y_preds = likelihood(f_preds)
    plt.plot(y_preds.stddev.cpu().numpy(), label='stddev')
    plt.plot(y_preds.sample().cpu().numpy(),'go', markersize=1, label='predicted')
    plt.plot(train_y.cpu(), 'ro', markersize=1, label='simulation', alpha=0.5)
    # plt.yscale('log')
    plt.legend()
    plt.show()

N = 1000
B = 7000

# t = torch.ones(x.shape[0]).to(DEVICE) * 1.25
ts, dt = np.linspace(1, 0, N, retstep=True)
color_array = generate_color_array(start_hue=200 / 360, end_hue=300 / 360, num_colors=N // 100 + 1)
# dt = - 1.25 / N
samples = []
for _ in range(0, 2):
    x = torch.randn([B, 7]) * sigma(1.0)
    x = x.to(DEVICE)
    for i in tqdm(range(N)):
        z = torch.randn_like(x).to(DEVICE)
        t = torch.ones(x.shape[0]) * ts[i]
        t = t.to(DEVICE)
        with torch.no_grad():
            x = x - g(t).view(-1, 1) ** 2 * model(x, t) * dt + g(t).view(-1, 1) * z * (-dt) ** (1 / 2)
        t += dt
        # if i % 100 == 0 or i == N - 1:
        #     y = x.cpu().detach().numpy()
        #     j = i // 100
        #     # distances_y = pdist(np.column_stack((y[:, 1], y[:, 2], y[:, 3])))
        #     # histc1, binsc1 = np.histogram(distances_y, bins=50, density=True)
        #     # logbinsc1 = np.logspace(np.log10(binsc1[0]), np.log10(binsc1[-1]), len(binsc1))
        #     y = (y * std) + mean
        #     m = y[:, 0]
        #     hist, bins = np.histogram(m, bins=50, density=True)
        #     logbinsc1 = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
        #     distances_y = m
        #
        #     if i == 0:
        #         plt.hist(distances_y, bins=logbinsc1, alpha=0.8, density=True, edgecolor=color_array[j], linewidth=1.5,
        #                  histtype='step', label='0 iter')
        #     elif i == N - 1:
        #         plt.hist(distances_y, bins=logbinsc1, alpha=0.8, density=True, edgecolor=color_array[j], linewidth=1.5,
        #                  histtype='step', label=f'{N} iter')
        #     else:
        #         plt.hist(distances_y, bins=logbinsc1, alpha=0.8, density=True, edgecolor=color_array[j], linewidth=1.5,
        #                  histtype='step')


    #
    with torch.no_grad():
        y_preds_1 = likelihood(gp_model(x))
    # # r = torch.from_numpy(np.random.normal(0, 1, (3000, 7))).float().cuda()
    # # y_preds_r = likelihood(gp_model(r))
    # #
    # with torch.no_grad():
    #     plt.plot(y_preds_1.stddev.cpu().numpy(), label='stddev')
    #     plt.plot(y_preds_1.mean.cpu().numpy(),'go', markersize=1, label='SDE')
    #     # plt.plot(y_preds_r.mean.cpu().numpy(), 'ro', markersize=1, label='random')
    #
    #     plt.legend()
    #     plt.show()

    mask = y_preds_1.mean.squeeze(0) > 0.4
    x = x[mask]

    x = x.cpu().detach().numpy()
    x = (x * std) + mean
    x[:, 0] = np.exp(x[:, 0])
    mask = x[:, 0] > 0.3
    x = x[mask]

    print(x.shape)

    samples.append(x)

x = np.concatenate((samples[0], samples[1]))
#x = np.concatenate((samples[0], samples[1], samples[2], samples[3]))

# x = torch.from_numpy(x).float().cuda()
# y_preds = likelihood(gp_model(x))
#
#
# with torch.no_grad():
#     plt.plot(y_preds.stddev.cpu().numpy(), label='stddev')
#     plt.plot(y_preds.mean.cpu().numpy(),'go', markersize=1, label='predicted')
#    # plt.yscale('log')
#     plt.legend()
#     plt.show()


# Set the axis labels and title
# plt.xscale('log')
# plt.yscale('log')
# # plt.xlabel(r'Inter-particle distance, $d$ (au)', fontsize=14)
# plt.xlabel(r'Mass Distribution, $M$ (Solar Masses, $M_\odot$)', fontsize=14)
# # plt.axis(xmin=1e-4)
# plt.ylabel(r'Probability Density Function, $f(d)$', fontsize=14)
# plt.title(f'{i + 1} iterations', fontsize=16)
# plt.legend(loc='lower left')
# plt.savefig(f'{model_path}/ipd_training.png')
# # Show the plot
# plt.show()

# x = x.cpu().detach().numpy()

# Define labels
labels = ['x', 'y', 'z']
figure = corner.corner(x[:, 1:4], color='red', labels=labels, show_titles=True, bins=20)
# corner.corner(X, color='blue', fig=figure, bins=40)
plt.plot([], [], label='simulation', color='blue')
plt.plot([], [], label='SDE sampling', color='red')
plt.legend(loc='upper right')
plt.suptitle(f'SDE Sampling Corner Plot')
plt.savefig(f'{model_path}/corner_xyz.png')
plt.show()

Xr, name = x1, 'm1.e4'
x1[:, 0] = np.exp(x1[:, 0])
Xr2, name2 = clusters.next_train(return_name=True)
Xr2[:, 0] = np.exp(Xr2[:, 0])

np.save(f'{model_path}/x.sample', x)
print('VR = ', virial_ratio(x))

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
plt.hist(distances_1, bins=logbins, alpha=0.3, density=True, label=name[:-4])
plt.hist(distances_2, bins=logbins2, alpha=0.3, density=True, label=name2[:-4])
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
plt.hist(v, bins=logbins, alpha=0.5, density=True, label=name[:-4])
plt.hist(v2, bins=logbins2, alpha=0.5, density=True, label=name2[:-4])
plt.hist(vc1, bins=logbinsc1, alpha=0.8, density=True, edgecolor='blue', linewidth=1.5, histtype='step', label='SDE')
# plt.hist(vc2, bins=logbinsc2, alpha=0.8, density=True, edgecolor='orange', linewidth=1.5, histtype='step', label='MCMC')
# plt.hist(vc3, bins=logbinsc3, alpha=0.8, density=True, edgecolor='darkred', linewidth=1.5, histtype='step', label='Random Walk MC + SIM.')

# Set the axis labels and title
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Velocity, $v$', fontsize=14)
plt.axis(xmin=0.1)
plt.ylabel(r'Distribution, $f(v)$', fontsize=14)
plt.title('Normalized Distribution of Velocities', fontsize=16)
plt.legend(loc='lower left')
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


plt.hist(v, bins=logbins, alpha=0.5, density=True, label=name[:-4])
plt.hist(v2, bins=logbins2, alpha=0.5, density=True, label=name2[:-4])
plt.hist(mc1, bins=logbinsc1, alpha=0.8, density=True, edgecolor='blue', linewidth=1.5, histtype='step', label='SDE')
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
plt.savefig(f'{model_path}/md.png')
plt.show()

print(min(mc1))
