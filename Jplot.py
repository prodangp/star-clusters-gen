import matplotlib.pyplot as plt
from matplotlib import cycler


colors = cycler('color',
                ['#EE6666', '#3388BB', '#9988DD',
                 '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
       axisbelow=True, prop_cycle=colors)

plt.rc('xtick', direction='out', color='black', labelsize=16)
plt.rc('ytick', direction='out', color='black', labelsize=16)
font = {'size': 16}
plt.rc('font', **font)
plt.rc('patch', edgecolor='#E6E6E6')


# Your data
x = [1, 2.5, 5, 7.5, 10, 15]
y1 = [2349, 2313, 2228, 2279, 2170, 2161]
y2 = [1.32, 1.33, 1.71, 1.62, 1.88, 1.93]

fig, ax2 = plt.subplots(figsize=(8,6))

ax1 = ax2.twinx()  # instantiate a second axes that shares the same x-axis

color = '#EE6666'
ax2.set_xlabel('maximum jump size (pc)', position=(0.5, -0.1), fontsize=18)
ax1.set_ylabel('# binaries, upper limit', color=color, fontsize=18)
ax1.plot(x, y1, color=color, linewidth=3)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(1500, 2500)  # Setting limit for y2

color = '#3388BB'
ax2.set_ylabel('virial ratio', color=color, fontsize=18)  # we already handled the x-label with ax1
ax2.plot(x, y2, color=color, linewidth=3)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(1.25, 2.5)  # Setting limit for y1

plt.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()