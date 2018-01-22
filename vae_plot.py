import matplotlib.pyplot as plt
import numpy as np

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def save_scattered_image(z, id, name=None, N=10, xlim=None, ylim=None):
    print(z.shape)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(z[:, 0], z[:, 1], c=id, marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
    plt.colorbar(ticks=range(N))
    axes = plt.gca()
    if xlim:
        axes.set_xlim(xlim)
    if ylim:
        axes.set_ylim(ylim)
    plt.grid(True)
    if name:
        plt.savefig(name)
        print("save fig as:", name)
    else:
        plt.show()