import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.cm as cm
import seaborn as sns
import h5py
import numpy as np

def plot_image_grid(images_array, image_shape, cmap=plt.cm.jet, row_titles=None, name='my_fig.pdf', cbar=True):
    array = np.asarray(images_array)
    vmin = np.min(images_array)
    vmax = np.max(images_array)
    nrows, ncols = array.shape[:-1]
    f = plt.figure(figsize=(2 * ncols, 2 * nrows))
    grid = gs.GridSpec(nrows, ncols)
    axes = [[plt.subplot(grid[i, j]) for j in range(ncols)] for i in range(nrows)]
    for i in range(nrows):
        for j in range(ncols):
            sns.heatmap(np.reshape(array[i][j], image_shape),
                        ax=axes[i][j], cmap=cmap, cbar=cbar, vmin=vmin, vmax=vmax, square=True)
            axes[i][j].set(yticks=[])
            axes[i][j].set(xticks=[])

    if row_titles is not None:
        for i in range(nrows):
            axes[i][0].set_ylabel(row_titles[i], fontsize=36)

    plt.tight_layout()
    f.savefig(name)
    plt.show(f)
    plt.close(f)