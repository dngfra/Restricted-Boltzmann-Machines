import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.cm as cm
import seaborn as sns
import h5py
import numpy as np
import cmocean
import datetime
import math
from itertools import combinations_with_replacement

def factors(num):
    factors=[]
    for i in range(1,num+1):
        if num%i==0:
           factors.append(i)
    return factors

def best_grid(n_plots):
    for i in combinations_with_replacement(factors(n_plots), 2):
        dist = 100000
        dist_new = abs(i[0] - i[1])
        if dist_new < dist and i[0] * i[1] == n_plots:
            best = i
    return best

def plot_image_grid(images_array, image_shape, n_pictures, cmap=cmocean.cm.balance, save=False, row_titles=None, name='grid'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.pdf'):
    """
    Useful tool to display receptive fields of single hidden neurons or to plot multiple samples
    :param images_array: array, All the pictures as vectors on rows
    :param image_shape: tuple, usually is the square root of the number of columns
    :param n_pictures: int,  The number of pictures that you want to show of images_array
    :param save: bool, True to save, default = False
    :param cmap: string
    :param row_titles: string,
    :param name: 'string' name of the file to save, default grid+datetime
    :param cbar: 'string'
    :return:
    """
    array = np.asarray(images_array)[:n_pictures]
    vmin = np.min(array)
    vmax = np.max(array)

    ncols,nrows = best_grid(n_pictures)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows))


    for idx,ax in enumerate(axes.flat):
        ax.set_axis_off()
        im2 = plt.imshow(np.reshape(array[idx,:],image_shape), cmap=cmap, vmin=vmin, vmax=vmax)
        im = sns.heatmap(np.reshape(array[idx,:],image_shape),ax=ax,vmin=vmin,cmap = cmap, cbar=False, vmax=vmax, square = True)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.03, hspace=0.03)
    cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.95)
    #cbar.set_ticks(np.arange(0, 1.1, 0.5))
    #cbar.set_ticklabels(['low', 'medium', 'high'])

    if save:
        plt.savefig(name, format='pdf')
    plt.show()



def plot_single_image(image_array, image_shape,save=False, name='image'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.pdf'):
    data = np.asarray(image_array).reshape(image_shape)
    plt.imshow(data, cmap='Greys')
    if save:
        plt.savefig(name)
    plt.show()

def plot_input_sample(input_array,sample_array, image_shape,save=False, name='input_sample'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.pdf' ):
    inpt = np.asarray(input_array)
    sample = np.asarray(sample_array)
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].imshow(inpt.reshape(image_shape), cmap='Greys')
    axes[0].set_title("Input")
    axes[1].imshow(sample.reshape(image_shape), cmap='Greys')
    axes[1].set_title("Sample")
    if save:
        plt.savefig(name)
    plt.show()
