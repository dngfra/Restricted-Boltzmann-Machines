
import sys
#sys.path.append("/Users/fdangelo/PycharmProjects/myRBM")
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import Binarizer
from my_RBM_tf2 import RBM
import deepdish as dd
from bokeh.plotting import figure
from bokeh.io import export_svgs

machine = RBM(32*32, 600, 100, (32, 32), 32,'cd')
machine.from_saved_model('/cluster/home/fdangelo/Restricted-Boltzmann-Machines/logs/scalars/1604-004823')
datah5 = dd.io.load('/Users/fdangelo/PycharmProjects/myRBM/data/ising/ising_data_complete.hdf5')
data_bin ={}
datah5_norm = {}
#Take spin up as standard configuration
keys = list(datah5.keys())
binarizer = Binarizer(threshold=0)
for key in keys:
    datah5_norm[key] = np.array([np.where(np.sum(slice)<0,-slice,slice) for slice in datah5[key]])
    data_bin[key] = np.array([binarizer.fit_transform(slice) for slice in datah5_norm[key]]).reshape(datah5_norm[key].shape[0],-1).astype(np.float32)

magn_key_mean = []
for key in keys:
    print(data_bin[key].shape[0])
    magn_23 = np.array([np.mean(data_bin[key][i]) for i in range(data_bin[key].shape[0])])
    magn_key_mean.append(np.mean(magn_23))

steps =[200,1000,10000,100000,1000000]
for i in steps:

    fantasy_particle2,_,_,list_state1 = machine.sample(data_bin[keys[1]][12],n_step_MC=i)
    list_magn1 = [np.mean(np.array(i)) for i in list_state1]
    fantasy_particle2,_,_,list_state4 = machine.sample(data_bin[keys[4]][12],n_step_MC=i)
    list_magn4 = [np.mean(np.array(i)) for i in list_state4]
    fantasy_particle2,_,_,list_state5 = machine.sample(data_bin[keys[5]][12],n_step_MC=i)
    list_magn5 = [np.mean(np.array(i)) for i in list_state5]

    # create a new plot with default tools, using figure
    from bokeh.models import ColumnDataSource, Whisker
    p2 = figure(title = 'Magnetization of visible states of steps in the same markov chain',plot_width=1000, plot_height=600, y_range=(0.0, 1.01))
    x = np.arange(len(list_magn1))
    # add a circle renderer with x and y coordinates, size, color, and alpha
    p2.circle(x,list_magn1, size=7, line_color="navy", fill_color="green", line_width=0.1,  fill_alpha=0.5, legend = 'Magnetization samples T=2.186995' )
    p2.circle(x,list_magn4, size=7, line_color="navy", fill_color="yellow", line_width=0.1 , fill_alpha=0.5, legend = 'Magnetization samples T=2.269184')
    p2.circle(x,list_magn5, size=7, line_color="navy", fill_color="red", line_width=0.1,  fill_alpha=0.5, legend = 'Magnetization samples T=3.000000')

    #p2.circle(np.arange(len(magn_train)),magn_train, size=7, line_color="navy", fill_color="yellow", fill_alpha=0.5, legend = 'train')
    p2.line(x, magn_key_mean[1],line_color="green", line_width=3, line_alpha=1, legend='Magnetization data T=2.186995' )
    p2.line(x, magn_key_mean[4],line_color="yellow", line_width=3, line_alpha=1, legend='Magnetization data T=2.269184' )
    p2.line(x, magn_key_mean[5],line_color="red", line_width=3, line_alpha=1, legend='Magnetization data T=3.000000' )
    p2.yaxis.axis_label = "Magnetization"
    p2.xaxis.axis_label = "Markov chain steps"
    p2.legend.location = "bottom_right"
    p2.legend.click_policy="hide"
    p2.output_backend = "svg"
    export_svgs(p2, filename=str(len(list_magn1))+"magn_step.svg")


    E_1 = []
    E_4 = []
    E_5 = []
    for a in list_state1:
        E_1.append(machine.energy(tf.reshape(a,(1024,)))[0])
    for a in list_state4:
        E_4.append(machine.energy(tf.reshape(a,(1024,)))[0])
    for a in list_state5:
        E_5.append(machine.energy(tf.reshape(a,(1024,)))[0])


    p3 = figure(plot_width=800, plot_height=400)
    # add a circle renderer with x and y coordinates, size, color, and alpha
    p3.line(np.arange(len(E_1)), E_1,line_color="green", line_width=2, line_alpha=0.6, legend='Energy samples T=2.186995' )
    p3.line(np.arange(len(E_4)), E_4,line_color="yellow", line_width=2, line_alpha=0.6, legend='Energy samples T=2.269184' )
    p3.line(np.arange(len(E_5)), E_5,line_color="red", line_width=2, line_alpha=0.6, legend='Energy samples T=3.000000' )

    p3.yaxis.axis_label = "Energy"
    p3.xaxis.axis_label = "Mc step"
    p3.legend.location = "bottom_right"
    p3.output_backend = "svg"
    export_svgs(p3, filename=str(len(E_1))+"ener_step.svg")

    p4 = figure(plot_width=800, plot_height=400)
    # add a circle renderer with x and y coordinates, size, color, and alpha
    p4.circle(E_1,list_magn1, size=7, line_color="navy", fill_color="green", line_width=0.1,  fill_alpha=0.5, legend = 'Samples T=2.186995')
    p4.circle(E_4,list_magn4, size=7, line_color="navy", fill_color="yellow", line_width=0.1 , fill_alpha=0.5, legend = 'Samples T=2.269184')
    p4.circle(E_5,list_magn5, size=7, line_color="navy", fill_color="red", line_width=0.1,  fill_alpha=0.5, legend = 'Samples T=3.000000')


    p4.yaxis.axis_label = "Magnetization"
    p4.xaxis.axis_label = "Energy"

    p4.legend.location = "bottom_left"
    p4.output_backend = "svg"
    export_svgs(p4, filename=str(len(list_magn1))+"magn_energy.svg")
