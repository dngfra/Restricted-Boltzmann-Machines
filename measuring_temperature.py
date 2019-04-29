import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np
from sklearn.preprocessing import Binarizer
from sklearn.model_selection import train_test_split
import deepdish as dd
import itertools
import os
import datetime

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(32, 32)),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(17, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def create_conv_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(32, 32, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(17, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


#preprocessing the data
datah5 = dd.io.load('/Users/fdangelo/PycharmProjects/myRBM/data/ising/ising_data_L32.hdf5')

binarizer = Binarizer(threshold=0)
keys = list(datah5.keys())
for key in keys:
    datah5[key] = np.array([np.where(np.sum(slice)<0,-slice,slice) for slice in datah5[key]])
    datah5[key] = np.array([binarizer.fit_transform(slice) for slice in datah5[key]])


#class_names = ['T=1.000000', 'T=2.186995', 'T=2.261435', 'T=2.268900', 'T=2.269184', 'T=3.000000']
#class_names = ['T=1.000000','T=2.269184', 'T=3.000000'] #accuracy 0.99
#class_names = ['T=1.000000','T=2.186995','T=2.269184', 'T=3.000000'] #accuracy 0.85

class_names = [keys[i] for i in [4,6,7,8,9,10,11,12,16]]
#the correspondent label for the class will be the index in the previous list so label 0 ==> T=1.0000000

#We have to concatenate all the data and also create all the correspondent labels

data = datah5[class_names[0]]
for temperature in class_names[1:]:
    data = np.concatenate([data,datah5[temperature]])

#to create the correspondent label we just need a list [0,0,...,0,1,...,1,...]
class_labels = np.asarray(list(itertools.chain.from_iterable(itertools.repeat(x, datah5[keys[0]].shape[0]) for x in range(0,len(class_names)))))

#Split the dataset into test and train
ising_train, ising_test, temp_train, temp_test = train_test_split(data, class_labels, test_size=0.2, random_state=42)


current_time = datetime.datetime.now().strftime("%d%m-%H%M%S")
log_dir = 'measuring_temp_logs/scalars/' + current_time + '/train'
checkpoint_path = "measuring_temp_logs/checkpoints/cp" + current_time +".ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
l_dir = os.path.dirname(log_dir)

# Create checkpoint callback to save our trained model every epoch
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_weights_only=True,verbose=1, period=5)
tf_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir,histogram_freq = 50)
#Now we can create our neural network and print out a summary of the architecture
neural_network = create_conv_model()

filename = 'measuring_temp_logs/model' + current_time
# Open the file
with open(filename + 'report.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    neural_network.summary(print_fn=lambda x: fh.write(x + '\n'))
conv = True
#reshape for conv nets
if conv == True:
    ising_train = ising_train.reshape((ising_train.shape[0], 32, 32, 1))
    ising_test = ising_test.reshape((ising_test.shape[0], 32, 32, 1))

#Start training the model by testing every epoch the accuracy of the class prediction on the test set

neural_network.fit(ising_train, temp_train , epochs=100,batch_size=64,validation_data = (ising_test,temp_test),callbacks = [cp_callback,tf_callback ])

