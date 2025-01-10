# This program accomplishes the training of the EEG2Code model and subsequently predicts sequences from the test data.
# In the runtime configuration, input the file to be processed, such as 'hljvmc.mat'.

from __future__ import division
import keras
import time
import os
import gc
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io as sio
import tensorflow as tf
from keras import optimizers, Input
from keras import initializers
from keras.backend import permute_dimensions
from keras.constraints import max_norm
from keras.models import load_model, Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Permute, Flatten, Dense, BatchNormalization, Activation, Dropout, \
    AveragePooling2D, DepthwiseConv2D, Add, concatenate, Reshape
from keras.utils import np_utils

import keras
import time
import os
import gc
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io as sio
import tensorflow as tf
from keras import optimizers, Input
from keras import initializers
from keras.backend import permute_dimensions
from keras.constraints import max_norm
from keras.models import load_model, Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Permute, Flatten, Dense, BatchNormalization, Activation, Dropout, \
    AveragePooling2D, DepthwiseConv2D, Add, concatenate, Reshape
from keras.utils import np_utils
from tensorflow.python.keras.layers import SpatialDropout2D, Lambda, SeparableConv2D


def construct_model(windowSize, numberChannels):
    """
    Constructs the EEG2Code CNN model.

    Args:
        windowSize (int): The size of the time window for EEG data.
        numberChannels (int): The number of EEG channels.

    Returns:
        model (keras.Model): The constructed CNN model.
    """
    dropoutrate = 0.85
    input_layer = Input(shape=(windowSize, numberChannels, 1))
    unit1 = Conv2D(filters=21,
                   kernel_size=(160, 1),
                   data_format='channels_last',
                   strides=(2, 1),
                   padding='same')(input_layer)
    unit1 = BatchNormalization()(unit1)
    unit1 = Activation('relu')(unit1)
    unit1 = DepthwiseConv2D((1, 21),
                            use_bias=False,
                            depth_multiplier=1,
                            data_format='channels_last',
                            depthwise_constraint=max_norm(1.))(unit1)
    unit1 = BatchNormalization()(unit1)
    unit1 = Activation('relu')(unit1)
    unit1 = Dropout(dropoutrate)(unit1)

    unit2 = Conv2D(filters=21,
                   kernel_size=(80, 1),
                   strides=(2, 1),
                   data_format='channels_last',
                   padding='same')(input_layer)
    unit2 = BatchNormalization()(unit2)
    unit2 = Activation('relu')(unit2)
    unit2 = DepthwiseConv2D((1, 21),
                            use_bias=False,
                            depth_multiplier=1,
                            data_format='channels_last',
                            depthwise_constraint=max_norm(1.))(unit2)
    unit2 = BatchNormalization()(unit2)
    unit2 = Activation('relu')(unit2)
    unit2 = Dropout(dropoutrate)(unit2)

    unit3 = Conv2D(filters=21,
                   kernel_size=(40, 1),
                   strides=(2, 1),
                   padding='same')(input_layer)
    unit3 = BatchNormalization()(unit3)
    unit3 = Activation('relu')(unit3)
    unit3 = DepthwiseConv2D((1, 21),
                            use_bias=False,
                            depth_multiplier=1,
                            depthwise_constraint=max_norm(1.))(unit3)
    unit3 = BatchNormalization()(unit3)
    unit3 = Activation('relu')(unit3)
    unit3 = Dropout(dropoutrate)(unit3)
    input_layer2 = AveragePooling2D((2, 1))(input_layer)
    input_layer2=permute_dimensions(input_layer2,(0,1,3,2))
    l1unit = concatenate([input_layer2, unit1, unit2, unit3], axis=-1)
    l1out = AveragePooling2D((2, 1))(l1unit)

    # 3
    unit1 = Conv2D(filters=21,
                   kernel_size=(40, 1),
                   strides=(2, 1),
                   padding='same')(l1out)
    unit1 = BatchNormalization()(unit1)
    unit1 = Activation('relu')(unit1)
    unit1 = Dropout(dropoutrate)(unit1)

    unit2 = Conv2D(filters=21,
                   kernel_size=(20, 1),
                   strides=(2, 1),
                   padding='same')(l1out)
    unit2 = BatchNormalization()(unit2)
    unit2 = Activation('relu')(unit2)
    unit2 = Dropout(dropoutrate)(unit2)

    unit3 = Conv2D(filters=21,
                   kernel_size=(10, 1),
                   strides=(2, 1),
                   padding='same')(l1out)
    unit3 = BatchNormalization()(unit3)
    unit3 = Activation('relu')(unit3)
    unit3 = Dropout(dropoutrate)(unit3)

    l1out = AveragePooling2D((2, 1))(l1out)
    l2unit = concatenate([l1out, unit1, unit2, unit3])
    l2out = AveragePooling2D((2, 1))(l2unit)

    # 4
    l3unit = Conv2D(filters=147,
                    kernel_size=(10, 1),
                    strides=(5, 1),
                    padding='same')(l2out)
    l3unit = BatchNormalization()(l3unit)
    l3out = Activation('relu')(l3unit)
    l3out = AveragePooling2D((2, 1))(l3out)
    l3out = Dropout(dropoutrate)(l3out)
    # 5
    l5unit = Flatten()(l3out)
    l5out = Dense(147, activation='relu')(l5unit)
    l5out = Dropout(dropoutrate)(l5out)
    # layer6
    output_layer = Dense(2, activation='softmax')(l5out)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def load_matfile(filename, windowSize, datax, datay):
    """
    Loads and preprocesses the MATLAB file containing EEG data.

    Args:
        filename (str): The path to the MATLAB file.
        windowSize (int): The size of the time window for EEG data.
        datax (str): The key for training data in the MATLAB file.
        datay (str): The key for training labels in the MATLAB file.

    Returns:
        data_x_train (np.array): Preprocessed training data.
        data_y_train (np.array): Preprocessed training labels.
        data_x_val (np.array): Preprocessed validation data.
        data_y_val (np.array): Preprocessed validation labels.
        data_x_test (np.array): Preprocessed test data.
    """
    mat_contents = sio.loadmat(filename)
    train_data_x = np.array(mat_contents[datax])
    train_data_y = np.array(mat_contents[datay])
    test_data_x = np.array(mat_contents['test_data_x'])

    channels = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21]) - 1
    train_data_x = train_data_x[:, channels, :]
    test_data_x = test_data_x[:, channels, :]

    # Split train data into 250ms (150 sample) windows
    data_x_train = []
    data_y_train = []
    for ii in range(train_data_x.shape[0]):
        trialdata = train_data_x[ii, :, :].squeeze().transpose()
        data_x_windows = np.zeros((train_data_x.shape[2], windowSize, train_data_x.shape[1]))
        for t in range(windowSize):
            data_x_windows[:, t, :] = np.roll(trialdata, -t, axis=0)
        data_x_windows = data_x_windows[0:(data_x_windows.shape[0] - windowSize), :, :]

        bitdata = train_data_y[ii, :].squeeze().transpose()
        bitdata = np_utils.to_categorical(bitdata, 2)
        for t in range(data_x_windows.shape[0]):
            data_x_train.append(data_x_windows[t, :, :].squeeze())
            data_y_train.append(bitdata[t])
    data_x_train = np.array(data_x_train);
    data_y_train = np.array(data_y_train);

    # Split train data into train and validation sets (equal size)
    data_x_train = data_x_train.reshape(-1, data_x_train.shape[1], data_x_train.shape[2], 1)
    data_y_train = data_y_train.reshape(-1, 2)
    if (len(data_x_train) % 2) != 0:
        data_x_train = data_x_train[:-1]
        data_y_train = data_y_train[:-1]
    x_split = np.array_split(data_x_train, 2)
    y_split = np.array_split(data_y_train, 2)
    data_x_train = x_split[0]
    data_y_train = y_split[0]
    data_x_val = x_split[1]
    data_y_val = y_split[1]

    # Split test data into 250ms (150 sample) windows
    data_x_test = np.zeros((test_data_x.shape[0], test_data_x.shape[2] - windowSize, windowSize, test_data_x.shape[1]))
    for ii in range(test_data_x.shape[0]):
        trialdata = test_data_x[ii, :, :].squeeze().transpose()
        data_x_windows = np.zeros((test_data_x.shape[2], windowSize, test_data_x.shape[1]))
        for t in range(windowSize):
            data_x_windows[:, t, :] = np.roll(trialdata, -t, axis=0)
        data_x_windows = data_x_windows[0:(data_x_windows.shape[0] - windowSize), :, :]
        data_x_test[ii, :, :, :] = data_x_windows.reshape(-1, data_x_windows.shape[0], data_x_windows.shape[1],
                                                          data_x_windows.shape[2])

    return data_x_train, data_y_train, data_x_val, data_y_val, data_x_test


def downsample(arr, n):
    """
    Downsamples an array by averaging over every n elements.

    Args:
        arr (np.array): The input array.
        n (int): The downsampling factor.

    Returns:
        np.array: The downsampled array.
    """
    end = n * int(len(arr) / n)
    return np.mean(arr[:end].reshape(-1, n), 1)


# Main execution
num_args = len(sys.argv) - 1
for i in range(num_args):
    MATLAB_FILE = sys.argv[i+1]
    MODEL_FILE = MATLAB_FILE[:-4] + '.hdf5'

    ## PARAMETERS
    WINDOW_SIZE = 160  # Equals 250ms at 600Hz sampling rate
    lr = 0.0005  # Learning rate
    batchsize = 256  # Batch size
    epochs = 10  # Number of epochs

    ## LOAD DATA
    DATA_X = 'train_datax3012'
    DATA_Y = 'train_data_y3012'
    (data_x_train, data_y_train, data_x_val, data_y_val, data_x_test) = load_matfile(MATLAB_FILE, WINDOW_SIZE, DATA_X, DATA_Y)

    ## CREATE AND TRAIN EEG2Code CNN MODEL
    model = construct_model(data_x_train.shape[1], data_x_train.shape[2])
    adam = tf.keras.optimizers.Adam(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    history = model.fit(data_x_train, data_y_train, batch_size=batchsize, epochs=epochs,
                        validation_data=(data_x_val, data_y_val), callbacks=[
            keras.callbacks.ModelCheckpoint(MODEL_FILE, monitor='val_loss', verbose=0, save_best_only=True,
                                            save_weights_only=False, mode='auto', period=1)])
    tf.keras.backend.clear_session()

    ## PREDICT ON TEST DATA
    preddata1 = np.zeros((data_x_test.shape[0], data_x_test.shape[1], 2))
    for ii in range(data_x_test.shape[0]):
        data_x_test_run = data_x_test[ii, :, :, :].squeeze()
        x = data_x_test_run.reshape(-1, data_x_test_run.shape[1], data_x_test_run.shape[2], 1)
        # Perform EEG2Code prediction (sample-wise)
        preddata = model.predict(data_x_test_run.reshape(-1, data_x_test_run.shape[1], data_x_test_run.shape[2], 1))
        preddata1[ii, :] = preddata.squeeze()

    ## SAVE RESULTS
    RESULT_FILE = MATLAB_FILE[:-4] + 'result.mat'  # Here, name the output file.
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.keras.backend.clear_session()
    del data_x_train, data_y_train, data_x_val, data_y_val, data_x_test, model
    gc.collect()
    scipy.io.savemat(RESULT_FILE, {'preddata1': preddata1})
