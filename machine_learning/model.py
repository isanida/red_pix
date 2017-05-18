import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from urllib import request
import pickle
import os
import gzip
import numpy as np
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

import keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras import optimizers


def load_data_set():
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    filename = 'C:/Users/Stef/Anchormen_Projects/Healthcare_Demo/Deeplearning_Exploration/data/mnist.pkl.gz'
    if not os.path.exists(filename):
        print("Downloading MNIST dataset...")
        request.urlretrieve(url, filename)
    with gzip.open(filename, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()

    X_train, y_train = data[0]
    X_val, y_val = data[1]
    X_test, y_test = data[2]
    X_train = X_train.reshape((X_train.shape[0], 1, 28, 28))
    X_val = X_val.reshape((X_val.shape[0], 1, 28, 28))
    X_test = X_test.reshape((X_test.shape[0], 1, 28, 28))
    y_train = y_train.astype(np.uint8)
    y_val = y_val.astype(np.uint8)
    y_test = y_test.astype(np.uint8)
    return X_train, y_train, X_val, y_val, X_test, y_test


def conv_net_lasagne(X_train: np.ndarray, y_train: np.ndarray):
    """
    Building a convolutional neural network with the lasagne package.
    """

    print("WARNING: Training this neural leads to serious memory issues")

    net1 = NeuralNet(
        layers=[('input', layers.InputLayer),
                ('conv2d1', layers.Conv2DLayer),
                ('maxpool1', layers.MaxPool2DLayer),
                ('conv2d2', layers.Conv2DLayer),
                ('maxpool2', layers.MaxPool2DLayer),
                ('dropout1', layers.DropoutLayer),
                ('dense', layers.DenseLayer),
                ('dropout2', layers.DropoutLayer),
                ('output', layers.DenseLayer),
                ],
        # input layer
        input_shape=(None, 1, 28, 28),
        # layer conv2d1
        conv2d1_num_filters=32,
        conv2d1_filter_size=(5, 5),
        conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
        conv2d1_W=lasagne.init.GlorotUniform(),
        # layer maxpool1
        maxpool1_pool_size=(2, 2),
        # layer conv2d2
        conv2d2_num_filters=32,
        conv2d2_filter_size=(5, 5),
        conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
        # layer maxpool2
        maxpool2_pool_size=(2, 2),
        # dropout1
        dropout1_p=0.5,
        # dense
        dense_num_units=256,
        dense_nonlinearity=lasagne.nonlinearities.rectify,
        # dropout2
        dropout2_p=0.5,
        # output
        output_nonlinearity=lasagne.nonlinearities.softmax,
        output_num_units=10,
        # optimization method params
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,
        max_epochs=10,
        verbose=1,
        )
    # Train the network
    nn = net1.fit(X_train, y_train)

    return nn


def conv_net_keras(n_categories: int, width: int, height: int, depth: int=3) -> keras.models.Sequential:
    """
    Build convolutional neural network with Keras
    """

    model = Sequential()
    model.add(Convolution2D(32, (5, 5), input_shape=(width, height, depth), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, (5, 5), input_shape=(width, height, depth), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_categories, activation='softmax'))

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    return model


def train_keras_model(model: keras.models, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> keras.models.Sequential:
    model.fit(X_train, y_train, batch_size=32, nb_epoch=10, verbose=1, **kwargs)
    return model


def main():

    np.random.seed(123)  # for reproducibility

    X_train, y_train, X_val, y_val, X_test, y_test = load_data_set()

    n_categories = len(np.unique(y_train))
    y_train = np_utils.to_categorical(y_train, n_categories)

    y_test = np_utils.to_categorical(y_test, n_categories)

    # plt.imshow(X_train[0][0], cm.binary)
    # plt.show()

    model = conv_net_keras(depth=1)
    print(model.output_shape)

    # Reshape X_train to (50000, 28, 28, 1)
    X_train = np.rollaxis(X_train, 1, 4)
    model = train_keras_model(model=model, X_train=X_train, y_train=y_train)

    # nn = conv_net_lasagne(X_train=X_train, y_train=y_train)

if __name__ == "__main__":
    main()





