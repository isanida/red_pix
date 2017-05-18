import os
import time
import numpy as np
import keras

from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from sklearn.model_selection import train_test_split

import src.preprocessing.image_processing as ip
import src.preprocessing.image_transformation as it
import src.utils.filemanager as fm

import src.machine_learning.model as model

SAMPLE_SIZE = -1
IMG_WIDTH = 200
IMG_HEIGHT = 200


def main_old():

    path_data = os.path.join(os.path.join(fm.PATH_HOME, "data"))
    path_output = os.path.join(fm.PATH_HOME, "output")
    data_paths = os.listdir(path_data)

    path_img_info = os.path.join(path_output, "img_info.csv")
    path_processed_imgs = os.path.join(path_output, "flattened_imgs.pkl")
    df_img_info = fm.read_csv(path_img_info)

    assert all([d in data_paths for d in ["train", "test"]])

    for path in data_paths:
        if path == "train":
            train_path = os.path.join(path_data, path)
            df_img_info = ip.get_train_image_info(train_path=train_path, df_current=df_img_info)
            print("The images are of type(s): {}".format(df_img_info["file_extension"].unique()))
            print("We have {} unique cervix types which are: {}".format(len(df_img_info["type"].unique()),
                                                                        df_img_info["type"].unique()))
            print('We have a total of {} images in the whole dataset'.format(df_img_info.shape[0]))
            fm.write_csv(df=df_img_info, path=path_img_info)

    # df_img_mat = fm.read_pickle(img_flatten_path)
    df_sample = it.sample_dataframe(df=df_img_info, size=SAMPLE_SIZE, random_state=10)

    images_array, types, file_names = it.process_images(df=df_sample, rescaled_dim=[IMG_WIDTH, IMG_HEIGHT])
    fm.write_pickle(images_array, path=path_processed_imgs, compress=True)

    # Train convolutional neural network
    X = images_array

    # Convert categorical values
    unique_types, y = np.unique(types, return_inverse=True)

    print(X.shape)
    print(y.shape)
    y = np_utils.to_categorical(y, len(unique_types))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    print(X_train.shape)
    print(y_train.shape)

    m = model.conv_net_keras(n_categories=len(unique_types), width=IMG_WIDTH, height=IMG_HEIGHT, depth=3)
    m = model.train_keras_model(model=m, X_train=X_train, y_train=y_train)


def main():

    train_data_dir = os.path.join(fm.PATH_HOME, "data", "train")
    validation_data_dir = os.path.join(fm.PATH_HOME, "data", "validation")
    nb_train_samples = 700
    nb_validation_samples = 300
    epochs = 50
    batch_size = 16

    if K.image_data_format() == 'channels_first':
        input_shape = (3, IMG_WIDTH, IMG_HEIGHT)
    else:
        input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation="sigmoid"))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                        batch_size=batch_size, class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                                  target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                                  batch_size=batch_size, class_mode='categorical')

    model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=epochs,
                        validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size)

    model.save_weights('first_try.h5')


if __name__ == "__main__":
    t0 = time.time()
    main()
    t1 = time.time()
    print("Processing time: {} seconds".format(round((t1-t0), 2)))
