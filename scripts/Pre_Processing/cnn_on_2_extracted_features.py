# -*- coding: utf-8 -*-
"""
======================================================
======================================================
DESCRIPTION:
This script contains the code for running and evaluating
the training of a CNN on 2 features extracted from the
images of the training set. The resulting model is not
effective and provides poor accuracy. In fact, the CNN
is ill-adapted to the given features. The 2 features we
work on here, are the the image distance and edges, as
described in the report. However, in general, for image
recognition, we should not need to manually extract
anything. The CNN should find any such features by itself.
That is why we quickly discarded this idea and did not
bother finding good CNN to get results on these features.
In the report, refer to [EDGES] and [DISTANCE].
======================================================
======================================================



======================================================
======================IMPORTS=========================
======================================================
"""

import os
import subprocess
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import ipykernel
import time
import imageio
import glob
from PIL import Image
from patchify import patchify, unpatchify
from sklearn.model_selection import train_test_split, KFold

from tensorflow.keras import layers, models
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Reshape, Conv2D, MaxPooling2D, LeakyReLU


"""
======================================================
======================CONFIG==========================
======================================================
"""

"""
# Use this if GPU error during launch
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
"""


"""
======================================================
======================CONSTANTS=======================
======================================================
"""


img_patch_size = 16  # must be a divisor of 400 = 4 * 4 * 5 * 5
img_shape = (400, 400)
NUM_EPOCHS = 50
# Which fraction of all available training data to take:
DATA_PORTION = 1


"""
======================================================
=====================FUNCTIONS========================
======================================================
"""


def extract_images(image_path):
    """
    Extract all images from 'image_path'
    All values are between 0 and 1
    """

    imgs = []
    for img_path in glob.glob(image_path + "/*.png"):
        img = imageio.imread(img_path)
        img = img / 255.0
        imgs.append(img.astype('float32'))

    return np.asarray(imgs)


def extract_labels(label_path):
    """
    Extract all labels from 'label_path'
    """

    imgs = []
    for img_path in glob.glob(label_path + "/*.png"):
        img = imageio.imread(img_path)
        # Formalize labels
        img[img <= 127] = 0
        img[img > 127] = 1
        imgs.append(img.astype('uint8'))

    return np.asarray(imgs)


def create_patches(data, patch_shape):
    """
    separate image into patches, data is a collection of images
    """

    imgs = []
    for i in range(data.shape[0]):
        if len(patch_shape) == 3:  # RGB images
            patches = patchify(data[i], patch_shape, step=patch_shape[0])
            patches = patches.reshape((-1, patch_shape[0], patch_shape[1], patch_shape[2]))
            imgs.extend(patches)
        else:
            patches = patchify(data[i], patch_shape, step=patch_shape[0])
            patches = patches.reshape((-1, patch_shape[0], patch_shape[1]))
            number_of_patches = patches.shape[0]
            patches = patches.reshape((number_of_patches, -1))
            imgs.extend(patches)

    return np.asarray(imgs)


def characterise_each_patch_as_road_or_not(labels):
    """
    Binary classification for each patches, a patch is considered as a road if
    he has more than 50% road on it

    @labels : array_like
    @array of patches
    """

    new_labels = np.zeros((labels.shape[0]))
    for i in range(labels.shape[0]):
        new_labels[i] = 1 if np.count_nonzero(labels[i]) > labels[i].shape[0] / 2 else 0

    return new_labels


def plot_accuracies_with_respect_to_epochs(history, file_name="plot.png"):
    """
   Creates a plot showing how the model's accuracies vary with epochs
   history gives the model results, with the CORRECT ACCURACY MEASURING UNITS
   The figure is saved under the specified name
    """

    plt.plot(history.history['binary_accuracy'], 'g', label="accuracy on train set")
    plt.plot(history.history['val_binary_accuracy'], 'r', label="accuracy on validation set")
    plt.grid(True)
    plt.title('Training Accuracy vs. Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    plt.savefig(file_name)


"""
======================================================
=====================TRAINING=========================
======================================================
"""


def train_model(train_images, test_images, train_labels, test_labels):
    """
    Train a predefined model with the given data

    Returns the model, accuracy over the test data, loss over the test data
    Model works on a patch
    """

    # create mini_patches
    patches_train_images = create_patches(train_images, (img_patch_size, img_patch_size, 2))
    patches_test_images = create_patches(test_images, (img_patch_size, img_patch_size, 2))
    patches_train_labels = create_patches(train_labels, (img_patch_size, img_patch_size))
    patches_test_labels = create_patches(test_labels, (img_patch_size, img_patch_size))

    print("train label shape: ", patches_train_labels.shape)
    patches_train_labels = characterise_each_patch_as_road_or_not(patches_train_labels)
    patches_test_labels = characterise_each_patch_as_road_or_not(patches_test_labels)

    # Model
    model = models.Sequential()

    # Inspired from https://fractalytics.io/rooftop-detection-with-keras-tensorflow
    model.add(
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
                      input_shape=(img_patch_size, img_patch_size, 2)))
    model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(Dropout(0.50))

    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(Dropout(0.50))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.25)])

    history = model.fit(patches_train_images, patches_train_labels, epochs=NUM_EPOCHS,
                        batch_size=512,
                        validation_data=(patches_test_images, patches_test_labels))

    test_loss, test_acc = model.evaluate(patches_test_images, patches_test_labels)

    return model, test_loss, test_acc, history


def train_test_split_training(images, labels, test_size):
    """
    Train the model with a simple split of the data:
    test_size of data is used for testing and 1 - test_size is used for training

    Displays a result

    Returns the model and its history
    """

    # Split data
    train_images, test_images, train_labels, test_labels = train_test_split(images,
                                                                            labels,
                                                                            test_size=test_size,
                                                                            shuffle=True)
    model, test_loss, test_acc, history = train_model(train_images, test_images, train_labels, test_labels)

    print("Accuracy = ", test_acc)
    print("Loss = ", test_loss)

    return model, history


"""
======================================================
=====================EXECUTION========================
======================================================
"""


def main():
    start = time.time()

    data_dir = '../../data/'
    train_data_filename_norm = data_dir + 'training/data_augmented_norm/'
    train_data_filename_distance = data_dir + 'training/data_augmented_distance/'
    train_labels_filename = data_dir + 'training/data_augmented_groundtruth/'

    # Retrieve images/ground-truths

    norm = extract_labels(train_data_filename_norm)[:, :, :, None]
    distance = extract_labels(train_data_filename_distance)[:, :, :, None]

    images = np.concatenate((norm, distance), axis=3)

    labels = extract_labels(train_labels_filename)

    # shrink data size
    indexes = np.arange(len(images))
    np.random.shuffle(indexes)
    images = images[indexes[0: int(DATA_PORTION * len(indexes))]]
    labels = labels[indexes[0: int(DATA_PORTION * len(indexes))]]

    print("labels shape: ", labels.shape)
    print("images shape: ", images.shape)

    model, history = train_test_split_training(images, labels, 0.1)

    plot_accuracies_with_respect_to_epochs(history, "cnn_on_2_extracted_features.png")

    end = time.time()
    print("Computation time: ", end - start)


if __name__ == '__main__':
    main()
