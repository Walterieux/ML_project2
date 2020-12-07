# -*- coding: utf-8 -*-
"""
======================================================
======================================================
DESCRIPTION:
This script contains the code for running and evaluating
the training of a a model resulting from 2 CNNs trained
one after the other. The 2nd CNN takes as input the result
from evaluating (predicting) using the 1st CNN. The 1st CNN
takes simply the image values as an input.
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
import matplotlib.pyplot as plt
from tensorflow import keras
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

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

"""
======================================================
======================CONSTANTS=======================
======================================================
"""

img_patch_size = 16  # must be a divisor of 400 = 4 * 4 * 5 * 5
img_shape = (400, 400)
NUM_EPOCHS_MODEL1 = 50
NUM_EPOCHS_MODEL2 = 50
# Which fraction of all available training data to take:
DATA_PORTION = 0.5
# Index of the ground truth for which we want to see our prediction
IMAGE_TO_TEST_INDEX = 0

"""
======================================================
=====================FUNCTIONS========================
======================================================
"""


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


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
    """Extract all labels from 'label_path'"""

    imgs = []
    for img_path in glob.glob(label_path + "/*.png"):
        img = imageio.imread(img_path)
        # Formalize labels
        img[img <= 127] = 0
        img[img > 127] = 1
        imgs.append(img.astype('uint8'))

    return np.asarray(imgs)


def create_patches(data, patch_shape):
    """separate image into patches, data is a collection of images"""

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

    Parameters
    ----------
    labels : array_like
        array of patches
    """

    new_labels = np.zeros((labels.shape[0]))
    for i in range(labels.shape[0]):
        new_labels[i] = 1 if np.count_nonzero(labels[i]) > labels[i].shape[0] / 2 else 0

    return new_labels


def represent_predicted_labels(given, first_pred, second_pred):
    """
    Draw the three images labels
    """

    comparator = np.concatenate(
        ((given * 255).astype('uint8'),
         (first_pred * 255).astype('uint8')),
        axis=1)
    comparator = np.concatenate(
        (comparator,
         (second_pred * 255).astype('uint8')),
        axis=1)
    img = Image.fromarray(comparator)
    img.show()


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

    Returns the two models, accuracy over the test data, loss over the test data
    """

    # create mini_patches
    patches_train_images = create_patches(train_images, (img_patch_size, img_patch_size, 3))
    patches_test_images = create_patches(test_images, (img_patch_size, img_patch_size, 3))
    patches_train_labels = create_patches(train_labels, (img_patch_size, img_patch_size))
    patches_test_labels = create_patches(test_labels, (img_patch_size, img_patch_size))

    print("train label shape: ", patches_train_labels.shape)
    patches_train_labels = characterise_each_patch_as_road_or_not(patches_train_labels)
    patches_test_labels = characterise_each_patch_as_road_or_not(patches_test_labels)

    # First model
    model1 = models.Sequential()

    # Inspired from https://fractalytics.io/rooftop-detection-with-keras-tensorflow
    model1.add(
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
                      input_shape=(img_patch_size, img_patch_size, 3)))
    model1.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model1.add(layers.MaxPool2D((2, 2)))
    model1.add(Dropout(0.20))

    model1.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model1.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model1.add(layers.MaxPool2D((2, 2)))
    model1.add(Dropout(0.20))

    model1.add(layers.Flatten())
    model1.add(layers.Dense(128, activation='relu'))
    model1.add(layers.Dense(1, activation='sigmoid'))

    model1.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['binary_accuracy'])

    history = model1.fit(patches_train_images, patches_train_labels, epochs=NUM_EPOCHS_MODEL1, batch_size=512,
                         validation_data=(patches_test_images, patches_test_labels))

    model1.evaluate(patches_test_images, patches_test_labels)

    predicted_train_labels = model1.predict(patches_train_images)
    predicted_train_labels = predicted_train_labels.reshape(-1, int(img_shape[0] / img_patch_size),
                                                            int(img_shape[1] / img_patch_size))

    print("predicted_train labels shape: ", predicted_train_labels.shape)

    # Second model
    model2 = models.Sequential()

    model2.add(
        layers.Conv2D(256, kernel_size=(4, 4), strides=(4, 4), activation='relu', padding='same',
                      input_shape=(int(img_shape[0] / img_patch_size), int(img_shape[1] / img_patch_size), 1)))

    model2.add(layers.Conv2D(256, kernel_size=(4, 4), strides=(4, 4), activation='relu'))
    model2.add(layers.AvgPool2D((2, 2), padding='same'))
    model2.add(Dropout(0.30))

    model2.add(layers.Conv2D(128, kernel_size=(4, 4), strides=(4, 4), activation='relu', padding='same'))
    model2.add(layers.AvgPool2D((2, 2), padding='same'))
    model2.add(Dropout(0.30))

    model2.add(layers.Flatten())
    model2.add(layers.Dense(64, activation='relu'))
    model2.add(layers.Dense(img_shape[0] * img_shape[1], activation='sigmoid'))

    model2.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.25)])

    # Use the predicted result with model1 to train model2
    pred_tr_lab_shape = predicted_train_labels.shape
    print("pred_tr_lab_shape: ", pred_tr_lab_shape)
    tr_lab_shape = train_labels.shape
    print("tr_lab_shape: ", tr_lab_shape)
    model2.fit(
        predicted_train_labels.reshape((pred_tr_lab_shape[0], pred_tr_lab_shape[1], pred_tr_lab_shape[2], 1)),
        train_labels.reshape(tr_lab_shape[0], -1),
        epochs=NUM_EPOCHS_MODEL2,
    )

    predicted_test_labels = model1.predict(patches_test_images)
    predicted_test_labels = predicted_test_labels.reshape(-1, int(img_shape[0] / img_patch_size),
                                                          int(img_shape[1] / img_patch_size))

    pred_te_lab_shape = predicted_test_labels.shape
    predicted_test_labels = predicted_test_labels.reshape(
        (pred_te_lab_shape[0], pred_te_lab_shape[1], pred_te_lab_shape[2], 1))
    predicted_labels = model2.predict(predicted_test_labels)
    predicted_labels = predicted_labels.reshape((predicted_labels.shape[0], img_shape[0], img_shape[1], 1))
    print("predicted_labels shape: ", predicted_labels.shape)

    test_loss, test_acc = model2.evaluate(predicted_test_labels, test_labels.reshape((test_labels.shape[0], -1)))

    represent_predicted_labels(test_labels[IMAGE_TO_TEST_INDEX],
                               predicted_test_labels[IMAGE_TO_TEST_INDEX].repeat(img_patch_size, axis=0).repeat(
                                   img_patch_size,
                                   axis=1).reshape(
                                   img_shape),
                               predicted_labels[0].reshape(img_shape))

    return model1, model2, test_loss, test_acc, history


def train_test_split_training(images, labels, test_size):
    """
    Train the model with a simple split of the data:
    test_size of data is used for testing and 1 - test_size is used for training

    Returns the two models
    """

    # Split data
    train_images, test_images, train_labels, test_labels = train_test_split(images,
                                                                            labels,
                                                                            test_size=test_size,
                                                                            shuffle=True)

    model1, model2, test_loss, test_acc, history = train_model(train_images, test_images, train_labels, test_labels)

    print("Accuracy = ", test_acc)
    print("Loss = ", test_loss)

    return model1, model2, history


"""
======================================================
=====================EXECUTION========================
======================================================
"""


def main():
    start = time.time()

    install("patchify")

    data_dir = '../../data/'
    train_data_filename = data_dir + 'training/data_augmented/'
    train_labels_filename = data_dir + 'training/data_augmented_groundtruth/'

    # Retrieve images/groundtruths
    images = extract_images(train_data_filename)
    labels = extract_labels(train_labels_filename)

    # shrink data size
    indexes = np.arange(len(images))
    np.random.shuffle(indexes)
    images = images[indexes[0: int(DATA_PORTION * len(indexes))]]
    labels = labels[indexes[0: int(DATA_PORTION * len(indexes))]]

    print("images shape: ", images.shape)
    print("labels shape: ", labels.shape)

    model1, model2, history = train_test_split_training(images, labels, 0.1)

    plot_accuracies_with_respect_to_epochs(history, "2_cnn_after_one_another.png")

    end = time.time()
    print("Computation time: ", end - start)


if __name__ == '__main__':
    main()
