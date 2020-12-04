import os
import subprocess
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
import ipykernel
import matplotlib.pyplot as plt
import time
import imageio
import glob
from PIL import Image
from patchify import patchify, unpatchify
from sklearn.model_selection import train_test_split, KFold

from tensorflow.keras import layers, models
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Reshape, Conv2D, MaxPooling2D, LeakyReLU, ReLU

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

img_patch_size = 16  # must be a divisor of 400 = 4 * 4 * 5 * 5
img_shape = (400, 400)
NUM_EPOCHS = 50


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
            patches = patchify(data[i], patch_shape, step=patch_shape[0])  # TODO change step to have context border: ex step=patch_shape[0] - border
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


def train_model(train_images, test_images, train_labels, test_labels):
    """
    Train a predefined model with the given data

    Returns the model, accuracy over the test data, loss over the test data
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
    model = models.Sequential()

    # literature https://www.kdnuggets.com/2017/11/understanding-deep-convolutional-neural-networks-tensorflow-keras.html/2
    # Inspired from https://fractalytics.io/rooftop-detection-with-keras-tensorflow

    #
    # https://www.learnopencv.com/understanding-alexnet/ dropout 0.5

    """ ~ Model from Stanford University: https://youtu.be/bNb2fEVKeEo?t=3683"""
    model.add(
        layers.Conv2D(64, kernel_size=(3, 3), padding='same',
                      input_shape=(img_patch_size, img_patch_size, 3)))  # TODO change this
    model.add(tf.keras.layers.ReLU())
    model.add(layers.Conv2D(64, kernel_size=(3, 3)))
    model.add(tf.keras.layers.ReLU())
    model.add(layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(Dropout(.25))  # Avoid overfitting

    model.add(layers.Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.ReLU())
    model.add(layers.Conv2D(128, kernel_size=(3, 3), padding='same'))
    model.add(tf.keras.layers.ReLU())
    model.add(layers.MaxPool2D((2, 2), padding='same'))
    model.add(Dropout(.25))

    """model.add(layers.Conv2D(64, kernel_size=(3, 3), padding='same'))  # TODO bigger kernel size?
    model.add(tf.keras.layers.ReLU())
    model.add(layers.Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(tf.keras.layers.ReLU())
    model.add(layers.MaxPool2D((2, 2), padding='same'))
    model.add(Dropout(.25))"""

    model.add(layers.Flatten())
    model.add(layers.Dense(256))
    model.add(tf.keras.layers.ReLU())
    #model.add(Dropout(.5))
    #model.add(layers.Dense(1024))
    #model.add(tf.keras.layers.ReLU())
    #model.add(Dropout(.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adamax',
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.25)])

    history = model.fit(patches_train_images,
                        patches_train_labels,
                        epochs=NUM_EPOCHS,
                        validation_data=(patches_test_images, patches_test_labels))

    plt.plot(history.history['binary_accuracy'], 'g', label="accuracy on train set")
    plt.plot(history.history['val_binary_accuracy'], 'r', label="accuracy on validation set")
    plt.grid(True)
    plt.title('Training Accuracy vs. Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    test_loss, test_acc = model.evaluate(patches_test_images, patches_test_labels)

    return model, test_loss, test_acc


def cross_validation_training(images, labels, num_folds):
    """
    Train model with cross validation

    Returns the two models with the best accuracy over the testing set
    """

    acc_per_fold = []
    loss_per_fold = []
    best_accuracy = 0
    best_model = None

    kfold = KFold(n_splits=num_folds, shuffle=True)
    for train_index, test_index in kfold.split(images, labels):

        # Split data
        train_images = images[train_index]
        test_images = images[test_index]
        train_labels = labels[train_index]
        test_labels = labels[test_index]

        model, test_loss, test_acc = train_model(train_images, test_images, train_labels, test_labels)

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model = model

        acc_per_fold.append(test_acc)
        loss_per_fold.append(test_loss)
    print("Mean accuracy = ", np.mean(acc_per_fold), " accuracy variance: ", np.var(acc_per_fold))
    print("Mean loss = ", np.mean(loss_per_fold), " loss variance: ", np.var(loss_per_fold))

    return best_model


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

    model, test_loss, test_acc = train_model(train_images, test_images, train_labels, test_labels)

    print("Accuracy = ", test_acc)
    print("Loss = ", test_loss)

    return model


def main():
    start = time.time()
    need_to_train = True

    install("patchify")

    data_dir = '../data/'
    train_data_filename = data_dir + 'training/images/'
    train_labels_filename = data_dir + 'training/groundtruth/'
    train_data_filename_norm = data_dir + 'training/data_augmented_norm/'
    train_data_filename_edges = data_dir + 'training/data_augmented_edges/'

    training_training_data_path = data_dir + 'training_training/data_augmented'
    training_training_labels_path = data_dir + 'training_training/data_augmented_groundtruth'
    training_test_data_path = data_dir + 'training_test/data_augmented'
    training_test__labels_path = data_dir + 'training_test/data_augmented_groundtruth'

    # Retrieve images/groundtruths
    # images = extract_images(train_data_filename)
    # labels = extract_labels(train_labels_filename)
    # norm = extract_images(train_data_filename_norm)[:, :, :, None]
    # edges = extract_images(train_data_filename_edges)[:, :, :, None]

    train_images = extract_images(training_training_data_path)
    test_images = extract_images(training_test_data_path)
    train_labels = extract_labels(training_training_labels_path)
    test_labels = extract_labels(training_test__labels_path)

    images = extract_images(train_data_filename)
    labels = extract_labels(train_labels_filename)

    # model = train_test_split_training(images, labels, 0.9)
    model, test_loss, test_acc = train_model(train_images, test_images, train_labels, test_labels)
    model.save("saved_model")


    end = time.time()
    print("Computation time: ", end - start)


if __name__ == '__main__':
    main()
