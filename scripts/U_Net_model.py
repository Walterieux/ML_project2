import os
import subprocess
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
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
from tensorflow_core.python.keras import Input
from tensorflow_core.python.keras.layers import Conv2DTranspose, concatenate
from tensorflow_core.python.keras.models import Model
from tensorflow_core.python.keras.optimizers import Adam

import scripts.create_submission_groundtruth
from scripts.convolution import create_patches_from_training_or_test

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

img_patch_size = 16  # must be a divisor of 400 = 4 * 4 * 5 * 5
border_size = 16
img_patch_with_border_size = img_patch_size + (2 * border_size)
img_shape = (400, 400)
NUM_EPOCHS = 100


def extract_images(image_path):
    """
    Extract all images from 'image_path'
    All values are between 0 and 1
    """

    imgs = []
    for img_path in glob.glob(image_path + "/*.png"):
        img = imageio.imread(img_path)
        # img = img / 255.0
        #img = normalize_image(img)
        imgs.append(img.astype('uint8'))

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


def normalize_image(image):
    """
    Return an array with values between 0 and 1
    """

    min = np.min(image)
    max = np.max(image)
    return (image - min) / (max - min)


def train_model(train_images, test_images, train_labels, test_labels):
    """
    Train a predefined model with the given data

    Returns the model, accuracy over the test data, loss over the test data
    """

    # shrink data size
    indexes = np.arange(len(train_images))
    np.random.shuffle(indexes)
    train_images = train_images[indexes[0: int(0.25 * len(indexes))]]
    train_labels = train_labels[indexes[0: int(0.25 * len(indexes))]]
    indexes = np.arange(len(test_images))
    np.random.shuffle(indexes)
    test_images = test_images[indexes[0: int(0.4 * len(indexes))]]
    test_labels = test_labels[indexes[0: int(0.4 * len(indexes))]]

    nb_train = np.prod(train_labels.shape)
    nb_test = np.prod(test_labels.shape)
    percentage_road = (np.mean(train_labels) * nb_train + np.mean(test_labels) * nb_test) / (
            nb_train + nb_test)
    print("percentage road: ", percentage_road)

    # create mini_patches
    patches_train_images = np.array(
        [create_patches_from_training_or_test(train_image) for train_image in
         train_images])
    patches_test_images = np.array(
        [create_patches_from_training_or_test(test_image) for test_image in
         test_images])
    patches_train_labels = np.array(
        [create_patches_from_training_or_test(train_label, rgb_binary=False) for
         train_label in train_labels])
    patches_test_labels = np.array(
        [create_patches_from_training_or_test(test_label, rgb_binary=False) for
         test_label in test_labels])

    print("patches_train_images shape: ", patches_train_images.shape)
    print("patches_train_labels shape: ", patches_train_labels.shape)

    # patches_train_labels = characterise_each_patch_as_road_or_not(patches_train_labels)
    # patches_test_labels = characterise_each_patch_as_road_or_not(patches_test_labels)

    # https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5
    def build_model(input_size, start_neurons):
        inputs = Input(input_size)
        conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(inputs)
        conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
        pool1 = MaxPooling2D((2, 2))(conv1)
        pool1 = Dropout(0.25)(pool1)

        conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
        conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
        pool2 = MaxPooling2D((2, 2))(conv2)
        pool2 = Dropout(0.5)(pool2)

        conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
        conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
        pool3 = MaxPooling2D((2, 2))(conv3)
        pool3 = Dropout(0.5)(pool3)

        conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
        conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
        pool4 = MaxPooling2D((2, 2))(conv4)
        pool4 = Dropout(0.5)(pool4)

        # Middle
        convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
        convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)

        deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
        uconv4 = concatenate([deconv4, conv4])
        uconv4 = Dropout(0.5)(uconv4)
        uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
        uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

        deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
        uconv3 = concatenate([deconv3, conv3])
        uconv3 = Dropout(0.5)(uconv3)
        uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
        uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

        deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
        uconv2 = concatenate([deconv2, conv2])
        uconv2 = Dropout(0.5)(uconv2)
        uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
        uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

        deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
        uconv1 = concatenate([deconv1, conv1])
        uconv1 = Dropout(0.5)(uconv1)
        uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
        uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)

        output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)

        model = Model(inputs=inputs, outputs=output_layer)

        model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

        return model

    model = build_model(input_size=(572, 572, 1), start_neurons=16)

    history = model.fit(history=model.fit(patches_train_images,
                                patches_train_labels,
                                batch_size=32,
                                epochs=NUM_EPOCHS,
                                validation_data=(patches_test_images, patches_test_labels)))


    plt.plot(history.history['accuracy'], 'g', label="accuracy on train set")
    plt.plot(history.history['val_accuracy'], 'r', label="accuracy on validation set")
    plt.grid(True)
    plt.title('Training Accuracy vs. Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


    training_test_predicted_labels = model.predict(patches_test_images)
    unpatched_labels = scripts.create_submission_groundtruth.unpatch_labels(training_test_predicted_labels,
                                                                            test_images.shape[0],
                                                                            img_shape)
    scripts.create_submission_groundtruth.save_labels(unpatched_labels,
                                                      "../data/training_test/data_augmented_predicted_labels/")

    test_loss, test_acc = model.evaluate(patches_test_images, patches_test_labels)

    return model, test_loss, test_acc


def cross_validation_training(images, labels, num_folds):
    """
    Train model with cross validation

    Returns the model with the best accuracy over the testing set
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

    Returns the model
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

    data_dir = '../data/'
    train_data_filename = data_dir + 'training/images/'
    train_labels_filename = data_dir + 'training/groundtruth/'
    train_data_filename_norm = data_dir + 'training/data_augmented_norm/'
    train_data_filename_edges = data_dir + 'training/data_augmented_edges/'

    training_training_data_path = data_dir + 'training_training/data_augmented'
    training_training_labels_path = data_dir + 'training_training/data_augmented_groundtruth'
    training_test_data_path = data_dir + 'training_test/data_augmented'
    training_test__labels_path = data_dir + 'training_test/data_augmented_groundtruth'

    # train_images, mean, std = center(extract_images(training_training_data_path))
    # test_images, _, _ = center(extract_images(training_test_data_path), mean, std, still_to_center=False)
    train_images = extract_images(training_training_data_path)
    test_images = extract_images(training_test_data_path)
    train_labels = extract_labels(training_training_labels_path)
    test_labels = extract_labels(training_test__labels_path)

    # model = train_test_split_training(images, labels, 0.9)
    model, test_loss, test_acc = train_model(train_images, test_images, train_labels, test_labels)
    model.save("saved_model")

    end = time.time()
    print("Computation time: ", end - start)


if __name__ == '__main__':
    main()
