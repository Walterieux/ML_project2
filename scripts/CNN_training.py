import os
import subprocess
import sys

import numpy as np
import tensorflow as tf
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

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

img_patch_size = 16  # must be a divisor of 400 = 4 * 4 * 5 * 5
img_shape = (400, 400)
NUM_EPOCHS = 1


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
        imgs.append(img)

    return np.asarray(imgs)


def extract_labels(label_path):
    """Extract all labels from 'label_path'"""

    imgs = []
    for img_path in glob.glob(label_path + "/*.png"):
        img = imageio.imread(img_path)
        # Formalize labels
        img[img <= 127] = 0
        img[img > 127] = 1
        imgs.append(img)

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


def train_model(train_images, test_images, train_labels, test_labels, epochs):
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
    train_labels = characterise_each_patch_as_road_or_not(patches_train_labels)
    test_labels = characterise_each_patch_as_road_or_not(patches_test_labels)

    # First model
    model1 = models.Sequential()

    # Inspired from https://fractalytics.io/rooftop-detection-with-keras-tensorflow
    model1.add(
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
                      input_shape=(img_patch_size, img_patch_size, 3)))
    model1.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model1.add(layers.MaxPool2D((2, 2)))
    model1.add(Dropout(0.20))  # Avoid overfitting

    model1.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model1.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model1.add(layers.MaxPool2D((2, 2)))
    model1.add(Dropout(0.20))

    model1.add(layers.Flatten())
    model1.add(layers.Dense(128, activation='relu'))
    model1.add(layers.Dense(1, activation='sigmoid'))

    # model.summary()

    model1.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['binary_accuracy'])

    model1.fit(patches_train_images, train_labels, epochs=epochs)

    predicted_train_labels = model1.predict(patches_train_images)
    predicted_train_labels = predicted_train_labels.reshape(-1, int(img_shape[0] / img_patch_size),
                                                            int(img_shape[1] / img_patch_size))

    print("predicted_train labels shape: ", predicted_train_labels.shape)

    # Second model
    model2 = models.Sequential()

    model2.add(
        layers.Conv2D(512, kernel_size=(5, 5), activation='relu', padding='same',
                      input_shape=(int(img_shape[0] / img_patch_size), int(img_shape[1] / img_patch_size), 1)))

    model2.add(layers.Conv2D(512, kernel_size=(5, 5), activation='relu'))
    # model2.add(layers.MaxPool2D((2, 2)))
    # model2.add(Dropout(0.20))

    model2.add(layers.Flatten())
    model2.add(layers.Dense(1024, activation='relu'))
    model2.add(layers.Dense(img_shape[0] * img_shape[1], activation='sigmoid'))

    test_loss, test_acc = model2.fit(predicted_train_labels, test_labels, verbose=1)

    predicted_test_labels = model1.predict(patches_test_images)
    predicted_test_labels = predicted_test_labels.reshape(-1, int(img_shape[0] / img_patch_size),
                                                          int(img_shape[1] / img_patch_size))

    predicted_labels = model2.predict(predicted_test_labels)
    print("predicted_labels shape: ", predicted_labels.shape)

    comparator = np.concatenate(
        ((test_images[0] * 255).astype('uint8'), (predicted_labels[0] * 255).astype('uint8')), axis=1)
    img = Image.fromarray(comparator)
    img.show()

    return model1, test_loss, test_acc  # TODO change model1 by ??? we have 2 models!


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

        model, test_loss, test_acc = train_model(train_images, test_images, train_labels, test_labels, NUM_EPOCHS)

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

    model, test_loss, test_acc = train_model(train_images, test_images, train_labels, test_labels, NUM_EPOCHS)

    print("Accuracy = ", test_acc)
    print("Loss = ", test_loss)

    return model


def main():
    start = time.time()
    need_to_train = True

    install("patchify")

    data_dir = '../data/'
    train_data_filename = data_dir + 'training/images_rotated/'
    train_labels_filename = data_dir + 'training/images_rotated_groundtruth/'

    # Retrieve images/groundtruths
    images = extract_images(train_data_filename)
    print("images shape: ", images.shape)
    labels = extract_labels(train_labels_filename)
    print("labels shape: ", labels.shape)

    if need_to_train:
        model = train_test_split_training(images, labels, 0.1)
        model.save("saved_model")
    else:
        model = keras.models.load_model("saved_model")

    # prediction

    # prediction = model.predict(test_images)
    # prediction[prediction < 0.5] = 0
    # prediction[prediction >= 0.5] = 1

    # show expected and predicted image
    # prediction_reshaped = prediction.reshape(
    #    (-1, int(img_shape[0] / img_patch_size), int(img_shape[1] / img_patch_size), img_patch_size, img_patch_size))
    # prediction_reshaped = unpatchify(prediction_reshaped[0], img_shape)

    # prediction_reshaped = prediction.reshape(-1, int(img_shape[0] / img_patch_size),
    #                                          int(img_shape[1] / img_patch_size))
    # print("test_labels_original_image0: ", test_labels_original_image0.shape)
    # print("prediction_reshaped: ", prediction_reshaped.shape)
    # prediction_reshaped = prediction_reshaped[0].repeat(img_patch_size, axis=0).repeat(img_patch_size, axis=1)
    # comparator = np.concatenate(
    #     ((test_labels_original_image0 * 255).astype('uint8'), (prediction_reshaped * 255).astype('uint8')), axis=1)
    # img = Image.fromarray(comparator)
    # img.show()

    end = time.time()
    print("Computation time: ", end - start)


if __name__ == '__main__':
    main()
