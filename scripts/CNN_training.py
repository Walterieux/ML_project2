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


def main():
    install("patchify")
    img_patch_size = 16  # must be a divisor of 400 = 4 * 4 * 5 * 5
    img_shape = (400, 400)
    num_folds = 10
    acc_per_fold = []
    loss_per_fold = []

    data_dir = '../data/'
    train_data_filename = data_dir + 'training/images_rotated/'
    train_labels_filename = data_dir + 'training/images_rotated_groundtruth/'

    # Retrieve images/groundtruths
    images = extract_images(train_data_filename)
    print("images shape: ", images.shape)
    labels = extract_labels(train_labels_filename)
    print("labels shape: ", labels.shape)

    start = time.time()

    kfold = KFold(n_splits=num_folds, shuffle=True)
    for train_index, test_index in kfold.split(images, labels):

        # Split data
        train_images = images[train_index]
        test_images = images[test_index]
        train_labels = labels[train_index]
        test_labels = labels[test_index]

        test_labels_original_image0 = test_labels[0]

        # create mini_patches
        train_images = create_patches(train_images, (img_patch_size, img_patch_size, 3))
        test_images = create_patches(test_images, (img_patch_size, img_patch_size, 3))
        train_labels = create_patches(train_labels, (img_patch_size, img_patch_size))
        test_labels = create_patches(test_labels, (img_patch_size, img_patch_size))

        print("train label shape: ", train_labels.shape)
        train_labels = characterise_each_patch_as_road_or_not(train_labels)
        test_labels = characterise_each_patch_as_road_or_not(test_labels)

        model = models.Sequential()

        # Inspired from https://fractalytics.io/rooftop-detection-with-keras-tensorflow
        model.add(
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
                          input_shape=(img_patch_size, img_patch_size, 3)))
        model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPool2D((2, 2)))
        model.add(Dropout(0.20))  # Avoid overfitting

        model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPool2D((2, 2)))
        model.add(Dropout(0.20))

        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.summary()

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['binary_accuracy'])

        model.fit(train_images, train_labels, epochs=10)

        # model.save("saved_model") TODO

        # model = keras.models.load_model("saved_model")

        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
        acc_per_fold.append(test_acc)
        loss_per_fold.append(test_loss)
        # print("Accuracy = ", test_acc)

        # prediction
        prediction = model.predict(test_images)
        # prediction[prediction < 0.5] = 0
        # prediction[prediction >= 0.5] = 1

        # show expected and predicted image
        # prediction_reshaped = prediction.reshape(
        #    (-1, int(img_shape[0] / img_patch_size), int(img_shape[1] / img_patch_size), img_patch_size, img_patch_size))
        # prediction_reshaped = unpatchify(prediction_reshaped[0], img_shape)

        prediction_reshaped = prediction.reshape(-1, int(img_shape[0] / img_patch_size),
                                                 int(img_shape[1] / img_patch_size))
        print("test_labels_original_image0: ", test_labels_original_image0.shape)
        print("prediction_reshaped: ", prediction_reshaped.shape)
        prediction_reshaped = prediction_reshaped[0].repeat(img_patch_size, axis=0).repeat(img_patch_size, axis=1)
        comparator = np.concatenate(
            ((test_labels_original_image0 * 255).astype('uint8'), (prediction_reshaped * 255).astype('uint8')), axis=1)
        img = Image.fromarray(comparator)
        img.show()

    end = time.time()
    print("Computation time: ", end - start)
    print("Mean accuracy = ", np.mean(acc_per_fold), " accuracy variance: ", np.var(acc_per_fold))
    print("Mean loss = ", np.mean(loss_per_fold), " loss variance: ", np.var(loss_per_fold))


if __name__ == '__main__':
    main()
