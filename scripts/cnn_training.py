import os
import subprocess
import sys

import numpy as np
import tensorflow as tf
import ipykernel
import imageio
import glob
from PIL import Image
from patchify import patchify, unpatchify
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, models

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


def main():
    install("patchify")
    img_patch_size = 20
    img_shape = (400, 400)

    data_dir = '../data/'
    train_data_filename = data_dir + 'training/images/'
    train_labels_filename = data_dir + 'training/groundtruth/'

    # Retrieve images/groundtruth and create mini patches
    images = extract_images(train_data_filename)
    labels = extract_labels(train_labels_filename)

    # Split data
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.1)

    # create mini_patches
    train_images = create_patches(train_images, (img_patch_size, img_patch_size, 3))
    test_images = create_patches(test_images, (img_patch_size, img_patch_size, 3))
    train_labels = create_patches(train_labels, (img_patch_size, img_patch_size))
    test_labels = create_patches(test_labels, (img_patch_size, img_patch_size))

    # Test: retrieve original train labels using unpatchify
    test_labels_original = test_labels.reshape(
        (-1, int(img_shape[0] / img_patch_size), int(img_shape[1] / img_patch_size), img_patch_size, img_patch_size))
    print("test lab origi: ", test_labels_original.shape)
    test_labels_original_image0 = unpatchify(test_labels_original[0], img_shape)

    # CNN
    model = models.Sequential()
    model.add(
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(img_patch_size, img_patch_size, 3)))
    model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(img_patch_size ** 2, activation='sigmoid'))

    model.summary()

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=2)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
    print("Accuracy = ", test_acc)

    # prediction
    prediction = model.predict(test_images)

    # show expected and predicted image
    prediction_reshaped = prediction.reshape(
        (-1, int(img_shape[0] / img_patch_size), int(img_shape[1] / img_patch_size), img_patch_size, img_patch_size))
    prediction_reshaped = unpatchify(prediction_reshaped[0], img_shape)
    comparator = np.concatenate(
        ((test_labels_original_image0 * 255).astype('uint8'), (prediction_reshaped * 255).astype('uint8')), axis=1)
    img = Image.fromarray(comparator)
    img.show()

    # Testing
    tot_black_pixels = np.sum(test_labels < 0.5)
    tot_pixels = test_labels.shape[0] * test_labels.shape[1]
    print(tot_black_pixels / tot_pixels * 100, "% of black pixels")

    """
    threshold = 0.5
    result[result > threshold] = 1
    result[result <= threshold] = 0
    img = Image.fromarray(result * 255, 'L')
    img.show()
    """


if __name__ == '__main__':
    main()
