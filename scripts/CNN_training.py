import os
import subprocess
import sys

import numpy as np
import tensorflow as tf
import imageio
import glob
from PIL import Image
from sklearn.model_selection import train_test_split

from tensorflow.keras import datasets, layers, models


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def extract_image(image_path):
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


def create_mini_patches(data, patch_shape):
    """separate image into patches, data is a collection of images"""
    imgs = []
    for i in range(data.shape[0]):
        from patchify import patchify
        if len(patch_shape) == 3:
            imgs.append(patchify(data[i], patch_shape, step=patch_shape[0]).reshape(
                (-1, patch_shape[0], patch_shape[1], patch_shape[2])))
        else:
            imgs.append(patchify(data[i], patch_shape, step=patch_shape[0]).reshape(
                (-1, patch_shape[0], patch_shape[1])))
    return np.asarray(imgs)


def main():
    install("patchify")
    img_patch_size = 100

    data_dir = '../data/'
    train_data_filename = data_dir + 'training/images/'
    train_labels_filename = data_dir + 'training/groundtruth/'

    # Retrieve images/groundtruth and create mini patches
    images = extract_image(train_data_filename)
    images = create_mini_patches(images, (img_patch_size, img_patch_size, 3))
    images = images.reshape((-1, img_patch_size, img_patch_size, 3))

    labels = extract_labels(train_labels_filename)
    labels = create_mini_patches(labels, (img_patch_size, img_patch_size))
    labels = labels.reshape((-1, img_patch_size, img_patch_size))
    labels = labels.reshape(labels.shape[0], -1)


    # Split data
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.1)

    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3, 3),  activation='linear', input_shape=(img_patch_size, img_patch_size, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='linear'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='linear'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='linear'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='sigmoid'))
    model.add(layers.Dense(img_patch_size ** 2))

    model.summary()

    model.compile(optimizer='ftrl',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.BinaryAccuracy()])

    history = model.fit(train_images, train_labels, epochs=4)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
    print("Accuracy = ", test_acc)

    # prediction test
    # print(test_images[0:1, :, :, :].shape)
    # img = Image.fromarray(model.predict(test_images[0:1, :, :, :]).reshape((100, 100)))
    # img.show()

    # acc = history.history['accuracy']
    # print(acc)


if __name__ == '__main__':
    main()
