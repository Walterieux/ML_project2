# TODO add description + good values
import glob
import time

import ipykernel
import imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from patchify import patchify
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.python.keras.layers import Dropout

from Post_Processing import create_submission_groundtruth

"""
# Use this if GPU error during launch
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
"""

img_patch_size = 16  # must be a divisor of 400 = 4 * 4 * 5 * 5
border_size = 16
img_patch_with_border_size = img_patch_size + (2 * border_size)
img_shape = (400, 400)
NUM_EPOCHS = 50


def extract_images(image_path):
    """
    Extract all images from 'image_path'
    All values are between 0 and 1
    """

    imgs = []
    for img_path in glob.glob(image_path + "/*.png"):
        img = imageio.imread(img_path)
        img = normalize_image(img)
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


def create_patches_with_border(data, patch_shape, border, batch_name=""):
    """separate image into patches, data is a collection of images"""

    imgs = []
    for i in range(data.shape[0]):
        # Add border to whole image
        img = add_border_3d(data[i], border)
        # Split image
        patches = patchify(img, (patch_shape[0] + (border * 2), patch_shape[1] + (border * 2), patch_shape[2]),
                           step=patch_shape[0])
        patches = patches.reshape((-1, patch_shape[0] + (border * 2), patch_shape[1] + (border * 2),
                                   patch_shape[2]))
        imgs.extend(patches)

    return np.asarray(imgs)


def add_border_3d(image, border):
    """
    Adds a black border of size border to a given image
    """
    return np.pad(image, ((border, border), (border, border), (0, 0)), 'symmetric')


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


def train_model(train_images, validation_images, test_images, train_labels, validation_labels, test_labels):
    """
    Train a predefined model with the given data

    Returns the model, accuracy over the test data, loss over the test data
    """

    # shrink data size
    indexes = np.arange(len(train_images))
    np.random.shuffle(indexes)
    train_images = train_images[indexes[0: int(0.30 * len(indexes))]]
    train_labels = train_labels[indexes[0: int(0.30 * len(indexes))]]
    indexes = np.arange(len(validation_images))
    np.random.shuffle(indexes)
    validation_images = validation_images[indexes[0: int(0.5 * len(indexes))]]
    validation_labels = validation_labels[indexes[0: int(0.5 * len(indexes))]]


    # create mini_patches
    patches_train_labels = create_patches(train_labels, (img_patch_size, img_patch_size))
    patches_validation_labels = create_patches(validation_labels, (img_patch_size, img_patch_size))
    patches_test_labels = create_patches(test_labels, (img_patch_size, img_patch_size))

    patches_train_images = create_patches_with_border(train_images, (img_patch_size, img_patch_size, 3), border_size,
                                                      "TRAIN IMAGES")
    patches_validation_images = create_patches_with_border(validation_images, (img_patch_size, img_patch_size, 3), border_size,
                                                      "TRAIN IMAGES")
    patches_test_images = create_patches_with_border(test_images, (img_patch_size, img_patch_size, 3), border_size,
                                                     "TEST IMAGES")

    patches_train_labels = characterise_each_patch_as_road_or_not(patches_train_labels)
    patches_validation_labels = characterise_each_patch_as_road_or_not(patches_validation_labels)
    patches_test_labels = characterise_each_patch_as_road_or_not(patches_test_labels)

    # First model
    model = models.Sequential()

    model.add(
        layers.Conv2D(64, kernel_size=(3, 3), padding='same',
                      input_shape=(img_patch_with_border_size, img_patch_with_border_size, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.SpatialDropout2D(rate=0.10))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.SpatialDropout2D(rate=0.10))
    model.add(layers.MaxPool2D((2, 2), padding='same'))
    model.add(Dropout(.10))

    model.add(layers.Conv2D(128, kernel_size=(3, 3), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.SpatialDropout2D(rate=0.10))
    model.add(layers.Conv2D(128, kernel_size=(3, 3), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.SpatialDropout2D(rate=0.10))
    model.add(layers.Conv2D(128, kernel_size=(3, 3), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.SpatialDropout2D(rate=0.10))
    model.add(layers.MaxPool2D((2, 2), padding='same'))
    model.add(Dropout(.10))

    model.add(layers.Conv2D(256, kernel_size=(3, 3), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.SpatialDropout2D(rate=0.10))
    model.add(layers.Conv2D(256, kernel_size=(3, 3), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.SpatialDropout2D(rate=0.10))
    model.add(layers.Conv2D(256, kernel_size=(3, 3), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.SpatialDropout2D(rate=0.10))
    model.add(layers.MaxPool2D((2, 2), padding='same'))
    model.add(Dropout(.10))

    model.add(layers.Flatten())

    model.add(layers.Dense(512))
    model.add(tf.keras.layers.ReLU())
    model.add(Dropout(.5))
    model.add(layers.Dense(512))
    model.add(tf.keras.layers.ReLU())
    model.add(Dropout(.5))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adamax',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(patches_train_images,
                        patches_train_labels,
                        batch_size=32,
                        epochs=NUM_EPOCHS,
                        validation_data=(patches_validation_images, patches_validation_labels))

    test_loss, test_acc = model.evaluate(patches_test_images, patches_test_labels)

    plt.plot(history.history['accuracy'], 'g', label="accuracy on train set")
    plt.plot(history.history['val_accuracy'], 'r', label="accuracy on validation set")
    plt.hlines(test_acc, 0, NUM_EPOCHS, label="accuracy on test set")
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('CNN_with_border_' + str(border_size) + '.png')
    plt.show()

    training_test_predicted_labels = model.predict(patches_test_images)
    unpatched_labels = create_submission_groundtruth.unpatch_labels(training_test_predicted_labels,
                                                                    test_images.shape[0],
                                                                    img_shape)
    create_submission_groundtruth.save_labels(unpatched_labels,
                                              "../../data/training_test/data_augmented_predicted_labels/")

    return model, test_loss, test_acc


def main():
    start = time.time()

    data_dir = '../../data/'
    training_training_data_path = data_dir + 'training_training/data_augmented'
    training_training_labels_path = data_dir + 'training_training/data_augmented_groundtruth'
    training_validation_data_path = data_dir + 'training_validation/data_augmented'
    training_validation_labels_path = data_dir + 'training_validation/data_augmented_groundtruth'
    training_test_data_path = data_dir + 'training_test/data_augmented'
    training_test__labels_path = data_dir + 'training_test/data_augmented_groundtruth'

    train_images = extract_images(training_training_data_path)
    validation_images = extract_images(training_validation_data_path)
    test_images = extract_images(training_test_data_path)
    train_labels = extract_labels(training_training_labels_path)
    validation_labels = extract_labels(training_validation_labels_path)
    test_labels = extract_labels(training_test__labels_path)

    train_model(train_images, validation_images, test_images, train_labels, validation_labels, test_labels)

    end = time.time()
    print("Computation time: ", end - start)


if __name__ == '__main__':
    main()
