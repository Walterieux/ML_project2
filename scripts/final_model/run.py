# -*- coding: utf-8 -*-
"""
======================================================
======================================================
DESCRIPTION:
This script contains the code for running and evaluating
the training of our final, best performing model. Its
precise description can be found in the report. Note that
the parameters used in this script are designed to be run
on a beefy GPU, with quite some RAM available. The basis
U-Net model this implementation is built on can be found here:
https://www.kaggle.com/phoenigs/u-net-dropout-augmentation-stratification
======================================================
======================================================
"""

import numpy as np
import ipykernel
import matplotlib.pyplot as plt
import time
import imageio
import glob
from PIL import Image

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Dropout, UpSampling2D, \
    BatchNormalization, ReLU, SpatialDropout2D

from Post_Processing.create_submission_groundtruth import extract_test_images, save_labels
from Post_Processing.mask_to_submission import masks_to_submission
from Pre_Processing import patches

"""
# Use this if GPU error during launch
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
"""

patch_shape = (256, 256, 3)
img_shape = (400, 400)
test_img_shape = (608, 608)
NUM_EPOCHS = 50


def extract_images(image_path):
    """
    Extract all images from 'image_path'
    All values are between 0 and 1
    """

    imgs = []
    for img_path in glob.glob(image_path + "/*.png"):
        img = imageio.imread(img_path)
        imgs.append(img.astype('float32'))

    return np.asarray(imgs)


def extract_labels(label_path):
    """
    Extract all labels from 'label_path'
    """

    imgs = []
    for img_path in glob.glob(label_path + "/*.png"):
        img = imageio.imread(img_path)
        img[img <= 127] = 0
        img[img > 127] = 1
        imgs.append(img.astype('uint8'))

    return np.asarray(imgs)


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


def train_model(train_images, validation_images, test_images, train_labels, validation_labels, test_labels):
    """
    Train a predefined model with the given data

    Returns the model, accuracy over the test data, loss over the test data
    """

    # create mini_patches
    patches_train_images = patches.create_patches(train_images, patch_shape)
    patches_validation_images = patches.create_patches(validation_images, patch_shape)
    patches_test_images = patches.create_patches(test_images, patch_shape)
    patches_train_labels = patches.create_patches(train_labels, (patch_shape[0], patch_shape[1]))
    patches_train_labels = patches_train_labels[:, :, :, None]
    patches_validation_labels = patches.create_patches(validation_labels, (patch_shape[0], patch_shape[1]))
    patches_validation_labels = patches_validation_labels[:, :, :, None]
    patches_test_labels = patches.create_patches(test_labels, (patch_shape[0], patch_shape[1]))
    patches_test_labels = patches_test_labels[:, :, :, None]

    def build_model(input_size=(256, 256, 3)):
        inputs = Input(input_size)

        drop_rate = 0.15

        conv1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = ReLU()(conv1)
        conv1 = SpatialDropout2D(drop_rate)(conv1)
        conv1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = ReLU()(conv1)
        conv1 = SpatialDropout2D(drop_rate)(conv1)

        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        pool1 = Dropout(drop_rate)(pool1)

        conv2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = ReLU()(conv2)
        conv2 = SpatialDropout2D(drop_rate)(conv2)
        conv2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = ReLU()(conv2)
        conv2 = SpatialDropout2D(drop_rate)(conv2)

        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        pool2 = Dropout(drop_rate)(pool2)

        conv3 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = ReLU()(conv3)
        conv3 = SpatialDropout2D(drop_rate)(conv3)
        conv3 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = ReLU()(conv3)
        conv3 = SpatialDropout2D(drop_rate)(conv3)

        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        pool3 = Dropout(drop_rate)(pool3)

        conv4 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = ReLU()(conv4)
        conv4 = SpatialDropout2D(drop_rate)(conv4)
        conv4 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = ReLU()(conv4)

        drop4 = Dropout(0.5)(conv4)

        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        pool4 = Dropout(drop_rate)(pool4)

        conv5 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = BatchNormalization()(conv5)
        conv5 = ReLU()(conv5)
        conv5 = SpatialDropout2D(drop_rate)(conv5)
        conv5 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = ReLU()(conv5)

        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))

        merge6 = concatenate([drop4, up6])

        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))

        merge7 = concatenate([conv3, up7])

        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))

        merge8 = concatenate([conv2, up8])

        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))

        merge9 = concatenate([conv1, up9])

        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(inputs=inputs, outputs=conv10)

        model.compile(optimizer='adamax', loss='binary_crossentropy', metrics=['accuracy'])

        return model

    model = build_model(input_size=patch_shape)

    history = model.fit(patches_train_images,
                        patches_train_labels,
                        batch_size=2,
                        epochs=NUM_EPOCHS,
                        validation_data=(patches_validation_images, patches_validation_labels))

    test_loss, test_acc = model.evaluate(patches_test_images, patches_test_labels)

    plt.plot(history.history['accuracy'], 'g', label="accuracy on train set")
    plt.plot(history.history['val_accuracy'], 'r', label="accuracy on validation set")
    plt.hlines(test_acc, 0, NUM_EPOCHS, label="accuracy on test set")
    plt.grid(True)
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('U_Net_patches_256.png')
    plt.show()

    # Save predictions on training_test in order to apply post_process
    training_test_predicted_labels = model.predict(patches_test_images)
    unpatched_labels = np.asarray(patches.get_output_from_patches(training_test_predicted_labels, img_shape))
    unpatched_labels = unpatched_labels.reshape((-1, *train_labels[0].shape))
    save_labels(unpatched_labels, "../../data/training_test/data_augmented_predicted_labels/")

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

    model, test_loss, test_acc = train_model(train_images, validation_images, test_images, train_labels,
                                             validation_labels, test_labels)
    # model.save("saved_model")

    # predict groundtruth for AiCrowd images
    submission_images = extract_test_images()
    submission_images_patches = patches.create_patches(submission_images, patch_shape)
    test_test_predicted_labels = model.predict(submission_images_patches)
    unpatched_labels = np.asarray(patches.get_output_from_patches(test_test_predicted_labels, test_img_shape))
    unpatched_labels = unpatched_labels.reshape((-1, *test_img_shape, 1))

    save_labels(unpatched_labels, "../../data/test_set_labels/")

    end = time.time()
    print("Computation time: ", end - start)

    submission_filename = '../submission.csv'
    image_filenames = []
    for i in range(1, 51):
        image_filename = '../../data/test_set_labels/satImage_' + '%.3d' % i + '.png'
        image_filenames.append(image_filename)
    masks_to_submission(submission_filename, *image_filenames)


if __name__ == '__main__':
    main()
