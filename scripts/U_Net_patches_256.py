import os
import subprocess
import sys

import numpy as np
import tensorflow as tf
import ipykernel
import matplotlib.pyplot as plt
import time
import imageio
import glob
from PIL import Image
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split, KFold

from keras import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout, UpSampling2D, \
    BatchNormalization, ReLU

import create_submission_groundtruth
from images_preproces import center
from patches import create_patches, get_output_from_patches

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

patch_shape = (256, 256, 3)
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
        # img = normalize_image(img)
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

    # shrink data size
    """
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
    """

    # create mini_patches
    patches_train_images = create_patches(train_images, patch_shape)
    patches_test_images = create_patches(test_images, patch_shape)
    patches_train_labels = create_patches(train_labels, (patch_shape[0], patch_shape[1]))
    patches_train_labels = patches_train_labels[:, :, :, None]
    patches_test_labels = create_patches(test_labels, (patch_shape[0], patch_shape[1]))
    patches_test_labels = patches_test_labels[:, :, :, None]

    print("patches_train_images shape: ", patches_train_images.shape)
    print("patches_train_labels shape: ", patches_train_labels.shape)

    # https://github.com/ArkaJU/U-Net-Satellite/blob/master/U-Net.ipynb
    def build_model(input_size=(256, 256, 3)):
        inputs = Input(input_size)

        conv1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = ReLU()(conv1)
        conv1 = Dropout(.10)(conv1)
        conv1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = ReLU()(conv1)
        conv1 = Dropout(.10)(conv1)

        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        pool1 = Dropout(.10)(pool1)

        conv2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = ReLU()(conv2)
        conv2 = Dropout(.10)(conv2)
        conv2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = ReLU()(conv2)
        conv2 = Dropout(.10)(conv2)

        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        pool2 = Dropout(.10)(pool2)

        conv3 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = ReLU()(conv3)
        conv3 = Dropout(.10)(conv3)
        conv3 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = ReLU()(conv3)
        conv3 = Dropout(.10)(conv3)

        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        pool3 = Dropout(.10)(pool3)

        conv4 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = ReLU()(conv4)
        conv4 = Dropout(.10)(conv4)
        conv4 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = ReLU()(conv4)

        drop4 = Dropout(0.5)(conv4)

        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

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

        model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

        return model

    model = build_model(input_size=patch_shape)

    history = model.fit(patches_train_images,
                        patches_train_labels,
                        batch_size=2,
                        epochs=NUM_EPOCHS,
                        validation_data=(patches_test_images, patches_test_labels))

    plt.plot(history.history['accuracy'], 'g', label="accuracy on train set")
    plt.plot(history.history['val_accuracy'], 'r', label="accuracy on validation set")
    plt.grid(True)
    plt.title('Training Accuracy vs. Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    plt.savefig('U_Net_patches_256.png')

    training_test_predicted_labels = model.predict(patches_test_images)
    unpatched_labels = get_output_from_patches(training_test_predicted_labels, img_shape)
    create_submission_groundtruth.save_labels(unpatched_labels,
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
