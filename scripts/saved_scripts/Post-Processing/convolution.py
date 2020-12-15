# -*- coding: utf-8 -*-

import glob
import os

import imageio
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

output_size = 388
input_size = 572


def separate_data():
    data_dir = '../../../data/'
    data_dir_training = data_dir + 'training/'
    data_dir_training_training = data_dir + 'training_training/'
    data_dir_training_validation = data_dir + 'training_validation/'
    data_dir_training_test = data_dir + 'training_test/'
    list_of_features = ["data_augmented/", "data_augmented_distance/", "data_augmented_edges/",
                        "data_augmented_groundtruth/", "data_augmented_norm/"]
    choices = np.random.choice(np.linspace(1, 100, 100), 80, replace=False).astype(int)
    choices_not_training = np.zeros(20).astype(int)

    counter = 0
    for index in range(1, 101):
        if np.isin(index, choices) == False:
            choices_not_training[counter] = int(index)
            counter += 1
    choices_validation = choices_not_training[0:10]
    choices_test = choices_not_training[-10:]
    # choices_test = np.delete(np.linspace(1,100,100).astype(int),choices ).astype(int)

    for number, feature in enumerate(list_of_features):
        directory_to_read = data_dir_training + feature

        directory_to_training = data_dir_training_training + feature
        data_dir_test = data_dir_training_test + feature
        data_dir_val = data_dir_training_validation + feature

        imgs_train = extract_img_from_list(directory_to_read, choices)
        imgs_test = extract_img_from_list(directory_to_read, choices_test)
        imgs_val = extract_img_from_list(directory_to_read, choices_validation)

        store_list_img(directory_to_training, imgs_train)
        store_list_img(data_dir_test, imgs_test)
        store_list_img(data_dir_val, imgs_val)


def extract_img_from_list(filename, list_of_number):
    imgs = []
    for i in list_of_number:
        for j in range(8):
            for k in range(2):
                number = (i - 1) * 8 + j + 1 + k * 800
                imageid = "satImage_%.3d" % number
                image_filename = filename + imageid + ".png"

                if os.path.isfile(image_filename):
                    img = imageio.imread(image_filename)
                    imgs.append(img)
    return np.asarray(imgs)


def reshape_higher_dim(patch, patch_size, image_size):
    """ input : @patch : array like, patch of image
                @patch_size : tuple, size of patch
                @image_size : tuple, size of image
        output: return an array with size image_size which is the patch reshaped
    """

    output_matrix = np.zeros(image_size)
    for i in range(image_size[0] // patch_size[0]):
        for j in range(image_size[1] // patch_size[1]):
            output_matrix[i * patch_size[0]:(i + 1) * patch_size[0], j * patch_size[1]:(j + 1) * patch_size[1]] = patch[
                i, j]
    return output_matrix


def extract_blocks(a, blocksize, keep_as_view=False):
    """input : @matrix like
               @blocksize : size of a block
               keep_as_view : binary indicates if it is needed to reshape the matrix in 4d or in 2d
       output : return an array with size a.shape/blockwise
    """

    M, N = a.shape
    b0, b1 = blocksize
    if keep_as_view == 0:
        return a.reshape(M // b0, b0, N // b1, b1).swapaxes(1, 2).reshape(-1, b0, b1)
    else:
        return a.reshape(M // b0, b0, N // b1, b1).swapaxes(1, 2)


def extract_images_test(filename, num_images):
    """
    Extract all images from 'filename with test'
    All values are between 0 and 1
    """

    imgs = []
    for i in range(1, num_images + 1):
        imageid = "test_%.1d/" % i + "test_%.1d" % i
        image_filename = filename + imageid + ".png"

        if os.path.isfile(image_filename):
            img = imageio.imread(image_filename)
            imgs.append(img)
    return np.asarray(imgs)


def extract_images(image_path, divide_by255=True):
    """
    Extract all images from 'image_path'
    All values are between 0 and 1
    """

    imgs = []
    for img_path in glob.glob(image_path + "/*.png"):
        img = imageio.imread(img_path)
        if divide_by255:
            img = img / 255.0
        imgs.append(img.astype('float32'))

    return np.asarray(imgs)


def submission_convolution(filename, image_list, filename_comparaison, comparaison=False):
    """input : @filename : where to store the images
    #        @image_list : list of images after CNN
    #        @filename_comparaison : where to store the images compared
    #        
    #
    #output : apply convolution to and threshold to  each image of image_list to know if road or not. store the images in filename """

    # four types of convolutions  left up & left right ,up & down, left down & right up , left & right
    convolutions_4 = np.zeros((4, 3, 3))
    convolutions_4[0, :, :] = np.identity(3)
    np.put_along_axis(convolutions_4[1, :, :], np.array([[1], [1], [1]]), 1, axis=1)
    np.put_along_axis(convolutions_4[2, :, :], np.array([[2], [1], [0]]), 1, axis=1)
    convolutions_4[3, 1, :] = 1
    patch_size = (16, 16)
    for number, image in enumerate(image_list):
        reshaped = extract_blocks(image, patch_size, keep_as_view=True)
        summed = np.mean(reshaped, axis=(2, 3))
        threshold_matrix = np.where(summed >= 0.2, 1, 0)
        allconv = np.zeros((4, summed.shape[0], summed.shape[1]))
        # thresholds are lower for up & down and left & right as it is more likely to happen
        thresholds = [1.1, 1.1, 1.1, 1.1]
        for i in range(4):
            allconv[i, :, :] = signal.convolve2d(summed, convolutions_4[i, :, :], boundary='symm', mode='same')
            allconv[i, :, :] = np.where(allconv[i, :, :] >= thresholds[i], 1, 0)
            allconv[i, :, :] = np.multiply(allconv[i, :, :], threshold_matrix)

        # road_in_patch = threshold_matrix
        road_in_patch = np.where(np.sum(allconv, axis=0) >= 1, 1, 0)
        patch_convolution = signal.convolve2d(road_in_patch, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]),
                                              boundary='symm', mode='same')
        road_in_patch = np.where(patch_convolution >= 6, 1, road_in_patch)
        road_in_patch = np.where(patch_convolution == 0, 0, road_in_patch)

        correct_patch = reshape_higher_dim(road_in_patch, patch_size, image.shape)
        if comparaison:
            save_comparaison(image, correct_patch, filename_comparaison, number + 1)
        save_img(filename, correct_patch, number + 1)


def store_list_img(filename, images_list):
    for number, image in enumerate(images_list):
        save_img(filename, image, number + 1)


def save_img(filename, image, number):
    """ @input : -filename : name of the directory where the images should be stored
                 -data_images: array of images that will be rotated from [45,90,135,..360] degrees
        @output: store the rotated images in the directory filename
    """

    imageid = "satImage_%.3d" % number
    image_filename = filename + imageid + ".png"
    if (np.array_equal(image, image.astype(bool))):
        imageio.imwrite(image_filename, (image * 255).astype(np.uint8))
    else:
        imageio.imwrite(image_filename, image.astype(np.uint8))


def save_comparaison(image, correct_patch, filename_comparaison, number):
    """#input : @original image : array like image
    #        @image : array like (after CNN)
    #        @correct_patch array like after  convolution
    #        @filename_comparaison : where to store the comparaison image
    #        @number : int index
    #output : store the comparaision image with index number in filename_comparaison"""

    plt.figure()
    plt.ioff()

    plt.subplot(121)
    plt.imshow(np.where(image >= 0.25, 1, 0), cmap='gray')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.subplot(122)
    plt.imshow(correct_patch, cmap=plt.cm.gray)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    to_save = filename_comparaison + "comparaison_%1.d" % number + ".png"

    fig = plt.gcf()
    fig.savefig(to_save, dpi=400)
    plt.close(fig)


data_dir = '../../../data/'
test_dir = data_dir + 'test_set_labels/'
# images = extract_images(test_dir)
correct_labels = data_dir + 'correct_labels/'
original_img = data_dir + 'test_set_images/'
filename_comparaison = data_dir + 'comparaisons/'
train_augmented = data_dir + 'training/data_augmented/'
groundtruth = data_dir + 'training/groundtruth/'
eff_convol = data_dir + 'effect_of_convolution/'
image_list = extract_images(data_dir)
submission_convolution(eff_convol, image_list, eff_convol, comparaison=True)
# separate_data()
