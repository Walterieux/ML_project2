# -*- coding: utf-8 -*-
# TODO add a description
import glob
import os

import numpy as np
from scipy import signal
from Post_Processing.utility import extract_blocks, reshape_higher_dim, save_comparison, save_img, extract_images


def submission_convolution(filename, image_list, filename_comparison, comparison=False):
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
        if comparison:
            save_comparison(image, correct_patch, filename_comparison, number + 1)
        save_img(filename, correct_patch, number + 1)


def main():
    data_dir = '../../data/'
    test_dir = data_dir + 'test_set_labels/'
    # images = extract_images(test_dir)
    correct_labels = data_dir + 'correct_labels/'
    original_img = data_dir + 'test_set_images/'
    filename_comparaison = data_dir + 'comparaisons/'
    train_augmented = data_dir + 'training/data_augmented/'
    groundtruth = data_dir + 'training/groundtruth/'
    eff_convol = data_dir + 'effect_of_convolution/'
    image_list = extract_images(data_dir)

    submission_convolution(eff_convol, image_list, eff_convol, comparison=True)


if __name__ == '__main__':
    main()
