# -*- coding: utf-8 -*-
"""
======================================================
======================================================
DESCRIPTION:
This script contains code that is useful for creating
and using the patches, described in our report (PATCHES).
We use the patchify library to split images into patches
and put them back together after training/prediction.
======================================================
======================================================
"""

from patchify import patchify, unpatchify
import numpy as np

training_size = 400
test_size = 608
img_patch_size = 256


def create_patches(data, patch_shape):
    """
    separate image into patches, data is a collection of images
    """

    imgs = []

    if data[0].shape[0] == test_size:
        step_length = (test_size - patch_shape[0]) // 2  # 176
    else:
        step_length = (training_size - patch_shape[0])

    for i in range(data.shape[0]):
        if len(patch_shape) == 3:  # RGB images
            patches = patchify(data[i], patch_shape, step=step_length)
            patches = patches.reshape((-1, patch_shape[0], patch_shape[1], patch_shape[2]))
            imgs.extend(patches)
        else:
            patches = patchify(data[i], patch_shape, step=step_length)
            patches = patches.reshape((-1, patch_shape[0], patch_shape[1]))
            imgs.extend(patches)

    return np.asarray(imgs)


def get_output_from_patches(patches_list, output_shape):
    """
    reconstructs an image given its patches

    @patches_list: array like [nb_images * number_patches, img_patch_size, img_patch_size ] :
    9 for test and 4 for training
    @output_shape : array_like :[training_size, training_size] or [test_size, test_size] depending
    of which kind of image you create
    """

    if output_shape[0] == training_size:
        nb_matrix_by_row = 2
    else:
        nb_matrix_by_row = 3
    reconstructed_images = []
    nb_elem_by_patch = nb_matrix_by_row ** 2
    for i in range(patches_list.shape[0] // nb_elem_by_patch):
        reconstructed_image = unpatchify(
            patches_list[i * nb_elem_by_patch: (i + 1) * nb_elem_by_patch].reshape(nb_matrix_by_row, nb_matrix_by_row,
                                                                                   img_patch_size, img_patch_size),
            output_shape)
        reconstructed_images.extend(reconstructed_image)

    return reconstructed_images


def get_output_from_patches_with_mean(patches_list, output_shape):
    """
    reconstructs an image given its patches, patches may overlap, in which case we average the overlapping values

    @patches_list: array like [nb_images * number_patches, img_patch_size, img_patch_size ] :
    9 for test and 4 for training
    @output_shape : array_like :[training_size, training_size] or [test_size, test_size] depending
    of which kind of image you create
    """

    if output_shape[0] == training_size:
        nb_matrix_by_row = 2
        step_length = (training_size - img_patch_size)

    else:
        nb_matrix_by_row = 3
        step_length = (test_size - img_patch_size) // 2

    nb_elem_by_patch = nb_matrix_by_row ** 2
    images = []
    for number in range(patches_list.shape[0] // nb_elem_by_patch):
        reconstructed_images = np.zeros(output_shape)
        nb_elem_images = np.zeros(output_shape)
        for i in range(nb_matrix_by_row):
            for j in range(nb_matrix_by_row):
                reconstructed_images[i * step_length: i * step_length + img_patch_size,
                j * step_length: j * step_length + img_patch_size] += patches_list[
                    number * nb_elem_by_patch + i * nb_matrix_by_row + j]
                nb_elem_images[i * step_length: i * step_length + img_patch_size,
                j * step_length: j * step_length + img_patch_size] += 1
        reconstructed_images = np.divide(reconstructed_images, nb_elem_images)
        images.extend([reconstructed_images])

    return images
