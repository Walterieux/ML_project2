# -*- coding: utf-8 -*-
"""
======================================================
======================================================
DESCRIPTION:
This script contains the methods and functions we used
to load, enhance and save our data set. It contains the
various data pre-processing techniques mentioned in our
report, in section III A.
======================================================
======================================================
"""

import csv

import numpy as np
from scipy.ndimage import filters, zoom
from PIL import Image, ImageOps
import os
import imageio


def change_color(img):
    """
    creates an image where the white pixels become red and black become gray
    """

    datas = img.getdata()

    new_image_data = []
    red_value = (255, 0, 0)
    gray_value = (100, 100, 100)
    for item in datas:
        # change all white (also shades of whites) pixels to red
        if item[0] in list(range(190, 256)):
            new_image_data.append(red_value)
        else:
            #
            new_image_data.append(gray_value)

    # update image data
    img.putdata(new_image_data)
    return img


def overwrite_with_images(filename, list_of_test, list_of_labels):
    """
    creates images which overlay our prediction and the original image
    """
    for number, image_test in enumerate(list_of_test):
        # convert L to RGB
        image_label = list_of_labels[number].convert("RGB")
        # change colors
        image_label = change_color(image_label)
        # add alpha channel in order to have transparency
        image_test.putalpha(1)
        image_label.putalpha(1)

        # compose images together
        alphaComposited = Image.alpha_composite(image_label, image_test)
        # save image
        save_img(filename, alphaComposited, number + 1, PIL=True)


def center_by_image(list_of_image):
    """
    centers a given list of images, as a numpy array
    centers w.r.t one image
    """

    centered_image = np.zeros(list_of_image.shape)
    for number, image in enumerate(list_of_image):
        mean = np.mean(image, axis=(0, 1))
        std = np.std(image, axis=(0, 1))
        centered_image[number] = (image - mean) / std
    return centered_image


def center(list_of_image, mean=None, sigma=None, still_to_center=True):
    """
    centers a given list of images, as a numpy array
    centers w.r.t all images
    """

    if still_to_center:
        sigma = np.std(list_of_image, axis=(0, 1, 2))
        mean = np.mean(list_of_image, axis=(0, 1, 2))

    return (list_of_image - mean) / sigma, mean, sigma


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
            img = Image.open(image_filename)
            imgs.append(img)
    return imgs


def read_images(filename, num_images):
    """
    reads the given images into an array

    @filename: name of the folder where the images are stored
    @num_images: number of images stored in the folder
    """

    imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            img = Image.open(image_filename)
            imgs.append(img)
    return imgs


def rotate_images(filename, data_images):
    """
    rotates the given images and stores them

    @filename: name of the folder where the images will be stored
    @data_images: list of images that will be rotated of [45,90,135..,360] degrees
    """

    for i, image in enumerate(data_images):
        for j in range(8):
            rotate_img = image.rotate((j + 1) * 45)
            if ((j + 1) * 45) % 90 != 0:
                rotate_img = rotate_img.crop((60, 60, 340, 340))
                rotate_img = rotate_img.resize((400, 400))
            save_img(filename, rotate_img, i * 8 + j + 1)


def mirror_images(filename, data_images):
    """
    mirrors the given images along the vertical axis and stores them

    @filename: name of the folder where the images will be stored
    @data_images: list of images to be mirrored
    """

    for i, image in enumerate(data_images):
        save_img(filename, ImageOps.mirror(image), 800 + i + 1)


def edges_images(filename, data_images):
    """
    applies a Sobel filter on the given images and stores the results

    @filename: name of the folder where the resulting images will be stored
    @data_images: list of images to be treated
    """

    for j, image in enumerate(data_images):
        image = ImageOps.grayscale(image)
        image = np.array(image, dtype="uint32")
        imx = np.zeros(np.shape(image))
        imy = np.zeros(np.shape(image))
        filters.sobel(image, 1, output=imx, cval=0.0)  # axis 1 is x
        filters.sobel(image, 0, output=imy, cval=0.0)  # axis 0 is y
        magnitude = np.sqrt(imx ** 2 + imy ** 2)
        save_img(filename, magnitude, j + 1)


def get_mean_gray_value(images_data, images_groundtruth):
    """
    computes the average color of a known road

    @images_data  list of images
    @images_ground: list of corresponding groundtruths
    """

    mean_value_gray = np.zeros(3)
    for i, image in enumerate(images_data):
        image = np.array(image)
        for j in range(3):
            mean_value_gray[j] += 1 / 100 * np.mean(image[images_groundtruth[i] != 0, j])

    return mean_value_gray


def distance_image(filename, images_data, mean_value_gray):
    """
    computes the distance of an image (for more details, see report, [DISTANCE])

    @filename: where the distance images will be stored
    @images_data  list of images
    @mean_value_gray: known average color of a road
    """

    for j, image in enumerate(images_data):
        image = np.array(image)
        norm_image = np.linalg.norm(image - mean_value_gray, axis=2)
        save_img(filename, norm_image, j + 1)


def save_img(filename, image, number, PIL=False):
    """
    saves an image, that can be given either as a pillow image (PIL) or a numpy array
    """

    imageid = "satImage_%.3d" % number
    image_filename = filename + imageid + ".png"
    image_filename_bmp = filename + imageid + ".bmp"
    if PIL:
        image.save(image_filename_bmp)

    else:
        if np.array_equal(image, image.astype(bool)):
            imageio.imwrite(image_filename, (image * 255).astype(np.uint8))
        else:
            imageio.imwrite(image_filename, image.astype(np.uint8))


data_dir = '../../data/'
train_data_filename = data_dir + 'training/images/'
train_labels_filename = data_dir + 'training/groundtruth/'
train_augmented = data_dir + 'training/data_augmented/'
train_data_norm = data_dir + 'training/data_augmented_norm/'
train_data_filename_edges = data_dir + 'training/data_augmented_edges/'

test_labels_filename = data_dir + 'test_set_labels/'

test_images_filename = data_dir + 'test_set_images/'
alpha_images_test = data_dir + 'alpha_composite/'

test_images = extract_images_test(test_images_filename, 50)

test_labels = read_images(test_labels_filename, 50)

overwrite_with_images(alpha_images_test, test_images, test_labels)

TRAINING_SIZE = 1600
