# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 10:22:08 2020

@author: jeang
"""
from convolution import extract_images_test, extract_images
from patchify import patchify, unpatchify
import numpy as np
training_size = 400
test_size = 608
img_patch_size = 256
def create_patches(data, patch_shape):
    """separate image into patches, data is a collection of images"""

    
    imgs = []
    
    if data[0].shape[0] == test_size: 
        step_length = (test_size - patch_shape[0])//2 #176
    else: 
        step_length = (training_size-patch_shape[0])
        
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
    """ patches array like [nb_images * number_patches, img_patch_size, img_patch_size ] : 9 for test and 4 for training
        output_shape : array_like :[training_size, training_size] or [test_size, test_size] depending 
        of which kind of image you create
    ------------------------output ----------------------------------
    return initial image """
    if output_shape[0] == training_size: 
        nb_matrix_by_row = 2
    else: 
        nb_matrix_by_row = 3
    reconstructed_images = []
    nb_elem_by_patch = nb_matrix_by_row**2
    for i in range(patches_list.shape[0]//nb_elem_by_patch):
        reconstructed_image = unpatchify(patches_list[i*nb_elem_by_patch: (i+1)*nb_elem_by_patch ].reshape(nb_matrix_by_row, nb_matrix_by_row, img_patch_size, img_patch_size), output_shape)
        reconstructed_images.extend(reconstructed_image)
        
    return reconstructed_images

    

    
    


    
    
data_dir = '../data/'
test_dir = data_dir + 'test_set_labels/'
images = extract_images(test_dir)
correct_labels = data_dir + 'correct_labels/'
original_img = data_dir + 'test_set_images/'
filename_comparaison = data_dir + 'comparaisons/'
train_augmented = data_dir + 'training/data_augmented/'
groundtruth = data_dir + 'training/groundtruth/'
list_groundtruth = extract_images(groundtruth, divide_by255=False)
#list_augmented = extract_images(train_augmented, divide_by255=True)
#original_images = extract_images_test(original_img, 50)

#patches_train_images = create_patches(original_images, (img_patch_size, img_patch_size, 3))

#patches_train_images_groundtruth = create_patches(list_groundtruth, (img_patch_size, img_patch_size))
#print( np.shape(get_output_from_patches(patches_train_images_groundtruth, (training_size, training_size))))


#print(np.shape(patches_train_images))
