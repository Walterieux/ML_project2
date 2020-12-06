# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 15:01:43 2020

@author: jeang
"""

from convolution import submission_convolution, extract_blocks, reshape_higher_dim, extract_images
import numpy as np
from sklearn.metrics import accuracy_score, sklearn.metrics.f1_score


def calculate_accuracy_and_f1_score(label_images, correct_labels) :
    """input: @label_images : list of imagse containing the labeled images reconstructed after the CNN and convolution
              @correct_labels : list of images containing the true labels.
       output : return the total accuracy and f1 score in a list (for each picture) 
                #also print the average f1 score and accuracy
    """
    #size of each patch 
    patch_size = (16,16)
    #initialisation of outputs
    accuracies = np.zeros(label_images.shape[0])
    f1_scores  = np.zeros()
    for number, image in enumerate(correct_labels):
        # simply transform the groundtruth data such that a block is considerer as road if more than 128 pixels are
        #road else considered as not a road.
        reshaped = extract_blocks(image, patch_size, keep_as_view=True)
        summed = np.mean(reshaped, axis=(2,3))
        summed = np.where(summed >= 128, 1, 0)
        correct_patch = reshape_higher_dim(summed , patch_size, image.shape)
        #convert 2d image array to 1d 
        correct_patch_1d = correct_patch.ravel()
        label_image_1d = label_images[0].ravel()
        #calculate the metrics for each image
        accuracies[number] = accuracy_score(correct_patch_1d, label_image_1d)
        f1_scores[number] = f1_score(correct_patch_1d, label_image_1d)
    print("the total accuracy is " , np.mean(accuracies) , "and the f1 score associated is given by", np.mean(f1_scores))
    return accuracies,f1


data_dir = '../data/'
test_groundtruth = data_dir + 'training_test/data_augmented_groundtruth/'
test_CNN = data_dir + 'training_test/predicted_labels/'
list_groundtruth = extract_images (test_groundtruth)
list_images_CNN = extract_images(test_CNN)
test_CNN_binary = data_dir + 'training_test/labels_binary/'
submission_convolution(test_CNN_binary, list_images_CNN, None, None, comparaison=False)
list_images_CNN_binary = extract_images (test_CNN_binary)
calculate_accuracy_and_f1_score(list_images_CNN_binary, list_groundtruth) 
