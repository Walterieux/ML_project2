# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 10:25:08 2020

@author: jeang
"""
import os
import numpy as np 
import imageio
from scipy import signal
import glob
from matplotlib import pyplot as plt


def separate_data() : 
    data_dir = '../data/' 
    data_dir_training = data_dir + 'training/'
    data_dir_training_training = data_dir + 'training_training/'
    data_dir_training_test = data_dir + 'training_test/'
    list_of_features = ["data_augmented/","data_augmented_distance/", "data_augmented_edges/", "data_augmented_groundtruth/", "data_augmented_norm/"]
    choices = np.random.choice(np.linspace(1,100,100),90,replace=False).astype(int)
    choices_test = np.delete(choices, np.array([1,2,5])).astype(int)
    for number, feature in enumerate(list_of_features):
        directory_to_read = data_dir_training + feature
        directory_to_training = data_dir_training_training + feature
        data_dir_test = data_dir_training_test + feature
        imgs_train = extract_img_from_list(directory_to_read,choices)
        imgs_test = extract_img_from_list(directory_to_read,choices_test)
        print(data_dir_training_test)
        store_list_img(directory_to_training,imgs_train)
        store_list_img(data_dir_test,imgs_test)
        

def extract_img_from_list(filename,list_of_number):
    imgs = []
    for i in list_of_number:
        for j in range(8):
            for k in range(2):
                number = (i-1)*8 + j + k *800
                imageid =  "satImage_%.3d" %number
                image_filename = filename + imageid + ".png"

                if os.path.isfile(image_filename):
                    img = imageio.imread(image_filename)
                    imgs.append(img)
    return np.asarray(imgs)

    return np.asarray(imgs)
def reshape_higher_dim(patch, patch_size, image_size):
    """ input : @patch : array like, patch of image 
                @patch_size : tuple, size of patch 
                @image_size : tuple, size of image 
        output: return an array with size image_size which is the patch reshaped
    """
    retour = np.zeros(image_size)
    for i in range(image_size[0]//patch_size[0]):
        for j in range(image_size[1]//patch_size[1]):
            retour[i*patch_size[0]:(i+1)*patch_size[0], j*patch_size[1]:(j+1)*patch_size[1]] = patch[i,j]
    return retour
def extract_blocks(a, blocksize, keep_as_view=False):
    M,N = a.shape
    b0, b1 = blocksize
    if keep_as_view==0:
        return a.reshape(M//b0,b0,N//b1,b1).swapaxes(1,2).reshape(-1,b0,b1)
    else:
        return a.reshape(M//b0,b0,N//b1,b1).swapaxes(1,2)

def extract_images_test(filename, num_images):
    """
    Extract all images from 'filename with test'
    All values are between 0 and 1
    """
    imgs = []
    for i in range(1, num_images + 1):
        imageid = "test_%.1d/" %i + "test_%.1d" %i
        image_filename = filename + imageid + ".png"

        if os.path.isfile(image_filename):
            img = imageio.imread(image_filename)
            imgs.append(img)
    return np.asarray(imgs)

    return np.asarray(imgs)
def extract_images(image_path):
    """
    Extract all images from 'image_path'
    All values are between 0 and 1
    """

    imgs = []
    for img_path in glob.glob(image_path + "/*.png"):
        img = imageio.imread(img_path)
        img = img / 255.0
        imgs.append(img.astype('float32'))

    return np.asarray(imgs)

def submission_convolution(filename, image_list, filename_comparaison, original_images):
    """input : @filename : where to store the images
    #        @image_list : list of images after CNN 
    #        @filename_comparaison : where to store the images compared
    #        @original_images : list of images before CNN
    #        
    #output : apply convolution to and threshold to  each image of image_list to know if road or not. store the images in filename """
    
    #four types of convolutions  left up & left right ,up & down, left down & right up , left & right 
    convolutions_4 = np.zeros((4,3,3))
    convolutions_4[0,:,:] = np.identity(3)
    np.put_along_axis(convolutions_4[1,:,:], np.array([[1],[1],[1]]), 1, axis=1)
    np.put_along_axis(convolutions_4[2,:,:], np.array([[2],[1],[0]]), 1, axis=1)
    convolutions_4[3,1,:] = 1
    
    for number,image in enumerate(image_list):
        reshaped = extract_blocks(image, (16,16), keep_as_view=True)
        summed = np.mean(reshaped, axis=(2,3))
        threshold_matrix = np.where(summed >= 0.15,1,0)
        allconv = np.zeros((4,summed.shape[0],summed.shape[1]))
        #thresholds are lower for up & down and left & right as it is more likely to happen
        thresholds=[1.3,1.2,1.3,1.2]
        for i in range(4):
            allconv[i,:,:] = signal.convolve2d(summed, convolutions_4[i,:,:], boundary='symm', mode='same')
            allconv[i,:,:] = np.where(allconv[i,:,:] >= thresholds[i], 1, 0)
            allconv[i,:,:] =np.multiply( allconv[i,:,:] , threshold_matrix)
            
        
        road_in_patch = np.where(np.sum(allconv, axis=0) >=1 , 1, 0)
        patch_convolution = signal.convolve2d(road_in_patch, np.array([[1,1,1],[1,0,1],[1,1,1]]), boundary='symm', mode='same')
        road_in_patch = np.where(patch_convolution>=6, 1, road_in_patch)
        road_in_patch = np.where(patch_convolution==0, 0, road_in_patch)
        

        correct_patch = reshape_higher_dim(road_in_patch , (16,16), image.shape)
        
        save_comparaison(original_images[number], image, correct_patch, filename_comparaison,number+1)
        save_img(filename,correct_patch , number+1)
        
def store_list_img(filename,images_list):
    for number, image in enumerate(images_list) : 
        save_img(filename,image,number+1)
def save_img(filename,image,number):
    """ @input : -filename : name of the directory where the images should be stored
                 -data_images: array of images that will be rotated from [45,90,135,..360] degrees
        @output: store the rotated images in the directory filename
    """
    imageid = "satImage_%.3d" % number
    image_filename = filename + imageid + ".png"
    if (np.array_equal(image, image.astype(bool))):
        imageio.imwrite(image_filename,(image*255).astype(np.uint8))
    else :
        imageio.imwrite(image_filename, image.astype(np.uint8))  

def save_comparaison(original_image, image, correct_patch, filename_comparaison,number):
    """#input : @original image : array like image
    #        @image : array like (after CNN)
    #        @correct_patch array like after  convolution
    #        @filename_comparaison : where to store the comparaison image
    #        @number : int index 
    #output : store the comparaision image with index number in filename_comparaison"""
    plt.figure()
    plt.ioff()
    plt.subplot(221)
    plt.imshow(original_image)
    plt.title("original",fontsize=10)
    plt.subplot(222)
    plt.imshow(image,cmap='gray')
    plt.title("after CNN",fontsize=10)
    plt.subplot(223)
    plt.imshow(np.where(image>=0.25,1,0),cmap='gray')
    plt.title("after CNN threshold",fontsize=10)
    plt.subplot(224)
    plt.imshow(correct_patch,cmap=plt.cm.gray)
    plt.title("after threshold and convolution",fontsize=10)
    tosave= filename_comparaison + "comparaison_%1.d" % number +".png"
    fig = plt.gcf()
    fig.savefig(tosave,dpi=300)
    plt.close(fig)
data_dir = '../data/'
test_dir = data_dir + 'test_set_labels/'
images = extract_images(test_dir) 
correct_labels = data_dir + 'correct_labels/'
original_img  = data_dir + 'test_set_images/'
filename_comparaison = data_dir + 'comparaisons/'



separate_data()


"""original_images = extract_images_test(original_img, 50)
images = extract_images(test_dir)
submission_convolution(correct_labels, images, filename_comparaison,original_images)"""
