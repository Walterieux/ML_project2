# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 18:09:52 2020

@author: jeang
"""
from scipy.ndimage import rotate
from scipy.misc import imread, imshow
import scipy.misc
#from cnn_training import * 
import numpy as np 
from scipy.ndimage import filters,zoom
from PIL import Image 
import os
def read_images(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = Image.open(image_filename)
            img = scipy.misc.imread(image_filename)
            imgs.append(img)
    return imgs
def rotate_images(filename,data_images): 
    number=0
    for i,image in enumerate(data_images):
        for j in range(8):
            rotate_img_1=rotate(image,(j+1)*45)
            #rotate_img_1 = image.rotate( (j+1)*45)
            #if ((j+1)*45) % 90 != 0 :
                #height=400
                #print(np.shape(rotate_img_1))
                #left = 4
                #top = height / 5
                #right = 154
                #bottom = 3 * height / 5
  
# Cropped image of above dimension  
# (It will not change orginal image)  
                #rotate_img_1 = rotate_img_1.crop((left, top, right, bottom)) 
                #newsize = (400, 400) 
                #rotate_img_1 = rotate_img_1.resize(newsize) 
                #print("after" ,np.shape(rotate_img_1))
            save_img(filename,rotate_img_1,number+j)
        number+=8

def edges_images(filename,data_images):
    for j,image in enumerate(data_images):
        imx = np.zeros(image.shape)
        imy = np.zeros(image.shape)
        filters.sobel(image,1,imx,cval=0.0)  # axis 1 is x
        filters.sobel(image,0,imy, cval=0.0) # axis 0 is y
        magnitude = np.sqrt(imx**2+imy**2)
        save_img(filename,np.where(magnitude>=0.16*np.max(magnitude),255,0),j)
        

def save_img(filename,image,number):
    imageid = "satImage_%.3d" % number
    image_filename = filename + imageid + ".png"
    scipy.misc.imsave(image_filename, image)


data_dir = '../data/'
train_data_filename = data_dir + 'training/images/'
train_labels_filename = data_dir + 'training/groundtruth/'
TRAINING_SIZE = 100
train_labels_filename = data_dir + 'training/groundtruth/'
train_data_filename_rotated = data_dir + 'training/images_rotated/'
train_data_filename_rotated_groundtruth = data_dir + 'training/images_rotated_groundtruth/'

train_data_filename_edges = data_dir + 'training/images_edges/'
train_data_filename_edges_rotated = data_dir + 'training/images_edges_rotated/'


data = read_images(train_data_filename, TRAINING_SIZE)
rotate_images(train_data_filename_rotated,data)
data_rotated = read_images(train_data_filename_rotated,300)

data_groundtruh = read_images(train_labels_filename,TRAINING_SIZE)
rotate_images(train_data_filename_rotated_groundtruth,data_groundtruh)

rotate_images(train_data_filename_rotated,data)

edges_images(train_data_filename_edges,data)
edges_images(train_data_filename_edges_rotated,data_rotated)