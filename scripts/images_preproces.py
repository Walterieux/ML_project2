# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 18:09:52 2020

@author: jeang
"""

import numpy as np 
from scipy.ndimage import filters,zoom
from PIL import Image 
import os
import imageio

def read_images(filename, num_images):
    """ @input: filename: name of the folder where the images are stored 
            num_images: number of images stored in the folder 
    @output: list of images stored in the folder
"""
    imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            img = Image.open(image_filename)
            imgs.append(img)
    return imgs



def rotate_images(filename,data_images): 
    """ @input: filename: name of the folder where the images will bestored 
            data_images: list of images that will be rotated of [45,90,135..,360] degrees 
    @output: images rotated stored in the folder
"""
    number=0
    for i,image in enumerate(data_images):
        for j in range(8):
            rotate_img=image.rotate((j+1)*45)
            if ((j+1)*45) % 90 != 0 :
                rotate_img = rotate_img.crop((60, 60, 340, 340)) 
                rotate_img = rotate_img.resize((400, 400)) 
            save_img(filename,rotate_img,number+j+1)
        number+=8
def edges_images(filename,data_images):
    """ @input: filename: name of the folder where the images will bestored 
            data_images: list of images that we will get the edges
    @output: images edges in the folder
"""

    for j,image in enumerate(data_images):
        image=np.array( image, dtype='uint32' )
        filters.sobel(image,1,imx,cval=0.0)  # axis 1 is x
        filters.sobel(image,0,imy, cval=0.0) # axis 0 is y
        magnitude = np.sqrt(imx**2+imy**2)
        save_img(filename,np.where(magnitude>=0.16*np.max(magnitude),255,0),j+1)
        

def save_img(filename,image,number):
    """ @input: filename: name of the folder where the images will bestored 
            image: image that will be stored (array)
            number: the image will be stored with filename  filename + imageid + ".png"
    @output: image  in the folder
"""
    imageid = "satImage_%.3d" % number
    image_filename = filename + imageid + ".png"
    imageio.imwrite(image_filename, image)


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
data_rotated = read_images(train_data_filename_rotated,800)

data_groundtruh = read_images(train_labels_filename,TRAINING_SIZE)

rotate_images(train_data_filename_rotated_groundtruth,data_groundtruh)

edges_images(train_data_filename_edges,data)
edges_images(train_data_filename_edges_rotated,data_rotated)