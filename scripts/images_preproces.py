# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 18:09:52 2020

@author: jeang
"""
from scipy.ndimage import rotate
from scipy.misc import imread, imshow
import scipy.misc
from CNN_training import * 
import numpy as np 
from scipy.ndimage import filters,zoom
from PIL import Image 
import imageio
""" @input: -filename: name of the directory where the images are stored 
            -num_images: number of images in the directory
    @output: list of 3-d array (RGB) of images 
"""
def read_images(filename, num_images):
    
    imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            img = Image.open(image_filename)
            imgs.append(img)
    return imgs

""" @input : -filename : name of the directory where the images should be stored 
             -data_images: array of images that will be rotated from [45,90,135,..360] degrees
    @output: store the rotated images in the directory filename
"""
def rotate_images(filename,data_images): 
    number=0
    for i,image in enumerate(data_images):
        for j in range(8):
            rotate_img_1 = image.rotate( (j+1)*45)
            if ((j+1)*45) % 90 != 0 :
                rotate_img_1 = rotate_img_1.crop(( 60, 60, 340, 340))  
                rotate_img_1 = rotate_img_1.resize((400, 400) ) 

            save_img(filename,rotate_img_1,number+j+1)
        number+=8
""" @input : -filename : name of the directory where the images should be stored 
             -data_images: array of images 
    @output: store the edges of the images in the directory filename
"""
def edges_images(filename,data_images):
    for j,image in enumerate(data_images):
        image=np.array( image, dtype='uint32' )
        imx = np.zeros(image.shape)
        imy = np.zeros(image.shape)
        filters.sobel(image,1,imx,cval=0.0)  # axis 1 is x
        filters.sobel(image,0,imy, cval=0.0) # axis 0 is y
        magnitude = np.sqrt(imx**2+imy**2)
        save_img(filename,np.where(magnitude>=0.16*np.max(magnitude),255,0),j+1)
    
""" @input : -filename : name of the directory where the image should be stored 
             -image: image object
             -number: the image will be stored in the directory with filemane  "satImage_%.3d" % number
"""
def save_img(filename,image,number):
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
#rotate_images(train_data_filename_rotated,data)
data_rotated = read_images(train_data_filename_rotated,800)
print(len(data_rotated))
data_groundtruh = read_images(train_labels_filename,TRAINING_SIZE)
#rotate_images(train_data_filename_rotated_groundtruth,data_groundtruh)


edges_images(train_data_filename_edges,data)
edges_images(train_data_filename_edges_rotated,data_rotated)