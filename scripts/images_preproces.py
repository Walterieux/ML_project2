# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 18:09:52 2020

@author: jeang
"""
import numpy as np 
from scipy.ndimage import filters,zoom
from PIL import Image,ImageOps
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
    for i,image in enumerate(data_images):
        for j in range(8):
            rotate_img=image.rotate((j+1)*45)
            if ((j+1)*45) % 90 != 0 :
                rotate_img = rotate_img.crop((60, 60, 340, 340)) 
                rotate_img = rotate_img.resize((400, 400)) 
            save_img(filename,rotate_img,i*8+j+1)
def mirror_images(filename,data_images):
    """ @input: filename: name of the folder where the images will bestored 
            data_images: list of images
    @output: store the mirror images in the folder 
"""
    for i,image in enumerate(data_images):
        save_img(filename,ImageOps.mirror(image),800+i+1)
def edges_images(filename,data_images):
    """ @input: filename: name of the folder where the images will bestored 
            data_images: list of images that we will get the edges
    @output: images edges in the folder
"""

    for j,image in enumerate(data_images):
        image=ImageOps.grayscale(image)
        image = np.array( image, dtype='uint32' )
        imx = filters.sobel(image,1,cval=0.0)  # axis 1 is x
        imy = filters.sobel(image,0, cval=0.0) # axis 0 is y
        magnitude = np.sqrt(imx**2+imy**2)
        save_img(filename,np.where(magnitude>=0.16*np.max(magnitude),1,0),j+1)
        
def rgb_mean_value_road(images_groundtruth,images): 
    """input: @images_groundtruth: list of the groundtruth images 
      @images: list of images 
    output: return the mean RGB value of the road 
    """
    number_road_pixels=number_of_road(images_groundtruth)
    mean_gray=np.zeros(3)
    for  i in range(len(images)):
        image=np.array( images[i], dtype='uint32' )
        image_gray=np.array(images_groundtruth[i], dtype= 'uint32')
        for j in range(3):
            mean_gray[j]+=np.sum(image[image_gray!=0,j])/number_road_pixels
    return mean_gray
            
        
def number_of_road(images_groundtruth): 
    """input: @images_groundtruth : list of images of the groundtruth
       output: return the number of pixels that are road segments in the training set 
   """
    counter=0
    for image in images_groundtruth: 
        image =np.array( image, dtype='uint8' )
        counter+=np.count_nonzero(image)
    return counter
def distance_matrix(filename,data_images,mean_value_gray):
    """ @input: filename: name of the folder where the images will bestored 
            data_images: list of images that we will get the edges
    @output: images edges in the folder
    """ 
    print("len in dst matrix :" , len(data_images))
    for j,image in enumerate(data_images):
        image = np.array( image, dtype='uint8' )
        norm_image = np.linalg.norm(image - mean_value_gray,axis=2)        
        image_to_save=np.where(norm_image<=0.1*np.max(norm_image),1,0)
        for i in range(1,399):
            for k in range(1,399):
                if (image_to_save[i-1,k]==1 and image_to_save[i+1,k]==1 and image_to_save[i,k+1]==1 and image_to_save[i,k-1]==1) : 
                    image_to_save[i,k]=1
        print(j)
        save_img(filename,image_to_save,j+1)

def save_img(filename,image,number):
    """ @input: filename: name of the folder where the images will bestored 
            image: image that will be stored (array)
            number: the image will be stored with filename  filename + imageid + ".png"
    @output: image  in the folder
"""
    imageid = "satImage_%.3d" % number
    image_filename = filename + imageid + ".png"
    if np.array_equal(image, image.astype(bool)):
        imageio.imwrite(image_filename,(image*255).astype(np.uint8))
    else:
        imageio.imwrite(image_filename, image)


data_dir = '../data/'
train_data_filename_rotated = data_dir + 'training/images_rotated/'
train_data_filename_rotated_groundtruth = data_dir + 'training/images_rotated_groundtruth/'
train_data_filename_edges_rotated = data_dir + 'training/images_edges_rotated/'
train_data_distance=data_dir + 'training/images_data_distance_rotated/'

data = read_images(train_data_filename_rotated, 1600)
#data_groundtruth = read_images(train_data_filename_rotated_groundtruth,800)


mean_value_gray=np.array([86.40274733, 84.48545069, 75.52089947])
#mean_value_gray=(rgb_mean_value_road(data_groundtruh,data))

#distance_matrix(train_data_distance,data,mean_value_gray)
edges_images(train_data_filename_edges_rotated,data)