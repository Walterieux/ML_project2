import imageio
from tensorflow import keras
import numpy as np

import cnn_training

from use_borders_mathieu import create_patches_with_border

img_patch_size = 16
border_size = 16
img_shape = (608, 608)


def extract_test_images():
    """
    Extracts all test test set images as array
    """

    imgs = []
    for i in range(1, 51):
        img_path = '../data/test_set_images/test_' + str(i) + '/test_' + str(i) + '.png'
        img = imageio.imread(img_path)
        img = cnn_training.normalize_image(img)
        imgs.append(img.astype('float32'))

    return np.asarray(imgs)


def unpatch_labels(data, images_num, img_shape):
    """
    Unpatchify the given data to the original image shape
    """
    
    labels = []
    num_pat_per_img = int((img_shape[0]/img_patch_size) ** 2)
    for i in range(0, images_num):
        patches = data[num_pat_per_img*i:num_pat_per_img*(i+1)]
        patches = patches.reshape(int(img_shape[0]/img_patch_size),
                                  int(img_shape[0]/img_patch_size))
        patches = patches.repeat(img_patch_size, axis=0).repeat(img_patch_size, axis=1)
        labels.append(patches)

    return np.asarray(labels)


def save_labels(labels, path):
    """
    Saves labels in the given directory path
    """

    for i, label in enumerate(labels):
        filename = path + 'satImage_' + '%.3d' % (i + 1) + '.png'
        imageio.imwrite(filename, (label * 255).round().astype('uint8'))


def main():
    """
    It loads the already trained model to predict labels from test_set_images
    Then saves labels into folder '../data/test_set_labels/'
    """

    test_images = extract_test_images()
    patches_images = create_patches_with_border(test_images, (img_patch_size, img_patch_size, 3), border_size,
                                                     "TEST IMAGES")

    model = keras.models.load_model("saved_model")

    predictions = model.predict(patches_images)
    predicted_labels = unpatch_labels(predictions, 50, img_shape)
    save_labels(predicted_labels, "../data/test_set_labels/")


if __name__ == '__main__':
    main()
