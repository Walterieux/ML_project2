import os
import subprocess
import sys

import numpy as np
import tensorflow as tf
import imageio
import glob
from PIL import Image

from tensorflow.keras import datasets, layers, models


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def extract_image(image_path):
    """
    Extract all images from 'image_path'
    All values are between 0 and 1
    """

    imgs = []
    for img_path in glob.glob(image_path + "/*.png"):
        img = imageio.imread(img_path)
        img = img / 255
        imgs.append(img)

    return np.asarray(imgs)


def extract_labels(label_path):
    """Extract all labels from 'label_path'"""

    imgs = []
    for img_path in glob.glob(label_path + "/*.png"):
        img = imageio.imread(img_path)
        imgs.append(img)

    return np.asarray(imgs)


data_dir = '../data/'
train_data_filename = data_dir + 'training/images/'
train_labels_filename = data_dir + 'training/groundtruth/'

images = extract_image(train_data_filename)

labels = extract_labels(train_labels_filename)
labels = labels.reshape(labels.shape[0], -1)

train_images = images[:90]
test_images = images[90:]

train_labels = labels[:90]
test_labels = labels[90:]

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(400, 400, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='sigmoid'))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='hard_sigmoid'))  # Change softmax
model.add(layers.Dense(160000))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['binary_accuracy'])

history = model.fit(train_images, train_labels, epochs=2,
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print("Accuracy = ", test_acc)

# acc = history.history['accuracy']
# print(acc)
