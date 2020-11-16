import os
import subprocess
import sys

import numpy as np
import tensorflow as tf
import imageio
import glob

from tensorflow.keras import datasets, layers, models


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def extract_image(image_path, max_value):
    """Extract all images from 'image_path', then divide all values by max_value"""

    imgs = []
    for img_path in glob.glob(image_path + "/*.png"):
        img = imageio.imread(img_path)
        img = img / max_value
        imgs.append(img)

    return np.asarray(imgs)


data_dir = '../data/'
train_data_filename = data_dir + 'training/images/'
train_labels_filename = data_dir + 'training/groundtruth/'

images = extract_image(train_data_filename, 255)
print(images.shape)
labels = extract_image(train_labels_filename, 1)


train_images = images[:90]
test_images = images[90:]

train_labels = labels[:90]
test_labels = labels[90:]

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(400, 400, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=4,
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print(test_acc)

acc = history.history['accuracy']
print(acc)
