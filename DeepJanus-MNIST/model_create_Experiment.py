import random
import numpy as np
import art.attacks.evasion as attacks
import os
import h5py
import art
import tensorflow as tf
import cv2
import time
import json

from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from config import num_classes, DATASET

mnist = keras.datasets.mnist
K = keras.backend

#parameters
batch_size = 5
epochs = 5

# input image dimensions, images are 28x28x1 numpy arrays with pixel values ranging from 0 to 255
img_rows, img_cols = 28, 28

# the data, 4 numpy arrays split between train and test sets
# x_train train data 60k
# y_train train label
# x_test test data 10k
# y_test test label, integer from 0 to 9
(_, _), (x_test, y_test) = mnist.load_data()
#x_train = x_train.tolist()

tf.compat.v1.disable_eager_execution()
print("Retrieving data...")

x_train = []
y_train = np.array([])
print("Adding DeepJanus results to data...")
runs_directory = "runs"
runs = os.listdir(runs_directory)
numb_adv = 0
for i in runs:
    numb_adv_run = 0
    #print("Next run...")
    files = os.listdir(runs_directory+"/"+i)
    for j in files:
        if j == "archive":
            image = os.listdir(runs_directory+"/"+i+"/"+j)
            image.sort()
            goOver = "NONE"
            label = None
            for k in image:
                # print("      Next file..."+ k)
                if "json" in k:
                    f = open(runs_directory+"/"+i+"/"+j+"/"+k)
                    jsonFile = json.load(f)
                    if jsonFile["expected_label"] == jsonFile["predicted_label"]:
                        goOver = k[:-5]
                        label = None
                    #        print("          Same labels...")
                    else:
                        goOver = "NONE"
                        label = jsonFile["expected_label"]
                #       print("          Not Same labels...")
                #  input("...")

                if "png" in k:
                    if goOver not in k:
                        #      print("          IN if Not same" + k)
                        img = cv2.imread(runs_directory+"/"+i+"/"+j+"/"+k,0)
                        img = img.tolist()
                        x_train.append(img)
                        y_train = np.append(y_train,label)
                        numb_adv_run += 1
                        numb_adv += 1
    print(f"Number of adversarial for run {i}: {numb_adv_run}")


x_train = np.asarray(x_train)
print(f"Number of adversarial by DeepJanus: {numb_adv}")


print("Ready to train with new data...")
# To be able to use the dataset in Keras API, we need 4-dims numpy arrays.
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# MODEL CONFIGURATION
# add layers to the model
pathModel = "cnnClassifier.h5"
model = keras.models.load_model(f"models/{pathModel}")
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# MODEL TRAINING
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# MODEL EVALUATION
# Compare how the trained model performs on the test dataset
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#  Exporting the entire model allows to checkpoint a model and resume training later—from the exact same state—without access to the original code.
model.save('models/cnnClassifierTuned3.h5')
print("Model saved!")
