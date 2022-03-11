import random
import numpy as np
import art.attacks.evasion as attacks
import os
import h5py
import art
import tensorflow as tf
import cv2
import time

from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from config import num_classes, DATASET
from art.utils import load_mnist

mnist = keras.datasets.mnist
K = keras.backend

batch_size = 128
epochs = 5

img_rows, img_cols = 28, 28

(_, _), (x_test_F, y_test_F), min_pixel_value, max_pixel_value = load_mnist()
x_test = x_test_F[:500]
y_test = y_test_F[:500]

tf.compat.v1.disable_eager_execution()
print("Retrieving data...")

x_train = []
y_train = []
print("Adding Adversarial ToolBox results to robust model...")
runs_directory = "toolbox_runs"
runs = os.listdir(runs_directory)
countD = -1
for i in runs:
    #print("Next run...")
    files = os.listdir(runs_directory+"/"+i)
    countD += 1
    if countD == 2:
        count = 0
        for j in files:
            im = cv2.imread(f"{runs_directory}/{i}/{j}",0)
            x_train.append(im)
            y_train.append(y_test[count])
            count += 1
        break

x_train = np.asarray(x_train)



x_train = np.expand_dims(np.asarray(x_train),-1)
y_train = np.asarray(y_train)

print("Ready to train with new data...")

#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

pathModel = "cnnClassifierTuned2.h5"
model = keras.models.load_model(f"models/{pathModel}")

# MODEL TRAINING
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# MODEL EVALUATION
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('models/cnnClassifierTunedAdvanced.h5')
print("Model saved!")
