import cv2
from tensorflow import keras
from config import MODEL, DATASET
from utils import input_reshape
import numpy as np
import h5py

model = keras.models.load_model(MODEL)
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

gtruth_set = range(len(x_train))
print("Length of train data...")
print(gtruth_set)
pred = model.predict_classes(input_reshape(x_test))
print("Model predicted on the train data...")
correct_pred = np.argwhere(pred == y_test)
correct_set = np.intersect1d(correct_pred,gtruth_set)
print("Check which data have been correctly predicted...")
correct_set = np.random.choice(correct_set,500)
print("Taking only 500 data from the train data correctly predicted...")

xn = x_test[correct_set]
yn = y_test[correct_set]

print(xn.shape)
f = h5py.File(DATASET, 'w')
f.create_dataset("xn", shape=(len(xn),28,28), data=xn)
f.create_dataset("yn", data=yn)
f.close()
print("New data saved and ready to be used by DeepJanus")


